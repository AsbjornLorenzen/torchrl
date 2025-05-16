# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import time
import random

import hydra
import torch

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from utils.logging import init_logging, log_evaluation, log_training
from utils.utils import DoneTransform
from models.gnn_actor import GNNActor
from models.gnn_critic import GNNCritic


# Custom environment creator that randomizes the number of agents
def create_env(cfg, num_envs, agent_count=None, seed=None):
    """Create a VMAS environment with a specified or random number of agents.
    
    Args:
        cfg: Configuration object
        num_envs: Number of parallel environments
        agent_count: If specified, use this number of agents, otherwise randomize
        seed: Random seed for the environment
    """
    # If agent_count is not specified, randomly choose between 4 and 8
    if agent_count is None:
        agent_count = random.randint(4, 8)
    
    env = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=num_envs,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=seed if seed is not None else cfg.seed,
        n_agents=agent_count,  # Set the number of agents here
        # Scenario kwargs
        # **cfg.env.scenario,
    )
    
    # Add standard transformations
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    
    return env


def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))


class VariableAgentCollector:
    """A wrapper around SyncDataCollector that creates a new environment 
    with a random number of agents for each collection iteration."""
    
    def __init__(self, cfg, policy, frames_per_batch, total_frames):
        self.cfg = cfg
        self.policy = policy
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self.frames_collected = 0
        self.collector = None
        self.current_agent_count = None
        self._current_collector_iter = None # Add a field to store the iterator

        # Create initial collector
        self._create_new_collector()


    def _create_new_collector(self):
        """Create a new collector with a random number of agents."""
        # Close previous collector if it exists
        if self.collector is not None:
            self.collector.shutdown()

        # Random agent count for this iteration
        self.current_agent_count = random.randint(4, 8)
        torchrl_logger.info(f"Creating collector with {self.current_agent_count} agents")

        # Create environment with the random agent count
        env = create_env(
            self.cfg,
            num_envs=self.cfg.env.vmas_envs,
            agent_count=self.current_agent_count
        )

        # Create collector
        self.collector = SyncDataCollector(
            env,
            self.policy,
            device=self.cfg.env.device,
            storing_device=self.cfg.train.device,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames - self.frames_collected, # Adjust total_frames for the new collector
            postproc=DoneTransform(reward_key=env.reward_key, done_keys=env.done_keys),
        )
        # Get an iterator from the new collector
        self._current_collector_iter = iter(self.collector)
        
        # Return the environment for potential further use
        return env

    def __iter__(self):
        return self

    def __next__(self):
        try:
            # Try to get next batch from the current collector's iterator
            tensordict_data = next(self._current_collector_iter)
            self.frames_collected += tensordict_data.numel() # Assuming numel gives batch size * sequence length
            return tensordict_data
        except StopIteration:
            # If current collector is exhausted, create a new one
            if self.frames_collected < self.total_frames:
                env = self._create_new_collector() # This will also update self._current_collector_iter
                # Get the first batch from the new collector's iterator
                tensordict_data = next(self._current_collector_iter)
                self.frames_collected += tensordict_data.numel()
                return tensordict_data
            else:
                # All frames collected, stop iteration for VariableAgentCollector
                raise StopIteration

    def update_policy_weights_(self):
        """Update the policy weights in the collector."""
        if self.collector is not None:
            self.collector.update_policy_weights_()

    def shutdown(self):
        """Shutdown the collector."""
        if self.collector is not None:
            self.collector.shutdown()
    
    def _create_new_collector(self):
        """Create a new collector with a random number of agents."""
        # Close previous collector if it exists
        if self.collector is not None:
            self.collector.shutdown()
        
        # Random agent count for this iteration
        self.current_agent_count = random.randint(4, 8)
        torchrl_logger.info(f"Creating collector with {self.current_agent_count} agents")
        
        # Create environment with the random agent count
        env = create_env(
            self.cfg, 
            num_envs=self.cfg.env.vmas_envs, 
            agent_count=self.current_agent_count
        )
        
        # Create collector
        self.collector = SyncDataCollector(
            env,
            self.policy,
            device=self.cfg.env.device,
            storing_device=self.cfg.train.device,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames - self.frames_collected,
            postproc=DoneTransform(reward_key=env.reward_key, done_keys=env.done_keys),
        )
        
        # Return the environment for potential further use
        return env
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            # Try to get next batch from current collector
            tensordict_data = next(self.collector)
            self.frames_collected += tensordict_data.numel()
            return tensordict_data
        except StopIteration:
            # If current collector is exhausted, create a new one
            if self.frames_collected < self.total_frames:
                env = self._create_new_collector()
                tensordict_data = next(self.collector)
                self.frames_collected += tensordict_data.numel()
                return tensordict_data
            else:
                raise StopIteration
    
    def update_policy_weights_(self):
        """Update the policy weights in the collector."""
        if self.collector is not None:
            self.collector.update_policy_weights_()
    
    def shutdown(self):
        """Shutdown the collector."""
        if self.collector is not None:
            self.collector.shutdown()


@hydra.main(version_base="1.1", config_path="", config_name="mappo_pot")
def train(cfg: "DictConfig"):  # noqa: F821
    # Device
    cfg.train.device = "cpu" if not torch.cuda.device_count() else "cuda:0"
    cfg.env.device = cfg.train.device

    # Seeding
    torch.manual_seed(cfg.seed)

    # Sampling
    cfg.env.vmas_envs = cfg.collector.frames_per_batch // cfg.env.max_steps
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size = cfg.collector.frames_per_batch

    # Create a temporary env to get specifications
    # We're using 8 agents here, but the model should work with any number
    temp_env = create_env(
        cfg=cfg, 
        num_envs=1, 
        agent_count=8  # Use a reasonable number for initial spec creation
    )
    env_spec = temp_env.observation_spec
    action_spec = temp_env.action_spec
    reward_key = temp_env.reward_key
    action_key = temp_env.action_key
    done_keys = temp_env.done_keys
    
    # Create evaluation environment with a specific number of agents (e.g., 10)
    eval_agent_count = cfg.eval.get("eval_agent_count", 10)  # Default to 10 if not specified
    env_test = create_env(
        cfg=cfg,
        num_envs=cfg.eval.evaluation_episodes,
        agent_count=eval_agent_count,
        seed=cfg.seed,
    )
    
    # Close the temporary environment
    temp_env.close()

    print(f"Observation spec: {env_spec}")
    print(f"Action spec: {action_spec}")
    
    # GNN POLICY - should work with variable number of agents since it uses graph neural networks
    gnn_hidden_dim = cfg.model.get("gnn_hidden_dim", 128)
    gnn_layers = cfg.model.get("gnn_layers", 2)
    k_neighbours = cfg.model.get("k_neighbours", None)  # Default to fully connected if not specified
    pos_indices_list = cfg.model.get("pos_indices", [0, 2])  # Default to first 2
    pos_indices = slice(pos_indices_list[0], pos_indices_list[1])
    
    actor_net = nn.Sequential(
        GNNActor(
            n_agent_inputs=env_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * action_spec.shape[-1],
            gnn_hidden_dim=gnn_hidden_dim,
            n_gnn_layers=gnn_layers,
            activation_class=nn.Tanh,
            k_neighbours=k_neighbours,
            pos_indices=pos_indices,
            share_params=cfg.model.shared_parameters,
            device=cfg.train.device,
        ),
        NormalParamExtractor(),
    )
    
    policy_module = TensorDictModule(
        actor_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    
    # Since we're working with a variable number of agents, we need to create a full_action_spec_unbatched
    # that doesn't depend on a specific number of agents
    full_action_spec_unbatched = {"agents": {"action": action_spec.unbind(0)[-1].unbind(0)[-1]}}
    
    lowest_action = torch.zeros_like(
        full_action_spec_unbatched["agents"]["action"].space.low, 
        device=cfg.train.device
    )
    
    policy = ProbabilisticActor(
        module=policy_module,
        spec=full_action_spec_unbatched,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": lowest_action,
            "high": full_action_spec_unbatched["agents"]["action"].space.high,
        },
        return_log_prob=True,
    )

    critic_module = GNNCritic(
        n_agent_inputs=env_spec["agents", "observation"].shape[-1],
        gnn_hidden_dim=gnn_hidden_dim,  # Use same GNN params as actor (can be configured separately if needed)
        gnn_layers=gnn_layers,
        activation_class=nn.Tanh,
        k_neighbours=None,
        pos_indices=pos_indices,
        share_params=cfg.model.shared_parameters,  # Kept for consistency, GNN shares anyway
        device=cfg.train.device,  # Pass device object
    )
    
    value_module = ValueOperator(
        module=critic_module,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")]
    )

    # Create custom collector that changes agent count for each iteration
    collector = VariableAgentCollector(
        cfg=cfg,
        policy=policy,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.train.minibatch_size,
    )

    # Loss
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value_module,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coef=cfg.loss.entropy_eps,
        normalize_advantage=False,
    )
    
    loss_module.set_keys(
        reward=reward_key,
        action=action_key,
        done=("agents", "done"),
        terminated=("agents", "terminated"),
        value=("agents", "state_value")
    )
    
    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=cfg.loss.gamma, lmbda=cfg.loss.lmbda
    )
    
    optim = torch.optim.Adam(loss_module.parameters(), cfg.train.lr)

    # Logging
    if cfg.logger.backend:
        model_name = (
            ("Het" if not cfg.model.shared_parameters else "")
            + ("MA" if cfg.model.centralised_critic else "I")
            + "PPO"
            + f"_VariableAgents"
        )
        logger = init_logging(cfg, model_name)

    total_time = 0
    total_frames = 0
    sampling_start = time.time()
    
    for i, tensordict_data in enumerate(collector):
        agent_count = collector.current_agent_count
        torchrl_logger.info(f"\nIteration {i}, Agent count: {agent_count}")

        sampling_time = time.time() - sampling_start

        with torch.no_grad():
            loss_module.value_estimator(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )
            
        current_frames = tensordict_data.numel()
        total_frames += current_frames
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        training_tds = []
        training_start = time.time()
        
        for _ in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                training_tds.append(loss_vals.detach())

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.train.max_grad_norm
                )
                training_tds[-1].set("grad_norm", total_norm.mean())

                optim.step()
                optim.zero_grad()

        # Update policy in the collector
        collector.update_policy_weights_()

        # Clear replay buffer for the next iteration with potentially different agent count
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.device),
            sampler=SamplerWithoutReplacement(),
            batch_size=cfg.train.minibatch_size,
        )

        training_time = time.time() - training_start

        iteration_time = sampling_time + training_time
        total_time += iteration_time
        training_tds = torch.stack(training_tds)

        # Logging
        if cfg.logger.backend:
            log_training(
                logger,
                training_tds,
                tensordict_data,
                sampling_time,
                training_time,
                total_time,
                i,
                current_frames,
                total_frames,
                step=i,
            )
            # Log the agent count for this iteration
            logger.experiment.log({"agent_count": agent_count}, step=i)

        # Evaluation
        if (
            cfg.eval.evaluation_episodes > 0
            and i % cfg.eval.evaluation_interval == 0
            and cfg.logger.backend
        ):
            evaluation_start = time.time()
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                env_test.frames = []
                rollouts = env_test.rollout(
                    max_steps=cfg.env.max_steps,
                    policy=policy,
                    callback=rendering_callback,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                    # We are running vectorized evaluation we do not want it to stop when just one env is done
                )

                evaluation_time = time.time() - evaluation_start

                log_evaluation(logger, rollouts, env_test, evaluation_time, step=i)
                # Log the agent count used for evaluation
                logger.experiment.log({"eval_agent_count": eval_agent_count}, step=i)

        if cfg.logger.backend == "wandb":
            logger.experiment.log({}, commit=True)
            
        sampling_start = time.time()
        
    collector.shutdown()
    if not env_test.is_closed:
        env_test.close()


if __name__ == "__main__":
    train()
