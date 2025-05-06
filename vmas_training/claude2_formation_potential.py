from __future__ import annotations

import time
import random  # Added for randomizing agent numbers
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
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from utils.logging import init_logging, log_evaluation, log_training
from utils.utils import DoneTransform
from models.gnn_actor_variable import GNNActorVariable
from models.gnn_critic import GNNCritic

# New class to handle variable number of agents
class VariableAgentsWrapper(TransformedEnv):
    def __init__(self, env, min_agents=4, max_agents=8, fixed_num_agents=None):
        super().__init__(env)
        self.base_env = env
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.fixed_num_agents = fixed_num_agents
        
        # Maximum number of agents we'll ever use (for evaluation too)
        self.max_possible_agents = max(max_agents, fixed_num_agents or 0)
        
        # Add agent_mask to the observation space
        self._update_specs()
    
    def _update_specs(self):
        # We add an agent_mask field to the observation space
        self.observation_spec = self.base_env.observation_spec.clone()
        # Add a boolean mask to indicate which agents are active
        self.observation_spec.set(
            ("agents", "agent_mask"), 
            torch.zeros(
                (*self.base_env.batch_size, self.max_possible_agents, 1), 
                dtype=torch.bool, 
                device=self.base_env.device
            )
        )
    
    def _reset(self, tensordict=None, **kwargs):
        # Reset the underlying environment
        td = self.base_env._reset(tensordict, **kwargs)
        
        # Choose number of active agents for this episode
        if self.fixed_num_agents is not None:
            num_active_agents = self.fixed_num_agents
        else:
            num_active_agents = random.randint(self.min_agents, self.max_agents)
        
        # Create a mask for active agents
        agent_mask = torch.zeros(
            (*self.base_env.batch_size, self.max_possible_agents, 1), 
            dtype=torch.bool, 
            device=self.base_env.device
        )
        
        # Set first num_active_agents to True (active)
        agent_mask[..., :num_active_agents, :] = True
        
        # Add the agent mask to the tensordict
        td.set(("agents", "agent_mask"), agent_mask)
        
        # Zero out observations for inactive agents
        obs = td.get(("agents", "observation"))
        obs = obs * agent_mask.expand_as(obs[:, :, :1])  # Use broadcasting to mask observations
        td.set(("agents", "observation"), obs)
        
        return td
    
    def _step(self, tensordict):
        # Get the agent mask
        agent_mask = tensordict.get(("agents", "agent_mask"))
        
        # Mask the actions for inactive agents (set to zero)
        actions = tensordict.get(self.base_env.action_key)
        masked_actions = actions * agent_mask.expand_as(actions[:, :, :1])
        tensordict.set(self.base_env.action_key, masked_actions)

        # TODO: I have to fix the forward() method such that it receives the agent mask

        # TODO: I have to fix the forward() method such that it receives the agent mask
        
        # Forward to the base environment
        td = self.base_env._step(tensordict)
        
        # Propagate the mask to the next state
        td.set(("agents", "agent_mask"), agent_mask)
        
        # Mask observations and rewards for inactive agents
        obs = td.get(("agents", "observation"))
        obs = obs * agent_mask.expand_as(obs[:, :, :1])
        td.set(("agents", "observation"), obs)
        
        rewards = td.get(self.base_env.reward_key)
        rewards = rewards * agent_mask
        td.set(self.base_env.reward_key, rewards)
        
        return td


def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))


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

    # Create env and env_test
    base_env = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.env.vmas_envs,
        n_agents=cfg.env.get("max_agents", 8),  # Use max agents as the base
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        # Scenario kwargs
        **cfg.env.scenario,
    )
    
    # Wrap with variable agents handler
    env = VariableAgentsWrapper(
        base_env,
        min_agents=cfg.env.get("min_agents", 4),
        max_agents=cfg.env.get("max_agents", 8),
        fixed_num_agents=None  # Random during training
    )
    
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )

    # For evaluation, we'll use a fixed number of agents
    base_env_test = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.eval.evaluation_episodes,
        n_agents=cfg.eval.get("num_agents", 10),  # Default to 10 for evaluation
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        # Scenario kwargs
        **cfg.env.scenario,
    )
    
    # Wrap with fixed agents for evaluation
    env_test = VariableAgentsWrapper(
        base_env_test,
        min_agents=cfg.env.get("min_agents", 4),
        max_agents=cfg.env.get("max_agents", 8),
        fixed_num_agents=cfg.eval.get("num_agents", 10)  # Fixed number for eval
    )

    print(f"In torchrl, the given env action spec is {env.action_spec}")
    
    # GNN POLICY - Modified to handle agent_mask
    gnn_hidden_dim = cfg.model.get("gnn_hidden_dim", 128)
    gnn_layers = cfg.model.get("gnn_layers", 2)
    k_neighbours = cfg.model.get("k_neighbours", None) # Default to fully connected if not specified
    pos_indices_list = cfg.model.get("pos_indices", [0, 2]) # Default to first 2
    pos_indices = slice(pos_indices_list[0], pos_indices_list[1])
    
    # Modify GNN actor to handle agent_mask
    actor_net = nn.Sequential(
        GNNActorVariable(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * env.action_spec.shape[-1],
            gnn_hidden_dim=gnn_hidden_dim,
            n_gnn_layers=gnn_layers,
            activation_class=nn.Tanh,
            k_neighbours=k_neighbours,
            pos_indices=pos_indices,
            share_params=cfg.model.shared_parameters,
            device=cfg.train.device,
            # Add agent_mask support in your GNN implementation
        ),
        NormalParamExtractor(),
    )
    
    policy_module = TensorDictModule(
        actor_net,
        in_keys=[("agents", "observation"), ("agents", "agent_mask")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    
    lowest_action = torch.zeros_like(env.full_action_spec_unbatched[("agents", "action")].space.low, device=cfg.train.device)
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.full_action_spec_unbatched,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": lowest_action,
            "high": env.full_action_spec_unbatched[("agents", "action")].space.high,
        },
        return_log_prob=True,
    )

    # Modify GNN critic to handle agent_mask
    critic_module = GNNCritic(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_layers=gnn_layers,
        activation_class=nn.Tanh,
        k_neighbours=None,
        pos_indices=pos_indices,
        share_params=cfg.model.shared_parameters,
        device=cfg.train.device,
        # Add agent_mask support in your GNN implementation
    )
    
    value_module = ValueOperator(
        module=critic_module,
        in_keys=[("agents", "observation"), ("agents", "agent_mask")],
        out_keys=[("agents", "state_value")]
    )

    collector = SyncDataCollector(
        env,
        policy,
        device=cfg.env.device,
        storing_device=cfg.train.device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        postproc=DoneTransform(reward_key=env.reward_key, done_keys=env.done_keys),
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
        reward=env.reward_key,
        action=env.action_key,
        done=("agents", "done"),
        terminated=("agents", "terminated"),
        # Add agent_mask to loss module keys
        agent_mask=("agents", "agent_mask"),
    )
    
    # Custom value estimator that respects agent_mask
    loss_module.make_value_estimator(
        ValueEstimators.GAE, 
        gamma=cfg.loss.gamma, 
        lmbda=cfg.loss.lmbda
    )
    
    optim = torch.optim.Adam(loss_module.parameters(), cfg.train.lr)

    # Logging
    if cfg.logger.backend:
        model_name = (
            ("Het" if not cfg.model.shared_parameters else "")
            + ("MA" if cfg.model.centralised_critic else "I")
            + "PPO"
            + f"_VarAgents{cfg.env.get('min_agents', 4)}-{cfg.env.get('max_agents', 8)}"
        )
        logger = init_logging(cfg, model_name)

    total_time = 0
    total_frames = 0
    sampling_start = time.time()
    for i, tensordict_data in enumerate(collector):
        torchrl_logger.info(f"\nIteration {i}")

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

        collector.update_policy_weights_()

        training_time = time.time() - training_start

        iteration_time = sampling_time + training_time
        total_time += iteration_time
        training_tds = torch.stack(training_tds)

        # More logs
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
                )

                evaluation_time = time.time() - evaluation_start

                log_evaluation(logger, rollouts, env_test, evaluation_time, step=i)

        if cfg.logger.backend == "wandb":
            logger.experiment.log({}, commit=True)
        sampling_start = time.time()
    collector.shutdown()
    if not env.is_closed:
        env.close()
    if not env_test.is_closed:
        env_test.close()


if __name__ == "__main__":
    train()
