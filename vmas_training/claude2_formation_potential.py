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
from models.gnn_critic_variable import GNNCriticVariable

# New class to handle variable number of agents
class DynamicAgentsManager:
    """
    Manages environments with a dynamically changing number of agents.
    Creates a new environment instance for each different agent count.
    """
    def __init__(
        self,
        scenario_name,
        num_envs,
        min_agents,
        max_agents,
        continuous_actions=True,
        max_steps=200,
        device="cpu",
        seed=None,
        **scenario_kwargs
    ):
        self.scenario_name = scenario_name
        self.num_envs = num_envs
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.continuous_actions = continuous_actions
        self.max_steps = max_steps
        self.device = device
        self.seed = seed
        self.scenario_kwargs = scenario_kwargs
        
        # Cache for created environments (key: num_agents)
        self.env_cache = {}
        
        # Current active environment
        self.current_num_agents = None
        self.current_env = None
        
        # Create initial environment with default agent count
        self.set_num_agents(max_agents)
    
    def set_num_agents(self, num_agents):
        """
        Set the active number of agents by either retrieving a cached environment
        or creating a new one with the specified number of agents.
        """
        if num_agents < self.min_agents or num_agents > self.max_agents:
            raise ValueError(f"Number of agents must be between {self.min_agents} and {self.max_agents}")
        
        self.current_num_agents = num_agents
        
        # If we already have an environment with this agent count, use it
        if num_agents in self.env_cache:
            print(f"Using cached environment with {num_agents} agents")
            self.current_env = self.env_cache[num_agents]
            return self.current_env
        
        # Otherwise create a new environment
        print(f"Creating new environment with {num_agents} agents")
        base_env = VmasEnv(
            scenario=self.scenario_name,
            num_envs=self.num_envs,
            n_agents=num_agents,  # Exactly the number we need
            continuous_actions=self.continuous_actions,
            max_steps=self.max_steps,
            device=self.device,
            seed=self.seed,
            **self.scenario_kwargs
        )
        
        # Apply standard transformations
        env = TransformedEnv(
            base_env,
            RewardSum(in_keys=[base_env.reward_key], out_keys=[("agents", "episode_reward")]),
        )
        
        # Cache for future use
        self.env_cache[num_agents] = env
        self.current_env = env
        
        return env
    
    def get_current_env(self):
        """Returns the currently active environment"""
        return self.current_env
    
    def get_current_num_agents(self):
        """Returns the current number of agents"""
        return self.current_num_agents
    
    def random_num_agents(self):
        """Randomly select a number of agents and set the environment accordingly"""
        num_agents = random.randint(self.min_agents, self.max_agents)
        return self.set_num_agents(num_agents)
    
    def get_specs(self):
        """Get the current environment's specifications"""
        return {
            "observation_spec": self.current_env.observation_spec,
            "action_spec": self.current_env.action_spec,
            "full_action_spec_unbatched": self.current_env.full_action_spec_unbatched,
            "reward_key": self.current_env.reward_key,
            "done_keys": self.current_env.done_keys,
        }
    
    def close_all(self):
        """Close all environments in the cache"""
        for env in self.env_cache.values():
            if not env.is_closed:
                env.close()


def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))


@hydra.main(version_base="1.1", config_path="", config_name="mappo_gnn")
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

    # Create dynamic environment manager for training
    env_manager = DynamicAgentsManager(
        scenario_name=cfg.env.scenario_name,
        num_envs=cfg.env.vmas_envs,
        min_agents=cfg.env.get("min_agents", 4),
        max_agents=cfg.env.get("max_agents", 8),
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        # Scenario kwargs can be added here
        # **cfg.env.scenario,
    )

    # Start with a random number of agents
    env = env_manager.random_num_agents()

    # Create evaluation environment with fixed number of agents
    eval_num_agents = cfg.eval.get("num_agents", 8)
    env_test = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.eval.evaluation_episodes,
        n_agents=eval_num_agents,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        # Scenario kwargs
        # **cfg.env.scenario,
    )
    
    # Apply standard transformations to eval env
    env_test = TransformedEnv(
        env_test,
        RewardSum(in_keys=[env_test.reward_key], out_keys=[("agents", "episode_reward")]),
    )

    # Get current env specs
    env_specs = env_manager.get_specs()
    

    print(f"In torchrl, the given env spec is {env_specs}")
    
    # GNN POLICY - Modified to handle agent_mask
    gnn_hidden_dim = cfg.model.get("gnn_hidden_dim", 128)
    gnn_layers = cfg.model.get("gnn_layers", 2)
    k_neighbours = cfg.model.get("k_neighbours", None) # Default to fully connected if not specified
    pos_indices_list = cfg.model.get("pos_indices", [0, 2]) # Default to first 2
    pos_indices = slice(pos_indices_list[0], pos_indices_list[1])

    # Create actor network
    actor_net = nn.Sequential(
        GNNActorVariable(
            n_agent_inputs=env_specs["observation_spec"]["agents", "observation"].shape[-1],
            n_agent_outputs=2 * env_specs["action_spec"].shape[-1],
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
    
    lowest_action = torch.zeros_like(
        env_specs["full_action_spec_unbatched"][("agents", "action")].space.low, 
        device=cfg.train.device
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env_specs["full_action_spec_unbatched"],
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env_specs["reward_key"].replace("reward", "action")],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": lowest_action,
            "high": env_specs["full_action_spec_unbatched"][("agents", "action")].space.high,
        },
        return_log_prob=True,
    )
    

    # Modify GNN critic to handle agent_mask
    critic_module = GNNCriticVariable(
        n_agent_inputs=env_specs["observation_spec"]["agents", "observation"].shape[-1],
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
        in_keys=[("agents", "observation")], # , ("agents", "agent_mask")
        out_keys=[("agents", "state_value")]
    )

    collector = None

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
        reward=env_specs["reward_key"],
        action=env_specs["reward_key"].replace("reward", "action"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
        value=("agents", "state_value")
    )

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

    initial_num_agents = random.randint(
        cfg.env.get("min_agents", 4), cfg.env.get("max_agents", 8)
    )

    total_time = 0
    total_frames = 0


    # Main training loop
    for i in range(cfg.collector.n_iters):
        # Set a new random number of agents for this iteration
        if i > 0:  # Skip the first iteration as we already initialized with random agents
            env = env_manager.random_num_agents()
        
        current_num_agents = env_manager.get_current_num_agents()
        torchrl_logger.info(f"\nIteration {i} (using {current_num_agents} agents)")
        
        # Create a new collector for the current environment
        if collector is not None:
            collector.shutdown()
            
        collector = SyncDataCollector(
            env,
            policy,
            device=cfg.env.device,
            storing_device=cfg.train.device,
            frames_per_batch=cfg.collector.frames_per_batch,
            total_frames=cfg.collector.frames_per_batch,  # Collect only one batch
            postproc=DoneTransform(reward_key=env_specs["reward_key"], done_keys=env_specs["done_keys"]),
        )
        
        # Collect data
        sampling_start = time.time()
        tensordict_data = next(iter(collector))
        sampling_time = time.time() - sampling_start

        # Process collected data
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

        # Training
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

        training_time = time.time() - training_start
        iteration_time = sampling_time + training_time
        total_time += iteration_time
        training_tds = torch.stack(training_tds)

        # Update the policy weights in collector
        collector.update_policy_weights_()

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
                    callback=rendering_callback if 'rendering_callback' in globals() else None,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                )

                evaluation_time = time.time() - evaluation_start
                log_evaluation(logger, rollouts, env_test, evaluation_time, step=i)

        if cfg.logger.backend == "wandb":
            logger.experiment.log({}, commit=True)
    
    # Clean up
    if collector is not None:
        collector.shutdown()
    env_manager.close_all()
    if not env_test.is_closed:
        env_test.close()


if __name__ == "__main__":
    train()
