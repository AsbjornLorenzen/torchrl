# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import time

import hydra
import torch

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.tensor_specs import DiscreteTensorSpec
from torchrl.data.tensor_specs import (
    UnboundedContinuousTensorSpec,
    BoundedContinuous,
    CompositeSpec,
    Categorical,
    Composite# For type checking if needed
)
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv, EnvBase
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type, MarlGroupMapType
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from utils.logging import init_logging, log_evaluation, log_training
from utils.utils import DoneTransform
from models.gnn_actor import GNNActor
from models.gnn_critic import GNNCritic
import random
import torch
from tensordict import TensorDict


def rendering_callback(env, td):
    # Ensure env.frames exists, might need initialization on env_test
    if not hasattr(env, 'frames'):
        env.frames = []
    # VMAS render call might need to be adapted if `env` is the TransformedEnv
    # Accessing the base VMAS env for rendering:
    base_env = env
    while hasattr(base_env, "env") and not isinstance(base_env, VmasEnv): # Unwrap
        base_env = base_env.env
    if isinstance(base_env, VmasEnv):
        env.frames.append(base_env.render(mode="rgb_array", agent_index_focus=None))
    elif hasattr(env, '_rendering_render'): # TorchRL's common render method
         env.frames.append(env._rendering_render(mode="rgb_array", **{}))



@hydra.main(version_base="1.1", config_path="", config_name="mappo_pot")
def train(cfg: "DictConfig"):  # noqa: F821
    # --- Configuration ---
    MIN_AGENTS_TRAIN = cfg.train.get("min_agents_train", 4) # Use hydra config or default
    MAX_AGENTS_TRAIN = cfg.env.scenario.n_agents # This n_agents in scenario config should be MAX for training
    EVAL_AGENTS = cfg.eval.get("eval_agents", MAX_AGENTS_TRAIN) # Eval agents from config or default to MAX

    # Device
    cfg.train.device = "cpu" if not torch.cuda.is_available() else "cuda:0" # Updated cuda check
    cfg.env.device = cfg.train.device

    # Seeding
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed) # Also seed python's random for agent number selection

    # Sampling
    cfg.env.vmas_envs = cfg.collector.frames_per_batch // cfg.env.max_steps
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size = cfg.collector.frames_per_batch


    # --- Create Training Environment with VariableAgentWrapper ---
    # 1. Base VmasEnv is initialized with MAX_AGENTS_TRAIN
    #    All its specs will be based on this maximum number.
    base_vmas_env = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.env.vmas_envs,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        group_map=MarlGroupMapType.ALL_IN_ONE_GROUP,
        **cfg.env.scenario, # Scenario kwargs might also contain n_agents, ensure consistency
    )

    # 2. Wrap with VariableAgentWrapper
    #    This wrapper will handle the dynamic number of agents.
    #    Its n_agents property will report MAX_AGENTS_TRAIN for spec compatibility.
    variable_agent_env = VariableAgentWrapper(
        base_vmas_env,
        max_agents=MAX_AGENTS_TRAIN,
        min_agents=MIN_AGENTS_TRAIN
    )

    # 3. Apply other transformations (e.g., RewardSum)
    #    Ensure keys used by transforms are correct for the MARL "agents" group.
    #    variable_agent_env.reward_key is ("agents", "reward")
    #    The output key for RewardSum should also be under "agents" group if it's per-agent episode reward.
    env = TransformedEnv(
        variable_agent_env,
        RewardSum(
            in_keys=[variable_agent_env.reward_key],
            out_keys=[(variable_agent_env.group_name, "episode_reward")] # e.g. ("agents", "episode_reward")
        ),
        # Optional: Add DoneTransform if needed, but be careful with its effect on done keys
        # DoneTransform(reward_key=variable_agent_env.reward_key, done_keys=variable_agent_env.done_keys)
    )

    # --- Create Test Environment ---
    # This environment uses a fixed number of agents for evaluation.
    eval_scenario_cfg = cfg.env.scenario.copy() # Use omegaconf.OmegaConf.to_container if it's a DictConfig
    if hasattr(eval_scenario_cfg, 'n_agents'): # Ensure we can modify n_agents if it's part of scenario
        eval_scenario_cfg.n_agents = EVAL_AGENTS
    else: # If n_agents is a direct kwarg to VmasEnv not in scenario dict
        pass # VmasEnv will take EVAL_AGENTS from its n_agents param

    env_test = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.eval.evaluation_episodes,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed + 1, # Different seed for test env
        group_map=MarlGroupMapType.ALL_IN_ONE_GROUP,
        **eval_scenario_cfg,
    )
    env_test.frames = [] # for rendering_callback

    torchrl_logger.info(f"Training Env action key: {env.action_key}, reward key: {env.reward_key}, done keys: {env.done_keys}")
    torchrl_logger.info(f"Training Env observation spec: {env.observation_spec}")
    torchrl_logger.info(f"Training Env action spec: {env.action_spec}")


    # Policy and Critic setup
    # The observation_spec from `env` (which is VariableAgentWrapper) includes ("agents", "active_mask").
    # The GNNs need to process ("agents", "observation"). The masking of observations for inactive
    # agents is handled inside the VariableAgentWrapper.
    # Action spec from `env` is for MAX_AGENTS_TRAIN.
    
    # Get the spec for a single agent's action part for NormalParamExtractor output size
    # env.action_spec is CompositeSpec({"agents": CompositeSpec({"action": <single_action_spec>}, shape=(MAX_AGENTS,))})
    # So, env.action_spec[env.action_key] should give the <single_action_spec>
    action_spec_for_group_agents = None
    # Check if env.action_spec is a CompositeSpec and contains the group_name key
    # This would be the case if spec "flattening" (due to deprecation warning) does not occur.
    if isinstance(env.action_spec, CompositeSpec) and env.group_name in env.action_spec.keys(include_nested=False):
        action_spec_for_group_agents = env.action_spec[env.group_name]
    # Check if env.action_spec is already the BoundedContinuous spec for the group
    # This is the current case based on your logs due to the "leaf-returning" behavior.
    elif isinstance(env.action_spec, (BoundedContinuous, UnboundedContinuousTensorSpec)) and \
         hasattr(env.action_spec, "shape") and len(env.action_spec.shape) >= 2: # Expect at least (N_agents, Action_dim)
        torchrl_logger.info(
            f"env.action_spec is directly a ContinuousTensorSpec (shape: {env.action_spec.shape}). "
            f"Assuming it's the action spec for the group '{env.group_name}' due to spec flattening."
        )
        action_spec_for_group_agents = env.action_spec
    else:
        raise ValueError(
            f"Unexpected structure for env.action_spec: {env.action_spec} (type: {type(env.action_spec)}). "
            f"Cannot determine action_spec_for_group_agents for group '{env.group_name}'."
        )

    # At this point, action_spec_for_group_agents should be a continuous tensor spec like
    # BoundedContinuous(shape=[batch_size, num_agents, action_dim_per_agent]) or
    # BoundedContinuous(shape=[num_agents, action_dim_per_agent]) if batch_size is squeezed.
    if not (isinstance(action_spec_for_group_agents, (BoundedContinuous, UnboundedContinuousTensorSpec)) and \
            hasattr(action_spec_for_group_agents, "shape") and \
            len(action_spec_for_group_agents.shape) >= 2): # Needs at least agent dim and action dim
        raise ValueError(
            f"action_spec_for_group_agents is not a ContinuousTensorSpec with at least 2 dims. "
            f"Got: {action_spec_for_group_agents} (type: {type(action_spec_for_group_agents)})"
        )

    # Extract a single agent's spec by indexing.
    # This will have shape [action_dim_per_agent] and the correct low/high bounds.
    if len(action_spec_for_group_agents.shape) == 3: # (Batch, N_agent, Action_dim)
        single_agent_action_spec = action_spec_for_group_agents[0, 0]
    elif len(action_spec_for_group_agents.shape) == 2: # (N_agent, Action_dim)
        single_agent_action_spec = action_spec_for_group_agents[0]
    else:
        raise ValueError(
            f"action_spec_for_group_agents has an unexpected number of dimensions: {action_spec_for_group_agents.shape}. "
            f"Expected 2 or 3 dimensions (Agent, Action_dim) or (Batch, Agent, Action_dim)."
        )

    if not hasattr(single_agent_action_spec, "shape"): # Should be guaranteed by above
        raise TypeError(f"single_agent_action_spec does not have a 'shape' attribute. Spec: {single_agent_action_spec}")

    action_dim = single_agent_action_spec.shape[-1]
    gnn_output_dim_per_agent = 2 * action_dim # For loc and scale of Normal distribution

    # Observation spec for GNN input
    # env.observation_spec["agents","observation"] gives the single-agent observation spec
    single_agent_obs_spec = env.observation_spec[env.group_name]["observation"]
    obs_dim_per_agent = single_agent_obs_spec.shape[-1]


    gnn_hidden_dim = cfg.model.get("gnn_hidden_dim", 128)
    gnn_layers = cfg.model.get("gnn_layers", 2)
    k_neighbours = cfg.model.get("k_neighbours", None)
    pos_indices_list = cfg.model.get("pos_indices", [0, 2])
    pos_indices = slice(pos_indices_list[0], pos_indices_list[1])

    actor_net = nn.Sequential(
        GNNActor(
            n_agent_inputs=obs_dim_per_agent,
            n_agent_outputs=gnn_output_dim_per_agent, # mu and sigma for each action dimension
            gnn_hidden_dim=gnn_hidden_dim,
            n_gnn_layers=gnn_layers,
            activation_class=nn.Tanh,
            k_neighbours=k_neighbours,
            pos_indices=pos_indices,
            share_params=cfg.model.shared_parameters,
            device=cfg.train.device,
        ),
        NormalParamExtractor(), # Extracts loc and scale
    )
    policy_module = TensorDictModule(
        actor_net,
        # Input to GNNActor is ("agents", "observation")
        in_keys=[(env.group_name, "observation")],
        # Output from NormalParamExtractor will be loc and scale
        out_keys=[(env.group_name, "loc"), (env.group_name, "scale")],
    )

    # Policy uses the full action spec from the environment (which is for MAX_AGENTS_TRAIN)
    # env.action_spec should be CompositeSpec(agents: CompositeSpec(action: UnboundedSpec(shape=(act_dim,)), shape=(N,)))
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec.clone(), # Full spec for all agents
        in_keys=[(env.group_name, "loc"), (env.group_name, "scale")],
        out_keys=[env.action_key], # e.g. ("agents", "action")
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": single_agent_action_spec.space.low,  # Use single agent bounds
            "high": single_agent_action_spec.space.high,
            "tanh_loc": False # common practice for TanhNormal
        },
        return_log_prob=True,
    )

    critic_net = GNNCritic( # Renamed from critic_module to critic_net for clarity
        n_agent_inputs=obs_dim_per_agent,
        # Output is a single value per agent if doing per-agent value estimation
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_layers=gnn_layers,
        activation_class=nn.Tanh,
        k_neighbours=k_neighbours, # Use consistent k_neighbours or configure separately
        pos_indices=pos_indices,
        share_params=cfg.model.shared_parameters,
        device=cfg.train.device,
    )
    value_module = ValueOperator(
        module=critic_net,
        in_keys=[(env.group_name, "observation")],
        out_keys=[(env.group_name, "state_value")] # e.g. ("agents", "state_value")
    )

    collector = SyncDataCollector(
        env, # This is the TransformedEnv wrapping VariableAgentWrapper
        policy,
        device=cfg.env.device, # Device for env interaction
        storing_device=cfg.train.device, # Device for storing data in buffer
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        # If DoneTransform is used, ensure its output keys match what loss expects.
        # postproc=DoneTransform(reward_key=env.reward_key, done_keys=env.done_keys), # Keep if your DoneTransform is robust
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.train.minibatch_size,
    )

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value_module,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coef=cfg.loss.entropy_eps,
        normalize_advantage=False, # Consider setting to True if advantages vary widely
    )
    # --- Critical: Set keys for the loss module ---
    # These keys must match what's in the TensorDict from the collector/replay_buffer.
    # `env.reward_key` and `env.action_key` are from the wrapper: ("agents", "reward/action")
    # `done` and `terminated` keys depend on VmasEnv output and any Transforms like DoneTransform.
    # If VMAS `done` is global (e.g., "done"), and no transform makes it per-agent,
    # then use "done". If `DoneTransform` or similar creates per-agent done/terminated under
    # the "agents" group, then use ("agents", "done").
    # The original code used ("agents", "done"). This implies an expectation of per-agent done signals.
    # For this refactoring, we'll keep it, assuming it's intended.
    # The VariableAgentWrapper itself does not make global 'done' per-agent.
    # It only masks rewards and observations.
    loss_module.set_keys(
        reward=env.reward_key,  # ("agents", "reward")
        action=env.action_key,  # ("agents", "action")
        done=(env.group_name, "done"), # Assumes per-agent "done" exists, e.g. ("agents", "done")
        terminated=(env.group_name, "terminated"), # Assumes per-agent "terminated" exists
        value=(env.group_name, "state_value"), # From value_module: ("agents", "state_value")
        sample_log_prob=(env.group_name, "sample_log_prob")
        # Add log_prob if policy doesn't automatically write it to standard key in td.
        # PPO loss usually expects "sample_log_prob" or similar. ProbabilisticActor adds "sample_log_prob".
        # Check if the key is ("agents", "sample_log_prob") or just "sample_log_prob" if not grouped by agent.
        # ProbabilisticActor's out_keys includes env.action_key, and if return_log_prob=True,
        # it adds "sample_log_prob". This is typically at the root.
        # If log_prob needs to be per agent for the loss, ensure policy module outputs it per agent.
        # For now, assume "sample_log_prob" is correctly handled or PPO uses policy directly.
    )
    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=cfg.loss.gamma, lmbda=cfg.loss.lmbda
    )
    optim = torch.optim.Adam(loss_module.parameters(), cfg.train.lr)

    # Logging
    logger = None
    if cfg.logger.backend:
        model_name = (
            ("GNNActor" if isinstance(actor_net[0], GNNActor) else "Het") # Simplified name
            + ("MA" if cfg.model.centralised_critic else "I") # Assuming centralised_critic means GNN critic uses all obs
            + "PPO"
        )
        logger = init_logging(cfg, model_name)

    total_time = 0
    total_frames = 0
    sampling_start_time = time.time()

    pbar = tqdm.tqdm(total=cfg.collector.total_frames) if cfg.train.get("progress_bar", True) else None

    for i, tensordict_data in enumerate(collector):
        pbar.update(tensordict_data.numel()) if pbar else None
        
        # Determine active agents for the *next* rollout.
        # This is done *before* the data processing for the current batch,
        # as `set_active_agents` will affect the *next* `_reset` call by the collector.
        next_n_agents = random.randint(MIN_AGENTS_TRAIN, MAX_AGENTS_TRAIN)
        
        # Access the VariableAgentWrapper instance.
        # `collector.env` is the TransformedEnv. `collector.env.env` is VariableAgentWrapper.
        current_env_for_wrapper = collector.env 
        actual_wrapper = None
        while hasattr(current_env_for_wrapper, "env"):
            if isinstance(current_env_for_wrapper, VariableAgentWrapper):
                actual_wrapper = current_env_for_wrapper
                break
            current_env_for_wrapper = current_env_for_wrapper.env
        if isinstance(current_env_for_wrapper, VariableAgentWrapper): # Check if the loop didn't find it but the base is it
             actual_wrapper = current_env_for_wrapper


        if actual_wrapper:
            actual_wrapper.set_active_agents(next_n_agents)
            current_num_active_agents = actual_wrapper.current_n_agents
        else:
            torchrl_logger.error("Could not find VariableAgentWrapper in the environment chain to set agent count.")
            current_num_active_agents = "N/A (Wrapper not found)"

        log_string = f"Iter {i}, {current_num_active_agents} agents for next rollout. "
        sampling_time = time.time() - sampling_start_time

        # Compute GAE advantages
        with torch.no_grad():
            # Ensure value_estimator gets the correct keys. It uses keys set in loss_module.
            loss_module.value_estimator(
                tensordict_data,
                params=loss_module.critic_network_params, # Pass learnable parameters
                target_params=loss_module.target_critic_network_params, # Pass target parameters
            )
        
        # `tensordict_data` now contains `("agents", "advantage")` and `("agents", "value_target")`
        # It also contains `("agents", "active_mask")` from the wrapper (for the current state).
        # And `("next", "agents", "active_mask")` for the next state.

        current_frames_collected = tensordict_data.numel() # Number of steps x num_envs
        total_frames += current_frames_collected
        
        # Add to replay buffer. Data is already on cfg.train.device due to collector's storing_device.
        # `tensordict_data` has shape [vmas_envs, max_steps]. Reshape to [-1] for buffer.
        replay_buffer.extend(tensordict_data.reshape(-1))

        training_iter_start_time = time.time()
        cumulative_loss_td = None

        for epoch in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                subdata = replay_buffer.sample() # Samples a batch of transitions

                # Get the active_mask for this specific batch of data.
                # This mask corresponds to the state `s_t`, not `s_{t+1}`.
                # Shape: [minibatch_size, max_agents, 1]
                active_mask_batch = subdata.get(variable_agent_env.active_mask_key)
                if active_mask_batch is None:
                    raise ValueError(f"Key {variable_agent_env.active_mask_key} not found in sampled subdata.")

                # --- Masking for Loss Calculation ---
                # ClipPPOLoss might not internally use the "active_mask".
                # We need to ensure that inactive agents do not contribute to the loss.
                # This can be done by:
                # 1. Masking advantages and value_targets (already done by wrapper for rewards, GAE uses these).
                #    Advantages for inactive agents should be zero if their rewards were zero and values are zero/ignored.
                #    The `loss_module.value_estimator` calculates GAE. If rewards for inactive agents
                #    were 0 and their value estimates (from value_module) are also 0 (due to zeroed obs),
                #    then advantage and value_target should already be 0 for them.
                #    Explicit masking here is a safeguard or if value network doesn't output exact zero.
                
                # Ensure advantage and value_target are shaped [minibatch_size, max_agents, 1 or feature_dim]
                # Mask is [minibatch_size, max_agents, 1]
                adv_key = (env.group_name, ValueEstimators.GAE.value) # Default is "advantage"
                val_target_key = (env.group_name, ValueEstimators.GAE.value_target) # Default is "value_target"

                if adv_key in subdata.keys(True,True):
                    subdata[adv_key] = subdata[adv_key] * active_mask_batch
                if val_target_key in subdata.keys(True,True):
                     subdata[val_target_key] = subdata[val_target_key] * active_mask_batch
                
                # 2. Masking log_probs for the policy loss.
                #    The `sample_log_prob` from ProbabilisticActor is usually at the root.
                #    If it's per-agent (e.g. ("agents", "sample_log_prob")), it also needs masking.
                #    If `ClipPPOLoss` sums/means over agent dimension, it needs masking.
                #    Assuming `ClipPPOLoss` handles this correctly if per-agent terms are zeroed.
                #    Alternatively, one could implement a masked_mean reduction.

                loss_vals_td = loss_module(subdata) # This tensordict contains 'loss_objective', 'loss_critic', 'loss_entropy'

                # The losses from ClipPPOLoss are typically scalars (already reduced).
                # If they are not, and are per-agent, they would need masking before reduction.
                # Example: if loss_objective was per-agent:
                # loss_objective = (loss_vals_td["loss_objective"] * active_mask_batch.squeeze(-1)).sum() / active_mask_batch.sum()

                total_loss = loss_vals_td["loss_objective"] + loss_vals_td["loss_critic"] + loss_vals_td["loss_entropy"]
                
                optim.zero_grad()
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.train.max_grad_norm
                )
                optim.step()

                # Store detached loss values for logging for this minibatch
                loss_vals_td = loss_vals_td.detach()
                loss_vals_td.set("grad_norm", grad_norm.mean()) # grad_norm could be a tensor if many param groups
                if cumulative_loss_td is None:
                    cumulative_loss_td = loss_vals_td
                else:
                    for key, value in loss_vals_td.items(): # Basic mean accumulation
                        cumulative_loss_td[key] = (cumulative_loss_td[key] + value) / 2


        collector.update_policy_weights_() # Update policy in collector after optimizer step
        training_time = time.time() - training_iter_start_time
        iteration_time = sampling_time + training_time
        total_time += iteration_time
        
        log_string += f"Samp T: {sampling_time:.2f}s, Train T: {training_time:.2f}s. "
        if cumulative_loss_td:
            log_string += (f"Losses: Obj={cumulative_loss_td['loss_objective']:.3f}, "
                           f"Crit={cumulative_loss_td['loss_critic']:.3f}, Ent={cumulative_loss_td['loss_entropy']:.3f}. ")
        torchrl_logger.info(log_string)


        if logger and cumulative_loss_td:
            log_training( # Your existing logging function
                logger,
                cumulative_loss_td.apply(lambda x: x.mean()), # Log mean of losses over epochs/minibatches
                tensordict_data, # Original collected data for other metrics
                sampling_time,
                training_time,
                total_time,
                i, # Iteration number
                current_frames_collected,
                total_frames,
                step=total_frames, # Use total_frames as global step for logger
            )

        # Evaluation
        if (
            cfg.eval.evaluation_episodes > 0
            and i % cfg.eval.evaluation_interval == 0
        ):
            evaluation_start_time = time.time()
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                env_test.frames = [] # Reset frames for this eval run
                eval_rollouts = env_test.rollout(
                    max_steps=cfg.env.max_steps,
                    policy=policy, # Use the current policy
                    callback=rendering_callback,
                    auto_cast_to_device=True, # Ensure policy and env_test are on same device
                    break_when_any_done=False, # Run all env_test num_envs to completion
                )
            evaluation_time = time.time() - evaluation_start_time
            if logger:
                log_evaluation(logger, eval_rollouts, env_test, evaluation_time, step=total_frames)
        
        if cfg.logger.backend == "wandb" and logger:
             logger.experiment.log({}, commit=True) # Ensure all data is flushed

        sampling_start_time = time.time() # Reset for next iteration's sampling time

    collector.shutdown()
    if not env.is_closed: env.close()
    if not env_test.is_closed: env_test.close()
    if pbar: pbar.close()
    torchrl_logger.info(f"Training ended. Total time: {total_time / 3600:.2f} hours.")





class VariableAgentWrapper(EnvBase):
    """
    A wrapper for multi-agent environments like VmasEnv to handle a variable
    number of active agents during execution, up to a predefined maximum.

    The wrapper maintains fixed tensor shapes based on `max_agents` but uses
    an "active_mask" to indicate which agents are currently participating.
    Observations, actions, and rewards for inactive agents are masked (typically zeroed out).

    Args:
        env (VmasEnv): The VmasEnv instance to wrap. Assumed to be initialized with n_agents = max_agents.
        max_agents (int): The maximum number of agents the environment can support (and tensor shapes are based on).
        min_agents (int): The minimum number of agents that can be active.
    """
    def __init__(self, env: EnvBase, max_agents: int, min_agents: int):
        if not isinstance(env, EnvBase):
            raise TypeError("The wrapped environment must be an instance of torchrl.envs.EnvBase.")

        # Initialize EnvBase with device and batch_size from the wrapped env
        super().__init__(device=env.device, batch_size=env.batch_size)

        self.env = env
        self.max_agents = max_agents
        self.min_agents = min_agents
        self._current_n_agents = self.max_agents # Default to max


        # --- Define Group Name and Keys ---
        # Determine the group name from the wrapped env (usually "agents" for ALL_IN_ONE_GROUP)
        # EnvBase properties like self.action_key depend on this.
        if not hasattr(env, "group_map") or not env.group_map:
             # Attempt to infer from spec structure, default to "agents"
             found_group = None
             if isinstance(env.action_spec, CompositeSpec) and len(env.action_spec.keys(False)) == 1:
                 found_group = list(env.action_spec.keys(False))[0]
             self.group_name = found_group if found_group else "agents"
             torchrl_logger.warning(f"Wrapped env has no group_map, inferred group_name='{self.group_name}'")
        elif len(env.group_map) == 1:
             self.group_name = list(env.group_map.keys())[0]
        else:
             # If multiple groups exist, this wrapper logic might need adaptation.
             # For now, assume a single group as per ALL_IN_ONE_GROUP.
             raise ValueError("VariableAgentWrapper currently assumes a single agent group in the wrapped env.")

        self.group_name = "agents"

        self.active_mask_key = (self.group_name, "active_mask")
        self.observation_key = (self.group_name, "observation") # Cache for convenience

        # --- Specs ---
        # Simply clone the specs from the wrapped environment. They already have the
        # correct structure (nested group), batch size, and max_agents dimension.
        self.action_spec = self.env.action_spec.clone()
        self.reward_spec = self.env.reward_spec.clone()
        self.done_spec = self.env.done_spec.clone() # Usually global
        self.observation_spec = self.env.observation_spec.clone() # Clone the whole CompositeSpec


        # --- Add the active_mask spec ---
        try:
            # Get the parent CompositeSpec for the agent group
            group_composite_spec = self.observation_spec[self.group_name]
            if not isinstance(group_composite_spec, CompositeSpec):
                 raise TypeError(f"Expected observation_spec['{self.group_name}'] to be CompositeSpec.")

            # Determine the full shape needed for the mask spec
            # It must match the parent composite's shape plus the mask's feature dim (1)
            # parent_shape is e.g., torch.Size([30, 8])
            parent_shape = group_composite_spec.shape
            mask_feature_shape = torch.Size([1]) # Single boolean mask per agent
            active_mask_full_shape = parent_shape + mask_feature_shape
            # e.g., torch.Size([30, 8, 1])

            # Create the DiscreteTensorSpec with the *full* required shape
            # Note: Use CategoricalSpec as DiscreteTensorSpec is deprecated
            active_mask_spec = Categorical(
                n=2, # Equivalent to boolean
                shape=active_mask_full_shape, # Full shape including batch and agent dims
                dtype=torch.bool,
                device=self.device
            )

            # Add the correctly shaped spec to the group composite spec
            group_composite_spec.set("active_mask", active_mask_spec)

            # Verification check (optional but good practice)
            if "active_mask" not in group_composite_spec.keys(False):
                raise RuntimeError("Failed to add 'active_mask' key to observation_spec group.")
            if group_composite_spec["active_mask"].shape != active_mask_full_shape:
                 raise RuntimeError(f"Added mask spec has wrong shape: {group_composite_spec['active_mask'].shape} vs {active_mask_full_shape}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to add active_mask_spec to the cloned observation_spec. "
                f"Check structure of env.observation_spec: {self.env.observation_spec}. Error: {e}"
            ) from e


        # Initialize the actual mask tensor (_active_mask_value) used at runtime
        # Shape: [*batch_dims, max_agents, 1]
        self._active_mask_value = torch.ones(
            *self.batch_size, self.max_agents, 1,
            dtype=torch.bool,
            device=self.device
        )
        self.set_active_agents(self._current_n_agents) # Initialize mask based on default

        # Log initialization success and key info
        torchrl_logger.info(
             f"VariableAgentWrapper initialized. Batch: {self.batch_size}, Device: {self.device}. "
             f"Group: '{self.group_name}'. Agents: {self.min_agents}-{self.max_agents} (current: {self._current_n_agents})."
        )
        # Log keys derived from EnvBase properties (which use self.group_name and the specs)
        torchrl_logger.debug(f" Wrapper Action key: {self.action_key}")
        torchrl_logger.debug(f" Wrapper Reward key: {self.reward_key}")
        torchrl_logger.debug(f" Wrapper Done keys: {self.done_keys}")
        torchrl_logger.debug(f" Wrapper Active mask key: {self.active_mask_key}")
        torchrl_logger.debug(f" Wrapper Observation Spec: {self.observation_spec}")

    def set_active_agents(self, n_agents: int):
        if not (self.min_agents <= n_agents <= self.max_agents):
            raise ValueError(f"Requested n_agents ({n_agents}) is out of configured range [{self.min_agents}, {self.max_agents}].")
        self._current_n_agents = n_agents

        # Create the core mask for the agent dimension
        mask_core = torch.arange(self.max_agents, device=self.device) < n_agents # Shape [max_agents]

        # Expand to full dimensions: [*batch_dims, max_agents, 1]
        # This shape is convenient for broadcasting with agent-specific data like observations (..., N, D_obs) or rewards (..., N, 1)
        self._active_mask_value = mask_core.view(1, self.max_agents, 1).expand(
            *self.batch_size, self.max_agents, 1
        ).clone()
        # print(f"VariableAgentWrapper: Set active agents to {self._current_n_agents}. Mask shape: {self._active_mask_value.shape} on device {self._active_mask_value.device}")


    def _get_current_active_mask(self) -> torch.Tensor:
        """Returns the active mask tensor, e.g., shape (*batch_size, max_agents, 1)."""
        return self._active_mask_value

    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        # Reset the underlying environment. It operates with self.max_agents.
        td_reset = self.env._reset(tensordict, **kwargs)

        current_mask = self._get_current_active_mask()

        # Add the active_mask to the outgoing tensordict for the initial observation
        # The spec for ("agents", "active_mask") is for a single agent (shape [1]),
        # TensorDict handles the batching and agent dimension.
        td_reset[self.active_mask_key] = current_mask # Shape [*B, N_max, 1]

        # Zero out observations for inactive agents
        # td_reset[("agents", "observation")] has shape [*B, N_max, obs_dim]
        # current_mask has shape [*B, N_max, 1], broadcasts correctly.
        if self.observation_key in td_reset.keys(include_nested=True):
            td_reset[self.observation_key] = td_reset[self.observation_key] * current_mask
        else:
            # This should not happen if VmasEnv conforms to standard "agents" grouping.
            raise KeyError(f"Observation key {self.observation_key} not found in td_reset from wrapped env. Keys: {td_reset.keys(True,True)}")

        return td_reset

    def _step(self, tensordict: TensorDict) -> TensorDict:
        current_mask = self._get_current_active_mask() # Shape [*B, N_max, 1]

        # Mask actions for inactive agents before passing to the underlying environment
        if self.action_key not in tensordict.keys(include_nested=True):
            raise KeyError(f"Action key {self.action_key} not found in input tensordict to _step. Keys: {tensordict.keys(True,True)}")

        # Make a copy if you need to preserve original actions from policy for logging,
        # but usually the effectively taken action (masked) is what's important.
        # original_actions = tensordict[self.action_key].clone()
        tensordict[("agents",self.action_key)] = tensordict[self.action_key] * current_mask # Zero out actions for inactive

        # Step the underlying environment with the (potentially masked) actions
        breakpoint()
        td_out = self.env._step(tensordict)

        # --- Process the output tensordict (td_out) ---

        # 1. Add the active_mask for the *next* state's observation
        # Key for the mask in the *next* part of the tensordict.
        next_active_mask_key = ("next", self.active_mask_key) # e.g., ("next", "agents", "active_mask")
        td_out[next_active_mask_key] = current_mask.clone() # Or re-fetch if it could change mid-step (unlikely here)

        # 2. Mask rewards for inactive agents in the *next* state
        # td_out["next", self.reward_key] e.g., ("next", "agents", "reward")
        next_reward_key_nested = ("next", self.reward_key)
        if next_reward_key_nested not in td_out.keys(include_nested=True):
            # This might happen if "reward" is not under "next" or structured differently.
            # VmasEnv with ALL_IN_ONE_GROUP should place ("next", "agents", "reward").
            torchrl_logger.warning(f"Reward key {next_reward_key_nested} not found in td_out from self.env._step. td_out keys: {td_out.keys(True,True)}")
        else:
            reward_val = td_out[next_reward_key_nested]
            # Ensure mask is broadcastable to reward shape.
            # If reward is [*B, N, D_rew] and mask is [*B, N, 1].
            # If reward is scalar per agent, D_rew=1, direct multiply works.
            # If reward is truly scalar per agent (e.g. shape [*B, N]), mask needs squeeze.
            if reward_val.dim() == current_mask.dim() and reward_val.shape[-1] == current_mask.shape[-1]: # Both are e.g. [B,N,1]
                 masked_reward = reward_val * current_mask
            elif reward_val.dim() == current_mask.dim() - 1 and current_mask.shape[-1] == 1: # Reward [B,N], Mask [B,N,1]
                 masked_reward = reward_val * current_mask.squeeze(-1)
            elif reward_val.dim() == current_mask.dim() and current_mask.shape[-1] == 1: # Reward [B,N,D_rew], Mask [B,N,1]
                 masked_reward = reward_val * current_mask # Broadcasting
            else:
                raise ValueError(f"Cannot broadcast reward (shape {reward_val.shape}) with mask (shape {current_mask.shape})")
            td_out[next_reward_key_nested] = masked_reward


        # 3. Zero out next_observations for inactive agents
        # td_out[("next", "agents", "observation")]
        next_obs_key_nested = ("next", self.observation_key)
        if next_obs_key_nested not in td_out.keys(include_nested=True):
            torchrl_logger.warning(f"Next observation key {next_obs_key_nested} not found in td_out from self.env._step. td_out keys: {td_out.keys(True,True)}")
        else:
            td_out[next_obs_key_nested] = td_out[next_obs_key_nested] * current_mask


        # 4. Done flags:
        # VmasEnv 'done' (and 'terminated') is typically global (e.g., td_out["done"] or td_out["next","done"]).
        # As such, it's usually not masked per individual agent's activity by this wrapper.
        # If 'done' were per-agent under ("next", "agents", "done"), it would need masking:
        #   per_agent_done_key = ("next", self.group_name, "done")
        #   if per_agent_done_key in td_out.keys(include_nested=True):
        #       td_out[per_agent_done_key] = td_out[per_agent_done_key] & current_mask.squeeze(-1) # Logical AND

        return td_out

    def _set_seed(self, seed: int | None):
        # Seed the underlying environment. EnvBase handles the call to _set_seed.
        self.env.set_seed(seed)

    @property
    def lib(self): # Delegate to underlying env if it has 'lib' (like VmasEnv for rendering)
        return getattr(self.env, 'lib', None)

    def render(self, *args, **kwargs): # Delegate common methods
        if hasattr(self.env, 'render'):
            return self.env.render(*args, **kwargs)
        raise NotImplementedError(f"Wrapped environment {type(self.env)} does not support render.")

    def close(self):
        if not self.is_closed: # property from EnvBase
            if hasattr(self.env, 'close'):
                self.env.close()
            super().close() # Marks this wrapper instance as closed

    @property
    def n_agents(self) -> int:
        # This property should reflect the tensor dimensions, which are based on max_agents.
        return self.max_agents

    @property
    def current_n_agents(self) -> int:
        # Returns the currently active number of agents.
        return self._current_n_agents

    # For completeness, delegate state_dict/load_state_dict if the wrapper itself had more state.
    # For now, only current_n_agents is specific to the wrapper's dynamic behavior beyond the env's state.
    def state_dict(self, **kwargs) -> dict:
        # Basic state dict, can be expanded if wrapper has more state
        wrapped_env_state = self.env.state_dict(**kwargs) if hasattr(self.env, 'state_dict') else {}
        return {
            "_current_n_agents": self._current_n_agents,
            "env_state_dict": wrapped_env_state
        }

    def load_state_dict(self, state_dict: dict, **kwargs):
        self._current_n_agents = state_dict["_current_n_agents"]
        self.set_active_agents(self._current_n_agents) # Crucial to re-apply mask state
        if hasattr(self.env, 'load_state_dict') and "env_state_dict" in state_dict:
            self.env.load_state_dict(state_dict["env_state_dict"], **kwargs)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"env={self.env}, "
                f"max_agents={self.max_agents}, min_agents={self.min_agents}, "
                f"current_n_agents={self._current_n_agents})")


if __name__ == "__main__":
    # For tqdm and other utilities that might be missing
    try:
        import tqdm
    except ImportError:
        print("tqdm not found, progress bar will be disabled. pip install tqdm")
        # Create a dummy tqdm if not found, so pbar related lines don't crash
        class dummy_tqdm:
            def __init__(self, *args, **kwargs): pass
            def update(self, *args, **kwargs): pass
            def close(self, *args, **kwargs): pass
        tqdm = dummy_tqdm

    train()
