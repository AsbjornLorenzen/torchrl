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
    CompositeSpec # For type checking if needed
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
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))


@hydra.main(version_base="1.1", config_path="", config_name="mappo_pot")
def train(cfg: "DictConfig"):  # noqa: F821
    # --- Configuration ---
    # MAX_AGENTS_TRAIN = 8
    # MAX AGENTS is set in the config udner scenario
    MIN_AGENTS_TRAIN = 4
    EVAL_AGENTS = 10
    MAX_AGENTS_TRAIN = cfg.env.scenario.n_agents
    # Ensure cfg.env.scenario.n_agents (or equivalent) is set to MAX_AGENTS_TRAIN for the training env

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
    # --- Integrate the wrapper ---
    env = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.env.vmas_envs,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        group_map=MarlGroupMapType.ALL_IN_ONE_GROUP,
        # Scenario kwargs
        **cfg.env.scenario,
    )
    print(f"ORg env has action keys {env.action_keys}")
    wrapped_env = VariableAgentWrapper(env, cfg.env.scenario.n_agents, MIN_AGENTS_TRAIN)
    env = TransformedEnv(
        wrapped_env,
        RewardSum(in_keys=[wrapped_env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    # env.action_key = wrapped_env.env.action_key

    eval_cfg = cfg.env.scenario
    eval_cfg.n_agents = EVAL_AGENTS

    env_test = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.eval.evaluation_episodes,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        # Scenario kwargs
        **eval_cfg,
    )

    print(f"In torchrl, the final env action key is: {env.action_key}")
    print(f"In torchrl, the final env action spec is: {env.action_spec}")
    print(f"In torchrl, the final env full_action_spec_unbatched is: {env.full_action_spec_unbatched}")


    # Ensure the action key is what we expect (a tuple for grouped actions)
    if not isinstance(env.action_key, tuple) or len(env.action_key) != 2:
        raise ValueError(
            f"env.action_key is expected to be a tuple like ('agents', 'action'), got {env.action_key}"
        )

    # Get the spec for the actual action tensor (not the composite container)
    # env.full_action_spec_unbatched will be a CompositeSpec, e.g. {("agents", "action"): UnbatchedAgentActionSpec}
    action_tensor_spec_unbatched = env.full_action_spec_unbatched[env.action_key]

    print(f"Action Tensor Spec (unbatched for one agent): {action_tensor_spec_unbatched}")
    print(f"Action Tensor Spec shape: {action_tensor_spec_unbatched.shape}") # Should be (action_dim,)

    expected_gnn_output_dim = 2 * action_tensor_spec_unbatched.shape[-1]
    print(f"Expected GNN output dim (for loc+scale): {expected_gnn_output_dim}")





    # GNN POLICY
    gnn_hidden_dim = cfg.model.get("gnn_hidden_dim", 128)
    gnn_layers = cfg.model.get("gnn_layers", 2)
    k_neighbours = cfg.model.get("k_neighbours", None) # Default to fully connected if not specified
    pos_indices_list = cfg.model.get("pos_indices", [0, 2]) # Default to first 2
    pos_indices = slice(pos_indices_list[0], pos_indices_list[1])
    # TODO: FIX POLICY MODULE TO HANDLE VARIABLE N_AGENTS
    actor_net = nn.Sequential(
        GNNActor(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * action_tensor_spec_unbatched.shape[-1],
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
    lowest_action = torch.zeros_like(action_tensor_spec_unbatched.space.low, device=cfg.train.device)
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.full_action_spec_unbatched,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": lowest_action,
            "high": action_tensor_spec_unbatched.space.high,
        },
        return_log_prob=True,
    )
    critic_module = GNNCritic(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        gnn_hidden_dim=gnn_hidden_dim, # Use same GNN params as actor (can be configured separately if needed)
        gnn_layers=gnn_layers,
        activation_class=nn.Tanh,
        k_neighbours=None,
        pos_indices=pos_indices,
        share_params=cfg.model.shared_parameters, # Kept for consistency, GNN shares anyway
        device=cfg.train.device, # Pass device object
    )
    value_module = ValueOperator(
        module=critic_module,
        in_keys=[("agents", "observation")],
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
        value=("agents", "state_value"),
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
        )
        logger = init_logging(cfg, model_name)

    total_time = 0
    total_frames = 0
    sampling_start = time.time()
    # --- Training Loop Modification ---
    for i, tensordict_data in enumerate(collector):
        # --- Determine agents for NEXT rollout ---
        # Do this *before* the collector runs again implicitly at the end of the loop
        # or explicitly if you manage collector steps manually.
        # If collector runs until frames_per_batch are met, setting it here affects
        # the *next* call to enumerate(collector).
        next_n_agents = random.randint(MIN_AGENTS_TRAIN, MAX_AGENTS_TRAIN)
        # Access the wrapper instance to set the number. Assumes 'env' passed to
        # collector holds the transformations and the wrapper is accessible.
        # If TransformedEnv hides it, you might need env.env or similar.
        # Let's assume env.env gets the VariableAgentWrapper instance
        if isinstance(collector.env, TransformedEnv):
            # Find the wrapper in the chain of transformations
            wrapper = collector.env
            while not isinstance(wrapper, VariableAgentWrapper) and hasattr(wrapper, "env"):
                 wrapper = wrapper.env
            if isinstance(wrapper, VariableAgentWrapper):
                wrapper.set_active_agents(next_n_agents)
            else:
                print("Warning: Could not find VariableAgentWrapper to set agent count.")
        elif isinstance(collector.env, VariableAgentWrapper):
             collector.env.set_active_agents(next_n_agents)


        torchrl_logger.info(f"\nIteration {i} (using {wrapper.current_n_agents if 'wrapper' in locals() else 'N/A'} agents)")
        sampling_time = time.time() - sampling_start

        # Should work, but advantages for inactive agents will be calculated based on
        # zero rewards/values (if value network outputs 0 for masked inputs).
        # These advantages should ideally be masked out before the PPO loss.
        # TODO: Fix zeroing out of values
        with torch.no_grad():
            loss_module.value_estimator(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )
        current_frames = tensordict_data.numel()
        active_agent_frames = tensordict_data["agents", "active_mask"].sum().item()
        total_frames += current_frames
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        training_tds = []
        training_start = time.time()
        for _ in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                subdata = replay_buffer.sample()
                active_mask_batch = subdata.get(("agents", "active_mask")) # Shape [minibatch, max_agents]

                # --- Loss Calculation with Masking ---
                # Option 1: Mask values *before* passing to loss (if loss doesn't support masks)
                # Advantage is often ('agents', 'advantage')
                if ("agents", "advantage") in subdata.keys(True):
                     subdata["agents", "advantage"] = subdata["agents", "advantage"] * active_mask_batch.unsqueeze(-1)
                # Value target is often ('agents', 'value_target')
                if ("agents", "value_target") in subdata.keys(True):
                     subdata["agents", "value_target"] = subdata["agents", "value_target"] * active_mask_batch.unsqueeze(-1)

                # Calculate loss - ClipPPOLoss might average over the agent dim.
                loss_vals = loss_module(subdata)
                training_tds.append(loss_vals.detach())


                # If loss_vals are already reduced means (like the default), masking *before* is better.
                # Let's assume masking *before* is sufficient or ClipPPOLoss needs custom modification.
                # We'll proceed assuming the loss function correctly handles zeroed advantages/targets
                # OR that pre-masking advantages/targets works.

                # Aggregate losses (assuming they are now correctly weighted or masked)

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                # Check if loss is valid (e.g., not NaN) due to masking/division issues
                if torch.isnan(loss_value):
                     print("Warning: NaN loss detected. Check masking and loss calculation.")
                     continue # Skip this batch

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
                    # We are running vectorized evaluation we do not want it to stop when just one env is done
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


class VariableAgentWrapper(EnvBase):
    def __init__(self, env: VmasEnv, max_agents: int, min_agents: int):
        # Store env instance early if needed for accessing its properties
        self.max_agents = max_agents
        self.min_agents = min_agents

        # 1. Get actual keys from VmasEnv
        vmas_action_key = env.action_key
        vmas_reward_key = env.reward_key
        vmas_done_keys_list = env.done_keys # This is a list of nested keys, e.g., [("agents", "done"), ("agents", "terminated")]

        # 2. Clone and configure specs
        cloned_action_spec = env.action_spec.clone()
        if cloned_action_spec is not None:
            cloned_action_spec.set_input_domain("action", vmas_action_key)

        cloned_reward_spec = env.reward_spec.clone()
        if cloned_reward_spec is not None:
            cloned_reward_spec.set_input_domain("reward", vmas_reward_key)

        cloned_done_spec = env.done_spec.clone()
        if cloned_done_spec is not None:
            # Map VmasEnv's done keys to TorchRL's standard internal done key names
            # This depends on how VmasEnv structures its done signals.
            # Example: If VmasEnv's first done_key is the primary 'terminated' signal
            if len(vmas_done_keys_list) > 0:
                 cloned_done_spec.set_input_domain("terminated", vmas_done_keys_list[0])
                 cloned_done_spec.set_input_domain("done", vmas_done_keys_list[0]) # Often "done" is an alias for "terminated"
            # If VmasEnv has a separate truncation signal as its second key:
            # if len(vmas_done_keys_list) > 1:
            #    cloned_done_spec.set_input_domain("truncated", vmas_done_keys_list[1])


        cloned_observation_spec = env.observation_spec.clone()
        # Add your active_mask_spec to the cloned_observation_spec
        # Assuming 'obs_spec_agents_obs' was correctly defined based on cloned_observation_spec
        active_mask_tensor_spec = UnboundedContinuousTensorSpec(
            shape=torch.Size([self.max_agents, 1]), # Ensure max_agents is available
            device=env.device,
            dtype=torch.bool,
        )
        if cloned_observation_spec is not None:
            cloned_observation_spec[("agents", "active_mask")] = active_mask_tensor_spec
        else:
            # Handle case where base observation_spec might be None, though unlikely for Vmas
            raise Exception("Base observatin_spec is None")


        # 3. Call super().__init__ with the configured specs
        super().__init__(
            device=env.device,
            batch_size=env.batch_size, # Ensure VmasEnv has batch_size attribute or handle appropriately
            action_spec=cloned_action_spec,
            reward_spec=cloned_reward_spec,
            done_spec=cloned_done_spec,
            observation_spec=cloned_observation_spec
        )

        self.env = env

        # 4. Lock keys to specs (optional, but good for clarity and robustness)
        # Use object.__setattr__ to bypass any custom __setattr__ in the hierarchy
        object.__setattr__(self, '_action_key_is_locked_to_spec', True)
        object.__setattr__(self, '_reward_key_is_locked_to_spec', True)
        object.__setattr__(self, '_done_keys_is_locked_to_spec', True)

        # 5. Initialize VariableAgentWrapper specific attributes
        # These attributes might need self.device and self.batch_size, which are set by super().__init__
        self._current_n_agents = self.max_agents # Or min_agents, depending on desired start
        self._active_mask = torch.ones(
            self.batch_size + (self.max_agents, 1), # self.batch_size is now set
            dtype=torch.bool,
            device=self.device # self.device is now set
        )
        # Initialize other necessary attributes for your wrapper's logic

        # --- Explicitly set the wrapper's keys to match the underlying VmasEnv's grouped keys ---
        print(f"Other env action key: {self.env.action_key}")
        # print(f"Current action key: {self.action_key}")
        # self.action_key = self.env.action_key # This should be ("agents", "action")
        # self.reward_key = self.env.reward_key # This should be ("agents", "reward")

        # self.observation_key = self.env.observation_key # If you use it explicitly

        # Your existing print statement for debugging:
        print(f"VariableAgentWrapper: self.action_key set to {self.action_key}")
        print(f"VariableAgentWrapper: self.reward_key set to {self.reward_key}")

        # --- Store the keys from the wrapped environment ---
        # print(f"Action key is {self.action_key}")
        # print(f"rew key is {self.reward_key}")

        obs_spec = self.observation_spec["agents","observation"]

        if not hasattr(obs_spec, 'shape'):
            raise TypeError(f"Expected observation_spec['agents', 'observation'] to have a 'shape' attribute, but got type {type(obs_spec)}")

        # Determine the shape for the mask.
        # Assuming obs_spec.shape is like (..., n_agents, obs_feature_dim)
        # We want the mask shape to be (..., n_agents)
        mask_shape = obs_spec.shape[:-1] # Remove the last dimension (observation features)

        # Create the spec for the active_mask.
        # It should be boolean (discrete with 2 values: 0 or 1)
        # and have the derived shape.
        active_mask_spec = DiscreteTensorSpec(
            n=2,                   # Represents boolean values (0 or 1)
            shape=mask_shape,
            dtype=torch.bool,      # Explicitly set dtype to boolean
            device=obs_spec.device # Optional: Keep the device consistent
        )

        self.active_mask_key = ("agents", "active_mask")
        # Assign the correctly defined spec
        self.observation_spec["agents", "active_mask"] = active_mask_spec

    def set_active_agents(self, n_agents: int):
        if not (self.min_agents <= n_agents <= self.max_agents):
             raise ValueError(f"n_agents must be between {self.min_agents} and {self.max_agents}")
        self._current_n_agents = n_agents
        # Create mask: True for active, False for inactive
        mask = torch.arange(self.max_agents, device=self.device) < n_agents
        # Expand mask to match batch size (num_envs)
        self._active_mask = mask.unsqueeze(0).expand(*self.env.batch_size, self.max_agents)
        print(f"Set active agents to {self._current_n_agents} for next rollout.")


    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        # Reset the underlying env
        td_reset = self.env._reset(tensordict, **kwargs)
        # Add the active mask
        td_reset[self.active_mask_key] = self._active_mask.clone()
        # Optional: Zero out observations for inactive agents (might help GNN)
        td_reset[("agents", "observation")][~self._active_mask] = 0.0
        return td_reset

    def _step(self, tensordict: TensorDict) -> TensorDict:
        # Mask actions sent to inactive agents (e.g., set to zero or a neutral action)
        action_key = self.env.action_key


        # --- Add a check for debugging ---
        if action_key not in tensordict.keys(include_nested=True):
            print("\n--- ERROR: Action key missing in VariableAgentWrapper._step ---")
            print(f"Timestamp: {time.time()}")
            print(f"Expected Action Key: {action_key}")
            print(f"TensorDict Keys Received: {tensordict.keys(True, True)}")
            print(f"TensorDict Shape: {tensordict.shape}")
            # Optionally print parts of the tensordict content if small enough
            # print(f"TensorDict content sample: {tensordict.exclude('observation')}")
            raise KeyError(f"Action key {action_key} not found in input tensordict to VariableAgentWrapper._step. Available keys: {tensordict.keys(True, True)}")
        # ---------------------------------


        original_actions = tensordict[action_key].clone()
        tensordict[action_key][~self._active_mask] = 0.0 # Zero actions for inactive

        # Step the underlying env
        td_step = self.env._step(tensordict)

        # Restore original actions in the output if needed for logging/buffer
        # td_step[action_key] = original_actions # Or keep masked action? Depends on loss.

        # Add the active mask for the *next* state
        td_step["next", "agents", "active_mask"] = self._active_mask.clone()

        # Mask results from inactive agents (rewards, potentially 'done' if per-agent)
        # Ensure rewards for inactive agents are 0
        reward_key = self.env.reward_key # e.g., ("agents", "reward")
        print(f"In wrapper, got env key {reward_key} but before had {self.reward_key}")

        if ("next", *reward_key) in td_step.keys(include_nested=True):
             td_step["next", reward_key][~self._active_mask] = 0.0
        elif reward_key in td_step.get("next", TensorDict({},[])).keys(include_nested=True):
              # Handle cases where reward might be nested differently under 'next'
              td_step["next", reward_key][~self._active_mask] = 0.0

        # Optional: Zero out next observations for inactive agents
        obs_key = ("agents", "observation")
        if obs_key in td_step["next", "agents"].keys(include_nested=False):
            td_step["next", obs_key][~self._active_mask] = 0.0

        # Handle 'done' and 'terminated' - if they are per-agent, mask them.
        # Vmas 'done' is usually global, but check your scenario.
        # If done is ("agents", "done"):
        #    td_step["next", "agents", "done"][~self._active_mask] = False # Or True? Depends. Usually False.
        #    td_step["next", "agents", "terminated"][~self._active_mask] = False

        return td_step

    def _set_seed(self, seed):
        # Seed the underlying env
        self.env._set_seed(seed)

    # --- Need to expose other methods/properties if used ---
    @property
    def n_agents(self):
         # Return the MAX agents, as the tensors shapes reflect this
         return self.max_agents

    @property
    def current_n_agents(self):
        # Return the currently active number
        return self._current_n_agents

    # Expose other relevant properties/methods from self.env if needed
    # E.g., render, close, state_dict, load_state_dict etc.

if __name__ == "__main__":
    train()
