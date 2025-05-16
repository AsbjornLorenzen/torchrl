import torch
from torchrl.envs import Transform, TransformedEnv
from tensordict import TensorDictBase, TensorDict
from __future__ import annotations

import time
import random # Import random

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
from torchrl.envs import RewardSum, TransformedEnv, Compose # Import Compose
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
# from torchrl.modules.models.multiagent import MultiAgentMLP # Not used
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from utils.logging import init_logging, log_evaluation, log_training # Assuming these are your custom utils
from utils.utils import DoneTransform # Assuming this is your custom util
from models.gnn_actor import GNNActor # Assuming these are your custom models
from models.gnn_critic import GNNCritic # Assuming these are your custom models


rom torchrl.specs import UnboundedContinuousTensorSpec, DiscreteTensorSpec, CompositeSpec, BinaryDiscreteTensorSpec

class VariableAgentWrapper(Transform):
    def __init__(
        self,
        in_keys: list[str],
        out_keys: list[str],
        min_n_agents: int,
        max_n_agents: int,
        max_total_agents: int, # Max agents the env is initialized with
        is_eval: bool = False,
        eval_n_agents: int | None = None,
    ):
        if not out_keys: # By default, modify in_keys if out_keys is not provided
            out_keys = in_keys
        super().__init__(in_keys=in_keys, out_keys=out_keys) # in_keys not strictly used here but good practice
        self.min_n_agents = min_n_agents
        self.max_n_agents = max_n_agents
        self.max_total_agents = max_total_agents
        self.is_eval = is_eval
        self.eval_n_agents = eval_n_agents if eval_n_agents is not None else max_total_agents

        if self.is_eval and self.eval_n_agents > self.max_total_agents:
            raise ValueError("eval_n_agents cannot exceed max_total_agents")
        if not self.is_eval and self.max_n_agents > self.max_total_agents:
            raise ValueError("max_n_agents for training cannot exceed max_total_agents")

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        # tensordict_reset is the output of env.reset()
        # We need to add/modify the active_mask here.
        device = tensordict_reset.device
        batch_size = tensordict_reset.batch_size

        if self.is_eval:
            current_n_active_agents_per_env = torch.full(
                batch_size, self.eval_n_agents, device=device, dtype=torch.long
            )
        else:
            current_n_active_agents_per_env = torch.randint(
                self.min_n_agents,
                self.max_n_agents + 1,
                batch_size,
                device=device,
            )

        active_mask = torch.zeros(
            *batch_size, self.max_total_agents, device=device, dtype=torch.bool
        )
        # Create mask: for each batch item, set first current_n_active_agents to True
        for i in range(batch_size[0]): # Assuming batch_size is like [N]
            active_mask[i, : current_n_active_agents_per_env[i]] = True
        
        tensordict_reset.set(("agents", "active_mask"), active_mask)
        # Store the count too, might be useful for averaging losses
        tensordict_reset.set(("agents", "n_active"), current_n_active_agents_per_env.unsqueeze(-1))


        # Mask initial observations for inactive agents (optional, but good practice)
        # Assuming observation key is ("agents", "observation")
        obs_key = ("agents", "observation")
        if obs_key in tensordict_reset.keys(include_nested=True):
            tensordict_reset[obs_key] = tensordict_reset[obs_key] * active_mask.unsqueeze(-1)
        
        return tensordict_reset

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # This is called before the policy.
        # If policy needs the mask, it's already there from _reset or _step's next_td handling
        # No action usually needed here unless you want to transform observations based on mask
        # *before* they hit the policy in a special way.
        return tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # This method handles both reset and step logic based on tensordict content
        if tensordict.get("is_init", None) is not None: # This indicates a reset call chain
             # For resets, the actual reset is done by the environment, _reset is for post-processing
            return self._reset(tensordict, tensordict.clone(False)) # Pass a clone for _reset to fill

        # --- This is for the step logic ---
        active_mask = tensordict.get(("agents", "active_mask"))
        n_active = tensordict.get(("agents", "n_active"))

        # 1. Mask actions for inactive agents BEFORE sending to env.step()
        # Assuming env.action_key is ("agents", "action")
        action_key = self.parent.action_key # Get action key from wrapped env
        if action_key in tensordict.keys(include_nested=True) and active_mask is not None:
             tensordict[action_key] = tensordict[action_key] * active_mask.unsqueeze(-1)

        # tensordict is now modified with masked actions, ready for env.step()
        # The actual env.step() is called by the collector or rollout utilities *after* this transform.
        # This transform is part of the env, so its `step` method implicitly calls this `forward`
        # and then the parent env's step.
        # What we return here is the tensordict that goes INTO the parent env's step.
        # The results of parent.step() will be in `next_tensordict`, handled by `self.step`

        return tensordict # Return the modified tensordict

    # This method is called with the data *after* the environment's step
    @torch.no_grad()
    def step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Apply _call to the input tensordict (actions are masked here)
        tensordict = self.forward(tensordict) # This will mask actions if not already
        
        # Get the active_mask from the *input* tensordict (current step)
        # because this mask determines which agents were supposed to act
        active_mask = tensordict.get(("agents", "active_mask"))
        n_active = tensordict.get(("agents", "n_active"))

        # Call the parent environment's step
        next_tensordict = self.parent.step(tensordict)

        # 2. Mask results (next_observation, reward, done) for inactive agents
        # And ensure the mask is propagated
        if active_mask is not None:
            next_tensordict.set(("next", "agents", "active_mask"), active_mask.clone())
            if n_active is not None:
                next_tensordict.set(("next", "agents", "n_active"), n_active.clone())


            obs_key = ("next", "agents", "observation")
            if obs_key in next_tensordict.keys(include_nested=True):
                next_tensordict[obs_key] = next_tensordict[obs_key] * active_mask.unsqueeze(-1)

            reward_key = ("next", self.parent.reward_key[1]) # e.g. ("next", ("agents", "reward")) -> ("next", "agents", "reward")
            if reward_key in next_tensordict.keys(include_nested=True):
                 # Ensure reward key is correctly constructed if it's nested
                path_to_reward = reward_key # if reward_key is already ('next', 'agents', 'reward')
                if isinstance(self.parent.reward_key, tuple) and len(self.parent.reward_key) > 1: # e.g. ("agents", "reward")
                    path_to_reward = ("next", *self.parent.reward_key)

                current_reward = next_tensordict.get(path_to_reward)
                next_tensordict.set(path_to_reward, current_reward * active_mask.unsqueeze(-1))


            # Handle 'done' for inactive agents. Setting them to 'done' can simplify GAE.
            # Assuming done_key is like ("next", "done") for global or ("next", "agents", "done") for per-agent
            # Your DoneTransform uses env.done_keys which could be complex.
            # Let's assume a per-agent done for simplicity here. If global, this part is harder.
            # If using your `DoneTransform`, it might need to be mask-aware or this masking of done needs care.
            # For ClipPPOLoss with agent-wise done:
            agent_done_key_path = None
            if ("agents", "done") in self.parent.done_keys: # from loss_module.set_keys
                agent_done_key_path = ("next", "agents", "done")
            
            if agent_done_key_path and agent_done_key_path in next_tensordict.keys(include_nested=True):
                agent_dones = next_tensordict.get(agent_done_key_path)
                # Inactive agents are effectively "done" from the start of their inactivity
                agent_dones = torch.where(active_mask.unsqueeze(-1), agent_dones, True)
                next_tensordict.set(agent_done_key_path, agent_dones)
            # If there's a global 'done', it should be true if ALL *active* agents are done.
            # This logic is usually handled by the environment or a wrapper like DoneTransform.
            # For now, we focus on masking rewards and observations.

        return next_tensordict

    def transform_observation_spec(self, observation_spec: CompositeSpec) -> CompositeSpec:
        # Add the active_mask to the observation spec
        if isinstance(observation_spec.get("agents", None), CompositeSpec):
            mask_spec = BinaryDiscreteTensorSpec(
                n=2, # True/False
                shape=torch.Size([self.max_total_agents]), # Per agent
                device=observation_spec.device,
                dtype=torch.bool,
            )
            n_active_spec = DiscreteTensorSpec( # n_active can be up to max_total_agents
                n=self.max_total_agents + 1, # number of agents from 0 to max_total_agents
                shape=torch.Size([1]), # scalar per environment
                device=observation_spec.device,
                dtype=torch.long,
            )
            observation_spec["agents"].set("active_mask", mask_spec)
            observation_spec["agents"].set("n_active", n_active_spec)

        return observation_spec
    
    # You might also need to implement transform_input_spec, transform_reward_spec etc.
    # if the mask needs to be reflected in all specs used by model/loss.
    # For now, observation_spec is often the most critical for model input.


@hydra.main(version_base="1.1", config_path="", config_name="mappo_pot")
def train(cfg: "DictConfig"):  # noqa: F821
    # Device
    cfg.train.device = "cpu" if not torch.cuda.device_count() else "cuda:0"
    cfg.env.device = cfg.train.device

    # Seeding
    torch.manual_seed(cfg.seed)

    # --- Key Change: Initialize with MAX_AGENTS_EVAL ---
    # Ensure your VmasEnv scenario config (cfg.env.scenario) uses cfg.env.max_agents_eval
    # For example, if scenario is 'balance', then cfg.env.scenario.n_agents = cfg.env.max_agents_eval
    # This needs to be set in your hydra config file.
    # Let's assume cfg.env.scenario already has n_agents reflecting max_agents_eval
    # or that VmasEnv takes n_agents directly if not in scenario kwargs.
    
    # For Vmas, `n_agents` is often a direct parameter of the scenario,
    # so you'd typically pass it like: `scenario_kwargs={"n_agents": cfg.env.max_agents_eval, ...}`
    # Update: VmasEnv takes n_agents directly for some scenarios if not specified in scenario_kwargs
    # Let's add it explicitly for clarity if it's a top-level VmasEnv param for your scenario
    # If 'n_agents' is part of cfg.env.scenario, ensure it's set to cfg.env.max_agents_eval there.
    # For the example, let's assume `cfg.env.scenario` will have `n_agents`.
    # A common Vmas pattern for n_agents:
    scenario_kwargs = dict(cfg.env.scenario)
    scenario_kwargs['n_agents'] = cfg.env.max_agents_eval # Explicitly set for clarity

    # Sampling
    cfg.env.vmas_envs = cfg.collector.frames_per_batch // cfg.env.max_steps
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size = cfg.collector.frames_per_batch


    # Create env and env_test
    # Pass the actual max number of agents the env should simulate
    base_env = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.env.vmas_envs,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        n_agents=cfg.env.max_agents_eval, # Explicitly set n_agents for VMAS
        # Scenario kwargs
        **scenario_kwargs, # This might also contain n_agents, ensure consistency
    )
    # Wrap with VariableAgentWrapper for training
    env = TransformedEnv(
        base_env,
        VariableAgentWrapper(
            in_keys=[], out_keys=[], # Not strictly used by this wrapper's logic
            min_n_agents=cfg.env.min_agents_train_lower,
            max_n_agents=cfg.env.max_agents_train_upper,
            max_total_agents=cfg.env.max_agents_eval,
            is_eval=False,
        ),
    )
    env = TransformedEnv(
        env, # Wrap the already wrapped env
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    # Note: The order of transforms matters. RewardSum should see masked rewards.

    base_env_test = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.eval.evaluation_episodes,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        n_agents=cfg.env.max_agents_eval, # Test env also initialized with max
        # Scenario kwargs
        **scenario_kwargs, # Ensure this also uses max_agents_eval for n_agents
    )
    # Wrap with VariableAgentWrapper for evaluation
    env_test = TransformedEnv(
        base_env_test,
        VariableAgentWrapper(
            in_keys=[], out_keys=[],
            min_n_agents=cfg.env.eval_n_agents, # Use a specific number for eval
            max_n_agents=cfg.env.eval_n_agents,
            max_total_agents=cfg.env.max_agents_eval,
            is_eval=True,
            eval_n_agents=cfg.env.eval_n_agents, # Number of agents for this eval run
        ),
    )
    # env_test = TransformedEnv(env_test, RewardSum(...)) # if needed for eval logging

    # --- Policy and Critic Networks ---
    # Initialized for cfg.env.max_agents_eval because observation_spec and action_spec
    # from the base_env (or wrapped env if spec doesn't change shape) will reflect that.
    # The GNNActor/Critic should be able to handle variable numbers of active agents
    # if their inputs (features) are zero for inactive agents.

    # The specs from `env` (which is wrapped) will now include `active_mask`.
    # The underlying observation and action specs will be for `max_agents_eval`.
    
    # Example: action_spec from wrapped env will still be for max_agents_eval
    # env.action_spec is fine, policy will output for max_agents_eval, wrapper masks it.
    # env.observation_spec["agents", "observation"] is for max_agents_eval, wrapper masks input.
    
    gnn_hidden_dim = cfg.model.get("gnn_hidden_dim", 128)
    gnn_layers = cfg.model.get("gnn_layers", 2)
    k_neighbours = cfg.model.get("k_neighbours", None)
    pos_indices_list = cfg.model.get("pos_indices", [0, 2]) 
    pos_indices = slice(pos_indices_list[0], pos_indices_list[1])

    # Use observation_spec from the base_env because the wrapper adds a mask
    # but the GNN itself operates on the raw observation features.
    # The wrapper ensures that for inactive agents, these features are zeroed out.
    actor_obs_in_key = ("agents", "observation") # Key the GNN will read
    critic_obs_in_key = ("agents", "observation")

    actor_net_module = GNNActor(
        n_agent_inputs=base_env.observation_spec["agents", "observation"].shape[-1], # Use base_env spec for feature size
        n_agent_outputs=2 * base_env.action_spec.shape[-1], # Use base_env spec for action size
        gnn_hidden_dim=gnn_hidden_dim,
        n_gnn_layers=gnn_layers,
        activation_class=nn.Tanh,
        k_neighbours=k_neighbours,
        pos_indices=pos_indices,
        share_params=cfg.model.shared_parameters,
        device=cfg.train.device,
    )
    actor_net = nn.Sequential(
        actor_net_module,
        NormalParamExtractor(),
    )
    policy_module = TensorDictModule(
        actor_net,
        in_keys=[actor_obs_in_key], # Read potentially masked obs
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    
    # lowest_action should use the spec from the environment the policy interacts with,
    # which is the wrapped 'env'. Its action_spec is for max_agents_eval.
    lowest_action = torch.zeros_like(env.action_spec[("agents", "action")].space.low, device=cfg.train.device)
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec, # Spec from the wrapped env (which is for max_total_agents)
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key], # This is ("agents", "action")
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": lowest_action,
            "high": env.action_spec[("agents", "action")].space.high,
        },
        return_log_prob=True, # Important for PPO
    )

    critic_net_module = GNNCritic(
        n_agent_inputs=base_env.observation_spec["agents", "observation"].shape[-1], # Use base_env spec
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_layers=gnn_layers,
        activation_class=nn.Tanh,
        k_neighbours=None,
        pos_indices=pos_indices,
        share_params=cfg.model.shared_parameters,
        device=cfg.train.device,
    )
    value_module = ValueOperator(
        module=critic_net_module,
        in_keys=[critic_obs_in_key], # Read potentially masked obs
        out_keys=[("agents", "state_value")]
    )

    collector = SyncDataCollector(
        env, # Use the wrapped env
        policy,
        device=cfg.env.device,
        storing_device=cfg.train.device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        postproc=DoneTransform(reward_key=env.reward_key, done_keys=env.done_keys), # DoneTransform should see masked rewards/dones
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
        entropy_coef=cfg.loss.entropy_eps, # We'll handle entropy masking carefully
        normalize_advantage=False, # Consider True if advantages vary a lot
    )
    loss_module.set_keys( # These keys must exist in the tensordict from the collector
        reward=env.reward_key, # e.g. ("agents", "reward")
        action=env.action_key, # e.g. ("agents", "action")
        done=("agents", "done"), # Make sure this is what DoneTransform produces
        terminated=("agents", "terminated"), # Make sure this is what DoneTransform produces
        # The sample_log_prob key is usually handled internally by ProbabilisticActor output
    )
    # The value estimator (GAE) will use the masked rewards and dones.
    # If value_module outputs 0 for inactive (zeroed obs) agents, advantage should be 0.
    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=cfg.loss.gamma, lmbda=cfg.loss.lmbda
    )
    optim = torch.optim.Adam(loss_module.parameters(), cfg.train.lr)

    # ... (logging setup) ...

    total_time = 0
    total_frames = 0
    sampling_start = time.time()
    for i, tensordict_data in enumerate(collector):
        torchrl_logger.info(f"\nIteration {i}")
        sampling_time = time.time() - sampling_start

        # tensordict_data now contains ("agents", "active_mask")
        # and ("agents", "observation"), ("agents", "reward") etc. are masked by VariableAgentWrapper

        with torch.no_grad():
            # GAE calculation: uses masked rewards. If critic outputs ~0 for masked obs, advantage will be ~0.
            loss_module.value_estimator(
                tensordict_data, # This tensordict has the mask and masked data
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )
        current_frames = tensordict_data.numel() # numel might be tricky if batch_size varies, but usually it's fixed per iter.
        total_frames += current_frames # Total frames processed by collector
        
        # Reshape and add to replay buffer
        # The active_mask is part of the data and will be sampled by the buffer
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        training_tds = []
        training_start = time.time()
        for _ in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                subdata = replay_buffer.sample() # subdata contains ("agents", "active_mask")
                active_mask_sample = subdata[("agents", "active_mask")] # Shape: [minibatch_size, max_total_agents]
                n_active_sample = subdata[("agents", "n_active")] # Shape: [minibatch_size, 1]
                
                # --- Loss Calculation with Masking ---
                loss_vals_td = loss_module(subdata) # This calls policy and critic

                # Policy loss and value loss should be okay if advantages are correctly zero for inactive agents.
                # Entropy is the main concern. ClipPPOLoss calculates it as:
                # entropy = self.actor_network.get_entropy() (if probabilistic actor)
                # loss_entropy = -self.entropy_coef * masked_ops.mean(entropy)
                # We need to ensure this mean is only over active agents or adjust.

                # Get per-agent entropy from the policy's distribution
                # The policy is `loss_module.actor_network`
                # After loss_module(subdata) is called, subdata is populated with intermediate results
                # like 'action_log_prob', and potentially 'dist_entropy' if the policy returns it
                # or if ClipPPOLoss computes and stores it. Let's assume we can get it from the distribution.
                
                # If policy is ProbabilisticActor, its last used distribution is stored.
                # This is a bit of a hack to get the distribution used inside loss_module call.
                # A cleaner way would be for ClipPPOLoss to accept a mask for entropy calculation.
                
                # The `loss_vals_td["loss_entropy"]` is typically `-entropy_coef * entropy.mean()`.
                # We need to re-calculate this if `entropy.mean()` was over all `max_total_agents`.
                # Let's get the distribution from the policy AFTER it has processed `subdata`
                # (which happens inside `loss_module(subdata)` when it computes log_probs)
                
                # The `loss_module` itself uses `policy(subdata_clone)` to get log_probs and potentially entropy
                # `policy.get_dist_params` is called, then `policy.dist_class` is used.
                # Let's assume `policy.get_entropy(average=False)` would give per-agent entropy.
                # Or, more robustly, if `loss_vals_td` contains `("agents", "dist_entropy")` [batch, n_agents, 1]
                # (you might need to ensure your ProbabilisticActor or its distribution provides this)

                # Simplest approach: Assume GNN outputs for inactive agents are such that
                # their predicted distributions have minimal or zero entropy, or that their
                # contribution to advantage (and thus policy gradient) is zero.
                # The most direct way to handle entropy is to make it part of the loss calculation
                # that can be masked.
                
                # If `loss_vals_td["loss_entropy"]` is already a scalar (mean entropy * coef):
                # Try to re-weight. Total possible entropy slots = batch_size * max_total_agents
                # Total active entropy slots = active_mask_sample.sum()
                # If the "mean" in loss_entropy was over max_total_agents, scale it.
                # This is an approximation.
                # current_loss_entropy = loss_vals_td["loss_entropy"]
                # mean_factor = self.max_total_agents / (n_active_sample.float().mean() + 1e-6) # approx
                # corrected_loss_entropy = current_loss_entropy * mean_factor

                # More robust: If your PPO loss can provide 'entropy_loss_per_agent'
                # Or, if 'dist_entropy' is available of shape (batch, n_agents) in subdata or from policy:
                with torch.no_grad(): # Get policy distribution parameters
                    policy(subdata) # Ensure distribution parameters are populated in subdata
                dist = policy.get_dist(subdata) # Get the distribution
                entropy_per_agent = dist.entropy() # Shape [batch_size, max_total_agents, 1] or [batch_size, max_total_agents]
                if entropy_per_agent.ndim > active_mask_sample.ndim: # e.g. [B,N,1] vs [B,N]
                    entropy_per_agent = entropy_per_agent.squeeze(-1)

                masked_entropy = entropy_per_agent * active_mask_sample
                # Average entropy over only active agents per batch item, then average over batch
                # Handle cases where n_active_sample could be zero for a batch item (though unlikely with min_n_agents > 0)
                sum_active_agents_per_env = active_mask_sample.sum(dim=1).float()
                avg_entropy_per_env = masked_entropy.sum(dim=1) / (sum_active_agents_per_env + 1e-6) # Avoid div by zero
                correct_mean_entropy = avg_entropy_per_env.mean() # Mean over batch

                loss_entropy = -loss_module.entropy_coef * correct_mean_entropy
                
                # Replace the potentially incorrect loss_entropy from the module
                loss_vals_td.set("loss_entropy", loss_entropy) # Update the tensordict

                loss_value = (
                    loss_vals_td["loss_objective"]
                    + loss_vals_td["loss_critic"]
                    + loss_vals_td["loss_entropy"] # Use the corrected one
                )

                loss_value.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.train.max_grad_norm
                )
                # Detach loss_vals_td before storing if it was not already
                training_tds.append(loss_vals_td.detach())
                training_tds[-1].set("grad_norm", total_norm.mean().detach())


                optim.step()
                optim.zero_grad()

        collector.update_policy_weights_()
        training_time = time.time() - training_start
        iteration_time = sampling_time + training_time
        total_time += iteration_time
        
        # Stack training_tds, being careful if they don't have identical keys throughout
        # This should be fine if loss_module always returns the same set of keys
        stacked_training_tds = torch.stack(training_tds)


        # ... (logging) ...
        # When logging, you might want to log n_active_agents to see the distribution
        # For example, in log_training:
        # logger.log_scalar("train/n_active_agents_mean", tensordict_data[("agents", "n_active")].float().mean().item(), step=step)

        if (
            cfg.eval.evaluation_episodes > 0
            and i % cfg.eval.evaluation_interval == 0
            and cfg.logger.backend
        ):
            evaluation_start = time.time()
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                env_test.frames = [] # Reset frames for rendering
                # The env_test is already wrapped with VariableAgentWrapper for evaluation
                rollouts = env_test.rollout(
                    max_steps=cfg.env.max_steps,
                    policy=policy,
                    callback=rendering_callback if cfg.eval.render else None, # Add render flag to cfg
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                )
            evaluation_time = time.time() - evaluation_start
            log_evaluation(logger, rollouts, env_test, evaluation_time, step=i) # Ensure env_test has n_agents attribute if logger uses it

            
            if not env_test.is_closed: # Close test env after eval
                env_test.close()


        if cfg.logger.backend == "wandb" and logger:
            logger.experiment.log({}, commit=True)
        
        # No need to restart sampling_start here, it's at the beginning of the loop
        # sampling_start = time.time()

    # Shutdown final training collector and env
    if "collector" in locals() and collector is not None:
        collector.shutdown()
    if "env" in locals() and env is not None and not env.is_closed:
        env.close()
    if "env_test" in locals() and env_test is not None and not env_test.is_closed: # Ensure test_env is also closed
        env_test.close()

        # ... (rest of the loop and shutdown) ...
    # ... (shutdown)




def rendering_callback(env, td):
    # Check if frames attribute exists, if not, create it
    if not hasattr(env, "frames"):
        env.frames = []
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))

# Helper function to create environment
def create_env(cfg, n_agents_override=None, is_eval=False):
    num_envs = cfg.eval.evaluation_episodes if is_eval else cfg.env.vmas_envs
    
    # Determine n_agents
    if n_agents_override is not None:
        current_n_agents = n_agents_override
    elif hasattr(cfg.env.scenario, "n_agents"): # If n_agents is part of scenario config
        current_n_agents = cfg.env.scenario.n_agents
    else: # Fallback or error, VMAS scenarios usually define n_agents
        # This might need adjustment based on how your VMAS scenario expects n_agents
        # For example, some scenarios have n_agents fixed.
        # If your scenario is like "balance" which expects a certain number,
        # you might need different scenario files or a scenario that adapts.
        # For this example, let's assume n_agents can be passed to the scenario config.
        # If n_agents is a direct kwarg to VmasEnv not in scenario:
        # current_n_agents = default_value # or handle error
        raise ValueError("n_agents not specified in cfg.env.scenario and not overridden")

    scenario_kwargs = cfg.env.scenario.copy() # Make a copy to modify
    # Update n_agents in scenario_kwargs if it's a scenario parameter
    # The exact key 'n_agents' might vary based on your specific VMAS scenario implementation
    # Common practice for VMAS scenarios is to have 'n_agents' as a parameter.
    if hasattr(cfg.env.scenario, "n_agents") or n_agents_override is not None:
         scenario_kwargs.n_agents = current_n_agents


    env = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=num_envs,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed + (1000 if is_eval else 0), # Ensure different seed for test env if needed
        # Scenario kwargs
        **scenario_kwargs, # Pass modified scenario arguments
        # If n_agents is a direct argument to VmasEnv, not part of scenario:
        # n_agents=current_n_agents,
    )
    
    # The reward key might change if n_agents changes and results in different group structure
    # However, with a single "agents" group, env.reward_key should remain ("agents", "reward")
    # Ensure DoneTransform uses the correct done_keys from the newly created env
    # and reward_key
    env = TransformedEnv(
        env,
        Compose( # Use Compose for multiple transforms
            RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
            # DoneTransform might need to be adapted if it assumes fixed done_keys structure.
            # However, env.done_keys should reflect the current env.
            DoneTransform(reward_key=env.reward_key, done_keys=env.done_keys)
        )
    )
    return env

if __name__ == "__main__":
    train()
