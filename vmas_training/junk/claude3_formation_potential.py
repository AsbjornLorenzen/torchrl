from __future__ import annotations
from torchrl.objectives.utils import distance_loss

import time
import random  # Added for randomizing agent numbers
import hydra
import torch

import torch.distributions as td
from torch.distributions.transforms import TanhTransform
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.transforms import Compose
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from utils.logging import init_logging, log_evaluation, log_training
from utils.utils import DoneTransform
from models.gnn_actor_variable import GNNActorVariable
from models.gnn_critic import GNNCritic



# TODO: ENSURE THIS IS WORKING. DO SOME DEBUGGING AND TESTS https://claude.ai/share/ff284144-4019-4957-9736-cec9cbcde913
class MaskedTanhNormal(TanhNormal):
    def __init__(self, loc, scale, upscale=5.0, low=None, high=None): # low/high for compatibility
        # Infer mask from loc. Assumes loc for masked agents is exactly 0.
        # A very small epsilon might be needed if loc can be near-zero for active agents.
        self._mask = (loc.abs().sum(dim=-1, keepdim=True) > 1e-8)

        # Ensure scale is positive and small for masked agents to avoid NaNs in log_prob
        # but keep it differentiable.
        scale_safe = torch.where(self._mask.expand_as(scale), scale, torch.ones_like(scale) * 1e-6)

        super().__init__(loc, scale_safe, upscale=upscale)
        # Store low and high if provided, for potential clipping if TanhNormal doesn't handle it perfectly.
        # TanhNormal output is already in (-1, 1) * upscale.
        # If your action space is different, you might need AffineTransform.
        self._low_bound = low
        self._high_bound = high

class SafeNormalParamExtractor(NormalParamExtractor):
    def forward(self, x):
        # Original behavior - split into mean and std
        loc, scale_params = x.chunk(2, dim=-1)
        scale = torch.nn.functional.softplus(scale_params)
        
        # Fix for masked agents - ensure non-zero scale values
        mask = torch.any(loc != 0, dim=-1, keepdim=True)
        scale_safe = scale * mask + 1e-6 * (~mask)
        
        return loc, scale_safe



class VariableAgentsPPOLoss(ClipPPOLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure default keys include agent_mask if you add it to default_keys
        # self.default_keys.update({"agent_mask": ("agents", "agent_mask")})
        self.agent_mask_key = ("agents", "agent_mask")


    def forward(self, tensordict: TensorDict) -> TensorDict:
        # Get the agent mask. Ensure it's present.
        # It should be (batch_size, num_agents_max, 1)
        agent_mask = tensordict.get(self.agent_mask_key)
        if agent_mask is None:
            raise ValueError("Agent mask not found in tensordict. Ensure it's collected and passed.")

        # --- Standard PPO calculations happen here ---
        # The super().forward will calculate advantage, ratio, etc.
        # We need to ensure these calculations respect the mask implicitly or explicitly.
        # GAE advantage calculation:
        # If critic values for masked agents are 0 and rewards are 0,
        # GAE for those agents will naturally be 0.
        # The value_estimator should ideally be aware of the mask or operate on masked inputs.
        # If your GNNCritic outputs 0 for masked agents, this should be fine.

        # Let PPO loss compute as usual first. It will compute for all max_agents.
        # tensordict_out = super().forward(tensordict.clone(False)) # clone(False) for safety if modified

        # --- Modification for masking losses ---
        # The PPO loss components (actor, critic, entropy) are typically summed or averaged
        # over agents. We need to ensure only active agents contribute.

        # Option 1: Modify inputs to super().forward (more involved)
        # Option 2: Modify the outputs of super().forward (simpler if losses are per-agent or can be masked)

        # Let's try to get the loss dictionary and then mask relevant parts.
        # We need to be careful about how advantage and value targets are computed.
        # If value_estimator used masked rewards/values, advantage for masked agents should be 0.

        # Make sure advantage is correctly shaped and masked if necessary
        # Advantage is usually (batch, max_agents, 1)
        adv = tensordict.get("advantage") # This should have been computed by value_estimator
        
        # The PPO loss actor part typically involves: surrogate = advantage * ratio
        # If advantage for masked agents is 0, then their surrogate loss is 0.
        # This is the cleanest way. Ensure your value estimator and critic produce 0s for masked agents.


        # We will let the parent compute the loss and then adjust based on mask.
        # The `loss_stat_keys` in ClipPPOLoss are actor_loss, critic_loss, entropy_loss.
        # These are usually means over batch and agents.
        # We need to ensure the mean is only over *active* agents.

        # To do this properly, we might need to override the loss computation more deeply.
        # The core of actor loss:
        # ratio = (log_prob - old_log_prob).exp()
        # surrogate1 = advantage * ratio
        # surrogate2 = advantage * torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
        # loss_actor = -torch.min(surrogate1, surrogate2)

        # If we mask loss_actor before averaging:
        # loss_actor = loss_actor * agent_mask.squeeze(-1) # Assuming loss_actor is (B, N_agent)
        # loss_actor = loss_actor.sum() / agent_mask.sum() # Mean over active agents

        # Let's try a slightly different approach by overriding the `_loss_actor` part
        # or by masking after the fact.

        # Simpler: rely on advantage and value_target being zero for masked agents.
        # If advantage[masked] = 0, then loss_objective components for them are 0.
        # If value_target[masked] = state_value[masked] = 0, then loss_critic for them is 0.
        # Entropy for masked agents (if their action dist is degenerate, e.g. always 0 action) could be 0 or negative.
        # It's safer to mask entropy term too.

        # Clone to avoid in-place modification issues if any part of super().forward expects originals
        td_clone = tensordict # tensordict.clone(False) # shallow clone
        
        # ---- Logits and old_log_prob processing ----
        # This is handled by the parent class if keys are set right.

        # ---- Value Estimation ----
        # This is typically done *before* calling loss.forward() in the training loop.
        # Ensure your ValueOperator (GNNCritic) outputs 0 for masked agents.

        # ---- Actor Loss ----
        # Based on torchrl.objectives.ppo.ClipPPOLoss._loss_actor
        log_prob = td_clone.get(self.keys.log_prob) # (B, N_max, ActDim) or (B, N_max)
        old_log_prob = td_clone.get(self.keys.old_log_prob) # same shape
        advantage = td_clone.get(self.keys.advantage) # (B, N_max, 1) or (B, N_max)

        # Ensure advantage has same number of dims as log_prob for broadcasting
        if advantage.ndim != log_prob.ndim:
            advantage = advantage.unsqueeze(-1).expand_as(log_prob)

        ratio = (log_prob - old_log_prob).exp()
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(
            ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
        )
        loss_actor = -torch.min(surr1, surr2) # Shape: (B, N_max, ActDim) or (B, N_max)

        # Mask the actor loss:
        # agent_mask could be (B, N_max, 1). Need to match loss_actor dimensions.
        mask_for_actor = agent_mask
        if loss_actor.ndim > mask_for_actor.ndim: # e.g. loss_actor has action_dim
            mask_for_actor = agent_mask.expand_as(loss_actor)
        elif loss_actor.ndim < mask_for_actor.ndim and mask_for_actor.shape[-1] == 1:
            mask_for_actor = agent_mask.squeeze(-1)


        loss_actor = loss_actor * mask_for_actor
        num_active_agents = mask_for_actor.sum()
        if num_active_agents > 0:
            loss_actor = loss_actor.sum() / num_active_agents
        else:
            loss_actor = torch.zeros_like(loss_actor).sum()


        # ---- Value Loss ----
        value = td_clone.get(self.keys.value) # (B, N_max, 1) or (B, N_max)
        value_target = td_clone.get(self.keys.value_target) # (B, N_max, 1) or (B, N_max)
        
        loss_critic = distance_loss(
            value,
            value_target,
            loss_function=self.loss_critic_type,
        ) # Shape: (B, N_max, 1) or (B, N_max)

        # Mask the critic loss:
        mask_for_critic = agent_mask
        if loss_critic.ndim > mask_for_critic.ndim:
             mask_for_critic = agent_mask.expand_as(loss_critic)
        elif loss_critic.ndim < mask_for_critic.ndim and mask_for_critic.shape[-1] == 1:
            mask_for_critic = agent_mask.squeeze(-1)

        loss_critic = loss_critic * mask_for_critic
        num_active_agents_critic = mask_for_critic.sum() # Could be different if value has different shape
        if num_active_agents_critic > 0:
             loss_critic = loss_critic.sum() / num_active_agents_critic
        else:
            loss_critic = torch.zeros_like(loss_critic).sum()
        

        # ---- Entropy Loss ----
        # Actor is ProbabilisticActor, distribution is on tensordict[self.actor_network.dist_key]
        # Or, if log_prob is directly from distribution.log_prob(action), entropy is harder.
        # Usually, entropy is directly from the distribution object.
        if self.entropy_bonus:
            try:
                dist = self.actor_network.get_dist(td_clone) # policy_module_tensordict
            except TypeError: # older TorchRL
                dist = self.actor_network.get_dist_params(td_clone)[0] # first element is the dist

            entropy = dist.entropy() # Shape (B, N_max, ActDim) or (B, N_max)
            if entropy.shape != agent_mask.shape:
                if entropy.ndim > agent_mask.ndim : # entropy has action dim, agent_mask does not
                     mask_for_entropy = agent_mask.expand_as(entropy)
                elif entropy.ndim < agent_mask.ndim and agent_mask.shape[-1] == 1: # agent_mask has extra 1
                     mask_for_entropy = agent_mask.squeeze(-1).expand_as(entropy)
                else: # Need careful broadcasting
                    mask_for_entropy = agent_mask.expand_as(entropy) # Hope for the best
            else:
                mask_for_entropy = agent_mask

            loss_entropy = -self.entropy_coef * entropy * mask_for_entropy
            num_active_agents_entropy = mask_for_entropy.sum()
            if num_active_agents_entropy > 0:
                loss_entropy = loss_entropy.sum() / num_active_agents_entropy
            else:
                loss_entropy = torch.zeros_like(loss_entropy).sum()
        else:
            loss_entropy = torch.zeros_like(loss_actor)

        total_loss = loss_actor + self.critic_coef * loss_critic + loss_entropy
        
        tensordict_out = TensorDict(
            {
                "loss_objective": loss_actor.detach(), # PPO use "loss_objective" for actor loss
                "loss_critic": loss_critic.detach(),
                "loss_entropy": loss_entropy.detach(),
                "loss_total": total_loss, # for backward
            },
            batch_size=td_clone.batch_size,
        )
        # For PPO, actual actor loss is often called "loss_objective"
        tensordict_out["loss_objective"] = loss_actor 
        tensordict_out["loss_total"] = total_loss # Ensure this exists for .backward()

        return tensordict_out


class VariableAgentsWrapper(TransformedEnv):
    def __init__(self, env, min_agents=4, max_agents=8, fixed_num_agents_eval=None):
        super().__init__(env)
        self.min_agents = min_agents
        self.max_agents = max_agents # This is the crucial fixed dimension
        self._fixed_num_agents_eval = fixed_num_agents_eval # For evaluation
        self._is_eval = False # Flag to indicate if we are in evaluation mode

        # This will be set before each episode/rollout
        self.current_num_active_agents = self.max_agents

        # Ensure the base env is created with max_agents
        if hasattr(self.base_env, "n_agents") and self.base_env.n_agents != self.max_agents:
            raise ValueError(
                f"Base environment must be initialized with max_agents ({self.max_agents}), "
                f"but got {self.base_env.n_agents}"
            )

        # Update observation and input specs to include agent_mask
        self._update_specs()

    def _update_specs(self):
        # Add agent_mask to observation_spec
        mask_spec_shape = (*self.base_env.batch_size, self.max_agents, 1)
        agent_mask_spec = torch.zeros(mask_spec_shape, dtype=torch.bool, device=self.device)

        # Grouped keys
        self.observation_spec = self.base_env.observation_spec.clone()
        self.observation_spec["agents", "agent_mask"] = agent_mask_spec.clone()

        if ("agents", "agent_mask") not in self.input_spec.keys(True, True):
             self.input_spec["agents", "agent_mask"] = agent_mask_spec.clone()


    def set_active_agents_for_rollout(self, num_agents: int | None):
        """
        Sets the number of active agents for the next environment reset(s).
        If num_agents is None, it will be randomized.
        """
        if self._is_eval and self._fixed_num_agents_eval is not None:
            self.current_num_active_agents = self._fixed_num_agents_eval
            # torchrl_logger.info(f"Evaluation mode: Active agents set to fixed {self.current_num_active_agents}")
        elif num_agents is None:
            self.current_num_active_agents = random.randint(self.min_agents, self.max_agents)
            # torchrl_logger.info(f"Randomizing active agents: Set to {self.current_num_active_agents}")
        else:
            if not (self.min_agents <= num_agents <= self.max_agents):
                raise ValueError(f"num_agents ({num_agents}) must be between min_agents ({self.min_agents}) and max_agents ({self.max_agents})")
            self.current_num_active_agents = num_agents
            # torchrl_logger.info(f"Active agents for next rollout set to {self.current_num_active_agents}")

    def train(self, mode: bool = True):
        self._is_eval = not mode
        return super().train(mode)

    def eval(self):
        self._is_eval = True
        if self._fixed_num_agents_eval is not None:
            self.set_active_agents_for_rollout(self._fixed_num_agents_eval) # Apply fixed num for eval
        return super().eval()


    def _make_agent_mask(self, batch_size):
        """Helper to create the agent mask."""
        agent_mask = torch.zeros(
            (*batch_size, self.max_agents, 1),
            dtype=torch.bool,
            device=self.device,
        )
        # Activate the first 'current_num_active_agents'
        agent_mask[..., : self.current_num_active_agents, :] = True
        return agent_mask

    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        # If fixed_num_agents_eval is set and we are in eval mode, use it.
        # Otherwise, current_num_active_agents should have been set by set_active_agents_for_rollout
        if self._is_eval and self._fixed_num_agents_eval is not None:
            num_to_activate = self._fixed_num_agents_eval
        else: # Use the number set by set_active_agents_for_rollout (or default if not called)
            num_to_activate = self.current_num_active_agents

        # torchrl_logger.info(f"Resetting with {num_to_activate} active agents. Is_eval: {self._is_eval}")


        # The base environment always operates with self.max_agents slots
        td_reset = self.base_env._reset(tensordict, **kwargs)

        # Determine batch_size from the reset tensordict
        # Assuming observation is like ("agents", "observation")
        obs_key = ("agents", "observation") # Or get it from spec
        batch_size = td_reset.get(obs_key).shape[:-2] # Get batch dimensions

        agent_mask = self._make_agent_mask(batch_size)
        td_reset.set(("agents", "agent_mask"), agent_mask)

        # Mask observations for inactive agents
        # Assuming observation is shaped (batch, n_agents, obs_dim)
        obs = td_reset.get(obs_key)
        # The mask needs to be expanded to match the observation dimension if obs_dim > 1
        # agent_mask for obs needs shape (batch, n_agents, obs_dim)
        # current agent_mask is (batch, n_agents, 1)
        expanded_mask_obs = agent_mask.expand_as(obs)
        obs = obs * expanded_mask_obs
        td_reset.set(obs_key, obs)
        return td_reset

    def _step(self, tensordict: TensorDict) -> TensorDict:
        # Agent mask should be in the input tensordict from the policy
        # If not, it might be the first step after reset where it's not yet passed by policy
        if ("agents", "agent_mask") in tensordict.keys(True, True):
            agent_mask = tensordict.get(("agents", "agent_mask"))
        else: # Should have been set by _reset, or needs to be created if stateful
             # For stateless, it should be passed. For now, let's assume it's there or from self.
            obs_key = ("agents", "observation")
            batch_size = tensordict.get(obs_key).shape[:-2]
            agent_mask = self._make_agent_mask(batch_size)
            tensordict.set(("agents", "agent_mask"), agent_mask)


        # Mask actions for inactive agents before passing to base_env
        # Assuming action is shaped (batch, n_agents, action_dim)
        action_key = self.base_env.action_key # e.g. ("agents", "action")
        actions = tensordict.get(action_key)

        # agent_mask for actions needs shape (batch, n_agents, action_dim)
        expanded_mask_act = agent_mask.expand_as(actions)
        masked_actions = actions * expanded_mask_act
        tensordict.set(action_key, masked_actions)

        # Step the base environment
        next_td = self.base_env._step(tensordict)

        # Propagate the agent_mask to the next tensordict
        next_td.set(("agents", "agent_mask"), agent_mask)

        # Mask next_observations for inactive agents
        # Assuming observation is ("agents", "observation")
        next_obs_key = ("next", "agents", "observation") # Standard for next state
        if next_obs_key in next_td.keys(True,True):
            next_obs = next_td.get(next_obs_key)
            expanded_mask_next_obs = agent_mask.expand_as(next_obs)
            next_obs = next_obs * expanded_mask_next_obs
            next_td.set(next_obs_key, next_obs)

        # Mask rewards for inactive agents
        # Assuming reward is ("agents", "reward")
        reward_key = self.base_env.reward_key
        rewards = next_td.get(reward_key)
        # reward_mask is (batch, n_agents, 1), which should broadcast correctly
        masked_rewards = rewards * agent_mask
        next_td.set(reward_key, masked_rewards)

        # Mask dones for inactive agents (important for GAE)
        # Typically, inactive agents should be considered 'done' or their 'done' signal irrelevant.
        # If their reward is 0 and value is 0, GAE will handle it.
        # For safety, ensure done state is consestent.
        # Let's assume done for inactive agents is True, or their contributions are zeroed.
        for dk_tuple in self.done_keys: # self.done_keys are like [("agents","done"), ("agents","terminated")]
            if dk_tuple in next_td.keys(True, True):
                done_val = next_td.get(dk_tuple)
                # For inactive agents, their 'done' status can be set to True or their values zeroed.
                # If rewards and values are zeroed, their specific done state might not matter as much for GAE.
                # However, for consistency:
                # done_val = done_val | (~agent_mask) # Mark inactive as done
                # Or ensure their contributions are zero. Since rewards are zeroed, this is often sufficient.
                # Let's ensure values for inactive agents are zeroed by the critic.

        return next_td

    def forward(self, tensordict: TensorDict) -> TensorDict:
        # This method is called by TransformedEnv._call
        # We need to ensure transformations are applied correctly.
        # The _step and _reset methods handle the core logic.
        # This primarily calls the base_env's forward and applies transforms.
        return super().forward(tensordict)

    # Optional: If you need to expose a way to get the true number of agents for logging
    def get_current_num_active_agents(self):
        return self.current_num_active_agents


def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))

class VmasEnvFactory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env_cache = {}  # Cache environments by agent count
        
    def get_env(self, n_agents):
        """Get an environment with the specified number of agents"""
        if n_agents not in self.env_cache:
            # Create a new environment with exactly n_agents
            base_env = VmasEnv(
                scenario=self.cfg.env.scenario_name,
                num_envs=self.cfg.env.vmas_envs,
                n_agents=n_agents,
                continuous_actions=True,
                max_steps=self.cfg.env.max_steps,
                device=self.cfg.env.device,
                seed=self.cfg.seed,
            )
            
            env = TransformedEnv(
                base_env,
                RewardSum(in_keys=[base_env.reward_key], 
                           out_keys=[("agents", "episode_reward")]),
            )
            
            self.env_cache[n_agents] = env
            
        return self.env_cache[n_agents]
    
    def close_all(self):
        """Close all environments in the cache"""
        for env in self.env_cache.values():
            if not env.is_closed:
                env.close()


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


    # --- Environment Setup ---
    # Base VMAS env always uses MAX_AGENTS
    max_agents_for_scenario = cfg.env.get("max_agents", 8)
    min_agents_for_scenario = cfg.env.get("min_agents", 4)

    base_env_train = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.env.vmas_envs, # This is batch_size for env
        n_agents=max_agents_for_scenario, # CRITICAL: Base env has max_agents
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
    )

    # Wrap with VariableAgentsWrapper
    variable_agents_env = VariableAgentsWrapper(
        base_env_train,
        min_agents=min_agents_for_scenario,
        max_agents=max_agents_for_scenario,
    )

    # Use group map to get keys if they are grouped
    reward_key_ingroup = base_env_train.reward_key # e.g., "reward"
    reward_key_grouped = reward_key_ingroup

    # Done keys are usually a list of tuples already, like [('agents', 'done'), ('agents', 'terminated')]
    # If not, map them:
    done_keys_grouped = []
    for dk_tuple_or_str in base_env_train.done_keys:
        if isinstance(dk_tuple_or_str, str): # e.g. "done"
            done_keys_grouped.append(("agents", dk_tuple_or_str))
        else: # e.g. ("agents", "done")
            done_keys_grouped.append(dk_tuple_or_str)


    env = TransformedEnv(
        variable_agents_env,
        Compose( # Use Compose for multiple transforms
            RewardSum(in_keys=[reward_key_grouped],
                      out_keys=[("agents", "episode_reward")]),
            DoneTransform(reward_key=reward_key_grouped, # Pass the correct, possibly grouped, reward key
                          done_keys=done_keys_grouped) # Pass correct done keys
        )
    )



    # --- Test Environment ---
    # Evaluation environment also uses VariableAgentsWrapper but with fixed_num_agents_eval
    eval_n_agents = cfg.eval.get("num_agents", max_agents_for_scenario)
    base_env_test = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.eval.evaluation_episodes,
        n_agents=max_agents_for_scenario, # Base test env also has max_agents
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed + 1, # Different seed for test env
    )
    variable_agents_env_test = VariableAgentsWrapper(
        base_env_test,
        min_agents=min_agents_for_scenario, # Min/max are for training randomization range
        max_agents=max_agents_for_scenario,
        fixed_num_agents_eval=eval_n_agents
    )
    env_test = TransformedEnv(
        variable_agents_env_test,
        Compose( # Use Compose for multiple transforms
             RewardSum(in_keys=[reward_key_grouped],
                      out_keys=[("agents", "episode_reward")]),
            # DoneTransform not strictly needed for eval rollout if not used for metric
        )
    )
    env_test.eval() # Set to evaluation mode
    
    # GNN POLICY - Modified to handle agent_mask
    gnn_hidden_dim = cfg.model.get("gnn_hidden_dim", 128)
    gnn_layers = cfg.model.get("gnn_layers", 2)
    k_neighbours = cfg.model.get("k_neighbours", None) # Default to fully connected if not specified
    pos_indices_list = cfg.model.get("pos_indices", [0, 2]) # Default to first 2
    pos_indices = slice(pos_indices_list[0], pos_indices_list[1])
    
    # Modify GNN actor to handle agent_mask
    actor_obs_key_tuple = ("agents", "observation") # Default for VMAS

    actor_net = nn.Sequential(
        GNNActorVariable(
            n_agent_inputs=env.observation_spec[actor_obs_key_tuple].shape[-1],
            n_agent_outputs=2 * env.action_spec.shape[-1], # Make sure key is right
            gnn_hidden_dim=gnn_hidden_dim,
            n_gnn_layers=gnn_layers,
            activation_class=nn.Tanh,
            k_neighbours=k_neighbours,
            pos_indices=pos_indices,
            share_params=cfg.model.shared_parameters,
            device=cfg.train.device,
            # Add agent_mask support in your GNN implementation
        ),
        SafeNormalParamExtractor(),
    )
    policy_module = TensorDictModule(
        actor_net,
        in_keys=[actor_obs_key_tuple], # Add ("agents", "agent_mask") if GNN uses it
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    
    lowest_action = torch.zeros_like(env.full_action_spec_unbatched[("agents", "action")].space.low, device=cfg.train.device)
    # Action spec for ProbabilisticActor needs to be for a single agent group, unbatched
    # It should reflect the action space of ONE agent slot out of MAX_AGENTS
    # env.action_spec["agents", "action"] is usually the right one for MA settings
    # print(f"Action spec for policy: {env.action_spec[('agents', 'action')]}")

    # TODO: Redo this to use at least action 0
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec, # Spec for one agent in the group
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[("agents","action")], # Use tuple key
        distribution_class=MaskedTanhNormal,
        distribution_kwargs={
            # TanhNormal typically outputs in (-1,1) range, then scaled by upscale
            # If your env has different action bounds, you might need an AffineTransform
            # or ensure MaskedTanhNormal's internal scaling matches.
            # For now, let's assume action space is symmetric and TanhNormal handles it.
            "upscale": torch.tensor(env.action_spec.space.high[0], device=cfg.train.device) # Assuming symmetric
        },
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"), # Grouped log_prob key
    )

    # Modify GNN critic to handle agent_mask
    critic_module_net = GNNCritic(
        n_agent_inputs=env.observation_spec[actor_obs_key_tuple].shape[-1],
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
        module=critic_module_net,
        in_keys=[actor_obs_key_tuple], # Add ("agents", "agent_mask") if GNN uses it
        out_keys=[("agents", "state_value")], # Grouped value key
    )

    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size = cfg.collector.frames_per_batch # Or larger for off-policy

    collector = SyncDataCollector(
        env, # The single, wrapped training environment
        policy,
        device=cfg.env.device,
        storing_device=cfg.train.device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        # postproc already in env via Compose
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.train.minibatch_size,
        # Ensure keys from collector are stored, including "agent_mask"
    )

    loss_module = VariableAgentsPPOLoss( # Use your custom loss
        actor_network=policy,
        critic_network=value_module,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coef=cfg.loss.entropy_eps,
        normalize_advantage=cfg.loss.get("normalize_advantage", False), # Add to config if needed
        loss_critic_type=cfg.loss.get("loss_critic_type", "smooth_l1"), # Add to config
    )

    loss_module.set_keys(
        reward=reward_key_grouped, # e.g. ("agents", "reward")
        action=("agents", "action"),
        sample_log_prob=("agents", "sample_log_prob"), # Make sure policy outputs this
        value=("agents", "state_value"),
        # done and terminated are usually nested under "next" by the collector for GAE
        # but PPO loss might need them at top level of sample. Check PPO specific needs.
        # For GAE, it needs td["next", group, done_key_leaf]
        # DoneTransform places them at root of agent group.
        done="done",  # Update to match the actual path in your TensorDict
        terminated="terminated",  # Update to match the actual path in your TensorDict
        agent_mask=("agents", "agent_mask") # CRITICAL
    )
    # Value estimator (GAE)
    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=cfg.loss.gamma, lmbda=cfg.loss.lmbda
    )
    
    optim = torch.optim.Adam(loss_module.parameters(), cfg.train.lr)
    env_factory = VmasEnvFactory(cfg)

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
    sampling_start = time.time()


    # --- Training Loop ---
    total_time = 0
    total_frames_collected = 0 # Renamed to avoid conflict with collector's total_frames
    
    for i in range(cfg.collector.n_iters):
        sampling_start = time.time()
        # Set number of active agents for this collection phase
        num_active_agents_for_rollout = random.randint(
            min_agents_for_scenario, max_agents_for_scenario
        )
        # Access VariableAgentsWrapper: env.env is VariableAgentsWrapper if it's the first wrapper
        # Or if multiple wrappers: env.get_nested_env_by_type(VariableAgentsWrapper)
        # Assuming VariableAgentsWrapper is the first one after base_env:
        env.base_env.set_active_agents_for_rollout(num_active_agents_for_rollout)

        torchrl_logger.info(
            f"\nIteration {i}: Starting data collection with "
            f"{env.base_env.get_current_num_active_agents()} active agents."
        )

        # Collect data
        # The collector will call env.reset(), which will use current_num_active_agents
        tensordict_data = collector.next() # Gets one frames_per_batch

        # Check for agent_mask in collected data
        if ("agents", "agent_mask") not in tensordict_data.keys(True,True):
            torchrl_logger.error("CRITICAL: agent_mask not found in collected data!")
        # else:
        #     torchrl_logger.info(f"Agent mask found, sum: {tensordict_data['agents', 'agent_mask'].sum()}")


        sampling_time = time.time() - sampling_start

        # --- GAE Computation ---
        # This should be done on the collected data *before* adding to replay buffer if GAE writes to it
        with torch.no_grad():
            loss_module.value_estimator( # This computes 'advantage' and 'value_target'
                tensordict_data,
                params=loss_module.critic_network_params, # Functional call
                target_params=loss_module.target_critic_network_params, # Functional call
            )
        
        current_frames = tensordict_data.numel() # frames_per_batch
        total_frames_collected += current_frames
        
        # Add to replay buffer. Data already has max_agents dimension.
        # data_view might not be needed if buffer handles batch_locked TensorDicts
        # Flattening along time dim if data is (time, batch_env, agents, ...)
        # Collector usually returns (batch_env * time, agents, ...) if frames_per_batch includes time
        # Assuming tensordict_data is already (total_frames_in_batch, max_agents, ...)
        replay_buffer.extend(tensordict_data.reshape(-1)) # Reshape to (N, max_agents, ...)

        # --- Training Phase ---
        training_start = time.time()
        training_tds_logs = [] # For logging
        for epoch in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                subdata = replay_buffer.sample() # Samples (minibatch_size, max_agents, ...)
                if ("agents", "agent_mask") not in subdata.keys(True,True):
                     torchrl_logger.error("CRITICAL: agent_mask not found in subdata for training!")
                     # This means it wasn't stored or retrieved correctly by buffer.
                # else:
                #      torchrl_logger.info(f"Sampled agent mask sum: {subdata['agents', 'agent_mask'].sum()}")


                loss_vals = loss_module(subdata) # Pass subdata to your custom loss
                loss_total = loss_vals["loss_total"] # Key from your VariableAgentsPPOLoss

                optim.zero_grad()
                loss_total.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.train.max_grad_norm
                )
                optim.step()

                # Log detached loss values
                log_td = TensorDict({k: v.detach() for k,v in loss_vals.items() if "loss" in k}, batch_size=[])
                log_td.set("grad_norm", grad_norm.mean().detach())
                training_tds_logs.append(log_td)
        
        training_time = time.time() - training_start
        
        collector.update_policy_weights_() # Update policy in collector if needed (for on-policy)

        iteration_time = sampling_time + training_time
        total_time += iteration_time
        
        # Stack logs
        if training_tds_logs:
            training_tds_stacked = torch.stack(training_tds_logs).apply(lambda x: x.mean(), batch_size=[])
        else:
            training_tds_stacked = TensorDict({}, batch_size=[])


        # --- Logging ---
        if logger and cfg.logger.backend:
            log_training(
                logger,
                training_tds_stacked, # Aggregated training losses
                tensordict_data,    # Data from collection phase (for episode stats)
                sampling_time,
                training_time,
                total_time,
                i, # iteration number
                current_frames,
                total_frames_collected,
                step=i, # Use iteration as step for logger
                active_agents_in_iteration=env.env.get_current_num_active_agents() # Log active agents
            )
            
        # --- Evaluation ---
        if (
            cfg.eval.evaluation_episodes > 0
            and i % cfg.eval.evaluation_interval == 0
            and logger and cfg.logger.backend
        ):
            evaluation_start = time.time()
            env_test.eval() # Set to evaluation mode (uses fixed_num_agents_eval)
            torchrl_logger.info(
                f"Evaluation: Starting with "
                f"{env_test.env.get_current_num_active_agents()} active agents."
            )
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                env_test.frames = [] # Reset frames for rendering
                rollouts = env_test.rollout(
                    max_steps=cfg.env.max_steps,
                    policy=policy,
                    callback=rendering_callback if cfg.eval.render else None,
                    auto_cast_to_device=True,
                    break_when_any_done=False, # Complete all envs
                )
            evaluation_time = time.time() - evaluation_start
            log_evaluation(logger, rollouts, env_test, evaluation_time, step=i)
            env.train() # Set training env back to train mode

        # For next iteration's data collection
        sampling_start = time.time() 

    collector.shutdown()
    if not env.is_closed:
        env.close()
    if not env_test.is_closed:
        env_test.close()
    if logger and cfg.logger.backend == "wandb":
        logger.experiment.finish()


if __name__ == "__main__":
    train()
