# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import time

import hydra
import torch
from torch import nn
from tensordict import TensorDict

# --- PyG Imports ---
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv # Example GNN layer
    from torch_geometric.data import Data, Batch
    _has_pyg = True
except ImportError:
    _has_pyg = False
    # Handle the case where PyG is not installed if needed
    print("PyTorch Geometric not found. GNN functionality will be unavailable.")
    # You might want to raise an error or fall back to MLP here

# --- TorchRL Imports ---
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
# Remove MultiAgentMLP import if no longer used elsewhere (or keep if critic still uses it)
# from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Assuming these utils exist from your original code
from utils.logging import init_logging, log_evaluation, log_training
from utils.utils import DoneTransform

# --- GNN Actor Module Definition ---
class GNNActorVariable(nn.Module):
    """
    A GNN-based actor network module for multi-agent RL.

    Assumes the input observation tensor for each agent contains its features,
    and positions are included (e.g., at the beginning) to build the graph.
    Handles batching by creating a PyG Batch object internally.
    """
    def __init__(
        self,
        n_agent_inputs: int,
        n_agent_outputs: int,
        gnn_hidden_dim: int = 128,
        n_gnn_layers: int = 2,
        activation_class=nn.Tanh,
        k_neighbours: float | None = None,
        pos_indices: slice = slice(0, 2), # Indices for XY position in observation
        share_params: bool = True, # GNNs inherently share params across nodes/agents
                                   # This flag is kept for consistency maybe? But GNN handles sharing.
        device = None, # Added device parameter
    ):
        super().__init__()
        if not _has_pyg:
            raise ImportError("PyTorch Geometric is required for GNNActor.")

        self.n_agent_inputs = n_agent_inputs
        self.n_agent_outputs = n_agent_outputs
        self.k_neighbours = k_neighbours
        self.pos_indices = pos_indices
        self.device = device # Store device

        # Define GNN layers
        self.gnn_layers = nn.ModuleList()
        input_dim = n_agent_inputs
        for _ in range(n_gnn_layers):
            # Using GCNConv as an example
            self.gnn_layers.append(GCNConv(input_dim, gnn_hidden_dim))
            input_dim = gnn_hidden_dim

        # Output MLP head for each agent
        self.output_mlp = nn.Linear(gnn_hidden_dim, n_agent_outputs)
        self.activation = activation_class()


    def _build_graph_batch(self, obs: torch.Tensor, agent_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Builds PyG batch graph data from batched observations, including self-loops.
        Takes into account agent_mask to only include active agents in the graph.

        Args:
            obs (torch.Tensor): Observation tensor of shape (batch_size, n_agents, obs_dim)
            agent_mask (torch.Tensor): Boolean mask [batch_size, n_agents, 1] where True indicates active agents

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor):
                - x (torch.Tensor): Node features, shape (total_active_agents, obs_dim)
                - edge_index (torch.Tensor): Edge indices, shape (2, num_total_edges)
                - batch_vector (torch.Tensor): Maps each node to its batch index, shape (total_active_agents)
        """
        batch_size, n_agents, obs_dim = obs.shape
        
        # Reshape agent_mask to [batch_size, n_agents]
        if agent_mask.dim() == 3 and agent_mask.size(2) == 1:
            agent_mask = agent_mask.squeeze(-1)
        
        # Count total active agents
        total_active_agents = agent_mask.sum().item()
        
        if total_active_agents == 0:  # Handle case with no active agents
            return (torch.empty(0, obs_dim, device=self.device),
                    torch.empty(2, 0, dtype=torch.long, device=self.device),
                    torch.empty(0, dtype=torch.long, device=self.device))
        
        # Create mapping from original indices to new condensed indices
        # For each batch element, create a tensor where active agents get sequential indices
        # and inactive agents get -1
        new_indices = torch.full((batch_size, n_agents), -1, dtype=torch.long, device=self.device)
        current_idx = 0
        for b in range(batch_size):
            active_count = 0
            for a in range(n_agents):
                if agent_mask[b, a]:
                    new_indices[b, a] = current_idx
                    current_idx += 1
                    active_count += 1
        
        # Extract only active agent observations
        x_list = []
        batch_vector_list = []
        pos_list = []
        batch_map = []  # Maps each active node to its batch index
        
        for b in range(batch_size):
            batch_active_mask = agent_mask[b]
            if batch_active_mask.sum() > 0:  # Only process batch elements with active agents
                x_list.append(obs[b, batch_active_mask])
                pos_list.append(obs[b, batch_active_mask, self.pos_indices])
                batch_vector_list.append(torch.full((batch_active_mask.sum(),), b, dtype=torch.long, device=self.device))
                batch_map.append(torch.arange(batch_active_mask.sum(), device=self.device) + 
                               (0 if not batch_map else batch_map[-1][-1] + 1))
        
        # Concatenate features and positions
        x = torch.cat(x_list, dim=0) if x_list else torch.empty(0, obs_dim, device=self.device)
        pos = torch.cat(pos_list, dim=0) if pos_list else torch.empty(0, len(self.pos_indices), device=self.device)
        batch_vector = torch.cat(batch_vector_list, dim=0) if batch_vector_list else torch.empty(0, dtype=torch.long, device=self.device)
        
        # If no agents are active, return empty tensors
        if total_active_agents == 0:
            return (x, 
                    torch.empty(2, 0, dtype=torch.long, device=self.device),
                    batch_vector)
        
        # Building edges only between active agents
        edge_index_list = []
        
        # Process each batch element separately
        current_offset = 0
        for b in range(batch_size):
            active_agents = agent_mask[b].sum().item()
            if active_agents == 0:
                continue
                
            # Get positions for this batch
            batch_pos = pos_list[b]  # Shape: [active_agents, pos_dim]
            
            # Calculate pairwise distances within this batch element
            batch_dist = torch.cdist(batch_pos, batch_pos, p=2)  # Shape: [active_agents, active_agents]
            
            # Find k-nearest neighbors for active agents
            if self.k_neighbours is not None and self.k_neighbours > 0 and active_agents > 1:
                # Limit k to number of active agents - 1 (for non-self connections)
                k_actual = min(self.k_neighbours + 1, active_agents)
                
                # Get top-k indices
                _, knn_idx = torch.topk(batch_dist, k=k_actual, dim=-1, largest=False, sorted=True)
                
                # Handle self-loops and k-NN edges separately
                if k_actual > 1:  # Ensure there are neighbors besides self
                    # Get indices of k-NN (excluding self at index 0)
                    neighbor_idx = knn_idx[:, 1:k_actual]  # Shape: [active_agents, k_actual-1]
                    source_idx = torch.arange(active_agents, device=self.device).view(-1, 1).expand(-1, k_actual-1)
                    
                    # Add offset for current batch
                    source_with_offset = source_idx + current_offset
                    target_with_offset = neighbor_idx + current_offset
                    
                    # Create edge index for k-NN
                    knn_edge_index = torch.stack([
                        source_with_offset.reshape(-1), 
                        target_with_offset.reshape(-1)
                    ], dim=0)
                    
                    edge_index_list.append(knn_edge_index)
            
            # Always add self-loops for active agents
            self_loop_indices = torch.arange(active_agents, device=self.device) + current_offset
            self_loops = torch.stack([self_loop_indices, self_loop_indices], dim=0)
            edge_index_list.append(self_loops)
            
            # Update offset for next batch
            current_offset += active_agents
        
        # Combine all edges
        edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.empty(2, 0, dtype=torch.long, device=self.device)
        
        return x, edge_index, batch_vector

    def forward(self, agent_observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GNN actor.
        
        Args:
            agent_observations (torch.Tensor): Tensor of shape (batch_size, n_agents, obs_dim)
            agent_mask (torch.Tensor): Boolean mask of shape (batch_size, n_agents, 1) or (batch_size, n_agents)
                                       where True indicates active agents
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_agents, n_agent_outputs)
                         with zeros for masked (inactive) agents
        """
        # Ensure input is on the correct device
        obs = agent_observations.to(dtype=torch.float32)
        # Get batch dimensions
        batch_size, n_agents, obs_dim = obs.shape
        agent_mask = torch.any(obs != 0, dim=2)
        # print(f"In gnn forward, got obs {obs[:,:,0]} and mask {agent_mask}")
        
        # Reshape agent_mask if needed
        if agent_mask.dim() == 3 and agent_mask.size(2) == 1:
            agent_mask = agent_mask.squeeze(-1)
        
        output_dim = self.n_agent_outputs
        
        # Handle empty batch/no agents case
        if batch_size == 0 or n_agents == 0:
            return torch.zeros(batch_size, n_agents, output_dim, device=self.device)
        
        # Count total active agents
        total_active_agents = agent_mask.sum().item()
        
        # If no active agents, return zeros
        if total_active_agents == 0:
            return torch.zeros(batch_size, n_agents, output_dim, device=self.device)
        
        # print(f"Agent mask is currently {agent_mask}")
        # Create a mapping from original agent indices to condensed indices
        condensed_indices = torch.full((batch_size, n_agents), -1, dtype=torch.long, device=self.device)
        current_idx = 0
        for b in range(batch_size):
            for a in range(n_agents):
                if agent_mask[b, a]:
                    condensed_indices[b, a] = current_idx
                    current_idx += 1
        
        # Build the graph structure for the batch (using only active agents)
        x, edge_index, batch_vector = self._build_graph_batch(obs, agent_mask)
        
        # If there are active agents, process through GNN
        if x.size(0) > 0:
            # Pass through GNN layers
            for layer in self.gnn_layers:
                x = layer(x, edge_index)
                x = self.activation(x)
            
            # Pass through final MLP head
            agent_outputs = self.output_mlp(x)  # Shape: (total_active_agents, output_dim)
        else:
            # This shouldn't happen due to earlier check, but just in case
            agent_outputs = torch.empty(0, output_dim, device=self.device)
        
        # Initialize output tensor with zeros (for all agents including masked ones)
        final_output = torch.zeros(batch_size, n_agents, output_dim, device=self.device)
        
        # Place the computed outputs in the correct positions according to the mask
        current_idx = 0
        for b in range(batch_size):
            for a in range(n_agents):
                if agent_mask[b, a]:
                    final_output[b, a] = agent_outputs[current_idx]
                    current_idx += 1
        
        return final_output

# --- Main Training Function Modification ---
@hydra.main(version_base="1.1", config_path="", config_name="mappo_ippo")
def train(cfg: "DictConfig"):  # noqa: F821
    if not _has_pyg:
         print("Cannot train with GNN Actor: PyTorch Geometric not installed.")
         return

    # Device
    # Ensure device is set early and consistently
    cfg.train.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.env.device = cfg.train.device # Assuming env runs on the same device for simplicity

    # Seeding
    torch.manual_seed(cfg.seed)

    # Sampling
    # ... (rest of your config setup)

    # Create env and env_test
    # ... (same as before) ...
    env = VmasEnv( # Make sure device is passed correctly
        scenario=cfg.env.scenario_name,
        num_envs=cfg.env.vmas_envs,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device, # Pass device here
        seed=cfg.seed,
        # Scenario kwargs
        **cfg.env.scenario,
    )
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    # ... (env_test setup) ...


    # --- Policy Modification ---
    # Check your observation spec to confirm position indices if needed
    print("Observation spec:", env.observation_spec[("agents", "observation")].shape[-1])
    # Make sure cfg.model contains GNN specific params like gnn_hidden_dim, gnn_radius etc.
    # Example: Add these to your mappo_ippo.yaml or defaults
    # model:
    #   shared_parameters: True
    #   centralised_critic: False # Or True
    #   gnn_hidden_dim: 128
    #   gnn_layers: 2
    #   gnn_radius: 5.0 # Or null for fully connected
    #   pos_indices: [0, 2] # Example: If positions are first 2 elements

    # Extract GNN parameters from config
    gnn_hidden_dim = cfg.model.get("gnn_hidden_dim", 128)
    gnn_layers = cfg.model.get("gnn_layers", 2)
    gnn_radius = cfg.model.get("gnn_radius", None) # Default to fully connected if not specified
    pos_indices_list = cfg.model.get("pos_indices", [0, 2]) # Default to first 2
    pos_indices = slice(pos_indices_list[0], pos_indices_list[1])


    actor_gnn_module = GNNActor(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=2 * env.action_spec.shape[-1], # For loc and scale
        n_agents=env.n_agents,
        gnn_hidden_dim=gnn_hidden_dim,
        n_gnn_layers=gnn_layers,
        activation_class=nn.Tanh, # Or cfg.model.activation
        gnn_radius=gnn_radius,
        pos_indices=pos_indices,
        share_params=cfg.model.shared_parameters, # Pass sharing info if needed
        device=cfg.train.device # Pass the device
    )

    # The rest of the actor network definition uses the GNN module
    actor_net = nn.Sequential(
        actor_gnn_module, # Use the GNN module here
        NormalParamExtractor(),
    ).to(cfg.train.device) # Ensure the whole sequential module is on the correct device


    policy_module = TensorDictModule(
        module=actor_net, # Pass the nn.Sequential containing the GNN
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    # policy_module = policy_module.to(cfg.train.device) # TensorDictModule usually handles device internally based on contained module

    # ProbabilisticActor definition remains the same
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.full_action_spec_unbatched,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.full_action_spec_unbatched[("agents", "action")].space.low,
            "high": env.full_action_spec_unbatched[("agents", "action")].space.high,
        },
        return_log_prob=True,
    )
    # policy = policy.to(cfg.train.device) # ProbabilisticActor also often handles device via its module


    # --- Critic ---
    # IMPORTANT: Decide if your critic also needs to be a GNN.
    # If cfg.model.centralised_critic is True, the critic needs access to all agent observations.
    # A centralized GNN critic would typically build one large graph using all agent info.
    # If cfg.model.centralised_critic is False (IPPO), the critic can remain a decentralized MLP
    # or become a decentralized GNN (similar structure to the actor GNN).
    # For now, let's assume the critic remains an MLP as in your original code.
    # If you want a GNN critic, you'll need a similar GNN module definition for it.

    # Using the original MultiAgentMLP for the critic for now
    # Need to re-import if you removed it earlier
    from torchrl.modules.models.multiagent import MultiAgentMLP
    critic_module = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralised=cfg.model.centralised_critic, # Important flag
        share_params=cfg.model.shared_parameters,
        device=cfg.train.device,
        depth=2,
        num_cells=256,
        activation_class=nn.Tanh,
    )
    value_module = ValueOperator(
        module=critic_module,
        # Adjust in_keys based on whether it's centralized or not
        in_keys=[("agents", "observation")], # Decentralized / IPPO case
        # For centralized critic, you might need a different key or process
        # the observation differently before passing it. VMAS might provide a specific
        # 'state' key for centralized critics, check env specs.
    ).to(cfg.train.device) # Ensure critic is on the device


    # --- Collector, Buffer, Loss, Optimizer, Logging ---
    # These parts generally remain the same, but ensure device consistency
    collector = SyncDataCollector(
        env,
        policy,
        device=cfg.env.device, # Use env device for collection step
        storing_device=cfg.train.device, # Use train device for storage
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        postproc=DoneTransform(reward_key=env.reward_key, done_keys=env.done_keys),
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.train.minibatch_size,
    )

    # Loss module needs the updated policy and value_module
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value_module,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coef=cfg.loss.entropy_eps,
        normalize_advantage=False, # As per original code
    ).to(cfg.train.device) # Ensure loss module is on the device

    loss_module.set_keys(
        reward=env.reward_key,
        action=env.action_key,
        # Ensure these done/terminated keys match VMAS output
        done=("agents", "done") if ("agents", "done") in env.done_keys else env.done_keys[0],
        terminated=("agents", "terminated") if ("agents", "terminated") in env.done_keys else env.done_keys[-1], # Best guess
        # It might be just "done" and "terminated" at the root level in VMAS? Check env spec.
    )
    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=cfg.loss.gamma, lmbda=cfg.loss.lmbda
    )
    # Optimizer targets the parameters of the loss module (which includes actor and critic)
    optim = torch.optim.Adam(loss_module.parameters(), cfg.train.lr)

    # --- Logging ---
    # ... (same as before, ensure logger is initialized correctly) ...

    # --- Training Loop ---
    # ... (The training loop logic remains largely the same) ...
    # Make sure data tensors are moved to cfg.train.device if necessary before loss calculation,
    # although the collector's storing_device and buffer's device should handle this.

    total_time = 0
    total_frames = 0
    sampling_start = time.time()
    for i, tensordict_data in enumerate(collector):
        torchrl_logger.info(f"\nIteration {i}")
        sampling_time = time.time() - sampling_start

        # Ensure data used for value estimation is on the correct device
        tensordict_data = tensordict_data.to(cfg.train.device)

        # GAE Calculation
        with torch.no_grad():
            loss_module.value_estimator(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params, # Not used by GAE but good practice
            )
        current_frames = tensordict_data.numel() # Num environments in batch
        current_frames_agents = current_frames * env.n_agents # Total agent steps if needed
        total_frames += current_frames_agents # Accumulate total agent steps

        # --- Training Epochs ---
        training_tds = []
        training_start = time.time()
        # Reshape data for buffer (batch_size * sequence_length, ...)
        # Ensure done/terminated shapes are handled correctly if they differ from reward
        # The view used depends on how DoneTransform structures the output
        data_view = tensordict_data.reshape(-1) # Flatten env and time dimensions
        replay_buffer.extend(data_view)

        for _ in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                # Sample should already be on cfg.train.device
                subdata = replay_buffer.sample()
                # Ensure subdata is on the correct device before passing to loss
                # Although buffer should handle this
                subdata = subdata.to(cfg.train.device)

                loss_vals = loss_module(subdata)
                training_tds.append(loss_vals.detach())

                loss_policy = loss_vals["loss_objective"]
                loss_critic = loss_vals["loss_critic"]
                loss_entropy = loss_vals["loss_entropy"]
                loss_value = loss_policy + loss_critic + loss_entropy

                loss_value.backward()

                # Grad clipping
                total_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.train.max_grad_norm
                )
                training_tds[-1].set("grad_norm", total_norm.mean()) # Log mean grad norm

                optim.step()
                optim.zero_grad()

        # Update collector policy weights
        collector.update_policy_weights_()

        training_time = time.time() - training_start
        # ... (Rest of the logging and evaluation loop) ...

    # ... (Cleanup) ...

if __name__ == "__main__":
    train()
