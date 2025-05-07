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
