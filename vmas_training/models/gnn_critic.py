# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

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
# (Keep relevant TorchRL imports if needed for context, but not strictly required for the module itself)
from tensordict.nn import TensorDictModule
from torchrl.modules import ValueOperator # ValueOperator is used to WRAP the critic, not BE the critic core

# --- GNN Critic Module Definition ---
class GNNCritic(nn.Module):
    """
    A GNN-based critic network module for multi-agent RL.

    Assumes the input observation tensor for each agent contains its features,
    and positions are included (e.g., at the beginning) to build the graph.
    Handles batching by creating a PyG Batch object internally.
    Outputs a single value estimate per agent.
    """
    def __init__(
        self,
        n_agent_inputs: int,
        gnn_hidden_dim: int = 128,
        n_gnn_layers: int = 2,
        activation_class=nn.Tanh,
        k_neighbours: float | None = None,
        pos_indices: slice = slice(0, 2), # Indices for XY position in observation
        share_params: bool = True, # GNNs inherently share params across nodes/agents
                                   # This flag is kept for consistency maybe? But GNN handles sharing.
        device = None, # Added device parameter
    ):
        """
        Initializes the GNNCritic module.

        Args:
            n_agent_inputs (int): Dimensionality of the observation input for each agent.
            gnn_hidden_dim (int, optional): Hidden dimension size for GNN layers. Defaults to 128.
            n_gnn_layers (int, optional): Number of GNN layers. Defaults to 2.
            activation_class (torch.nn.Module, optional): Activation function class. Defaults to nn.Tanh.
            k_neighbours (float | None, optional): Number of nearest neighbors to connect in the graph.
                                                   If None, a fully connected graph might be implied (or adjust logic).
                                                   Defaults to None.
            pos_indices (slice, optional): Slice object indicating the indices of position data
                                           within the agent observation vector. Defaults to slice(0, 2).
            share_params (bool, optional): Kept for API consistency, but GNN inherently shares parameters
                                           across nodes (agents). Defaults to True.
            device (torch.device or str, optional): Device to place tensors on. Defaults to None (uses default device).
        """
        super().__init__()
        if not _has_pyg:
            raise ImportError("PyTorch Geometric is required for GNNCritic.")

        self.n_agent_inputs = n_agent_inputs
        self.k_neighbours = k_neighbours
        self.pos_indices = pos_indices
        self.device = device # Store device

        # Define GNN layers
        self.gnn_layers = nn.ModuleList()
        input_dim = n_agent_inputs
        for _ in range(n_gnn_layers):
            # Using GCNConv as an example, same as actor
            self.gnn_layers.append(GCNConv(input_dim, gnn_hidden_dim))
            input_dim = gnn_hidden_dim

        # Output MLP head for each agent's value estimate
        # Output dimension is 1 for the value function
        self.output_mlp = nn.Linear(gnn_hidden_dim, 1)
        self.activation = activation_class()

    def _build_graph_batch(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Builds PyG batch graph data from batched observations.
        Identical to the actor's implementation.

        Args:
            obs (torch.Tensor): Observation tensor of shape (batch_size, n_agents, obs_dim)

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor):
                - x (torch.Tensor): Node features, shape (batch_size * n_agents, obs_dim)
                - edge_index (torch.Tensor): Edge indices, shape (2, num_total_edges)
                - batch_vector (torch.Tensor): Maps each node to its batch index, shape (batch_size * n_agents)
        """
        batch_size, n_agents, obs_dim = obs.shape
        if n_agents == 0: # Handle case with no agents
             return (torch.empty(0, obs_dim, device=self.device),
                     torch.empty(2, 0, dtype=torch.long, device=self.device),
                     torch.empty(0, dtype=torch.long, device=self.device))

        # Node features (flatten batch and agent dims)
        x = obs.reshape(batch_size * n_agents, obs_dim)

        # Extract positions for distance calculation
        pos = obs[:, :, self.pos_indices].to(self.device) # (batch_size, n_agents, 2 or 3)

        # --- Efficient Batched Graph Construction ---
        # Create indices for nodes within each batch element
        node_indices = torch.arange(n_agents, device=self.device)
        # Use meshgrid to get all pairs of nodes (row, col) within each agent group
        col, row = torch.meshgrid(node_indices, node_indices, indexing='ij')
        col = col.reshape(1, n_agents, n_agents) # Add batch dim for broadcasting
        row = row.reshape(1, n_agents, n_agents)

        # Calculate pairwise distances within each batch element
        # Shape: (batch_size, n_agents, n_agents)
        dist = torch.cdist(pos, pos, p=2)

        if self.k_neighbours is None or self.k_neighbours >= n_agents:
            # Fully connected graph within each batch element (excluding self-loops initially)
            adj = torch.ones(batch_size, n_agents, n_agents, dtype=torch.bool, device=self.device)
            adj.diagonal(dim1=-2, dim2=-1).fill_(False) # Remove self-loops for edge_index construction
            edge_index_list = [adj[b].nonzero().t() + b * n_agents for b in range(batch_size)]
            edge_index = torch.cat(edge_index_list, dim=1)
        elif self.k_neighbours > 0:
             # K-Nearest Neighbors graph construction
             knn_val, knn_idx = torch.topk(dist, k=self.k_neighbours + 1, dim=-1, largest=False, sorted=True)
             # knn_idx shape: (batch_size, n_agents, self.k + 1)

             # Exclude self-loops which should be the closest (index 0)
             neighbor_idx = knn_idx[..., 1:] # Shape: (batch_size, n_agents, self.k)
             source_idx = torch.arange(n_agents, device=self.device).view(1, -1, 1).expand(batch_size, -1, self.k_neighbours)

             # --- Construct edge_index for the batch ---
             # Flatten the source and target indices
             flat_source = source_idx.reshape(batch_size, -1) # (batch_size, n_agents * k)
             flat_target = neighbor_idx.reshape(batch_size, -1) # (batch_size, n_agents * k)

             # Add offsets for batching
             row_list = []
             col_list = []
             node_offset = 0
             for b in range(batch_size):
                 rows_b = flat_source[b] + node_offset
                 cols_b = flat_target[b] + node_offset
                 row_list.append(rows_b)
                 col_list.append(cols_b)
                 node_offset += n_agents

             row_edge = torch.cat(row_list) # Source nodes
             col_edge = torch.cat(col_list) # Target nodes
             edge_index = torch.stack([row_edge, col_edge], dim=0) # Shape (2, batch_size * n_agents * k)
        else: # k_neighbours == 0, no edges
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)


        # Create the batch vector (needed by some PyG layers/utilities, though not GCNConv directly)
        batch_vector = torch.arange(batch_size, device=self.device).repeat_interleave(n_agents)

        return x, edge_index, batch_vector


    def forward(self, agent_observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GNNCritic.

        Args:
            agent_observations (torch.Tensor): Observation tensor of shape
                                               (batch_size, n_agents, obs_dim).

        Returns:
            torch.Tensor: Value estimates tensor of shape (batch_size, n_agents, 1).
        """
        # Ensure input is on the correct device and dtype
        obs = agent_observations.to(device=self.device, dtype=torch.float32)

        # obs shape: (batch_size, n_agents, obs_dim)
        batch_size = obs.shape[0]
        n_agents = obs.shape[1]

        if batch_size == 0 or n_agents == 0: # Handle empty batch/no agents
            output_dim = self.output_mlp.out_features # Should be 1
            # Adjust n_agents based on input if possible, fallback to init value
            return torch.zeros(batch_size, n_agents, output_dim, device=self.device)

        # Build the graph structure for the batch
        x, edge_index, _ = self._build_graph_batch(obs) # We don't strictly need the batch_vector for GCNConv

        # Pass through GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            x = self.activation(x)

        # Pass through final MLP head to get value estimates
        agent_values = self.output_mlp(x) # Shape: (batch_size * n_agents, 1)

        # Reshape back to (batch_size, n_agents, 1)
        final_output = agent_values.view(batch_size, n_agents, 1)

        # Return the tensor; TensorDictModule/ValueOperator will handle putting it into the output key ('state_value')
        return final_output

# Example Usage (how you might integrate it with TensorDictModule and ValueOperator):
# Assuming 'device', 'n_agent_inputs', 'n_agents', 'cfg' (hydra config) are defined

# critic_module = GNNCritic(
#     n_agent_inputs=n_agent_inputs,
#     n_agents=n_agents,
#     gnn_hidden_dim=cfg.model.critic.gnn_hidden_dim, # Example config access
#     n_gnn_layers=cfg.model.critic.n_gnn_layers,
#     activation_class=getattr(nn, cfg.model.critic.activation), # Example config access
#     k_neighbours=cfg.model.critic.k_neighbours,
#     pos_indices=slice(cfg.env.state_spec["observation_spec"]["agents"]["observation"].shape[-1] - 2, # Example: Assuming pos is last 2 dims
#                       cfg.env.state_spec["observation_spec"]["agents"]["observation"].shape[-1]),
#     device=device,
# )

# # Wrap the GNN critic module with ValueOperator for TorchRL integration
# value_module = ValueOperator(
#     module=critic_module,
#     in_keys=[("agents", "observation")], # Specify input key from TensorDict
#     out_keys=[("agents", "state_value")] # Standard output key for value estimates
# )
# value_module.to(device)

