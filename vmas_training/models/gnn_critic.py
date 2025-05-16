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
    from torch_geometric.utils import add_self_loops as pyg_add_self_loops
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
        gnn_layers: int = 2,
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
        self.device = device if device is not None else torch.device('cpu')

        # Define GNN layers
        self.gnn_layers = nn.ModuleList()
        input_dim = n_agent_inputs
        for i in range(gnn_layers):
            # Initialize GCNConv with BOTH add_self_loops=False AND normalize=False
            self.gnn_layers.append(
                GCNConv(
                    input_dim,
                    gnn_hidden_dim,
                    add_self_loops=False, # Set to False
                    normalize=False      # Must be False
                )
            )
            input_dim = gnn_hidden_dim
        self.gnn_layers = self.gnn_layers.to(self.device)

        # Output MLP head for each agent's value estimate
        self.output_mlp = nn.Linear(gnn_hidden_dim, 1).to(self.device)
        self.activation = activation_class().to(self.device)

    def _build_graph_batch(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Builds PyG batch graph data from batched observations.
        Adds self-loops manually to the edge_index.
        Expects obs shape: (batch_size * time_or_other_dims, n_agents, obs_dim)
        """
        batch_size_eff, n_agents, obs_dim = obs.shape
        num_nodes = batch_size_eff * n_agents # Total number of nodes in the batch

        if n_agents == 0:
             return (torch.empty(0, obs_dim, device=self.device),
                     torch.empty(2, 0, dtype=torch.long, device=self.device),
                     torch.empty(0, dtype=torch.long, device=self.device))

        x = obs.reshape(num_nodes, obs_dim)
        pos = obs[:, :, self.pos_indices].to(self.device)

        node_indices = torch.arange(n_agents, device=self.device)
        dist = torch.cdist(pos, pos, p=2)

        # --- Graph Construction Logic ---
        if self.k_neighbours is None or self.k_neighbours <= 0 or self.k_neighbours >= n_agents:
            if self.k_neighbours is not None and self.k_neighbours <= 0:
                 edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            else: # Fully connected
                adj = torch.ones(batch_size_eff, n_agents, n_agents, dtype=torch.bool, device=self.device)
                adj.diagonal(dim1=-2, dim2=-1).fill_(False) # Start without self-loops
                edge_index_list = [adj[b].nonzero().t() + b * n_agents for b in range(batch_size_eff)]
                if not edge_index_list:
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                else:
                    edge_index = torch.cat(edge_index_list, dim=1)
        else: # K-Nearest Neighbors
             k_int = int(self.k_neighbours)
             # Request k neighbors (don't need +1 anymore as we don't rely on GCNConv adding loops)
             knn_val, knn_idx = torch.topk(dist, k=k_int, dim=-1, largest=False, sorted=True)

             # Filter out self-references if k is large enough to include them
             # Note: This simple knn_idx might contain self-references if k is large.
             # A more robust KNN would compute distances and explicitly exclude self before topk.
             # However, add_self_loops later handles duplicates correctly.

             source_idx = torch.arange(n_agents, device=self.device).view(1, -1, 1).expand(batch_size_eff, -1, k_int)

             flat_source = source_idx.reshape(batch_size_eff, -1)
             flat_target = knn_idx.reshape(batch_size_eff, -1) # Use knn_idx directly

             row_list = []
             col_list = []
             node_offset = 0
             for b in range(batch_size_eff):
                 # Filter out self-loops from KNN results before adding offset
                 b_source = flat_source[b]
                 b_target = flat_target[b]
                 non_self_loop_mask = (b_source != b_target)
                 rows_b = b_source[non_self_loop_mask] + node_offset
                 cols_b = b_target[non_self_loop_mask] + node_offset

                 row_list.append(rows_b)
                 col_list.append(cols_b)
                 node_offset += n_agents

             if not row_list:
                 edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
             else:
                 row_edge = torch.cat(row_list)
                 col_edge = torch.cat(col_list)
                 edge_index = torch.stack([row_edge, col_edge], dim=0)

        # --- Manually Add Self-Loops ---
        # pyg_add_self_loops returns edge_index and edge_weights (optional)
        # We only need the updated edge_index. It handles duplicates.
        edge_index_with_loops, _ = pyg_add_self_loops(edge_index, num_nodes=num_nodes)

        batch_vector = torch.arange(batch_size_eff, device=self.device).repeat_interleave(n_agents)

        # Return the edge_index *with* manually added self-loops
        return x, edge_index_with_loops, batch_vector


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
        original_shape = obs.shape
        n_dims = obs.dim()

        # --- Reshape input if necessary ---
        if n_dims == 4:
            # Input is likely (batch, time, n_agents, obs_dim)
            batch_size, time_steps, n_agents, obs_dim = original_shape
            # Merge batch and time dimensions
            obs_reshaped = obs.reshape(batch_size * time_steps, n_agents, obs_dim)
        elif n_dims == 3:
            # Input is likely (batch, n_agents, obs_dim)
            batch_size, n_agents, obs_dim = original_shape
            obs_reshaped = obs # No reshape needed, but use consistent variable name
        else:
            raise ValueError(f"GNNCritic received input with unexpected number of dimensions: {n_dims}. Expected 3 or 4.")

        # --- Handle Empty Input After Reshape ---
        current_batch_size = obs_reshaped.shape[0]
        current_n_agents = obs_reshaped.shape[1]

        if current_batch_size == 0 or current_n_agents == 0:
            output_dim = self.output_mlp.out_features # Should be 1
            # Determine the correct output shape based on the original input shape
            if n_dims == 4:
                 final_shape = (*original_shape[:-1], output_dim) # (batch, time, n_agents, 1)
            else: # n_dims == 3
                 final_shape = (*original_shape[:-1], output_dim) # (batch, n_agents, 1)
            return torch.zeros(final_shape, device=self.device, dtype=torch.float32)


        # --- Process the (now 3D) reshaped observations ---
        # Build the graph structure for the potentially merged batch
        # Pass current_n_agents to _build_graph_batch if it relies on it
        x, edge_index, _ = self._build_graph_batch(obs_reshaped)

        # Pass through GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            x = self.activation(x)

        # Pass through final MLP head to get value estimates
        # Output shape: (current_batch_size * current_n_agents, 1)
        agent_values = self.output_mlp(x)

        # --- Reshape output back to original dimensionality ---
        # Reshape based on the original input dimensions
        if n_dims == 4:
            # Reshape back to (batch_size, time_steps, n_agents, 1)
            final_output = agent_values.view(batch_size, time_steps, n_agents, 1)
        else: # n_dims == 3
            # Reshape back to (batch_size, n_agents, 1)
            final_output = agent_values.view(batch_size, n_agents, 1)

        return final_output


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

