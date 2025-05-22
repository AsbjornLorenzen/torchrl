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
import torch.nn.functional as F

# --- PyG Imports ---
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv # Example GNN layer
    from torch_geometric.data import Data, Batch
    _has_pyg = True
except ImportError:
    _has_pyg = False
    # Handle the case where PyG is not installed if needed
    print("PyTorch Geometric not found. GNN functionality will be unavailable.")
    # You might want to raise an error or fall back to MLP here

class PGATActor(nn.Module):
    """
    Position-based Graph Attention Network (PGAT) actor for UAV swarm control.
    Uses the existing _build_graph_batch function with modifications for PGAT.
    """
    def __init__(
        self,
        n_agent_inputs: int,
        n_agent_outputs: int,
        n_agents: int,
        gnn_hidden_dim: int = 128,
        n_gnn_layers: int = 2,
        activation_class=nn.Tanh,
        k_agents: int = 5,  # Number of agent neighbors in observation
        k_obstacles: int = 5,  # Number of obstacles in observation
        agent_pos_indices: slice = slice(0, 2),  # Position indices in agent features
        obstacle_pos_indices: slice = slice(0, 2),  # Position indices in obstacle features
        agent_attenuation: float = 1.0,  # c_a^W in the paper
        obstacle_attenuation: float = 1.0,  # c_o^W in the paper
        device=None,
    ):
        super().__init__()
        self.n_agent_inputs = n_agent_inputs
        self.n_agent_outputs = n_agent_outputs
        self.n_agents = n_agents
        self.k_agents = k_agents
        self.k_obstacles = k_obstacles
        self.agent_pos_indices = agent_pos_indices
        self.obstacle_pos_indices = obstacle_pos_indices
        self.agent_attenuation = agent_attenuation
        self.obstacle_attenuation = obstacle_attenuation
        self.device = device if device is not None else torch.device('cpu')
        
        # We'll use GAT layers instead of GCN layers
        self.gat_layers = nn.ModuleList()
        input_dim = n_agent_inputs
        for _ in range(n_gnn_layers):
            # Using GATConv with multiple attention heads
            self.gat_layers.append(
                GATConv(
                    in_channels=input_dim if _ == 0 else gnn_hidden_dim,
                    out_channels=gnn_hidden_dim // 4,  # Divide by number of heads
                    heads=4,
                    concat=True,
                    dropout=0.1
                )
            )
            input_dim = gnn_hidden_dim
        
        # Output MLP head
        self.output_mlp = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
            activation_class(),
            nn.Linear(gnn_hidden_dim, n_agent_outputs)
        )
        
        # Move all components to the device
        self.to(self.device)
    
    def _build_graph_batch(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Builds PyG batch graph data with position-based attention weights.
        Modified from the original _build_graph_batch to include edge_attr.

        Args:
            obs (torch.Tensor): Observation tensor of shape (batch_size, n_agents, obs_dim)

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
                - x (torch.Tensor): Node features, shape (batch_size * n_agents, obs_dim)
                - edge_index (torch.Tensor): Edge indices, shape (2, num_total_edges)
                - edge_attr (torch.Tensor): Edge attributes with attention weights
                - batch_vector (torch.Tensor): Maps each node to its batch index
        """
        batch_size, n_agents, obs_dim = obs.shape
        if n_agents == 0:  # Handle case with no agents
            return (torch.empty(0, obs_dim, device=self.device),
                    torch.empty(2, 0, dtype=torch.long, device=self.device),
                    torch.empty(0, 1, device=self.device),
                    torch.empty(0, dtype=torch.long, device=self.device))

        # Node features (flatten batch and agent dims)
        x = obs.reshape(batch_size * n_agents, obs_dim)

        # Extract positions for distance calculation
        pos = obs[:, :, self.agent_pos_indices].to(self.device)  # (batch_size, n_agents, 2 or 3)

        # Calculate pairwise distances within each batch element
        # Shape: (batch_size, n_agents, n_agents)
        dist = torch.cdist(pos, pos, p=2)

        # Find indices of agents and obstacles based on observation structure
        # This assumes the first k_agents are agents and the next k_obstacles are obstacles
        # Adjust these assumptions based on your exact observation structure
        
        # PGAT uses separate attention mechanisms for agents and obstacles
        # We'll use the k-nearest neighbors approach but apply different 
        # attention weights based on whether it's an agent-agent or agent-obstacle edge
        
        # If using k-nearest neighbors for edge construction
        if self.k_agents > 0:
            # Get k-nearest neighbors for each agent
            knn_val, knn_idx = torch.topk(dist, k=self.k_agents + 1, dim=-1, largest=False, sorted=True)
            
            # Get indices of the k-NN (excluding self - this creates the non-self-loop edges)
            neighbor_idx = knn_idx[..., 1:]  # Shape: (batch_size, n_agents, self.k_agents)
            source_idx = torch.arange(n_agents, device=self.device).view(1, -1, 1).expand(batch_size, -1, self.k_agents)
            
            # Flatten the source and target indices for k-NN edges
            flat_source_knn = source_idx.reshape(batch_size, -1)  # (batch_size, n_agents * k_agents)
            flat_target_knn = neighbor_idx.reshape(batch_size, -1)  # (batch_size, n_agents * k_agents)
            
            # Construct k-NN edge_index for the batch
            row_list_knn = []
            col_list_knn = []
            edge_attr_list = []  # For attention weights
            node_offset = 0
            
            for b in range(batch_size):
                # k-NN edges
                rows_b_knn = flat_source_knn[b] + node_offset
                cols_b_knn = flat_target_knn[b] + node_offset
                
                # Calculate attention weights using the PGAT formula (eq. 18)
                # For agent-agent connections
                for i, (source, target) in enumerate(zip(rows_b_knn, cols_b_knn)):
                    # Get original indices without batch offset
                    source_orig = source - node_offset
                    target_orig = target - node_offset
                    
                    # Calculate distance-based attention weight
                    distance = dist[b, source_orig, target_orig]
                    weight = torch.exp(-self.agent_attenuation * distance)
                    
                    row_list_knn.append(source)
                    col_list_knn.append(target)
                    edge_attr_list.append(weight)
                
                node_offset += n_agents
            
            row_edge_knn = torch.tensor(row_list_knn, device=self.device)
            col_edge_knn = torch.tensor(col_list_knn, device=self.device)
            
            knn_edge_index = torch.stack([row_edge_knn, col_edge_knn], dim=0)
            knn_edge_attr = torch.tensor(edge_attr_list, device=self.device).unsqueeze(1)
        else:
            # If k_agents is 0, there are no k-NN edges
            knn_edge_index = torch.empty(2, 0, dtype=torch.long, device=self.device)
            knn_edge_attr = torch.empty(0, 1, device=self.device)
            node_offset = batch_size * n_agents  # For batch_vector construction
        
        # Add self-loops with weight 1.0
        global_node_indices = torch.arange(batch_size * n_agents, device=self.device)
        self_loop_edge_index = torch.stack([global_node_indices, global_node_indices], dim=0)
        self_loop_edge_attr = torch.ones(batch_size * n_agents, 1, device=self.device)
        
        # Combine k-NN edges and self-loops
        edge_index = torch.cat([knn_edge_index, self_loop_edge_index], dim=1)
        edge_attr = torch.cat([knn_edge_attr, self_loop_edge_attr], dim=0)
        
        # Construct batch vector
        batch_vector_list = []
        for b in range(batch_size):
            batch_vector_list.append(torch.full((n_agents,), b, dtype=torch.long, device=self.device))
        batch_vector = torch.cat(batch_vector_list)
        
        return x, edge_index, edge_attr, batch_vector

    def forward(self, agent_observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the PGAT actor network.
        
        Args:
            agent_observations: Tensor of shape (batch_size, n_agents, obs_dim)
            
        Returns:
            Tensor of shape (batch_size, n_agents, n_agent_outputs)
        """
        # Ensure input is on the correct device
        obs = agent_observations.to(dtype=torch.float32, device=self.device)
        
        batch_size = obs.shape[0]
        n_agents = obs.shape[1]
        
        if batch_size == 0 or n_agents == 0:  # Handle empty batch/no agents
            return torch.zeros(batch_size, n_agents, self.n_agent_outputs, device=self.device)
        
        # Build the graph structure with attention weights
        x, edge_index, edge_attr, batch_vector = self._build_graph_batch(obs)
        
        # Pass through GAT layers
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, edge_attr)
            x = F.relu(x)
