import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


class ObservationConfig:
    """Configuration class to define how to extract Q/K/V from single agent observation"""
    def __init__(self, k_neighbors: int = 2, k_obstacles: int = 2):
        self.k_neighbors = k_neighbors
        self.k_obstacles = k_obstacles
        
        # Agent's own state indices (beginning of observation)
        self.agent_position_idx = slice(0, 2)          # [x, y] position
        self.agent_velocity_idx = slice(2, 4)          # [vx, vy] velocity
        self.grad_agents_idx = slice(4,6)
        self.grad_vol_idx = slice(6,8)
        self.grad_obs_idx = slice(8,10)
        self.grad_form_idx = slice(10,12)

        self.neighbor_block_idx = slice(12, 12 + k_neighbors * 6)
        self.neighbor_progress_idx = slice(12 + k_neighbors * 6, 12 + k_neighbors * 6 + k_neighbors)
        self.obstacle_block_idx = slice(12 + k_neighbors * 7, 12 + k_neighbors * 7 + k_obstacles * 2)
        end_neighbor_block = 12 + k_neighbors * 7 + k_obstacles * 2

        self.agent_ideal_dist = slice(end_neighbor_block, end_neighbor_block + 1)
        self.goal_vector_idx = slice(end_neighbor_block + 1, end_neighbor_block + 3)        # vector to goal
        self.formation_vector_idx = slice(end_neighbor_block + 3, end_neighbor_block + 5)   # vector to nearest formation point
        self.vel_to_form_idx = slice(end_neighbor_block + 5, end_neighbor_block + 6)        # dot product of velocity with vec to form 
        self.progress_idx = slice(end_neighbor_block + 6, end_neighbor_block + 7)           # progress over last 5
        
        # Individual neighbor/obstacle feature dimensions
        self.neighbor_obs_dim = 6
        self.obstacle_obs_dim = 2
        
    def get_agent_query_indices(self) -> List[slice]:
        """Define which parts of agent's OWN observation to use for Query"""
        return [
            self.agent_position_idx,
            self.agent_velocity_idx,
            self.grad_agents_idx,
            self.grad_vol_idx,
            self.grad_obs_idx,
            self.grad_form_idx,
            self.agent_ideal_dist,
            self.goal_vector_idx,
            self.formation_vector_idx,
            self.vel_to_form_idx,
            self.progress_idx
        ]

    def get_neighbor_key_indices(self):
        return [
            self.neighbor_block_idx
        ]

    def get_neighbor_value_indices(self):
        return [
            self.neighbor_block_idx,
            self.neighbor_progress_idx
        ]

    def get_obstacle_key_indices(self):
        return [
            self.obstacle_block_idx
        ]
    
    def get_obstacle_value_indices(self):
        return [
            self.obstacle_block_idx
        ]



class PGATCrossAttentionLayer(nn.Module):
    """
    PGAT Cross-Attention Layer implementing the paper's approach:
    - Query: Agent's own state  
    - Key/Value: Separate processing for neighbor agents and obstacles
    - Distance-based attention weights
    """
    def __init__(self, 
                 query_dim: int,
                 agent_key_dim: int,
                 agent_value_dim: int, 
                 obstacle_key_dim: int,
                 obstacle_value_dim: int,
                 out_channels: int,
                 heads: int = 4,
                 dropout: float = 0.0,
                 c_agent_decay: float = 1.0,
                 c_obstacle_decay: float = 2.0,
                 device=None):
        super().__init__()
        
        self.query_dim = query_dim
        self.agent_key_dim = agent_key_dim
        self.agent_value_dim = agent_value_dim
        self.obstacle_key_dim = obstacle_key_dim
        self.obstacle_value_dim = obstacle_value_dim
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.device = device if device is not None else torch.device('cpu')
        
        # Distance-based attention decay coefficients (learnable parameters)
        self.c_agent_decay = nn.Parameter(torch.tensor(c_agent_decay, device=self.device).clamp(min=0.1))
        self.c_obstacle_decay = nn.Parameter(torch.tensor(c_obstacle_decay, device=self.device).clamp(min=0.1))
        
        # Linear transformations for Query
        self.lin_query = nn.Linear(query_dim, heads * out_channels, bias=False, device=self.device)
        
        # Separate linear transformations for agent neighbors
        self.lin_agent_key = nn.Linear(agent_key_dim, heads * out_channels, bias=False, device=self.device)
        self.lin_agent_value = nn.Linear(agent_value_dim, heads * out_channels, bias=False, device=self.device)
        
        # Separate linear transformations for obstacles  
        self.lin_obstacle_key = nn.Linear(obstacle_key_dim, heads * out_channels, bias=False, device=self.device)
        self.lin_obstacle_value = nn.Linear(obstacle_value_dim, heads * out_channels, bias=False, device=self.device)
        
        # Output projection networks for concatenated features
        self.agent_proj = nn.Linear(heads * out_channels, out_channels, device=self.device)
        self.obstacle_proj = nn.Linear(heads * out_channels, out_channels, device=self.device)
        self.final_proj = nn.Linear(2 * out_channels, out_channels, device=self.device)  # For concatenation
        
        # Attention scaling factor
        self.scale = (out_channels // heads) ** -0.5
        
        self.reset_parameters()

    def reset_parameters(self):
        for module in [self.lin_query, self.lin_agent_key, self.lin_agent_value, 
                      self.lin_obstacle_key, self.lin_obstacle_value,
                      self.agent_proj, self.obstacle_proj, self.final_proj]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                query_features: torch.Tensor,           # [n_agents, query_dim]
                agent_key_features: torch.Tensor,       # [n_agents, k_neighbors, agent_key_dim] 
                agent_value_features: torch.Tensor,     # [n_agents, k_neighbors, agent_value_dim]
                obstacle_key_features: torch.Tensor,    # [n_agents, k_obstacles, obstacle_key_dim]
                obstacle_value_features: torch.Tensor,  # [n_agents, k_obstacles, obstacle_value_dim]
                agent_positions: torch.Tensor,          # [n_agents, 2]
                neighbor_positions: torch.Tensor,       # [n_agents, k_neighbors, 2]
                obstacle_positions: torch.Tensor,       # [n_agents, k_obstacles, 2]
                ) -> torch.Tensor:
        """
        Cross-attention between agents and their neighbors/obstacles
        """
        n_agents = query_features.shape[0]
        H, C = self.heads, self.out_channels
        
        # Transform queries (from agent's own state)
        query = self.lin_query(query_features).view(n_agents, H, C)  # [n_agents, heads, out_channels]
        
        # Process agent neighbors
        agent_attended = self._attend_to_agents(
            query, agent_key_features, agent_value_features, 
            agent_positions, neighbor_positions
        )  # [n_agents, out_channels]
        
        # Process obstacles  
        obstacle_attended = self._attend_to_obstacles(
            query, obstacle_key_features, obstacle_value_features,
            agent_positions, obstacle_positions  
        )  # [n_agents, out_channels]
        
        # Concatenate and project
        combined = torch.cat([agent_attended, obstacle_attended], dim=-1)  # [n_agents, 2*out_channels]
        output = self.final_proj(combined)  # [n_agents, out_channels]
        
        return output
    
    def _attend_to_agents(self, query, agent_key_features, agent_value_features, 
                         agent_positions, neighbor_positions):
        """Attend to neighboring agents"""
        n_agents, k_neighbors = agent_key_features.shape[:2]
        H, C = self.heads, self.out_channels
        
        if k_neighbors == 0:
            return torch.zeros(n_agents, C, device=self.device)
        
        # Transform keys and values
        agent_keys = self.lin_agent_key(agent_key_features).view(n_agents, k_neighbors, H, C)
        agent_values = self.lin_agent_value(agent_value_features).view(n_agents, k_neighbors, H, C)
        
        # Calculate distance-based attention weights
        # agent_positions: [n_agents, 2], neighbor_positions: [n_agents, k_neighbors, 2]
        distances = torch.norm(
            agent_positions.unsqueeze(1) - neighbor_positions, 
            dim=-1, keepdim=True
        )  # [n_agents, k_neighbors, 1]
        
        # Distance-based weights: exp(-c * distance)
        distance_weights = torch.exp(-self.c_agent_decay * distances)  # [n_agents, k_neighbors, 1]
        
        # Dot-product attention
        # query: [n_agents, H, C], agent_keys: [n_agents, k_neighbors, H, C]
        dot_attention = torch.einsum('nhc,nkhc->nhk', query, agent_keys) * self.scale  # [n_agents, H, k_neighbors]
        
        # Combine with distance weights
        # distance_weights: [n_agents, k_neighbors, 1] -> [n_agents, 1, k_neighbors]
        combined_attention = dot_attention * distance_weights.squeeze(-1).unsqueeze(1)  # [n_agents, H, k_neighbors]
        
        # Apply softmax
        attention_weights = F.softmax(combined_attention, dim=-1)  # [n_agents, H, k_neighbors]
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        # attention_weights: [n_agents, H, k_neighbors], agent_values: [n_agents, k_neighbors, H, C]
        attended = torch.einsum('nhk,nkhc->nhc', attention_weights, agent_values)  # [n_agents, H, C]
        
        # Project and aggregate across heads
        attended = attended.view(n_agents, -1)

        return self.agent_proj(attended)
    
    def _attend_to_obstacles(self, query, obstacle_key_features, obstacle_value_features,
                           agent_positions, obstacle_positions):
        """Attend to obstacles"""
        n_agents, k_obstacles = obstacle_key_features.shape[:2]
        H, C = self.heads, self.out_channels
        
        if k_obstacles == 0:
            return torch.zeros(n_agents, C, device=self.device)
        
        # Transform keys and values
        obstacle_keys = self.lin_obstacle_key(obstacle_key_features).view(n_agents, k_obstacles, H, C)
        obstacle_values = self.lin_obstacle_value(obstacle_value_features).view(n_agents, k_obstacles, H, C)
        
        # Calculate distance-based attention weights
        distances = torch.norm(
            agent_positions.unsqueeze(1) - obstacle_positions, 
            dim=-1, keepdim=True
        )  # [n_agents, k_obstacles, 1]
        
        distance_weights = torch.exp(-self.c_obstacle_decay * distances)  # [n_agents, k_obstacles, 1]
        
        # Dot-product attention  
        dot_attention = torch.einsum('nhc,nkhc->nhk', query, obstacle_keys) * self.scale  # [n_agents, H, k_obstacles]
        
        # Combine with distance weights
        combined_attention = dot_attention * distance_weights.squeeze(-1).unsqueeze(1)  # [n_agents, H, k_obstacles]
        
        # Apply softmax
        attention_weights = F.softmax(combined_attention, dim=-1)  # [n_agents, H, k_obstacles]  
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        attended = torch.einsum('nhk,nkhc->nhc', attention_weights, obstacle_values)  # [n_agents, H, C]
        
        # Project and aggregate across heads
        attended = attended.view(n_agents, -1)  # [n_agents, H*C]

        return self.obstacle_proj(attended)

class MultiLayerEnhancedPGATActor(nn.Module):
    def __init__(
        self,
        obs_config: ObservationConfig,
        total_obs_dim: int,
        n_agent_outputs: int,
        gnn_hidden_dim: int = 128,
        n_gnn_layers: int = 2,
        n_attention_heads: int = 4,
        k_neighbours: Optional[int] = None,
        k_obstacles: Optional[int] = None,
        dropout: float = 0.0,
        pos_indices: slice = slice(0, 2),
        device=None,
    ):
        super().__init__()
        
        self.obs_config = obs_config
        self.total_obs_dim = total_obs_dim
        self.n_agent_outputs = n_agent_outputs
        self.k_neighbours = k_neighbours or obs_config.k_neighbors
        self.k_obstacles = k_obstacles or obs_config.k_obstacles
        self.pos_indices = pos_indices
        self.device = device if device is not None else torch.device('cpu')
        self.n_gnn_layers = n_gnn_layers
        
        # Calculate dimensions
        self.query_dim = self._calculate_feature_dim(obs_config.get_agent_query_indices())
        self.agent_key_dim = self._calculate_feature_dim(obs_config.get_neighbor_key_indices())
        self.agent_value_dim = self._calculate_feature_dim(obs_config.get_neighbor_value_indices())
        self.obstacle_key_dim = self._calculate_feature_dim(obs_config.get_obstacle_key_indices())
        self.obstacle_value_dim = self._calculate_feature_dim(obs_config.get_obstacle_value_indices())
        
        # Build PGAT layers
        self.pgat_layers = nn.ModuleList()
        
        for i in range(n_gnn_layers):
            if i == 0:
                # First layer uses extracted features from observations
                layer = PGATCrossAttentionLayer(
                    query_dim=self.query_dim,
                    agent_key_dim=self.agent_key_dim,
                    agent_value_dim=self.agent_value_dim,
                    obstacle_key_dim=self.obstacle_key_dim,  
                    obstacle_value_dim=self.obstacle_value_dim,
                    out_channels=gnn_hidden_dim,
                    heads=n_attention_heads,
                    dropout=dropout,
                    device=self.device
                )
            else:
                # Subsequent layers use hidden representations
                layer = PGATCrossAttentionLayer(
                    query_dim=gnn_hidden_dim,
                    agent_key_dim=gnn_hidden_dim, 
                    agent_value_dim=gnn_hidden_dim,
                    obstacle_key_dim=gnn_hidden_dim,
                    obstacle_value_dim=gnn_hidden_dim,
                    out_channels=gnn_hidden_dim,
                    heads=n_attention_heads,
                    dropout=dropout,
                    device=self.device
                )
            self.pgat_layers.append(layer)
        
        # Neighbor/obstacle embedding projections for subsequent layers
        self.neighbor_projections = nn.ModuleList([
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim) 
            for _ in range(n_gnn_layers - 1)
        ])
        self.obstacle_projections = nn.ModuleList([
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim) 
            for _ in range(n_gnn_layers - 1)
        ])
        
        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(gnn_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, n_agent_outputs)
        ).to(self.device)
        
    def _calculate_feature_dim(self, slice_list: List[slice]) -> int:
        """Calculate total dimension from list of slices"""
        total_dim = 0
        for s in slice_list:
            total_dim += s.stop - s.start
        return total_dim
    
    def _extract_features_from_obs(self, obs: torch.Tensor, slice_list: List[slice]) -> torch.Tensor:
        """Extract and concatenate features from observation using slice list"""
        features = []
        for s in slice_list:
            features.append(obs[..., s])
        return torch.cat(features, dim=-1)

    def _extract_neighbor_features(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract neighbor and obstacle features"""
        batch_size = obs.shape[0]
        
        # Extract raw features
        neighbor_keys_raw = self._extract_features_from_obs(obs, self.obs_config.get_neighbor_key_indices())
        neighbor_values_raw = self._extract_features_from_obs(obs, self.obs_config.get_neighbor_value_indices())
        obstacle_keys_raw = self._extract_features_from_obs(obs, self.obs_config.get_obstacle_key_indices())
        obstacle_values_raw = self._extract_features_from_obs(obs, self.obs_config.get_obstacle_value_indices())
        
        # Reshape to separate individual neighbors/obstacles
        agent_key_features = neighbor_keys_raw.view(batch_size, self.k_neighbours, -1)
        agent_value_features = neighbor_values_raw.view(batch_size, self.k_neighbours, -1)
        obstacle_key_features = obstacle_keys_raw.view(batch_size, self.k_obstacles, -1) 
        obstacle_value_features = obstacle_values_raw.view(batch_size, self.k_obstacles, -1)
        
        return agent_key_features, agent_value_features, obstacle_key_features, obstacle_value_features

    def _extract_positions(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract positions for agents, neighbors, and obstacles"""
        batch_size = obs.shape[0]
        
        # Extract agent's own position
        agent_positions = obs[:, self.pos_indices]  # [batch_size, 2]
        
        # Extract neighbor positions from the neighbor block
        neighbor_block = obs[:, self.obs_config.neighbor_block_idx]  
        neighbor_block = neighbor_block.view(batch_size, self.k_neighbours, self.obs_config.neighbor_obs_dim)  
        neighbor_positions = neighbor_block[:, :, :2]  # Assuming first 2 are positions
        
        # Extract obstacle positions
        obstacle_block = obs[:, self.obs_config.obstacle_block_idx]  
        obstacle_positions = obstacle_block.view(batch_size, self.k_obstacles, 2)  
        
        return agent_positions, neighbor_positions, obstacle_positions
        
    def forward(self, agent_observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-layer PGAT
        """
        batch_size, n_agents = agent_observations.shape[0], agent_observations.shape[1]
        
        obs = agent_observations.to(dtype=torch.float32, device=self.device)
        
        if batch_size == 0 or n_agents == 0:
            return torch.zeros(batch_size, n_agents, self.n_agent_outputs, device=self.device)
        
        # Flatten for processing: [batch_size * n_agents, obs_dim]
        x_flat = obs.reshape(batch_size * n_agents, -1)
        
        # Extract query features (agent's own state)
        query_features = self._extract_features_from_obs(x_flat, self.obs_config.get_agent_query_indices())
        
        # Extract positions (these stay constant across layers)
        agent_pos, neighbor_pos, obstacle_pos = self._extract_positions(x_flat)
        
        # Initialize agent embeddings
        x = query_features  # Start with query features
        
        # Track neighbor and obstacle embeddings separately
        agent_key_feat, agent_val_feat, obs_key_feat, obs_val_feat = self._extract_neighbor_features(x_flat)
        
        # Pass through PGAT layers
        for i, layer in enumerate(self.pgat_layers):
            if i == 0:
                # First layer: use original features
                x = layer(
                    query_features=x,
                    agent_key_features=agent_key_feat,
                    agent_value_features=agent_val_feat,
                    obstacle_key_features=obs_key_feat,
                    obstacle_value_features=obs_val_feat,
                    agent_positions=agent_pos,
                    neighbor_positions=neighbor_pos,
                    obstacle_positions=obstacle_pos
                )
            else:
                # Subsequent layers: project previous embeddings
                # Project agent embeddings to neighbor/obstacle spaces
                neighbor_embeddings = self.neighbor_projections[i-1](x).unsqueeze(1).expand(-1, self.k_neighbours, -1)
                obstacle_embeddings = self.obstacle_projections[i-1](x).unsqueeze(1).expand(-1, self.k_obstacles, -1)
                
                x = layer(
                    query_features=x,
                    agent_key_features=neighbor_embeddings,
                    agent_value_features=neighbor_embeddings,
                    obstacle_key_features=obstacle_embeddings,
                    obstacle_value_features=obstacle_embeddings,
                    agent_positions=agent_pos,
                    neighbor_positions=neighbor_pos,
                    obstacle_positions=obstacle_pos
                )
            
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Apply output MLP
        agent_outputs = self.output_mlp(x)
        
        return agent_outputs.view(batch_size, n_agents, -1)


