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
        
        # Ego Agent Position (Query)
        self.ego_agent_position_idx = slice(0, 2)  # [x, y] position
        
        # Other Ego-Agent Features (to be concatenated after PGAT output)
        self.ego_agent_velocity_idx = slice(2, 4)  # [vx, vy] velocity
        self.ego_grad_agents_idx = slice(4, 6)
        self.ref_point_vel_idx = slice(6, 8)
        self.ego_form_vec_idx = slice(8, 10)
        self.ego_vor_vec_idx = slice(10, 12)
        
        # Neighbor Features (for Keys & Values) - FIXED: Each neighbor has 7 dimensions
        self.neighbor_block_raw_idx = slice(12, 12 + k_neighbors * 7)  # Changed from * 6 to * 7
        
        # Within each neighbor's 7-dim block
        self.neighbor_positions_in_block_idx = slice(0, 2)  # First 2 dims are relative positions
        self.neighbor_velocities_in_block_idx = slice(2, 4)  # Next 2 dims are velocities
        self.neighbor_vec_to_form_in_block_idx = slice(4, 6)  # Next 2 dims are vec_to_form
        self.neighbor_progress_in_block_idx = slice(6, 7)  # Last dim is progress (NEW)
        
        # Obstacle Features (for Keys & Values) - FIXED: Updated to account for correct neighbor block size
        self.obstacle_positions_raw_idx = slice(12 + k_neighbors * 7, 12 + k_neighbors * 7 + k_obstacles * 2)
        
        # Other Ego-Agent Features continued - FIXED: Updated starting index
        end_neighbor_obstacle_block = 12 + k_neighbors * 7 + k_obstacles * 2
        self.ego_agent_ideal_dist_idx = slice(end_neighbor_obstacle_block, end_neighbor_obstacle_block + 1)
        self.ego_goal_vector_idx = slice(end_neighbor_obstacle_block + 1, end_neighbor_obstacle_block + 3)
        
        # Reference Point Features (Ego-Agent's relation to its formation goal) - FIXED: Updated starting index
        ref_point_features_start_idx = end_neighbor_obstacle_block + 3
        self.ego_guiding_vector_idx = slice(ref_point_features_start_idx, ref_point_features_start_idx + 2)
        self.ego_vel_to_form_idx = slice(ref_point_features_start_idx + 2, ref_point_features_start_idx + 3)
        self.ego_progress_idx = slice(ref_point_features_start_idx + 3, ref_point_features_start_idx + 4)
        
        # Individual neighbor/obstacle feature dimensions - FIXED: Updated neighbor dimension
        self.neighbor_obs_dim = 7  # Changed from 6 to 7
        self.obstacle_obs_dim = 2
        
    def get_agent_query_indices(self) -> List[slice]:
        """Define which parts of agent's OWN observation to use for Query"""
        # For PGAT, Query is just the ego agent position
        return [self.ego_agent_position_idx]
    
    def get_neighbor_key_indices(self) -> List[slice]:
        """Return indices for neighbor positions (keys)"""
        # We'll extract positions from the neighbor block in PGATActor
        return [self.neighbor_block_raw_idx]
    
    def get_obstacle_key_indices(self) -> List[slice]:
        """Return indices for obstacle positions (keys)"""
        return [self.obstacle_positions_raw_idx]
    
    def get_reference_point_feature_indices(self) -> List[slice]:
        """Return indices for reference point features"""
        return [
            self.ego_guiding_vector_idx,
            self.ego_vel_to_form_idx,
            self.ego_progress_idx
        ]

    def get_ego_grad_feature_indices(self) -> List[slice]:
        """Return indices to gradients"""
        return [
            # NOTE: In modified env, grad_agents is actually all grads summed
            self.ego_grad_agents_idx,
            # self.ref_point_vel_idx,
            # self.ego_form_vec_idx,
            # self.ego_vor_vec_idx,
        ]
    
    def get_other_ego_feature_indices(self) -> List[slice]:
        """Return indices for other ego-agent features"""
        return [
            # self.ego_agent_velocity_idx,
            # self.ego_grad_agents_idx, # THIS IS THE GRAD INDEX WE NEED TO INCLUDE
            self.ref_point_vel_idx,
            self.ego_form_vec_idx,
            self.ego_vor_vec_idx,
            self.ego_agent_ideal_dist_idx,
            # self.ego_goal_vector_idx,
            # self.ego_guiding_vector_idx,
            # self.ego_vel_to_form_idx,
            # self.ego_progress_idx,
        ]

    def get_new_query_indices(self) -> List[slice]:
        """Return indices for reference point features"""
        return [
            self.ego_agent_position_idx,
            self.ego_agent_velocity_idx,
            # self.ego_guiding_vector_idx,
            self.ego_agent_ideal_dist_idx,
            # self.ego_goal_vector_idx,
            # self.ego_vel_to_form_idx,
            # self.ego_progress_idx,
        ]

    def get_query_dim(self) -> int:
        """Get dimension for query (ego agent position)"""
        # return 11
        return 5

    def get_grad_feature_dim(self) -> int:
        return 2
    
    def get_neighbor_key_dim(self) -> int:
        """Get dimension for neighbor keys (positions)"""
        return 2
    
    def get_neighbor_value_dim(self) -> int:
        """Get dimension for neighbor values"""
        # vel(2) + vec_to_form(2) + progress(1) + relative_pos(2) = 7
        return 10
    
    def get_obstacle_key_dim(self) -> int:
        """Get dimension for obstacle keys (positions)"""
        return 2
    
    def get_obstacle_value_dim(self) -> int:
        """Get dimension for obstacle values (relative positions)"""
        return 5
    
    def get_reference_point_feature_dim(self) -> int:
        """Get dimension for reference point features"""
        # formation_vector(2) + vel_to_form(1) + progress(1) = 4
        return 4
    
    def get_other_ego_feature_dim(self) -> int:
        """Get dimension for other ego features"""
        # grad_agents(2) + grad_vol(2) + grad_obs(2) + 
        # grad_form(2) + ideal_dist(1) + goal_vector(2) = 13
        return 7


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
        self.register_parameter('c_agent_decay', nn.Parameter(torch.tensor(c_agent_decay).clamp(min=0.1)))

        self.c_obstacle_decay = nn.Parameter(torch.tensor(c_obstacle_decay, device=self.device).clamp(min=0.1))
        
        # Linear transformations for Query
        self.lin_query = nn.Linear(query_dim, heads * out_channels, bias=True)
        
        # Separate linear transformations for agent neighbors
        self.lin_agent_key = nn.Linear(agent_key_dim, heads * out_channels, bias=True, device=self.device)
        self.lin_agent_value = nn.Linear(agent_value_dim, heads * out_channels, bias=True, device=self.device)
        
        # Separate linear transformations for obstacles  
        self.lin_obstacle_key = nn.Linear(obstacle_key_dim, heads * out_channels, bias=True, device=self.device)
        self.lin_obstacle_value = nn.Linear(obstacle_value_dim, heads * out_channels, bias=True, device=self.device)
        
        # Output projection networks for attended features
        self.agent_proj = nn.Linear(heads * out_channels, out_channels, device=self.device)
        self.obstacle_proj = nn.Linear(heads * out_channels, out_channels, device=self.device)
        # Removed self.final_proj as per requirements
        
        # Attention scaling factor
        self.scale = (out_channels // heads) ** -0.5
        
        self.reset_parameters()

    def reset_parameters(self):
        for module in [self.lin_query, self.lin_agent_key, self.lin_agent_value, 
                      self.lin_obstacle_key, self.lin_obstacle_value,
                      self.agent_proj, self.obstacle_proj]:
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
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention between agents and their neighbors/obstacles
        Returns: (agent_attended, obstacle_attended) tensors separately
        """
        n_agents = query_features.shape[0]
        H, C = self.heads, self.out_channels
        
        # Transform queries (from agent's own state)
        query = self.lin_query(query_features).reshape(n_agents, H, C)  # [n_agents, heads, out_channels]
        
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
        
        if torch.any(torch.isnan(agent_attended)):
            breakpoint()
        if torch.any(torch.isnan(obstacle_attended)):
            breakpoint()
        # Return both attended features separately
        return agent_attended, obstacle_attended


    def _attend_to_agents(self, query, agent_key_features, agent_value_features, 
                         agent_positions, neighbor_relative_positions):
        """Attend to neighboring agents - FIXED for collision avoidance"""
        n_agents, k_neighbors = agent_key_features.shape[:2]
        H, C = self.heads, self.out_channels
        
        device = query.device
        if k_neighbors == 0:
            return torch.zeros(n_agents, C, device=device)

        # Use relative positions directly for distance calculation
        distances = torch.norm(neighbor_relative_positions, dim=-1, keepdim=True)
        distances = torch.clamp(distances, min=1e-3)
        
        # FIXED: More attention to closer neighbors (inverse relationship)
        # This encourages the network to pay more attention to collision threats
        # distance_weights = 1.0 / (torch.clamp(self.c_agent_decay, min=0.5, max=10.0) * distances)

        distance_weights = torch.exp(-torch.clamp(self.c_agent_decay,min=0.5,max=10.0) * distances)  # [n_agents, k_obstacles, 1]

        # Alternative: Remove distance weighting entirely and let attention learn
        # distance_weights = torch.ones_like(distances)
        
        # Rest of the attention mechanism...
        agent_key_flat = agent_key_features.reshape(-1, agent_key_features.shape[-1])
        agent_value_flat = agent_value_features.reshape(-1, agent_value_features.shape[-1])
        
        agent_keys = self.lin_agent_key(agent_key_flat).reshape(n_agents, k_neighbors, H, C)
        agent_values = self.lin_agent_value(agent_value_flat).reshape(n_agents, k_neighbors, H, C)
        
        # Dot-product attention
        dot_attention = torch.einsum('nhc,nkhc->nhk', query, agent_keys) * self.scale
        
        # Combine with distance weights
        combined_attention = dot_attention * distance_weights.squeeze(-1).unsqueeze(1)
        
        attention_weights = F.softmax(combined_attention, dim=-1)
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)
        
        attended = torch.einsum('nhk,nkhc->nhc', attention_weights, agent_values)
        attended = attended.reshape(n_agents, -1)

        return self.agent_proj(attended)

    def _attend_to_obstacles(self, query, obstacle_key_features, obstacle_value_features,
                           agent_positions, obstacle_positions):
        """Attend to obstacles"""
        n_agents, k_obstacles = obstacle_key_features.shape[:2]
        H, C = self.heads, self.out_channels
        
        if k_obstacles == 0:
            return torch.zeros(n_agents, C, device=self.device)
        
        # Transform keys and values
        obstacle_keys = self.lin_obstacle_key(obstacle_key_features).reshape(n_agents, k_obstacles, H, C)
        obstacle_values = self.lin_obstacle_value(obstacle_value_features).reshape(n_agents, k_obstacles, H, C)
        
        # Calculate distance-based attention weights
        distances = torch.norm(obstacle_positions, dim=-1, keepdim=True)
        distances = torch.clamp(distances, min=1e-3)
    
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
        attended = attended.reshape(n_agents, -1)  # [n_agents, H*C]

        return self.obstacle_proj(attended)


class PGATActor(nn.Module):
    def __init__(
        self,
        obs_config: ObservationConfig,
        total_obs_dim: int,
        n_agent_outputs: int,
        gnn_hidden_dim: int = 256,
        n_gnn_layers: int = 1,
        n_attention_heads: int = 4,
        k_neighbors: Optional[int] = None,
        k_obstacles: Optional[int] = None,
        dropout: float = 0.0,
        pos_indices: slice = slice(0, 2),
        device=None,
        init_bias_value: float = 0.0,  # Bias the output towards higher values
    ):
        super().__init__()
        
        self.obs_config = obs_config
        self.total_obs_dim = total_obs_dim
        self.n_agent_outputs = n_agent_outputs
        self.k_neighbors = k_neighbors or obs_config.k_neighbors
        self.k_obstacles = k_obstacles or obs_config.k_obstacles
        self.pos_indices = pos_indices
        self.device = device if device is not None else torch.device('cpu')
        self.n_gnn_layers = n_gnn_layers
        self.gnn_hidden_dim = gnn_hidden_dim

        # Try to fix stability with these:
        self.init_bias_value = init_bias_value
        self.register_buffer('prev_output', None)
        
        # Calculate dimensions from ObservationConfig
        self.query_dim = obs_config.get_query_dim()  # 2 for ego position
        self.agent_key_dim = obs_config.get_neighbor_key_dim()  # 2 for neighbor positions
        self.agent_value_dim = obs_config.get_neighbor_value_dim()  # 7 for neighbor values
        self.obstacle_key_dim = obs_config.get_obstacle_key_dim()  # 2 for obstacle positions
        self.obstacle_value_dim = obs_config.get_obstacle_value_dim()  # 2 for obstacle relative positions
        self.ref_point_feature_dim = obs_config.get_reference_point_feature_dim()  # 4
        self.other_ego_feature_dim = obs_config.get_other_ego_feature_dim()  # 13
        
        # Build PGAT layers
        self.pgat_layers = nn.ModuleList()
        
        for i in range(n_gnn_layers):
            if i == 0:
                # First layer uses extracted features from observations
                layer = PGATCrossAttentionLayer(
                    query_dim=self.query_dim,  # 2 for ego position
                    agent_key_dim=self.agent_key_dim,  # 2 for neighbor positions
                    agent_value_dim=self.agent_value_dim,  # 7 for neighbor values
                    obstacle_key_dim=self.obstacle_key_dim,  # 2 for obstacle positions
                    obstacle_value_dim=self.obstacle_value_dim,  # 2 for obstacle relative positions
                    out_channels=gnn_hidden_dim,
                    heads=n_attention_heads,
                    dropout=dropout,
                    device=self.device
                )
            else:
                # Subsequent layers use hidden representations for query, but original features for K/V
                layer = PGATCrossAttentionLayer(
                    query_dim=gnn_hidden_dim,  # Query from previous layer
                    agent_key_dim=self.agent_key_dim,  # Still 2 for neighbor positions
                    agent_value_dim=self.agent_value_dim,  # Still 7 for neighbor values
                    obstacle_key_dim=self.obstacle_key_dim,  # Still 2 for obstacle positions
                    obstacle_value_dim=self.obstacle_value_dim,  # Still 2 for obstacle relative positions
                    out_channels=gnn_hidden_dim,
                    heads=n_attention_heads,
                    dropout=dropout,
                    device=self.device
                )
            self.pgat_layers.append(layer)
        
        # MLP for processing reference point features
        self.ego_mlp = nn.Sequential(
            nn.Linear(self.other_ego_feature_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim)
        ).to(self.device)
        
        # Projection layer for multi-layer GAT (if needed)
        if n_gnn_layers > 1:
            # Projects concatenated [agent_attended, obstacle_attended, ref_point_features] back to gnn_hidden_dim
            self.multi_layer_query_proj = nn.Sequential(
                nn.Linear(2 * gnn_hidden_dim, 2 * gnn_hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * gnn_hidden_dim, gnn_hidden_dim)
            )
        
        # Output MLP
        # Input: agent_attended + obstacle_attended + ref_point_features + other_ego_features
        # = gnn_hidden_dim + gnn_hidden_dim + gnn_hidden_dim + other_ego_feature_dim
        # = 3 * gnn_hidden_dim + other_ego_feature_dim
        # TODO: FIX DIM
        output_mlp_input_dim = 3 * gnn_hidden_dim # + self.other_ego_feature_dim
        self.output_mlp = nn.Sequential(
            nn.Linear(output_mlp_input_dim, 2 * gnn_hidden_dim),  # 512
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * gnn_hidden_dim, gnn_hidden_dim),        # 256
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),       # 128
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim // 2, n_agent_outputs)      # Final output
        ) 

        # Initialize the network weights
        self._initialize_weights()

        # Layer norm modules
        self.gat_agent_norms = nn.ModuleList([
            nn.LayerNorm(gnn_hidden_dim, device=self.device) 
            for _ in range(n_gnn_layers)
        ])
        self.gat_obstacle_norms = nn.ModuleList([
            nn.LayerNorm(gnn_hidden_dim, device=self.device) 
            for _ in range(n_gnn_layers)
        ])
        self.ego_norm = nn.LayerNorm(gnn_hidden_dim, device=self.device)
        self.combined_gat_norm = nn.LayerNorm(2 * gnn_hidden_dim, device=self.device)


    def _initialize_weights(self):
        """Custom weight initialization to bias towards higher outputs"""
        
        # Initialize final layer of main MLP to output higher values
        with torch.no_grad():
            # Set bias to positive values to encourage higher outputs
            # self.output_mlp[-1].bias.fill_(self.init_bias_value)
            nn.init.normal_(self.output_mlp[-1].weight, mean=0.0, std=0.05)
        
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


    def _extract_neighbor_features(self, obs: torch.Tensor, ego_positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract neighbor key and value features - FIXED for proper collision avoidance"""
        batch_size = obs.shape[0]
        
        # Extract neighbor block
        neighbor_block = obs[:, self.obs_config.neighbor_block_raw_idx]
        neighbor_block = neighbor_block.reshape(batch_size, self.k_neighbors, self.obs_config.neighbor_obs_dim)
        
        # Extract neighbor RELATIVE positions (don't convert to absolute)
        neighbor_relative_positions = neighbor_block[:, :, self.obs_config.neighbor_positions_in_block_idx]
        
        # For keys, use relative positions directly (spatial relationships matter more than absolute positions)
        agent_key_features = neighbor_relative_positions  # [batch, k_neighbors, 2]
        
        # Extract other neighbor features
        neighbor_velocities = neighbor_block[:, :, self.obs_config.neighbor_velocities_in_block_idx]
        neighbor_vec_to_form = neighbor_block[:, :, self.obs_config.neighbor_vec_to_form_in_block_idx]
        neighbor_progress = neighbor_block[:, :, self.obs_config.neighbor_progress_in_block_idx]
        
        # Values: include relative positions, velocities, and collision-relevant info
        distances = torch.norm(neighbor_relative_positions, dim=-1, keepdim=True)
        unit_directions = neighbor_relative_positions / (distances + 1e-8)
        
        agent_value_features = torch.cat([
            neighbor_relative_positions,  # Where they are relative to me
            distances,                    # How far they are (crucial for collision avoidance)
            unit_directions,              # Which direction they are
            neighbor_velocities,          # How they're moving
            neighbor_vec_to_form,
            neighbor_progress
        ], dim=-1)  # [batch, k_neighbors, 9] instead of 7
        
        # Return relative positions for distance calculations
        return agent_key_features, agent_value_features, neighbor_relative_positions

    def _extract_obstacle_features(self, obs: torch.Tensor, ego_positions: torch.Tensor):
        """Extract obstacle features - enhanced for collision avoidance"""
        batch_size = obs.shape[0]
        
        # Extract obstacle positions (these should already be relative in your observation)
        obstacle_relative_positions = obs[:, self.obs_config.obstacle_positions_raw_idx]
        obstacle_relative_positions = obstacle_relative_positions.reshape(batch_size, self.k_obstacles, 2)
        
        # Keys: relative positions
        obstacle_key_features = obstacle_relative_positions
        
        # Values: enhanced with distance and direction info
        distances = torch.norm(obstacle_relative_positions, dim=-1, keepdim=True)
        unit_directions = obstacle_relative_positions / (distances + 1e-8)
        
        obstacle_value_features = torch.cat([
            obstacle_relative_positions,  # Where they are
            distances,                    # How far (crucial for collision avoidance)
            unit_directions              # Which direction
        ], dim=-1)  # [batch, k_obstacles, 5] instead of 2
        
        return obstacle_key_features, obstacle_value_features, obstacle_relative_positions

        
    def forward(self, agent_observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-layer PGAT
        """
        batch_size, n_agents = agent_observations.shape[0], agent_observations.shape[1]
        
        device = agent_observations.device
        obs = agent_observations.to(dtype=torch.float32, device=device)
        
        if batch_size == 0 or n_agents == 0:
            return torch.zeros(batch_size, n_agents, self.n_agent_outputs, device=self.device)
        
        # Flatten for processing: [batch_size * n_agents, obs_dim]
        x_flat = obs.reshape(batch_size * n_agents, -1)
        
        # Extract query features (ego agent position)
        # query_input_L0 = self._extract_features_from_obs(x_flat, self.obs_config.get_agent_query_indices())
        # TEST: Try using vec to formation as query
        query_input_L0 = self._extract_features_from_obs(x_flat, self.obs_config.get_new_query_indices())
        
        # Extract ego positions
        ego_positions = x_flat[:, self.obs_config.ego_agent_position_idx]  # [batch_size * n_agents, 2]
        
        # Extract neighbor features
        agent_keys_input, agent_values_input, neighbor_positions = self._extract_neighbor_features(x_flat, ego_positions)
        
        # Extract obstacle features
        obstacle_keys_input, obstacle_values_input, obstacle_positions = self._extract_obstacle_features(x_flat, ego_positions)
        
        # Extract reference point features
        # ref_point_features_input = self._extract_features_from_obs(x_flat, self.obs_config.get_reference_point_feature_indices())
        
        # Extract other ego features
        other_ego_features_input = self._extract_features_from_obs(x_flat, self.obs_config.get_other_ego_feature_indices())
        grad_ego_features_input = self._extract_features_from_obs(x_flat, self.obs_config.get_ego_grad_feature_indices())
        
        # Process reference point features through MLP (once, outside the loop)
        # processed_ref_point_features = self.ref_point_mlp(ref_point_features_input)  # [batch_size * n_agents, gnn_hidden_dim]
        processed_ego_features = self.ego_mlp(other_ego_features_input)
        processed_ego_features = self.ego_norm(processed_ego_features)
        
        # GAT layer processing
        current_query = query_input_L0  # Start with ego position
        agent_attended_final = None
        obstacle_attended_final = None
        
        for i, layer in enumerate(self.pgat_layers):
            # Get attention outputs
            agent_attended, obstacle_attended = layer(
                query_features=current_query,
                agent_key_features=agent_keys_input,
                agent_value_features=agent_values_input,
                obstacle_key_features=obstacle_keys_input,
                obstacle_value_features=obstacle_values_input,
                agent_positions=ego_positions,
                neighbor_positions=neighbor_positions,
                obstacle_positions=obstacle_positions
            )

            agent_attended = self.gat_agent_norms[i](agent_attended)
            obstacle_attended = self.gat_obstacle_norms[i](obstacle_attended)
            
            # Store final layer outputs
            if i == self.n_gnn_layers - 1:
                agent_attended_final = agent_attended
                obstacle_attended_final = obstacle_attended
            
            # Prepare query for next layer (if not last layer)
            if i < self.n_gnn_layers - 1:
                if self.n_gnn_layers > 1:
                    # Combine outputs with reference point features for next query
                    combined_features = torch.cat([agent_attended, obstacle_attended], dim=-1) # , processed_ref_point_features
                    current_query = F.relu(self.multi_layer_query_proj(combined_features))
                    current_query = F.dropout(current_query, p=0.1, training=self.training)
                else:
                    # If only one layer, this won't be executed
                    current_query = agent_attended
        
        # Combine final GAT outputs with reference point features
        combined_gat_output = torch.cat([
            agent_attended_final, 
            obstacle_attended_final, 
            # processed_ref_point_features
        ], dim=-1)  # [batch_size * n_agents, 3 * gnn_hidden_dim]
        
        # Apply ReLU and dropout
        combined_gat_output = self.combined_gat_norm(combined_gat_output)
        combined_gat_output = F.relu(combined_gat_output)
        combined_gat_output = F.dropout(combined_gat_output, p=0.1, training=self.training)
        
        # Concatenate with other ego features
        final_features = torch.cat([combined_gat_output, processed_ego_features], dim=-1) # grad_ego_features_input

        # BEGIN NEW APPROACH 
        # Main network output
        main_output = self.output_mlp(final_features)
        
        # if self.use_residual_baseline:
        #     # Baseline output (simple heuristic behavior)
        #     baseline_output = self.baseline_mlp(grad_ego_features_input)
        #     
        #     # Gating mechanism
        #     gate = self.gate_mlp(final_features)
        #     
        #     # Combine main output and baseline
        #     agent_outputs = gate * main_output + (1 - gate) * baseline_output
        # else:
        agent_outputs = main_output
        if torch.any(torch.isnan(agent_outputs)):
            breakpoint()
        
        return agent_outputs.reshape(batch_size, n_agents, -1)

        # OLD APPROACH:
        # Apply output MLP
        # agent_outputs = self.output_mlp(final_features)

