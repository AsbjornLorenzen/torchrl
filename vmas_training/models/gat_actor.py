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



class SimplePGATLayer(nn.Module):
    """
    Simple PGAT following the paper:
    - Agent-agent attention via message passing (your existing approach)  
    - Agent-obstacle attention via direct computation (no edges needed)
    """
    def __init__(self,
                 query_dim: int,
                 agent_key_dim: int, 
                 agent_value_dim: int,
                 obstacle_dim: int,  # Obstacle features per obstacle
                 out_channels: int,
                 heads: int = 4,
                 k_obstacles: int = 2,
                 dropout: float = 0.0,
                 device=None):
        super().__init__()
        
        self.out_channels = out_channels
        self.heads = heads
        self.k_obstacles = k_obstacles
        self.dropout = dropout
        self.device = device if device is not None else torch.device('cpu')
        
        # Your existing GAT layer for agent-agent attention
        from torch_geometric.nn import GATConv
        self.agent_gat = GATConv(
            in_channels=agent_key_dim,  # Will be set properly
            out_channels=out_channels // 2,  # Split output channels
            heads=heads,
            concat=True,
            dropout=dropout
        ).to(device)
        
        # Obstacle attention components (direct computation)
        self.obstacle_query_proj = nn.Linear(query_dim, heads * (out_channels // 4), bias=False).to(device)
        self.obstacle_key_proj = nn.Linear(obstacle_dim, heads * (out_channels // 4), bias=False).to(device)
        self.obstacle_value_proj = nn.Linear(obstacle_dim, heads * (out_channels // 4), bias=False).to(device)
        
        # Learnable decay coefficients (paper's c_a^W and c_o^W)
        self.c_obstacle_decay = nn.Parameter(torch.tensor(2.0))
        
        # Final combination
        total_features = (out_channels // 2) * heads + (out_channels // 4) * heads
        self.final_proj = nn.Linear(total_features, out_channels).to(device)
        
    def compute_obstacle_attention(self, 
                                 query_features: torch.Tensor,     # [num_agents, query_dim]
                                 obstacle_features: torch.Tensor,  # [num_agents, k_obstacles, obstacle_dim]
                                 agent_positions: torch.Tensor,    # [num_agents, 2]
                                 obstacle_positions: torch.Tensor): # [num_agents, k_obstacles, 2]
        """
        Compute obstacle attention directly (following paper's equations 18-19)
        No message passing - just direct attention computation
        """
        num_agents = query_features.shape[0]
        H = self.heads
        C_obs = self.out_channels // 4
        
        if self.k_obstacles == 0 or obstacle_features.shape[1] == 0:
            return torch.zeros(num_agents, H * C_obs, device=self.device)
        
        # Project features
        query_proj = self.obstacle_query_proj(query_features).view(num_agents, H, C_obs)  # [num_agents, H, C_obs]
        obstacle_key = self.obstacle_key_proj(obstacle_features).view(num_agents, self.k_obstacles, H, C_obs)
        obstacle_value = self.obstacle_value_proj(obstacle_features).view(num_agents, self.k_obstacles, H, C_obs)
        
        # Compute attention scores (dot product)
        query_expanded = query_proj.unsqueeze(1)  # [num_agents, 1, H, C_obs]
        attention_scores = (query_expanded * obstacle_key).sum(dim=-1)  # [num_agents, k_obstacles, H]
        
        # Add position-based weighting (paper's equation 18: W_ko = exp(-c_o^W ||p_k - p_o||))
        distances = torch.norm(
            agent_positions.unsqueeze(1) - obstacle_positions,  # [num_agents, k_obstacles, 2]
            dim=-1
        )  # [num_agents, k_obstacles]
        
        distance_weights = torch.exp(-self.c_obstacle_decay * distances)  # [num_agents, k_obstacles]
        
        # Combine attention with distance weights
        attention_scores = attention_scores * distance_weights.unsqueeze(-1)  # [num_agents, k_obstacles, H]
        
        # Softmax normalization (paper's equation 19)
        attention_weights = F.softmax(attention_scores, dim=1)  # Normalize over obstacles
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)
        
        # Weighted sum of obstacle values
        obstacle_out = (attention_weights.unsqueeze(-1) * obstacle_value).sum(dim=1)  # [num_agents, H, C_obs]
        
        return obstacle_out.view(num_agents, H * C_obs)
    
    def forward(self,
                x: torch.Tensor,                    # [num_agents, feature_dim] - node features
                edge_index: torch.Tensor,           # [2, num_edges] - agent-agent edges only
                query_features: torch.Tensor,       # [num_agents, query_dim]
                obstacle_features: torch.Tensor,    # [num_agents, k_obstacles, obstacle_dim]
                agent_positions: torch.Tensor,      # [num_agents, 2] 
                obstacle_positions: torch.Tensor):  # [num_agents, k_obstacles, 2]
        """
        Forward pass combining agent-agent and agent-obstacle attention
        """
        # Agent-agent attention (your existing approach)
        agent_out = self.agent_gat(x, edge_index)  # [num_agents, (out_channels//2) * heads]
        
        # Agent-obstacle attention (direct computation)
        obstacle_out = self.compute_obstacle_attention(
            query_features, obstacle_features, agent_positions, obstacle_positions
        )  # [num_agents, (out_channels//4) * heads]
        
        # Combine results (following paper's equation 21)
        combined = torch.cat([agent_out, obstacle_out], dim=-1)
        output = F.relu(self.final_proj(combined))
        
        return output


class YourEnhancedPGATActor(nn.Module):
    """
    Minimal changes to your existing code to add obstacle attention
    """
    def __init__(self,
                 obs_config,
                 total_obs_dim: int,
                 n_agent_outputs: int,
                 gnn_hidden_dim: int = 128,
                 n_gnn_layers: int = 2,
                 n_attention_heads: int = 4,
                 k_neighbours: Optional[int] = None,
                 dropout: float = 0.0,
                 pos_indices: slice = slice(0, 2),
                 device=None):
        super().__init__()
        
        self.obs_config = obs_config
        self.total_obs_dim = total_obs_dim
        self.n_agent_outputs = n_agent_outputs
        self.k_neighbours = k_neighbours or obs_config.k_neighbors
        self.pos_indices = pos_indices
        self.device = device if device is not None else torch.device('cpu')
        
        # Calculate dimensions
        self.query_dim = self._calculate_feature_dim(obs_config.get_agent_query_indices())
        self.agent_key_dim = self._calculate_feature_dim(obs_config.get_neighbor_key_indices())
        self.agent_value_dim = self._calculate_feature_dim(obs_config.get_neighbor_value_indices())
        self.obstacle_dim = obs_config.obstacle_obs_dim  # Assuming this exists
        
        # Build layers
        self.gat_layers = nn.ModuleList()
        
        for i in range(n_gnn_layers):
            if i == 0:
                # First layer: PGAT with obstacle attention
                layer = SimplePGATLayer(
                    query_dim=self.query_dim,
                    agent_key_dim=self.agent_key_dim,  # Will need adjustment for GATConv
                    agent_value_dim=self.agent_value_dim,
                    obstacle_dim=self.obstacle_dim,
                    out_channels=gnn_hidden_dim,
                    heads=n_attention_heads,
                    k_obstacles=obs_config.k_obstacles,
                    dropout=dropout,
                    device=self.device
                )
            else:
                # Subsequent layers: regular GAT
                from torch_geometric.nn import GATConv
                layer = GATConv(
                    gnn_hidden_dim, gnn_hidden_dim,
                    heads=n_attention_heads, concat=True, dropout=dropout
                ).to(self.device)
                
            self.gat_layers.append(layer)
            
        # Output MLP (unchanged)
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
    
    def _extract_obstacle_data(self, obs: torch.Tensor):
        """
        Extract obstacle features and positions from observations
        
        Args:
            obs: [batch_size * n_agents, obs_dim]
            
        Returns:
            obstacle_features: [batch_size * n_agents, k_obstacles, obstacle_dim]
            obstacle_positions: [batch_size * n_agents, k_obstacles, 2]
        """
        batch_size_times_agents = obs.shape[0]
        k_obstacles = self.obs_config.k_obstacles
        
        # Extract obstacle block from observation
        obstacle_block = self._extract_features_from_obs(obs, self.obs_config.get_obstacle_key_indices())
        
        if obstacle_block.shape[-1] == 0 or k_obstacles == 0:
            return (torch.zeros(batch_size_times_agents, 0, self.obstacle_dim, device=self.device),
                    torch.zeros(batch_size_times_agents, 0, 2, device=self.device))
        
        # Reshape to [batch_size * n_agents, k_obstacles, obstacle_dim]  
        obstacle_features = obstacle_block.view(batch_size_times_agents, k_obstacles, self.obstacle_dim)
        
        # Extract positions (assume first 2 dims are relative position to agent)
        obstacle_positions = obstacle_features[..., :2]
        
        return obstacle_features, obstacle_positions
    
    def _build_graph_batch(self, obs: torch.Tensor) -> torch.Tensor:
        """Build agent-agent edges only (your existing implementation)"""
        # Your existing graph building code - no changes needed!
        batch_size, n_agents, obs_dim = obs.shape
        
        if n_agents == 0:
            return torch.empty(2, 0, dtype=torch.long, device=self.device)
        
        pos = obs[:, :, self.pos_indices].to(self.device)
        dist = torch.cdist(pos, pos, p=2)
        
        # k-NN construction (your existing code)
        if self.k_neighbours is not None and self.k_neighbours > 0:
            knn_val, knn_idx = torch.topk(dist, k=min(self.k_neighbours + 1, n_agents), 
                                         dim=-1, largest=False, sorted=True)
            
            if knn_idx.shape[-1] > 1:
                neighbor_idx = knn_idx[..., 1:]
                k_actual = neighbor_idx.shape[-1]
            else:
                return torch.empty(2, 0, dtype=torch.long, device=self.device)
            
            if k_actual > 0:
                source_idx = torch.arange(n_agents, device=self.device).view(1, -1, 1).expand(batch_size, -1, k_actual)
                
                flat_source = source_idx.reshape(batch_size, -1)
                flat_target = neighbor_idx.reshape(batch_size, -1)
                
                row_list, col_list = [], []
                node_offset = 0
                for b in range(batch_size):
                    rows_b = flat_source[b] + node_offset
                    cols_b = flat_target[b] + node_offset
                    row_list.append(rows_b)
                    col_list.append(cols_b)
                    node_offset += n_agents
                
                edge_index = torch.stack([torch.cat(row_list), torch.cat(col_list)], dim=0)
            else:
                edge_index = torch.empty(2, 0, dtype=torch.long, device=self.device)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=self.device)
        
        # Add self-loops
        global_node_indices = torch.arange(batch_size * n_agents, device=self.device)
        self_loop_edge_index = torch.stack([global_node_indices, global_node_indices], dim=0)
        edge_index = torch.cat([edge_index, self_loop_edge_index], dim=1)
        
        return edge_index
        
    def forward(self, agent_observations: torch.Tensor) -> torch.Tensor:
        """Forward pass - minimal changes to your existing code"""
        batch_size, n_agents = agent_observations.shape[0], agent_observations.shape[1]
        
        obs = agent_observations.to(dtype=torch.float32, device=self.device)
        
        if batch_size == 0 or n_agents == 0:
            return torch.zeros(batch_size, n_agents, self.n_agent_outputs, device=self.device)
        
        # Build agent-agent graph (no changes - your existing code)
        edge_index = self._build_graph_batch(obs)
        
        # Flatten observations
        x = obs.reshape(batch_size * n_agents, -1)
        
        # Extract features for first layer
        query_features = self._extract_features_from_obs(x, self.obs_config.get_agent_query_indices())
        agent_key_features = self._extract_features_from_obs(x, self.obs_config.get_neighbor_key_indices()) 
        
        # Extract obstacle data (NEW - this is the main addition)
        obstacle_features, obstacle_positions = self._extract_obstacle_data(x)
        
        # Extract agent positions
        agent_positions = x[:, self.pos_indices]
        
        # Pass through layers
        for i, layer in enumerate(self.gat_layers):
            if i == 0:
                # First layer: PGAT with obstacle attention
                x = layer(
                    agent_key_features,  # Node features for agent GAT
                    edge_index,          # Agent-agent edges only
                    query_features,      # Query features
                    obstacle_features,   # Obstacle features
                    agent_positions,     # Agent positions
                    obstacle_positions   # Obstacle positions
                )
            else:
                # Subsequent layers: regular GAT
                x = layer(x, edge_index)
                
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Output MLP (unchanged)
        agent_outputs = self.output_mlp(x)
        
        return agent_outputs.view(batch_size, n_agents, -1)




class EnhancedPositionAwareGATLayer(MessagePassing):
    """
    Enhanced GAT layer with explicit Query/Key/Value extraction from observations
    """
    def __init__(self, 
                 query_dim: int, 
                 key_dim: int, 
                 value_dim: int,
                 out_channels: int, 
                 heads: int = 4, 
                 concat: bool = True, 
                 dropout: float = 0.0,
                 c_agent_decay: float = 1.0,
                 c_obstacle_decay: float = 2.0,
                 bias: bool = True, 
                 device=None):
        super().__init__(aggr='add', node_dim=0)
        
        self.query_dim = query_dim
        self.key_dim = key_dim  
        self.value_dim = value_dim
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.device = device if device is not None else torch.device('cpu')

        # Learnable decay coefficients (following paper's c_a^W and c_o^W)
        self.c_agent_decay = nn.Parameter(torch.tensor(c_agent_decay))
        self.c_obstacle_decay = nn.Parameter(torch.tensor(c_obstacle_decay))

        # Linear transformations for agents (used in message passing)
        self.lin_query = nn.Linear(query_dim, heads * out_channels, bias=False).to(self.device)
        self.lin_agent_key = nn.Linear(agent_key_dim, heads * out_channels, bias=False).to(self.device)
        self.lin_agent_value = nn.Linear(agent_value_dim, heads * out_channels, bias=False).to(self.device)
        
        # Linear transformations for obstacles (processed directly)
        self.lin_obstacle_key = nn.Linear(obstacle_key_dim, heads * out_channels, bias=False).to(self.device)
        self.lin_obstacle_value = nn.Linear(obstacle_value_dim, heads * out_channels, bias=False).to(self.device)
        
        # Output projection
        final_dim = heads * out_channels if concat else out_channels
        self.lin_out = nn.Linear(final_dim, out_channels, bias=True).to(self.device)
        
        # Attention scaling
        self.scale = (out_channels // heads) ** -0.5
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_query.weight)
        nn.init.xavier_uniform_(self.lin_key.weight)
        nn.init.xavier_uniform_(self.lin_value.weight)
        nn.init.xavier_uniform_(self.lin_out.weight)
        if hasattr(self.lin_out, 'bias') and self.lin_out.bias is not None:
            nn.init.zeros_(self.lin_out.bias)
            
    def forward(self, 
                query_features: torch.Tensor,
                key_features: torch.Tensor, 
                value_features: torch.Tensor,
                edge_index: torch.Tensor,
                positions: Optional[torch.Tensor] = None,  # ADD THIS PARAMETER
                edge_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            positions: Node positions [num_nodes, 2] for position-based attention
        """
        H, C = self.heads, self.out_channels
        
        # Transform to Q/K/V
        query = self.lin_query(query_features).view(-1, H, C)
        key = self.lin_key(key_features).view(-1, H, C)
        value = self.lin_value(value_features).view(-1, H, C)
        
        out = self.propagate(
            edge_index, 
            query=query, 
            key=key, 
            value=value,
            pos=positions,
            edge_type=edge_type,
            size=None
        )
        
        # Rest remains the same...
        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)
            
        out = self.lin_out(out)
        return out

        
    def message(self, query_i: torch.Tensor, key_j: torch.Tensor, 
                value_j: torch.Tensor, index: torch.Tensor,
                ptr: Optional[torch.Tensor], size_i: Optional[int],
                edge_type: Optional[torch.Tensor] = None,
                pos_i: Optional[torch.Tensor] = None,
                pos_j: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Enhanced message function with position-based attention"""
        
        if pos_i is not None and pos_j is not None:
            # Paper-style position-based attention
            dist = torch.norm(pos_i - pos_j, dim=-1, keepdim=True)
            # Separate decay coefficients for agents vs obstacles
            c_decay = 1.0  # You could make this learnable or edge-type dependent
            pos_weight = torch.exp(-c_decay * dist)
            
            # Standard attention
            alpha = (query_i * key_j).sum(dim=-1) * self.scale
            
            # Combine with position-based weights
            alpha = alpha * pos_weight.squeeze(-1)
        else:
            # Standard attention
            alpha = (query_i * key_j).sum(dim=-1) * self.scale
        
        # Apply softmax and continue as before
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = value_j * alpha.unsqueeze(-1)
        
        return out

class EnhancedPGATActor(nn.Module):
    """
    Enhanced PGAT Actor with nearest neighbor edge construction and proper Q/K/V extraction
    """
    def __init__(
        self,
        obs_config: ObservationConfig,
        total_obs_dim: int,               # Total observation dimension including neighbors/obstacles
        n_agent_outputs: int,
        gnn_hidden_dim: int = 128,
        n_gnn_layers: int = 2,
        n_attention_heads: int = 4,
        k_neighbours: Optional[int] = None,
        k_obstacles: Optional[int] = None,
        dropout: float = 0.0,
        pos_indices: slice = slice(0, 2),  # Position indices in observation
        device=None,
    ):
        """
        Args:
            obs_config: Configuration for observation indexing
            total_obs_dim: Total dimension of each agent's observation
            n_agent_outputs: Dimension of agent output (actions)
            pos_indices: Slice indicating where positions are in the observation
            ... other args
        """
        super().__init__()
        
        self.obs_config = obs_config
        self.total_obs_dim = total_obs_dim
        self.n_agent_outputs = n_agent_outputs
        self.k_neighbours = k_neighbours or obs_config.k_neighbors
        self.k_obstacles = k_obstacles or obs_config.k_obstacles
        self.pos_indices = pos_indices
        self.device = device if device is not None else torch.device('cpu')
        
        # Calculate Q/K/V dimensions based on observation config
        self.query_dim = self._calculate_feature_dim(obs_config.get_agent_query_indices())
        
        # Key dimension: neighbor features + obstacle features
        neighbor_key_dim = self._calculate_feature_dim(obs_config.get_neighbor_key_indices())
        obstacle_key_dim = self._calculate_feature_dim(obs_config.get_obstacle_key_indices())
        self.key_dim = neighbor_key_dim + obstacle_key_dim
        
        # Value dimension: neighbor features + obstacle features  
        neighbor_value_dim = self._calculate_feature_dim(obs_config.get_neighbor_value_indices())
        obstacle_value_dim = self._calculate_feature_dim(obs_config.get_obstacle_value_indices())
        self.value_dim = neighbor_value_dim + obstacle_value_dim
        
        print(f"Q/K/V dimensions: {self.query_dim}/{self.key_dim}/{self.value_dim}")
        
        # Build GAT layers
        self.gat_layers = nn.ModuleList()
        
        for i in range(n_gnn_layers):
            if i == 0:
                # First layer uses extracted Q/K/V from observations
                layer = EnhancedPositionAwareGATLayer(
                    query_dim=self.query_dim,
                    key_dim=self.key_dim,
                    value_dim=self.value_dim,
                    out_channels=gnn_hidden_dim,
                    heads=n_attention_heads,
                    concat=True,
                    dropout=dropout,
                    device=self.device
                )
            else:
                # Subsequent layers use hidden representations
                layer = EnhancedPositionAwareGATLayer(
                    query_dim=gnn_hidden_dim,
                    key_dim=gnn_hidden_dim,
                    value_dim=gnn_hidden_dim,
                    out_channels=gnn_hidden_dim,
                    heads=n_attention_heads,
                    concat=True,
                    dropout=dropout,
                    device=self.device
                )
            self.gat_layers.append(layer)
            
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

    def _extract_positions(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract positions from flattened observations"""
        batch_size = obs.shape[0] // self.total_obs_dim if obs.dim() == 1 else obs.shape[0]
        
        if obs.dim() == 2:  # Already flattened: [batch_size * n_agents, obs_dim]
            positions = obs[:, self.pos_indices]  # [batch_size * n_agents, 2]
        else:  # 3D tensor: [batch_size, n_agents, obs_dim]
            positions = obs[:, :, self.pos_indices].reshape(-1, 2)  # Flatten to [batch_size * n_agents, 2]
        
        return positions



    def _extract_qkv_features_separated(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract Q/K/V features with separate agent and obstacle processing
        Returns: (query, agent_key, agent_value, obstacle_key, obstacle_value)
        """
        # Query: agent's own state
        query_features = self._extract_features_from_obs(obs, self.obs_config.get_agent_query_indices())
        
        # Separate agent and obstacle features
        agent_key_features = self._extract_features_from_obs(obs, self.obs_config.get_neighbor_key_indices())
        agent_value_features = self._extract_features_from_obs(obs, self.obs_config.get_neighbor_value_indices())
        
        obstacle_key_features = self._extract_features_from_obs(obs, self.obs_config.get_obstacle_key_indices())
        obstacle_value_features = self._extract_features_from_obs(obs, self.obs_config.get_obstacle_value_indices())
        
        return query_features, agent_key_features, agent_value_features, obstacle_key_features, obstacle_value_features


    def _build_graph_batch(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph edges based on k-nearest neighbors using agent positions.
        Adapted from the GNNActor implementation.
        
        Args:
            obs (torch.Tensor): Observation tensor of shape (batch_size, n_agents, obs_dim)
            
        Returns:
            tuple(torch.Tensor, torch.Tensor):
                - edge_index (torch.Tensor): Edge indices, shape (2, num_total_edges)
                - edge_type (torch.Tensor): Edge types (all zeros for agent-agent edges)
        """
        batch_size, n_agents, obs_dim = obs.shape
        
        if n_agents == 0:  # Handle case with no agents
            return (torch.empty(2, 0, dtype=torch.long, device=self.device),
                    torch.empty(0, dtype=torch.long, device=self.device))
        
        # Extract positions for distance calculation
        pos = obs[:, :, self.pos_indices].to(self.device)  # (batch_size, n_agents, 2)
        
        # Calculate pairwise distances within each batch element
        # Shape: (batch_size, n_agents, n_agents)
        dist = torch.cdist(pos, pos, p=2)
        
        # Find k-nearest neighbors (including self for now)
        if self.k_neighbours is not None and self.k_neighbours > 0:
            # Get k+1 nearest neighbors (including self)
            knn_val, knn_idx = torch.topk(dist, k=min(self.k_neighbours + 1, n_agents), 
                                         dim=-1, largest=False, sorted=True)
            
            # Remove self-connections (first element is always self with distance 0)
            if knn_idx.shape[-1] > 1:
                neighbor_idx = knn_idx[..., 1:]  # Shape: (batch_size, n_agents, k)
                k_actual = neighbor_idx.shape[-1]
            else:
                neighbor_idx = torch.empty(batch_size, n_agents, 0, dtype=torch.long, device=self.device)
                k_actual = 0
            
            if k_actual > 0:
                # Create source indices for each agent
                source_idx = torch.arange(n_agents, device=self.device).view(1, -1, 1).expand(batch_size, -1, k_actual)
                
                # Flatten the source and target indices for k-NN edges
                flat_source_knn = source_idx.reshape(batch_size, -1)  # (batch_size, n_agents * k)
                flat_target_knn = neighbor_idx.reshape(batch_size, -1)  # (batch_size, n_agents * k)
                
                # Construct k-NN edge_index for the batch with proper offsets
                row_list_knn = []
                col_list_knn = []
                node_offset = 0
                for b in range(batch_size):
                    # k-NN edges with batch offset
                    rows_b_knn = flat_source_knn[b] + node_offset
                    cols_b_knn = flat_target_knn[b] + node_offset
                    row_list_knn.append(rows_b_knn)
                    col_list_knn.append(cols_b_knn)
                    node_offset += n_agents
                
                row_edge_knn = torch.cat(row_list_knn)  # Source nodes for k-NN
                col_edge_knn = torch.cat(col_list_knn)  # Target nodes for k-NN
                
                knn_edge_index = torch.stack([row_edge_knn, col_edge_knn], dim=0)
            else:
                knn_edge_index = torch.empty(2, 0, dtype=torch.long, device=self.device)
        else:
            # If k_neighbours is 0 or None, create empty edge index
            knn_edge_index = torch.empty(2, 0, dtype=torch.long, device=self.device)
        
        # Add self-loops for all agents
        global_node_indices = torch.arange(batch_size * n_agents, device=self.device)
        self_loop_edge_index = torch.stack([global_node_indices, global_node_indices], dim=0)
        
        # Combine k-NN edges and self-loops
        edge_index = torch.cat([knn_edge_index, self_loop_edge_index], dim=1)
        
        # Create edge types (0 for k-NN edges, 0 for self-loops - all same type)
        edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long, device=self.device)
        
        return edge_index, edge_type
        
    def forward(self, agent_observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with nearest neighbor graph construction and proper Q/K/V extraction
        
        Args:
            agent_observations: Each agent's observation
                               [batch_size, n_agents, total_obs_dim]
                               
        Returns:
            Actor outputs [batch_size, n_agents, n_agent_outputs]
        """
        batch_size, n_agents = agent_observations.shape[0], agent_observations.shape[1]
        
        # Ensure input is on correct device
        obs = agent_observations.to(dtype=torch.float32, device=self.device)
        
        if batch_size == 0 or n_agents == 0:
            return torch.zeros(batch_size, n_agents, self.n_agent_outputs, device=self.device)
        
        # Build graph structure based on spatial proximity
        edge_index, edge_type = self._build_graph_batch(obs)
        
        # Extract positions
        positions = obs[:, :, self.pos_indices].reshape(batch_size * n_agents, -1)
        
        # Flatten observations for graph processing
        # Node features: [batch_size * n_agents, obs_dim]
        x = obs.reshape(batch_size * n_agents, -1)
        
        # Pass through GAT layers
        for i, layer in enumerate(self.gat_layers):
            if i == 0:
                # First layer: extract Q/K/V from observations
                query_features, key_features, value_features = self._extract_qkv_features(x)
                x = layer(query_features, key_features, value_features, edge_index, positions, edge_type)
            else:
                # Subsequent layers: use hidden representations for Q/K/V
                x = layer(x, x, x, edge_index, positions, edge_type)
            
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
    
        # Apply output MLP
        agent_outputs = self.output_mlp(x)
        
        # Reshape back to batch format
        return agent_outputs.view(batch_size, n_agents, -1)


# Example usage showing nearest neighbor approach with proper Q/K/V extraction
def create_formation_pgat_example():
    """Example of how to set up the enhanced PGAT for formation control with nearest neighbor edges"""
    
    # Configure your observation structure
    k_neighbors, k_obstacles = 2, 2
    obs_config = ObservationConfig(k_neighbors=k_neighbors, k_obstacles=k_obstacles)
    
    # Calculate total observation dimension
    agent_self_dim = 23  # Updated based on obs_config indices
    total_obs_dim = agent_self_dim + (k_neighbors * 7) + (k_obstacles * 2)
    
    n_outputs = 2  # action dimensions [ax, ay]
    
    # Create the network
    actor = EnhancedPGATActor(
        obs_config=obs_config,
        total_obs_dim=total_obs_dim,
        n_agent_outputs=n_outputs,
        k_neighbours=4,  # Number of nearest neighbors for graph construction
        pos_indices=slice(0, 2),  # Position is at the beginning of observation
    )
    
    return actor


if __name__ == "__main__":
    # Test the enhanced implementation with proper Q/K/V extraction
    actor = create_formation_pgat_example()
    
    # Example input
    batch_size, n_agents = 2, 6
    total_obs_dim = 52
    
    agent_observations = torch.randn(batch_size, n_agents, total_obs_dim)
    # Set positions to be more realistic (agents spread out)
    agent_observations[:, :, 0:2] = torch.randn(batch_size, n_agents, 2) * 5
    
    # Forward pass
    actions = actor(agent_observations)
    print(f"Output shape: {actions.shape}")  # Should be [2, 6, 2]
    
    print("\nEnhanced PGAT with proper Q/K/V extraction:")
    print("- Query: extracted from agent's own state (position, velocity, gradients, goal info)")
    print("- Key: extracted from neighbor and obstacle observations")  
    print("- Value: extracted from neighbor and obstacle observations (including progress)")
    print("- Graph connectivity: k-nearest neighbors based on spatial proximity")


