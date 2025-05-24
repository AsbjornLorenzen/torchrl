
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import torch.nn.functional as F
from typing import Optional, Tuple


class ObservationConfig:
    """Configuration class to define how to extract Q/K/V from observations"""
    def __init__(self):
        # Define observation structure indices
        # These should be configured based on your specific observation space
        
        # Current agent observation indices
        self.agent_position_idx = slice(0, 2)          # [x, y] position
        self.agent_velocity_idx = slice(2, 4)          # [vx, vy] velocity
        self.formation_vector_idx = slice(4, 6)        # vector to nearest formation point
        self.formation_progress_idx = slice(6, 9)      # progress in last few steps [3 timesteps]
        self.formation_error_idx = slice(9, 10)        # current formation error
        
        # Neighbor agent observation indices (per neighbor)
        self.neighbor_position_idx = slice(0, 2)       # relative position to neighbor
        self.neighbor_velocity_idx = slice(2, 4)       # neighbor's velocity
        self.neighbor_formation_role_idx = slice(4, 5) # neighbor's formation role/ID
        self.neighbor_distance_idx = slice(5, 6)       # distance to neighbor
        self.neighbor_formation_progress_idx = slice(6, 9)  # neighbor's formation progress
        
        # Obstacle observation indices (per obstacle)
        self.obstacle_position_idx = slice(0, 2)       # relative position to obstacle
        self.obstacle_size_idx = slice(2, 3)           # obstacle size/radius
        self.obstacle_distance_idx = slice(3, 4)       # distance to obstacle
        self.obstacle_urgency_idx = slice(4, 5)        # avoidance urgency
        
    def get_agent_query_indices(self) -> List[slice]:
        """Define which parts of agent observation to use for Query"""
        return [
            self.agent_position_idx,
            self.agent_velocity_idx,
            self.formation_vector_idx,
            self.formation_progress_idx
        ]
    
    def get_neighbor_key_indices(self) -> List[slice]:
        """Define which parts of neighbor observation to use for Key"""
        return [
            self.neighbor_position_idx,
            self.neighbor_velocity_idx,
            self.neighbor_distance_idx
        ]
    
    def get_neighbor_value_indices(self) -> List[slice]:
        """Define which parts of neighbor observation to use for Value"""
        return [
            self.neighbor_position_idx,
            self.neighbor_velocity_idx,
            self.neighbor_formation_role_idx,
            self.neighbor_formation_progress_idx
        ]
    
    def get_obstacle_key_indices(self) -> List[slice]:
        """Define which parts of obstacle observation to use for Key"""
        return [
            self.obstacle_position_idx,
            self.obstacle_size_idx,
            self.obstacle_distance_idx
        ]
    
    def get_obstacle_value_indices(self) -> List[slice]:
        """Define which parts of obstacle observation to use for Value"""
        return [
            self.obstacle_position_idx,
            self.obstacle_size_idx,
            self.obstacle_urgency_idx
        ]



class PositionAwareGATLayer(MessagePassing):
    """
    Custom GAT layer that uses positions as Query/Key and features as Values.
    This implements the specific attention mechanism described in the paper.
    """
    def __init__(self, in_channels: int, out_channels: int, pos_dim: int = 2, 
                 heads: int = 4, concat: bool = True, dropout: float = 0.0,
                 bias: bool = True, device=None):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pos_dim = pos_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.device = device if device is not None else torch.device('cpu')
        
        # Linear transformations for queries (position-based)
        self.lin_query = nn.Linear(pos_dim, heads * out_channels, bias=False).to(self.device)
        
        # Linear transformations for keys (position-based)
        self.lin_key = nn.Linear(pos_dim, heads * out_channels, bias=False).to(self.device)
        
        # Linear transformations for values (feature-based)
        self.lin_value = nn.Linear(in_channels, heads * out_channels, bias=False).to(self.device)
        
        # Output projection
        if concat:
            self.lin_out = nn.Linear(heads * out_channels, out_channels, bias=bias).to(self.device)
        else:
            self.lin_out = nn.Linear(out_channels, out_channels, bias=bias).to(self.device)
            
        # Attention scaling factor
        self.scale = (out_channels // heads) ** -0.5
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_query.weight)
        nn.init.xavier_uniform_(self.lin_key.weight)
        nn.init.xavier_uniform_(self.lin_value.weight)
        nn.init.xavier_uniform_(self.lin_out.weight)
        if hasattr(self.lin_out, 'bias') and self.lin_out.bias is not None:
            nn.init.zeros_(self.lin_out.bias)
            
    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor,
                edge_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_channels]
            pos: Node positions [num_nodes, pos_dim]
            edge_index: Edge indices [2, num_edges]
            edge_type: Optional edge types (0 for agent-agent, 1 for agent-obstacle)
        """
        H, C = self.heads, self.out_channels
        
        # Transform positions to queries and keys
        query = self.lin_query(pos).view(-1, H, C)  # [num_nodes, heads, out_channels]
        key = self.lin_key(pos).view(-1, H, C)
        
        # Transform features to values
        value = self.lin_value(x).view(-1, H, C)
        
        # Propagate messages
        out = self.propagate(edge_index, query=query, key=key, value=value, 
                           size=None, edge_type=edge_type)
        
        # Reshape and apply output projection
        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)
            
        out = self.lin_out(out)
        
        return out
        
    def message(self, query_i: torch.Tensor, key_j: torch.Tensor, 
                value_j: torch.Tensor, index: torch.Tensor,
                ptr: Optional[torch.Tensor], size_i: Optional[int],
                edge_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention-weighted messages.
        query_i: Queries from target nodes [num_edges, heads, out_channels]
        key_j: Keys from source nodes [num_edges, heads, out_channels]
        value_j: Values from source nodes [num_edges, heads, out_channels]
        """
        # Compute attention scores
        alpha = (query_i * key_j).sum(dim=-1) * self.scale  # [num_edges, heads]
        
        # Apply softmax to get attention weights
        alpha = softmax(alpha, index, ptr, size_i)
        
        # Apply dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Weight values by attention
        out = value_j * alpha.unsqueeze(-1)  # [num_edges, heads, out_channels]
        
        return out


class PGATActor(nn.Module):
    """
    Position-aware Graph Attention Network Actor for multi-agent formation control.
    
    This implementation follows the paper's approach where:
    - Current agent position (p_k) is used as Query
    - Neighboring agent/obstacle positions (p_l, p_o) are used as Keys
    - High-dimensional observation features (f_kl, f_ko) are used as Values
    """
    def __init__(
        self,
        n_agent_inputs: int,
        n_agent_outputs: int,
        gnn_hidden_dim: int = 128,
        n_gnn_layers: int = 2,
        n_attention_heads: int = 4,
        activation_class=nn.Tanh,
        k_neighbours: Optional[int] = None,
        k_obstacles: Optional[int] = None,
        agent_pos_indices: slice = slice(0, 2),
        neighbor_agent_pos_indices: Optional[slice] = None,
        obstacle_pos_indices: Optional[slice] = None,
        share_params: bool = True,
        dropout: float = 0.0,
        device=None,
    ):
        """
        Args:
            n_agent_inputs: Dimension of agent observation
            n_agent_outputs: Dimension of agent output (actions)
            gnn_hidden_dim: Hidden dimension for GNN layers
            n_gnn_layers: Number of GAT layers
            n_attention_heads: Number of attention heads
            activation_class: Activation function class
            k_neighbours: Number of neighboring agents (if None, use all)
            k_obstacles: Number of obstacles considered (if None, use all available)
            agent_pos_indices: Slice indicating position indices in own observation
            neighbor_agent_pos_indices: Slice for neighbor positions (if different from agent)
            obstacle_pos_indices: Slice for obstacle positions in observation
            share_params: Whether to share parameters (kept for compatibility)
            dropout: Dropout rate for attention
            device: Torch device
        """
        super().__init__()
        
        try:
            import torch_geometric
        except ImportError:
            raise ImportError("PyTorch Geometric is required for PGATActor.")
            
        self.n_agent_inputs = n_agent_inputs
        self.n_agent_outputs = n_agent_outputs
        self.k_neighbours = k_neighbours
        self.k_obstacles = k_obstacles
        self.agent_pos_indices = agent_pos_indices
        self.neighbor_agent_pos_indices = neighbor_agent_pos_indices or agent_pos_indices
        self.obstacle_pos_indices = obstacle_pos_indices or agent_pos_indices
        self.device = device if device is not None else torch.device('cpu')
        self.dropout = dropout
        
        # Infer position dimension from indices
        pos_dim = agent_pos_indices.stop - agent_pos_indices.start
        
        # Build GAT layers using custom position-aware attention
        self.gat_layers = nn.ModuleList()
        input_dim = n_agent_inputs
        
        for i in range(n_gnn_layers):
            if i == 0:
                # First layer: use position-aware attention
                layer = PositionAwareGATLayer(
                    input_dim, 
                    gnn_hidden_dim,
                    pos_dim=pos_dim,
                    heads=n_attention_heads,
                    concat=True,
                    dropout=dropout,
                    device=self.device
                )
            else:
                # Subsequent layers: can use standard GAT or continue with position-aware
                layer = PositionAwareGATLayer(
                    gnn_hidden_dim,
                    gnn_hidden_dim,
                    pos_dim=pos_dim,
                    heads=n_attention_heads,
                    concat=True,
                    dropout=dropout,
                    device=self.device
                )
            self.gat_layers.append(layer)
            
        self.activation = activation_class().to(self.device)
        
        # Output MLP
        mlp_hidden_dim = 256
        self.output_mlp = nn.Sequential(
            nn.Linear(gnn_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, n_agent_outputs)
        ).to(self.device)
        
    def _extract_positions_and_features(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract agent positions, neighbor positions, and obstacle positions from observations.
        
        Returns:
            agent_pos: Current agent positions [batch_size, n_agents, pos_dim]
            neighbor_features: Features for all observations [batch_size, n_agents, obs_dim]
            obstacle_pos: Obstacle positions if available
        """
        # Extract current agent positions
        agent_pos = obs[:, :, self.agent_pos_indices]
        
        # For now, we'll use the full observation as features
        # In practice, you might want to extract specific feature subsets
        features = obs
        
        # Extract obstacle positions if specified
        # This assumes obstacles are encoded in the same observation tensor
        # You may need to adapt this based on your specific observation structure
        obstacle_pos = None
        if self.obstacle_pos_indices is not None and self.k_obstacles is not None:
            obstacle_pos = obs[:, :, self.obstacle_pos_indices]

        neighbor_pos = obs[:,:, self.neighbor_agent_pos_indices]
            
        return agent_pos, features, obstacle_pos
        
    def _build_graph_with_attention_info(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Build graph structure with position information for attention mechanism.
        
        Returns:
            x: Node features
            pos: Node positions  
            edge_index: Edge connectivity
            batch_vector: Batch assignment
            edge_type: Edge types (agent-agent vs agent-obstacle)
        """
        batch_size, n_agents, obs_dim = obs.shape
        
        if n_agents == 0:
            return (torch.empty(0, obs_dim, device=self.device),
                    torch.empty(0, 2, device=self.device),
                    torch.empty(2, 0, dtype=torch.long, device=self.device),
                    torch.empty(0, dtype=torch.long, device=self.device),
                    None)
                    
        # Extract positions and features
        agent_pos, features, obstacle_pos = self._extract_positions_and_features(obs)
        
        # Flatten for graph construction
        x = features.reshape(batch_size * n_agents, obs_dim)
        pos = agent_pos.reshape(batch_size * n_agents, -1)
        
        # Build edges based on k-nearest neighbors
        dist = torch.cdist(agent_pos, agent_pos, p=2)
        
        edge_list = []
        edge_type_list = []
        
        # Agent-agent edges
        if self.k_neighbours is not None and self.k_neighbours > 0:
            # Find k nearest neighbors
            _, knn_idx = torch.topk(dist, k=self.k_neighbours + 1, dim=-1, largest=False)
            neighbor_idx = knn_idx[..., 1:]  # Exclude self
            
            # Build edge index for each batch
            for b in range(batch_size):
                offset = b * n_agents
                for i in range(n_agents):
                    for j in range(self.k_neighbours):
                        if neighbor_idx[b, i, j] < n_agents:  # Valid neighbor
                            edge_list.append([i + offset, neighbor_idx[b, i, j].item() + offset])
                            edge_type_list.append(0)  # Agent-agent edge
        
        # Add self-loops
        for i in range(batch_size * n_agents):
            edge_list.append([i, i])
            edge_type_list.append(0)  # Treat self-loops as agent edges
            
        # Convert to tensor
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t()
            edge_type = torch.tensor(edge_type_list, dtype=torch.long, device=self.device)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=self.device)
            edge_type = None
            
        # Batch vector
        batch_vector = torch.cat([
            torch.full((n_agents,), b, dtype=torch.long, device=self.device)
            for b in range(batch_size)
        ])
        
        return x, pos, edge_index, batch_vector, edge_type
        
    def forward(self, agent_observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GAT actor network.
        
        Args:
            agent_observations: Observations tensor [batch_size, n_agents, obs_dim]
            
        Returns:
            Actor outputs [batch_size, n_agents, n_agent_outputs]
        """
        obs = agent_observations.to(dtype=torch.float32, device=self.device)
        batch_size, n_agents = obs.shape[0], obs.shape[1]
        
        if batch_size == 0 or n_agents == 0:
            return torch.zeros(batch_size, n_agents, self.n_agent_outputs, device=self.device)
            
        # Build graph with position information
        x, pos, edge_index, batch_vector, edge_type = self._build_graph_with_attention_info(obs)
        
        # Pass through GAT layers with position-aware attention
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, pos, edge_index, edge_type)
            if i < len(self.gat_layers) - 1:  # Don't activate after last layer
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        # Output through MLP
        agent_outputs = self.output_mlp(x)
        
        # Reshape back to batch format
        final_output = agent_outputs.view(batch_size, n_agents, -1)
        
        return final_output
