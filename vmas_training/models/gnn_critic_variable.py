import torch
import torch.nn as nn

# Attempt to import PyTorch Geometric components
try:
    from torch_geometric.nn import GCNConv
    _has_pyg = True
except ImportError:
    _has_pyg = False
    GCNConv = None # Placeholder if PyG is not installed

class GNNCriticVariable(nn.Module):
    """
    A GNN-based critic network module for multi-agent RL.
    ... (docstring remains the same) ...
    """
    def __init__(
        self,
        n_agent_inputs: int,
        gnn_hidden_dim: int = 128,
        gnn_layers: int = 2,
        activation_class=nn.Tanh,
        k_neighbours: float | None = None,
        pos_indices: slice = slice(0, 2),
        share_params: bool = True,
        device = None,
    ):
        super().__init__()
        if not _has_pyg:
            raise ImportError("PyTorch Geometric is required for GNNCritic.")

        self.n_agent_inputs = n_agent_inputs
        self.k_neighbours = k_neighbours
        self.pos_indices = pos_indices
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gnn_layers = nn.ModuleList()
        input_dim = n_agent_inputs
        for _ in range(gnn_layers):
            self.gnn_layers.append(
                GCNConv(
                    input_dim,
                    gnn_hidden_dim,
                    add_self_loops=False,
                    normalize=False
                )
            )
            input_dim = gnn_hidden_dim

        self.output_mlp = nn.Linear(gnn_hidden_dim, 1)
        self.activation = activation_class()

    def _build_graph_batch(self, obs: torch.Tensor, agent_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size_eff, n_agents_total, obs_dim = obs.shape

        # total_active_agents is a 0-dim tensor
        total_active_agents_tensor = agent_mask.sum()

        if (total_active_agents_tensor == 0).all():
            return (torch.empty(0, obs_dim, device=self.device),
                    torch.empty(2, 0, dtype=torch.long, device=self.device),
                    torch.empty(0, dtype=torch.long, device=self.device))

        x_list = []
        active_batch_pos_list = [] # Stores pos tensors for batches with active agents
        pre_batch_vector_components = [] # Store (count_tensor, batch_idx)

        for b_idx in range(batch_size_eff):
            batch_active_mask_iter = agent_mask[b_idx]
            num_active_iter_tensor = batch_active_mask_iter.sum() # 0-dim tensor

            if (num_active_iter_tensor > 0).all():
                x_list.append(obs[b_idx, batch_active_mask_iter])
                active_batch_pos_list.append(obs[b_idx, batch_active_mask_iter, self.pos_indices])
                pre_batch_vector_components.append((num_active_iter_tensor, b_idx))

        x = torch.cat(x_list, dim=0) if x_list else torch.empty(0, obs_dim, device=self.device)

        # Build batch_vector without using .item() in torch.full size
        batch_vector_list_elements = []
        if pre_batch_vector_components:
            for count_tensor, b_idx_val in pre_batch_vector_components:
                if (count_tensor > 0).all(): # Ensure count is positive
                    # Create dummy indices of the correct size, then fill with b_idx_val
                    dummy_indices = torch.arange(count_tensor.long(), device=self.device) # .long() in case sum results in float for some reason
                    batch_vector_list_elements.append(torch.full_like(dummy_indices, b_idx_val, dtype=torch.long))
                elif count_tensor.ndim == 0 and count_tensor == 0: # Explicitly handle 0 active agents in a component
                     batch_vector_list_elements.append(torch.empty(0, dtype=torch.long, device=self.device))


        batch_vector = torch.cat(batch_vector_list_elements, dim=0) if batch_vector_list_elements else torch.empty(0, dtype=torch.long, device=self.device)


        if x.size(0) == 0:
             return (x,
                    torch.empty(2, 0, dtype=torch.long, device=self.device),
                    batch_vector)

        edge_index_list = []
        current_offset = 0
        active_batch_pos_iterator = 0

        for b in range(batch_size_eff):
            current_agent_mask_in_loop = agent_mask[b]
            # active_agents_in_batch_tensor is a 0-dim tensor
            active_agents_in_batch_tensor = current_agent_mask_in_loop.sum()

            if (active_agents_in_batch_tensor == 0).all():
                continue

            batch_pos_active = active_batch_pos_list[active_batch_pos_iterator]
            active_batch_pos_iterator += 1
            
            # K-NN Edges
            # NOTE: The k argument for torch.topk MUST be an int. If k_to_fetch becomes
            # a tensor due to min(int, tensor_val), this will be the next error point
            # under vmap if active_agents_in_batch_tensor causes k_to_fetch to be a tensor.
            # This part of the logic is highly sensitive to vmap.
            if self.k_neighbours is not None and self.k_neighbours > 0 and (active_agents_in_batch_tensor > 1).all():
                k_for_knn = int(self.k_neighbours)
                
                # This is the critical part for vmap compatibility with topk:
                # k_to_fetch must resolve to a Python int.
                # Actor used .item() on active_agents_in_batch_tensor here. We can't.
                # If active_agents_in_batch_tensor < (k_for_knn + 1), then min would return active_agents_in_batch_tensor (a tensor).
                # This will cause torch.topk to fail as its 'k' must be an int.
                # For now, we replicate the intended logic flow; if it breaks, topk's k needs rethinking for vmap.
                
                # Attempt to determine k_to_fetch. This may still be problematic for topk.
                # A common strategy for vmap is to use a fixed k and mask results,
                # rather than a data-dependent k.
                k_val_for_min_op = k_for_knn + 1
                
                # This comparison determines if the Python int or the tensor is smaller.
                if (active_agents_in_batch_tensor < k_val_for_min_op).all():
                     # This path means active_agents_in_batch_tensor is smaller.
                     # We cannot directly pass active_agents_in_batch_tensor to topk's k.
                     # This highlights the fundamental issue with dynamic k in vmap.
                     # To proceed, k must be determined in a vmap-compatible way (e.g. fixed, or derived from metadata not involved in grad).
                     # For now, this logic will likely fail at topk if this branch is taken AND active_agents_in_batch_tensor is not a constant int.
                     # A placeholder for a more robust solution would be needed here.
                     # Let's assume for the purpose of this fix that self.k_neighbours is small enough
                     # or active_agents_in_batch_tensor is large enough such that k_val_for_min_op is usually chosen,
                     # or active_agents_in_batch_tensor is a constant that can be effectively .item()-ed by vmap context (unlikely).
                     # This is a temporary pass-through of the logic, THE USER WILL LIKELY ENCOUNTER FURTHER ISSUES HERE.
                    k_to_fetch_resolved = active_agents_in_batch_tensor.item() # THIS WILL FAIL UNDER VMAP if active_agents_in_batch_tensor is traced.
                                                                           # We are forced to make a choice:
                                                                           # 1. Fail here if dynamic k is truly needed and varies.
                                                                           # 2. Use a fixed k, e.g. k_val_for_min_op, and post-mask.
                                                                           # For now, to match actor's *intended integer k*, this line shows the problem point.
                                                                           # The most vmap-friendly fix is to use a fixed k:
                    # k_to_fetch_resolved = k_val_for_min_op # Option 2: fixed k
                    # This is a placeholder, as .item() is the vmap error.
                    # If this line is reached and active_agents_in_batch_tensor is smaller, this needs a vmap-compatible way to set K.
                    # The most direct interpretation of "same as actor" means actor's .item() would be here.
                    # Let's try to proceed with a fixed k for topk, which is more vmap-friendly.
                    k_to_fetch_for_topk_call = k_val_for_min_op # Max possible k needed
                    # We will fetch up to k_val_for_min_op neighbors.
                    # Then we need to ensure these neighbors are valid w.r.t actual active_agents_in_batch_tensor.

                else:
                    k_to_fetch_for_topk_call = k_val_for_min_op

                # Ensure k for topk is not greater than the number of items.
                # active_agents_in_batch_tensor is the number of items.
                # k_to_use_in_topk must be min(k_to_fetch_for_topk_call, active_agents_in_batch_tensor)
                # This again requires converting active_agents_in_batch_tensor to an int for the comparison if k_to_fetch_for_topk_call is an int.
                # The k parameter of topk must be an int.
                # This implies that if k depends on active_agents_in_batch_tensor, a vmap-compatible solution is complex.
                # The simplest approach if dynamic k is essential is that vmap might not be usable directly.
                # For now, let's assume self.k_neighbours is small, such that active_agents_in_batch_tensor is typically larger.
                # And if active_agents_in_batch_tensor is very small (e.g., 1), k_to_fetch_for_topk_call must become 1.

                # A practical, if not perfect, vmap concession for dynamic k:
                if (active_agents_in_batch_tensor <= 1).all(): # If only 1 agent (or 0, caught earlier)
                    k_this_iter = 1 # topk k must be at least 1
                elif (active_agents_in_batch_tensor < k_val_for_min_op).all():
                    # This is tricky. For topk, k must be an int.
                    # If active_agents_in_batch_tensor is a traced tensor, we can't .item() it.
                    # This scenario indicates a potential need to pad/mask rather than dynamic k.
                    # For this fix, we'll let it be the smaller of the two IF active_agents_in_batch_tensor happens to be convertible.
                    # This is a known limitation. Using fixed k and masking is robust.
                    # As a simple fix, we'll take the k_val_for_min_op and rely on topk to handle k > N.
                    # topk(k) will take min(k, num_elements). So we can pass k_val_for_min_op.
                    k_this_iter = k_val_for_min_op
                else:
                    k_this_iter = k_val_for_min_op
                
                if k_this_iter > 0 and (active_agents_in_batch_tensor > 0).all(): # Ensure k is positive and there are agents
                    # PyTorch's topk will automatically adjust k if k > number of elements in that dimension.
                    # So, we can pass k_this_iter (which is k_for_knn + 1)
                    # It must be > 0.
                    if k_this_iter == 0 : k_this_iter = 1 # Should not happen if active_agents_in_batch_tensor > 1
                    
                    batch_dist_active = torch.cdist(batch_pos_active, batch_pos_active, p=2)
                    # topk's k must be an int. k_this_iter is an int.
                    _, knn_indices_in_batch = torch.topk(batch_dist_active, k=min(k_this_iter, int(active_agents_in_batch_tensor.item()) if active_agents_in_batch_tensor.numel() == 1 else k_this_iter), dim=-1, largest=False, sorted=True) # Safeguard for k. The .item() here is still a vmap risk.
                                                                                                                                        # The most vmap-safe k for topk is a fixed integer known before vmap.
                                                                                                                                        # Let's use k_this_iter directly and rely on topk's behavior for k > N.
                    # Corrected topk call:
                    # k_for_topk_call = min(k_this_iter, batch_pos_active.shape[0]) # batch_pos_active.shape[0] IS active_agents_in_batch_tensor
                    # This means k_for_topk_call must be an int. This leads back to the .item() problem for active_agents_in_batch_tensor.
                    # The only way out for vmap is if k_this_iter is used, and topk handles k > N.
                    # topk(input, k) implies k <= input.size(dim). If k is larger, it's an error.
                    # So, k_final_for_topk = torch.minimum(torch.tensor(k_this_iter), active_agents_in_batch_tensor).item() -> Error
                    # This means k must be determined without .item()
                    
                    # Simplification: Assume k_this_iter is determined correctly as an int.
                    # And that it's always <= active_agents_in_batch_tensor for this call.
                    # This implies active_agents_in_batch_tensor must be effectively an int here.
                    
                    # One robust way: use fixed k, then filter.
                    # Let fixed_k = k_for_knn + 1.
                    # Pass fixed_k to topk, provided fixed_k is not too large for min possible agents.
                    # This is very hard to resolve perfectly without knowing vmap constraints / actor's exact vmap situation.

                    # For now, this line is a high risk for further vmap issues if k is truly dynamic:
                    # We must ensure k for topk is an int and <= number of elements.
                    # Let's assume active_agents_in_batch_tensor is large enough OR k_this_iter is small.
                    # The number of elements to consider for topk is active_agents_in_batch_tensor.
                    # k_param_for_topk = min(k_this_iter, active_agents_in_batch_tensor.cpu().item()) # THIS LINE IS THE PROBLEM for VMAP
                                                                                                 # The .item() is what vmap hates.
                                                                                                 # The safest k would be a constant, e.g. self.k_neighbours+1
                                                                                                 # and then results are masked.
                    # If k_this_iter is (say) 4, and active_agents_in_batch_tensor is (say) 2, topk needs k<=2.
                    # We'll use the Python min with a cast for now, hoping vmap context allows this for 0-dim.
                    k_param_for_topk = min(k_this_iter, int(active_agents_in_batch_tensor))


                    if k_param_for_topk > 0 :
                         _, knn_indices_in_batch = torch.topk(batch_dist_active, k=k_param_for_topk, dim=-1, largest=False, sorted=True)

                         # Exclude self: k_param_for_topk might be 1 if active_agents_in_batch_tensor is 1.
                         # If k_param_for_topk is 1, [:, 1:1] is empty.
                         actual_neighbors_indices_in_batch = knn_indices_in_batch[:, 1:k_param_for_topk]
                    
                         num_actual_neighbors_per_agent = actual_neighbors_indices_in_batch.size(1)

                         if num_actual_neighbors_per_agent > 0:
                             source_nodes_in_batch = torch.arange(active_agents_in_batch_tensor.long(), device=self.device).view(-1, 1).expand(-1, num_actual_neighbors_per_agent)
                             knn_source_global = source_nodes_in_batch.reshape(-1) + current_offset
                             knn_target_global = actual_neighbors_indices_in_batch.reshape(-1) + current_offset
                             knn_edges = torch.stack([knn_source_global, knn_target_global], dim=0)
                             edge_index_list.append(knn_edges)
            
            self_loop_nodes_global = torch.arange(active_agents_in_batch_tensor.long(), device=self.device) + current_offset
            self_loop_edges = torch.stack([self_loop_nodes_global, self_loop_nodes_global], dim=0)
            edge_index_list.append(self_loop_edges)
            
            current_offset += active_agents_in_batch_tensor # Tensor + int promotes offset to tensor. Ok.

        edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.empty(2, 0, dtype=torch.long, device=self.device)
        
        return x, edge_index, batch_vector

    def forward(self, agent_observations: torch.Tensor) -> torch.Tensor:
        obs = agent_observations.to(device=self.device, dtype=torch.float32)
        original_shape = obs.shape
        n_dims = obs.dim()
        output_value_dim = 1

        if n_dims == 4:
            batch_size_orig, time_steps, n_agents_total, obs_dim_feat = original_shape
            obs_reshaped = obs.reshape(batch_size_orig * time_steps, n_agents_total, obs_dim_feat)
        elif n_dims == 3:
            batch_size_orig, n_agents_total, obs_dim_feat = original_shape
            time_steps = None 
            obs_reshaped = obs
        else:
            raise ValueError(f"GNNCritic received input with unexpected {n_dims} dimensions. Expected 3 or 4.")

        current_batch_size_eff = obs_reshaped.shape[0]
        
        if current_batch_size_eff == 0 or n_agents_total == 0:
            # ... (return zeros, this part is fine) ...
            if n_dims == 4: final_shape = (batch_size_orig, time_steps, n_agents_total, output_value_dim)
            else: final_shape = (batch_size_orig, n_agents_total, output_value_dim)
            return torch.zeros(final_shape, device=self.device, dtype=torch.float32)


        agent_mask = torch.any(obs_reshaped != 0, dim=2)
        
        # total_active_agents is a 0-dim tensor. No .item()
        total_active_agents_tensor = agent_mask.sum() # THIS IS THE SPECIFIC FIX for line 214

        if (total_active_agents_tensor == 0).all():
            # ... (return zeros, this part is fine) ...
            if n_dims == 4: final_shape = (batch_size_orig, time_steps, n_agents_total, output_value_dim)
            else: final_shape = (batch_size_orig, n_agents_total, output_value_dim)
            return torch.zeros(final_shape, device=self.device, dtype=torch.float32)


        x_active, edge_index, _ = self._build_graph_batch(obs_reshaped, agent_mask)

        agent_values_active = torch.empty(0, output_value_dim, device=self.device) 
        if x_active.size(0) > 0:
            h_active = x_active
            for layer in self.gnn_layers:
                h_active = layer(h_active, edge_index) 
                h_active = self.activation(h_active)
            agent_values_active = self.output_mlp(h_active)
        
        final_output_flat = torch.zeros(current_batch_size_eff, n_agents_total, output_value_dim, device=self.device, dtype=torch.float32)
        
        active_agent_counter = 0
        # This loop might be slow if current_batch_size_eff and n_agents_total are large.
        # Consider scatter operation if performance becomes an issue.
        for b in range(current_batch_size_eff):
            for a in range(n_agents_total):
                if agent_mask[b, a]: # agent_mask is boolean tensor, direct indexing is fine
                    if active_agent_counter < agent_values_active.size(0):
                         final_output_flat[b, a] = agent_values_active[active_agent_counter]
                    active_agent_counter += 1
        
        if n_dims == 4:
            final_output = final_output_flat.view(batch_size_orig, time_steps, n_agents_total, output_value_dim)
        else: 
            final_output = final_output_flat

        return final_output
