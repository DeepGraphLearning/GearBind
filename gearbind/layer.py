import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min

from torchdrug import layers, data
from torchdrug.layers import functional


class DDGAttention(nn.Module):

    def __init__(self, input_dim, output_dim, value_dim=16, query_key_dim=16, num_heads=12):
        super(DDGAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.value_dim = value_dim
        self.query_key_dim = query_key_dim
        self.num_heads = num_heads

        self.query = nn.Linear(input_dim, query_key_dim*num_heads, bias=False)
        self.key   = nn.Linear(input_dim, query_key_dim*num_heads, bias=False)
        self.value = nn.Linear(input_dim, value_dim*num_heads, bias=False)

        self.out_transform = nn.Linear(
            in_features = (num_heads*value_dim) + (num_heads*(3+3+1)),
            out_features = output_dim,
        )
        self.layer_norm = nn.LayerNorm(output_dim)

    def _alpha_from_logits(self, logits, mask, inf=1e5):
        """
        Args:
            logits: Logit matrices, (N, L_i, L_j, num_heads).
            mask:   Masks, (N, L).
        Returns:
            alpha:  Attention weights.
        """
        N, L, _, _ = logits.size()
        mask_row = mask.view(N, L, 1, 1).expand_as(logits)      # (N, L, *, *)
        mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)     # (N, L, L, *)
        
        logits = torch.where(mask_pair, logits, logits-inf)
        alpha = torch.softmax(logits, dim=2)  # (N, L, L, num_heads)
        alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
        return alpha

    def _heads(self, x, n_heads, n_ch):
        """
        Args:
            x:  (..., num_heads * num_channels)
        Returns:
            (..., num_heads, num_channels)
        """
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)

    def forward(self, x, pos_CA, pos_CB, frame, mask):
        # Attention logits
        query = self._heads(self.query(x), self.num_heads, self.query_key_dim)    # (N, L, n_heads, head_size)
        key = self._heads(self.key(x), self.num_heads, self.query_key_dim)      # (N, L, n_heads, head_size)
        logits_node = torch.einsum('blhd, bkhd->blkh', query, key)
        alpha = self._alpha_from_logits(logits_node, mask)  # (N, L, L, n_heads)

        value = self._heads(self.value(x), self.num_heads, self.value_dim)  # (N, L, n_heads, head_size)
        feat_node = torch.einsum('blkh, bkhd->blhd', alpha, value).flatten(-2)
        
        rel_pos = pos_CB.unsqueeze(1) - pos_CA.unsqueeze(2)  # (N, L, L, 3)
        atom_pos_bias = torch.einsum('blkh, blkd->blhd', alpha, rel_pos)  # (N, L, n_heads, 3)
        feat_distance = atom_pos_bias.norm(dim=-1)
        feat_points = torch.einsum('blij, blhj->blhi', frame, atom_pos_bias)  # (N, L, n_heads, 3)
        feat_direction = feat_points / (feat_points.norm(dim=-1, keepdim=True) + 1e-10)
        feat_spatial = torch.cat([
            feat_points.flatten(-2), 
            feat_distance, 
            feat_direction.flatten(-2),
        ], dim=-1)

        feat_all = torch.cat([feat_node, feat_spatial], dim=-1)

        feat_all = self.out_transform(feat_all)  # (N, L, F)
        feat_all = torch.where(mask.unsqueeze(-1), feat_all, torch.zeros_like(feat_all))
        if x.shape[-1] == feat_all.shape[-1]:
            x_updated = self.layer_norm(x + feat_all)
        else:
            x_updated = self.layer_norm(feat_all)

        return x_updated


class GeometricAttention(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, value_dim=16, query_key_dim=16, 
                num_query_points=8, num_value_points=8, num_heads=12, activation=None):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.pair_feat_dim = pair_feat_dim
        self.value_dim = value_dim
        self.query_key_dim = query_key_dim
        self.num_query_points = num_query_points
        self.num_value_points = num_value_points
        self.num_heads = num_heads

        # Node
        self.proj_query = nn.Linear(node_feat_dim, query_key_dim*num_heads, bias=False)
        self.proj_key   = nn.Linear(node_feat_dim, query_key_dim*num_heads, bias=False)
        self.proj_value = nn.Linear(node_feat_dim, value_dim*num_heads, bias=False)

        # Pair
        self.proj_pair_bias = nn.Linear(pair_feat_dim, num_heads, bias=False)

        self.spatial_coef = nn.Parameter(torch.full([1, 1, 1, self.num_heads], fill_value=np.log(np.exp(1.) - 1.)), requires_grad=True)

        # Output
        self.out_transform = nn.Linear(
            in_features = (num_heads*pair_feat_dim) + (num_heads*value_dim) + (num_heads*(3+3+1)),
            out_features = node_feat_dim,
        )
        self.layer_norm = nn.LayerNorm(node_feat_dim)

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

    def _alpha_from_logits(self, logits, mask, inf=1e5):
        """
        Args:
            logits: Logit matrices, (N, L_i, L_j, num_heads).
            mask:   Masks, (N, L).
        Returns:
            alpha:  Attention weights.
        """
        N, L, _, _ = logits.size()
        mask_row = mask.view(N, L, 1, 1).expand_as(logits)      # (N, L, *, *)
        mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)     # (N, L, L, *)
        
        logits = torch.where(mask_pair, logits, logits-inf)
        alpha = torch.softmax(logits, dim=2)  # (N, L, L, num_heads)
        alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
        return alpha

    def _heads(self, x, n_heads, n_ch):
        """
        Args:
            x:  (..., num_heads * num_channels)
        Returns:
            (..., num_heads, num_channels)
        """
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)

    def _normalize_vector(self, v, dim, eps=1e-6):
        return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)

    def _node_logits(self, x):
        query_l = self._heads(self.proj_query(x), self.num_heads, self.query_key_dim)    # (N, L, n_heads, qk_ch)
        key_l = self._heads(self.proj_key(x), self.num_heads, self.query_key_dim)      # (N, L, n_heads, qk_ch)

        query_l = query_l.permute(0, 2, 1, 3)   # (N,L1,H,C) -> (N,H,L1,C)
        key_l = key_l.permute(0, 2, 3, 1)       # (N,L2,H,C) -> (N,H,C,L2)

        logits = torch.matmul(query_l, key_l)   # (N,H,L1,L2)
        logits = logits.permute(0, 2, 3, 1)     # (N,L1,L2,H)

        return logits

    def _pair_logits(self, z):
        logits_pair = self.proj_pair_bias(z)
        return logits_pair

    def _beta_logits(self, p_CB):
        qk = p_CB[:, :, None, :].expand(-1, -1, self.num_heads, -1)
        sum_sq_dist = ((qk.unsqueeze(2) - qk.unsqueeze(1)) ** 2).sum(-1)    # (N, L, L, n_heads)
        gamma = F.softplus(self.spatial_coef)
        logtis_beta = sum_sq_dist * ((-1 * gamma * np.sqrt(2 / 9)) / 2)
        return logtis_beta

    def _pair_aggregation(self, alpha, z):
        N, L = z.shape[:2]
        feat_p2n = torch.einsum("blkh, blkc->blhc", alpha, z)
        return feat_p2n.reshape(N, L, -1)

    def _node_aggregation(self, alpha, x):
        N, L = x.shape[:2]
        value_l = self._heads(self.proj_value(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, v_ch)
        feat_node = torch.einsum("blkh, bkhc->blhc", alpha, value_l)
        return feat_node.reshape(N, L, -1)

    def _beta_aggregation(self, alpha, graph, p_CB):
        N, L = p_CB.size()[:2]
        v = p_CB[:, :, None, :].expand(N, L, self.num_heads, 3) # (N, L, n_heads, 3)
        aggr = torch.einsum('blkh, bkhd->blhd', alpha, v)

        atom_pos, _ = functional.variadic_to_padded(graph.atom_pos, graph.num_nodes)
        frame, _ = functional.variadic_to_padded(graph.frame, graph.num_nodes)
        feat_points = (aggr - atom_pos[:, :, None, graph.atom_name2id["CA"]])
        feat_points = torch.einsum('blij, blhj->blhi', frame, feat_points)  # (N, L, n_heads, 3)
        feat_distance = feat_points.norm(dim=-1)
        feat_direction = self._normalize_vector(feat_points, dim=-1, eps=1e-4)

        feat_spatial = torch.cat([
            feat_points.reshape(N, L, -1), 
            feat_distance.reshape(N, L, -1), 
            feat_direction.reshape(N, L, -1),
        ], dim=-1)

        return feat_spatial

    def forward(self, graph, x, z, mask):
        pos_CB = torch.where(
            graph.atom_pos_mask[:, graph.atom_name2id["CB"], None].expand(-1, 3).bool(),
            graph.atom_pos[:, graph.atom_name2id["CB"]],
            graph.atom_pos[:, graph.atom_name2id["CA"]],
        )
        pos_CB, _ = functional.variadic_to_padded(pos_CB, graph.num_nodes)

        # Attention logits
        logits_node = self._node_logits(x)
        logits_pair = self._pair_logits(z)
        logits_spatial = self._beta_logits(pos_CB)
        # Summing logits up and apply `softmax`.
        logits_sum = logits_node + logits_pair + logits_spatial
        alpha = self._alpha_from_logits(logits_sum * np.sqrt(1 / 3), mask)  # (N, L, L, n_heads)

        # Aggregate features
        feat_p2n = self._pair_aggregation(alpha, z)
        feat_node = self._node_aggregation(alpha, x)
        feat_spatial = self._beta_aggregation(alpha, graph, pos_CB)

        # Finally
        feat_all = self.out_transform(torch.cat([feat_p2n, feat_node, feat_spatial], dim=-1)) # (N, L, F)
        feat_all = torch.where(mask.unsqueeze(-1), feat_all, torch.zeros_like(feat_all))
        if self.activation:
            feat_all = self.activation(feat_all)
        x_updated = self.layer_norm(x + feat_all)
        return x_updated


class PerResidueEncoder(nn.Module):

    num_residue_type = len(data.Protein.residue2id)
    num_atom_names = len(data.Protein.atom_name2id)

    def __init__(self, hidden_dim):
        super(PerResidueEncoder, self).__init__()
        self.aatype_embed = nn.Embedding(self.num_residue_type, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.num_residue_type * self.num_atom_names * 3 + hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, graph):
        crd14 = (graph.atom_pos - graph.atom_pos[:, None, graph.atom_name2id["CA"]])
        crd14 = torch.einsum('bij, bkj->bki', graph.frame, crd14)  # (*, 14, 3)
        crd14_mask = graph.atom_pos_mask.unsqueeze(-1).expand_as(crd14).bool()
        crd14 = torch.where(crd14_mask, crd14, torch.zeros_like(crd14))

        residue_type = graph.residue_type
        residue_type[residue_type == -1] = self.num_residue_type - 1
        aa_expand  = residue_type[:, None, None, None].expand(-1, self.num_residue_type, self.num_atom_names, 3)
        rng_expand = torch.arange(0, self.num_residue_type)[None, :, None, None]
        rng_expand = rng_expand.expand(graph.num_node, -1, self.num_atom_names, 3).to(aa_expand)
        place_mask = (aa_expand == rng_expand)
        crd_expand = crd14[:, None, :, :].expand(-1, self.num_residue_type, self.num_atom_names, 3)
        crd_expand = torch.where(place_mask, crd_expand, torch.zeros_like(crd_expand))
        crd_feat = crd_expand.reshape(-1, self.num_residue_type * self.num_atom_names * 3)

        aa_feat = self.aatype_embed(residue_type) # (*, hidden_dim)

        out_feat = self.mlp(torch.cat([crd_feat, aa_feat], dim=-1))
        return out_feat
