import functools
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter_min, scatter_sum
from torch_cluster import knn, nearest

from torchdrug import core, data, layers, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("layers.InterfaceGraph")
class InterfaceGraph(nn.Module, core.Configurable):

    def __init__(self, entity_level='node', cutoff=10.0):
        super(InterfaceGraph, self).__init__()
        self.entity_level = entity_level
        self.cutoff = cutoff

    def get_interface(self, a, b):
        nearest_b_indices = nearest(a.node_position, b.node_position, a.node2graph, b.node2graph)
        nearest_distance = (a.node_position - b.node_position[nearest_b_indices]).norm(dim=-1)
        is_interface_atom = nearest_distance < self.cutoff

        is_interface_resiude = scatter_max(is_interface_atom.long(), a.atom2residue)[0]
        is_interface_atom = is_interface_resiude[a.atom2residue].bool()
        return is_interface_atom
    
    def forward(self, graph):
        entity_a = graph.subgraph(graph.entity_a)
        entity_b = graph.subgraph(graph.entity_b)
        interface_mask_a = self.get_interface(entity_a, entity_b)
        interface_mask_b = self.get_interface(entity_b, entity_a)
        mask = torch.zeros(graph.num_node, dtype=torch.bool, device=graph.device)
        mask[graph.entity_a] = interface_mask_a
        mask[graph.entity_b] = interface_mask_b
        mask[graph.is_mutation] = 1

        if self.entity_level in ["node", "atom"]:
            graph = graph.subgraph(mask)
        else:
            graph = graph.subresidue(mask)

        return graph


@R.register("layers.AtomPositionGather")
class AtomPositionGather(nn.Module, core.Configurable):

    def from_3_points(self, p_x_axis, origin, p_xy_plane, eps=1e-10):
        """
            Adpated from torchfold
            Implements algorithm 21. Constructs transformations from sets of 3 
            points using the Gram-Schmidt algorithm.
            Args:
                x_axis: [*, 3] coordinates
                origin: [*, 3] coordinates used as frame origins
                p_xy_plane: [*, 3] coordinates
                eps: Small epsilon value
            Returns:
                A transformation object of shape [*]
        """
        p_x_axis = torch.unbind(p_x_axis, dim=-1)
        origin = torch.unbind(origin, dim=-1)
        p_xy_plane = torch.unbind(p_xy_plane, dim=-1)

        e0 = [c1 - c2 for c1, c2 in zip(p_x_axis, origin)]
        e1 = [c1 - c2 for c1, c2 in zip(p_xy_plane, origin)]

        denom = torch.sqrt(sum((c * c for c in e0)) + eps)
        e0 = [c / denom for c in e0]
        dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
        e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
        denom = torch.sqrt(sum((c * c for c in e1)) + eps)
        e1 = [c / denom for c in e1]
        e2 = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ]

        rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
        rots = rots.reshape(rots.shape[:-1] + (3, 3))

        return rots

    def forward(self, graph):
        residue_mask = \
            scatter_add((graph.atom_name == graph.atom_name2id["N"]).float(), graph.atom2residue, dim_size=graph.num_residue) + \
            scatter_add((graph.atom_name == graph.atom_name2id["CA"]).float(), graph.atom2residue, dim_size=graph.num_residue) + \
            scatter_add((graph.atom_name == graph.atom_name2id["C"]).float(), graph.atom2residue, dim_size=graph.num_residue)
        residue_mask = (residue_mask == 3)
        atom_mask = residue_mask[graph.atom2residue] & (graph.atom_name == graph.atom_name2id["CA"])
        graph = graph.subresidue(residue_mask)

        atom_pos = torch.full((graph.num_residue, len(graph.atom_name2id), 3), float("inf"), dtype=torch.float, device=graph.device)
        atom_pos[graph.atom2residue, graph.atom_name] = graph.node_position
        atom_pos_mask = torch.zeros((graph.num_residue, len(graph.atom_name2id)), dtype=torch.bool, device=graph.device)
        atom_pos_mask[graph.atom2residue, graph.atom_name] = 1

        graph = graph.subgraph(graph.atom_name == graph.atom_name2id["CA"])
        frame = self.from_3_points(
            atom_pos[:, graph.atom_name2id["N"]],
            atom_pos[:, graph.atom_name2id["CA"]],
            atom_pos[:, graph.atom_name2id["C"]]
        ).transpose(-1, -2)

        graph.view = 'residue'
        with graph.residue():
            graph.atom_pos = atom_pos
            graph.atom_pos_mask = atom_pos_mask
            graph.frame = frame

        return graph, atom_mask


@R.register("layers.geometry.BackboneNode")
class BackboneNode(nn.Module, core.Configurable):
    """
    Construct only alpha carbon atoms.
    """

    def forward(self, graph):
        """
        Return a subgraph that only consists of alpha carbon nodes.

        Parameters:
            graph (Graph): :math:`n` graph(s)
        """
        mask = ((graph.atom_name == data.Protein.atom_name2id["N"]) |
                (graph.atom_name == data.Protein.atom_name2id["CA"]) |
                (graph.atom_name == data.Protein.atom_name2id["C"]) ) & (graph.atom2residue != -1)
        residue2num_atom = graph.atom2residue[mask].bincount(minlength=graph.num_residue)
        residue_mask = residue2num_atom > 0
        mask = mask & residue_mask[graph.atom2residue]
        graph = graph.subgraph(mask).subresidue(residue_mask)

        return graph


@R.register("layers.KNNMutationSite")
class KNNMutationSite(nn.Module, core.Configurable):
    
    def __init__(self, k=64):
        super(KNNMutationSite, self).__init__()
        self.k = k

    def forward(self, graph):
        # Construct mutation site graphs
        mutation_mask = graph.is_mutation.to(torch.bool) & (graph.atom_name == graph.atom_name2id["CA"])

        center_position = graph.node_position[mutation_mask]
        mut2graph = graph.node2graph[mutation_mask]
        # Calculate distance to center position in each graph
        center_indices = nearest(graph.node_position, center_position, graph.node2graph, mut2graph)
        dist_to_center = ((graph.node_position - center_position[center_indices])**2).sum(-1)
        dist_to_center[mutation_mask] = 0.0
        _, selected_index = functional.variadic_topk(dist_to_center, graph.num_nodes, self.k, largest=False)
        start = graph.num_cum_nodes - graph.num_nodes
        selected_index += start.view(-1, 1)
        selected_index = selected_index.view(-1) 
        node_mask = torch.zeros(graph.num_node, dtype=torch.bool, device=graph.device)
        node_mask[selected_index] = 1
        # graph = graph.subgraph(node_mask)
        return node_mask
