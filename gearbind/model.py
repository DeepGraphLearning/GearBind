from collections.abc import Sequence

import torch
from torch import nn
from torch import autograd

from torch_scatter import scatter_add

from torchdrug import core, layers, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from . import layer, geometry


@R.register("models.BindModel")
class BindModel(nn.Module, core.Configurable):

    def __init__(self, model, num_mlp_layer=2):
        super(BindModel, self).__init__()
        self.model = model
        self.num_mlp_layer = num_mlp_layer

        hidden_dims = [model.output_dim] * (num_mlp_layer - 1)
        self.mlp = layers.MLP(model.output_dim * 2, hidden_dims + [1])

    def forward(self, batch, all_loss=None, metric=None):
        mutant = batch["mutant"]
        mutant_output = self.model(mutant, mutant.node_feature.float(), all_loss=all_loss, metric=metric)

        wild_type = batch["wild_type"]
        wild_type_output = self.model(wild_type, wild_type.node_feature.float(), all_loss=all_loss, metric=metric)
        
        wild_type_output = wild_type_output["graph_feature"]
        mutant_output = mutant_output["graph_feature"]

        outputs = torch.cat([mutant_output, wild_type_output], dim=-1)
        pred = self.mlp(outputs)
        outputs = torch.cat([wild_type_output, mutant_output], dim=-1)
        pred = pred - self.mlp(outputs)
        
        return {
            'ddG': pred, 
            'wild_type_feature': wild_type_output, 
            'mutant_feature': mutant_output
        }


@R.register("models.GearBind")
class GearBind(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation, edge_input_dim=None, num_angle_bin=None,
                 short_cut=False, batch_norm=False, activation="relu", concat_hidden=False, readout="sum",
                 use_attn=True):
        super(GearBind, self).__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.edge_dims = [edge_input_dim] + self.dims[:-1]
        self.num_relation = num_relation
        self.num_angle_bin = num_angle_bin
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.batch_norm = batch_norm
        self.use_attn = use_attn

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.GeometricRelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                   None, batch_norm, activation))
        self.atom_position_gather = geometry.AtomPositionGather()
        if use_attn:
            self.attn_layers = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.attn_layers.append(layer.DDGAttention(self.dims[i+1], self.dims[i+1]))
        if num_angle_bin:
            self.spatial_line_graph = layers.SpatialLineGraph(num_angle_bin)
            self.edge_layers = nn.ModuleList()
            for i in range(len(self.edge_dims) - 1):
                self.edge_layers.append(layers.GeometricRelationalGraphConv(
                    self.edge_dims[i], self.edge_dims[i + 1], num_angle_bin, None, batch_norm, activation))

        if batch_norm:
            self.batch_norm_layers = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.batch_norm_layers.append(nn.BatchNorm1d(self.dims[i + 1]))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        residue_graph, node_mask = self.atom_position_gather(graph)
        pos_CA, _ = functional.variadic_to_padded(residue_graph.node_position, residue_graph.num_nodes, value=0)
        pos_CB = torch.where(
            residue_graph.atom_pos_mask[:, residue_graph.atom_name2id["CB"], None].expand(-1, 3),
            residue_graph.atom_pos[:, residue_graph.atom_name2id["CB"]],
            residue_graph.atom_pos[:, residue_graph.atom_name2id["CA"]]
        )
        pos_CB, _ = functional.variadic_to_padded(pos_CB, residue_graph.num_nodes, value=0)
        frame, _ = functional.variadic_to_padded(residue_graph.frame, residue_graph.num_nodes, value=0)
        
        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(graph)
            edge_input = line_graph.node_feature.float()

        hiddens = []
        layer_input = input
        for i in range(len(self.layers)):
            hidden = self.layers[i](graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            if self.num_angle_bin:
                edge_hidden = self.edge_layers[i](line_graph, edge_input)
                node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
                update = scatter_add(edge_hidden * graph.edge_weight.unsqueeze(-1), node_out, dim=0, dim_size=graph.num_node * self.num_relation) 
                update = update.view(graph.num_node, self.num_relation * edge_hidden.shape[1])
                update = self.layers[i].linear(update)
                update = self.layers[i].activation(update)
                hidden = hidden + update
                edge_input = edge_hidden
            if self.batch_norm:
                hidden = self.batch_norm_layers[i](hidden)

            if self.use_attn:
                x, mask = functional.variadic_to_padded(hidden[node_mask], residue_graph.num_nodes, value=0)
                residue_hidden = self.attn_layers[i](x, pos_CA, pos_CB, frame, mask.bool())
                residue_hidden = functional.padded_to_variadic(residue_hidden, residue_graph.num_nodes)
                hidden[node_mask] += residue_hidden

            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(residue_graph, node_feature[node_mask])

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }


@R.register("models.DDGPredictor")
class DDGPredictor(nn.Module, core.Configurable):

    def __init__(self, hidden_dim=512, num_layers=4, pair_dim=128, max_relpos=100, 
                num_neighbors=128, num_heads=12, activation=None):
        super(DDGPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.pair_dim = pair_dim
        self.max_relpos = max_relpos
        self.num_layers = num_layers
        self.output_dim = 2
        self.atom_position_gather = geometry.AtomPositionGather()
        self.mutation_site_graph = geometry.KNNMutationSite(k=num_neighbors)

        self.relpos_embedding = nn.Embedding(max_relpos*2+2, pair_dim)
        self.residue_encoder = layer.PerResidueEncoder(hidden_dim)

        self.blocks = nn.ModuleList([
            layer.GeometricAttention(hidden_dim, pair_dim, num_heads=num_heads, activation=activation) for _ in range(num_layers)
        ])

        # Readout
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.project = nn.Linear(hidden_dim, 1, bias=False)

    def encode(self, graph):
        chain_id = graph.entity_a.long() * 2 + graph.entity_b.long()
        chain_id, _ = functional.variadic_to_padded(chain_id, graph.num_nodes) # batch_size * num_nodes * 1
        same_chain = (chain_id.unsqueeze(1) == chain_id.unsqueeze(2))

        seqpos, _ = functional.variadic_to_padded(graph.residue_number, graph.num_nodes)   # batch_size * num_nodes * 1
        relpos = seqpos.unsqueeze(1) - seqpos.unsqueeze(2)
        relpos = relpos.clamp(min=-self.max_relpos, max=self.max_relpos) + self.max_relpos
        relpos = torch.where(same_chain, relpos, torch.full_like(relpos, fill_value=self.max_relpos*2+1))
        pair_input = self.relpos_embedding(relpos)

        # Residue encoder
        input = self.residue_encoder(graph)
        x, mask = functional.variadic_to_padded(input, graph.num_nodes)

        for block in self.blocks:
            x = block(graph, x, pair_input, mask.bool())
        return x, mask

    def forward(self, batch, all_loss=None, metric=None):
        wild_type, _ = self.atom_position_gather(batch["wild_type"])
        mutant, _ = self.atom_position_gather(batch["mutant"])
        assert (wild_type.num_nodes == mutant.num_nodes).all()
        assert (wild_type.num_nodes == wild_type.num_residues).all()

        is_valid = (wild_type.num_nodes > 0) & (scatter_add(wild_type.is_mutation.float(), wild_type.node2graph, dim=0, dim_size=wild_type.batch_size) > 0)
        wild_type = wild_type[is_valid]
        mutant = mutant[is_valid]
        node_mask = self.mutation_site_graph(wild_type)
        wild_type = wild_type.subresidue(node_mask)
        mutant = mutant.subresidue(node_mask)

        output_wt, mask  = self.encode(wild_type)
        output_mt, mask = self.encode(mutant)

        feat_wm = torch.cat([output_wt, output_mt], dim=-1)
        feat_mw = torch.cat([output_mt, output_wt], dim=-1)
        feat_diff = self.mlp(feat_wm) - self.mlp(feat_mw)       # (N, L, F)
        
        per_residue_ddg = (self.project(feat_diff) * mask.float().unsqueeze(-1)).squeeze(-1)   # (N, L)
        ddG = per_residue_ddg.sum(dim=1, keepdim=True)    # (N,)

        result = torch.zeros((batch["wild_type"].batch_size, 1), dtype=torch.float, device=wild_type.device)
        result[is_valid] = ddG
        feat_wt = output_wt.sum(dim=1)
        feat_mt = output_mt.sum(dim=1)
        _feat_wt = torch.zeros((batch["wild_type"].batch_size, feat_wt.shape[1]), dtype=torch.float, device=wild_type.device)
        _feat_mt = torch.zeros((batch["mutant"].batch_size, feat_wt.shape[1]), dtype=torch.float, device=mutant.device)
        _feat_wt[is_valid] = feat_wt
        _feat_mt[is_valid] = feat_mt

        return {
            'ddG': result,
            'wild_type_feature': _feat_wt,
            'mutant_feature': _feat_mt,
        }
