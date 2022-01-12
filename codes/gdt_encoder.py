from codes.gdt_layers import GDTLayer, RGDTLayer
from torch import nn
from torch import Tensor
from codes.gnn_utils import EmbeddingLayer
import torch


class GDTEncoder(nn.Module):
    def __init__(self, config):
        super(GDTEncoder, self).__init__()
        self.config = config
        self.graph_encoder = nn.ModuleList()
        self.graph_encoder.append(module=GDTLayer(in_ent_feats=self.config.node_emb_dim,
                                                  out_ent_feats=self.config.hidden_dim,
                                                  num_heads=self.config.head_num,
                                                  hop_num=self.config.gnn_hop_num,
                                                  alpha=self.config.alpha,
                                                  layer_num=self.config.layers,
                                                  feat_drop=self.config.feat_drop,
                                                  attn_drop=self.config.attn_drop,
                                                  edge_drop=self.config.edge_drop,
                                                  negative_slope=self.config.negative_slope,
                                                  degree_norm=self.config.degree_norm,
                                                  residual=self.config.residual,
                                                  ppr_diff=self.config.ppr_diff))

        for _ in range(1, self.config.layers):
            self.graph_encoder.append(module=GDTLayer(in_ent_feats=self.config.hidden_dim,
                                                      out_ent_feats=self.config.hidden_dim,
                                                      num_heads=self.config.head_num,
                                                      hop_num=self.config.gnn_hop_num,
                                                      alpha=self.config.alpha,
                                                      edge_drop=self.config.edge_drop,
                                                      layer_num=self.config.layers,
                                                      feat_drop=self.config.feat_drop,
                                                      attn_drop=self.config.attn_drop,
                                                      degree_norm=self.config.degree_norm,
                                                      negative_slope=self.config.negative_slope,
                                                      residual=self.config.residual,
                                                      ppr_diff=self.config.ppr_diff))
        self.classifier = nn.Linear(in_features=self.config.hidden_dim, out_features=self.config.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.classifier.weight, gain=gain)

    def forward(self, graph, inputs: Tensor):
        h = inputs
        for _ in range(self.config.layers):
            h = self.graph_encoder[_](graph, h)
        logits = self.classifier(h)
        return logits


class RGDTEncoder(nn.Module):
    def __init__(self, config):
        super(RGDTEncoder, self).__init__()
        self.config = config
        if self.config.proj_emb_dim > 0:
            self.rel_ember = EmbeddingLayer(num=self.config.num_relations, dim=self.config.rel_emb_dim,
                                            project_dim=self.config.proj_emb_dim)
            self.ent_ember = EmbeddingLayer(num=self.config.num_entities, dim=self.config.node_emb_dim,
                                            project_dim=self.config.proj_emb_dim)
            ent_in_dim = rel_in_dim = self.config.proj_emb_dim
        else:
            self.rel_ember = EmbeddingLayer(num=self.config.num_relations, dim=self.config.rel_emb_dim)
            self.ent_ember = EmbeddingLayer(num=self.config.num_entities, dim=self.config.node_emb_dim)
            ent_in_dim = self.config.node_emb_dim
            rel_in_dim = self.config.rel_emb_dim
        self.graph_encoder = nn.ModuleList()
        self.graph_encoder.append(module=RGDTLayer(in_ent_feats=ent_in_dim,
                                                   out_ent_feats=self.config.hidden_dim,
                                                   in_rel_feats=rel_in_dim,
                                                   num_heads=self.config.head_num,
                                                   hop_num=self.config.gnn_hop_num,
                                                   alpha=self.config.alpha,
                                                   layer_num=self.config.layers,
                                                   feat_drop=self.config.feat_drop,
                                                   attn_drop=self.config.attn_drop,
                                                   edge_drop=self.config.edge_drop,
                                                   negative_slope=self.config.negative_slope,
                                                   residual=self.config.residual,
                                                   ppr_diff=self.config.ppr_diff))

        for _ in range(1, self.config.layers):
            self.graph_encoder.append(module=GDTLayer(in_ent_feats=self.config.hidden_dim,
                                                      out_ent_feats=self.config.hidden_dim,
                                                      num_heads=self.config.head_num,
                                                      hop_num=self.config.gnn_hop_num,
                                                      alpha=self.config.alpha,
                                                      layer_num=self.config.layers,
                                                      feat_drop=self.config.feat_drop,
                                                      attn_drop=self.config.attn_drop,
                                                      edge_drop=self.config.edge_drop,
                                                      negative_slope=self.config.negative_slope,
                                                      degree_norm=self.config.degree_norm,
                                                      residual=self.config.residual,
                                                      ppr_diff=self.config.ppr_diff))
        self.drop_out = nn.Dropout(self.config.out_drop)
        self.classifier = nn.Linear(in_features=self.config.hidden_dim, out_features=self.config.num_classes)
        self.reset_parameters()
        self.dummy_param = nn.Parameter(torch.empty(0))

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.classifier.weight, gain=gain)

    def init_graph_ember(self, ent_emb: Tensor = None, rel_emb: Tensor = None, rel_freeze=False, ent_freeze=False):
        if rel_emb is not None:
            self.rel_ember.init_with_tensor(data=rel_emb, freeze=rel_freeze)
        if ent_emb is not None:
            self.ent_ember.init_with_tensor(data=ent_emb, freeze=ent_freeze)

    def forward(self, graph):
        assert graph.number_of_nodes() <= self.ent_ember.num
        e_h = self.ent_ember(torch.arange(graph.number_of_nodes()).to(self.dummy_param.device))
        r_h = self.rel_ember(torch.arange(self.config.num_relations).to(self.dummy_param.device))
        h = self.graph_encoder[0](graph, e_h, r_h)
        for _ in range(1, self.config.layers):
            h = self.graph_encoder[_](graph, h)
        logits = self.classifier(self.drop_out(h))
        return logits
