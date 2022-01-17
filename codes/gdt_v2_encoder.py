from codes.gdt_v2_layers import GDTLayer, RGDTLayer
from torch import nn
from torch import Tensor
from codes.gnn_utils import EmbeddingLayer
import torch.nn.functional as F
from torch.nn import BatchNorm1d
import torch


class GDTEncoder(nn.Module):
    def __init__(self, config):
        super(GDTEncoder, self).__init__()
        self.config = config
        self.graph_encoder = nn.ModuleList()
        self.graph_encoder.append(module=GDTLayer(in_ent_feats=self.config.node_emb_dim,
                                                  out_ent_feats=self.config.hidden_dim,
                                                  num_heads=self.config.head_num,
                                                  layer_idx=1,
                                                  hop_num=self.config.gnn_hop_num,
                                                  alpha=self.config.alpha,
                                                  concat=self.config.concat,
                                                  layer_num=self.config.layers,
                                                  feat_drop=self.config.feat_drop,
                                                  attn_drop=self.config.attn_drop,
                                                  edge_drop=self.config.edge_drop,
                                                  residual=self.config.residual,
                                                  ppr_diff=self.config.ppr_diff))
        hidden_dim = 4 * self.config.hidden_dim if self.config.concat else self.config.hidden_dim
        for _ in range(1, self.config.layers):
            self.graph_encoder.append(module=GDTLayer(in_ent_feats=hidden_dim,
                                                      out_ent_feats=self.config.hidden_dim,
                                                      num_heads=self.config.head_num,
                                                      layer_idx=_ + 1,
                                                      hop_num=self.config.gnn_hop_num,
                                                      alpha=self.config.alpha,
                                                      edge_drop=self.config.edge_drop,
                                                      layer_num=self.config.layers,
                                                      feat_drop=self.config.feat_drop,
                                                      attn_drop=self.config.attn_drop,
                                                      concat=self.config.concat,
                                                      residual=self.config.residual,
                                                      rescale_res=self.config.rescale_res,
                                                      ppr_diff=self.config.ppr_diff))
        self.classifier = nn.Linear(in_features=hidden_dim, out_features=self.config.num_classes)
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
        self.ent_ember = EmbeddingLayer(num=self.config.num_entities, dim=self.config.node_emb_dim)
        if self.config.node_emb_dim == self.config.rel_emb_dim:
            self.rel_ember = EmbeddingLayer(num=self.config.num_relations, dim=self.config.node_emb_dim)
            rel_in_dim = self.config.node_emb_dim
        else:
            if self.config.proj_emb_dim == -1:
                self.rel_ember = EmbeddingLayer(num=self.config.num_relations, dim=self.config.rel_emb_dim)
                rel_in_dim = self.config.rel_emb_dim
            else:
                self.rel_ember = EmbeddingLayer(num=self.config.num_relations, dim=self.config.rel_emb_dim,
                                                project_dim=self.config.proj_emb_dim)
                rel_in_dim = self.config.rel_emb_dim

        ent_in_dim = self.config.node_emb_dim
        self.graph_encoder = nn.ModuleList()
        self.graph_encoder.append(module=RGDTLayer(in_ent_feats=ent_in_dim,
                                                   out_ent_feats=self.config.hidden_dim,
                                                   in_rel_feats=rel_in_dim,
                                                   layer_idx=1,
                                                   num_heads=self.config.head_num,
                                                   hop_num=self.config.gnn_hop_num,
                                                   alpha=self.config.alpha,
                                                   layer_num=self.config.layers,
                                                   feat_drop=self.config.feat_drop,
                                                   attn_drop=self.config.attn_drop,
                                                   edge_drop=self.config.edge_drop,
                                                   concat=self.config.concat,
                                                   residual=self.config.residual,
                                                   ppr_diff=self.config.ppr_diff))
        hidden_dim = 4 * self.config.hidden_dim if self.config.concat else self.config.hidden_dim
        for _ in range(1, self.config.layers):
            self.graph_encoder.append(module=GDTLayer(in_ent_feats=hidden_dim,
                                                      layer_idx=_+1,
                                                      out_ent_feats=self.config.hidden_dim,
                                                      num_heads=self.config.head_num,
                                                      hop_num=self.config.gnn_hop_num,
                                                      alpha=self.config.alpha,
                                                      layer_num=self.config.layers,
                                                      feat_drop=self.config.feat_drop,
                                                      attn_drop=self.config.attn_drop,
                                                      edge_drop=self.config.edge_drop,
                                                      concat=self.config.concat,
                                                      residual=self.config.residual,
                                                      rescale_res=self.config.rescale_res,
                                                      ppr_diff=self.config.ppr_diff))
        self.norm_layer = BatchNorm1d(num_features=hidden_dim)
        self.drop_out = nn.Dropout(0.1)
        self.classifier = nn.Linear(in_features=hidden_dim, out_features=self.config.num_classes)
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
        logits = self.classifier(self.drop_out(F.relu(self.norm_layer(h))))
        return logits
