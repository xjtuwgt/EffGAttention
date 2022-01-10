import torch
from codes.gdt_layers import GDTLayer
from codes.gated_gdt_layer import GatedGDTLayer
from torch import nn
from torch import Tensor
from dgl.nn.pytorch.utils import Identity
from codes.gnn_utils import EmbeddingLayer, small_init_gain_v2
from torch.nn import LayerNorm


class GDTEncoder(nn.Module):
    def __init__(self, config):
        super(GDTEncoder, self).__init__()
        self.config = config
        if self.config.central_emb:
            self.central_emb_layer = EmbeddingLayer(num=self.config.max_degree, dim=self.config.degree_emb_dim)
            self.feature_map = nn.Linear(in_features=self.config.node_emb_dim, out_features=self.config.degree_emb_dim)
            input_dim = self.config.degree_emb_dim
        else:
            input_dim = self.config.node_emb_dim
            self.central_emb_layer = None
            self.feature_map = None

        self.graph_encoder = nn.ModuleList()
        self.graph_encoder.append(module=GDTLayer(in_ent_feats=input_dim,
                                                  out_ent_feats=self.config.hidden_dim,
                                                  num_heads=self.config.head_num,
                                                  hop_num=self.config.gnn_hop_num,
                                                  alpha=self.config.alpha,
                                                  top_k=self.config.top_k,
                                                  top_p=self.config.top_p,
                                                  sparse_mode=self.config.sparse_mode,
                                                  layer_num=self.config.layers,
                                                  feat_drop=self.config.feat_drop,
                                                  attn_drop=self.config.attn_drop,
                                                  residual=self.config.residual,
                                                  ppr_diff=self.config.ppr_diff))

        for _ in range(1, self.config.layers):
            self.graph_encoder.append(module=GDTLayer(in_ent_feats=self.config.hidden_dim,
                                                      out_ent_feats=self.config.hidden_dim,
                                                      num_heads=self.config.head_num,
                                                      hop_num=self.config.gnn_hop_num,
                                                      alpha=self.config.alpha,
                                                      top_k=self.config.top_k,
                                                      top_p=self.config.top_p,
                                                      sparse_mode=self.config.sparse_mode,
                                                      layer_num=self.config.layers,
                                                      feat_drop=self.config.feat_drop,
                                                      attn_drop=self.config.attn_drop,
                                                      residual=self.config.residual,
                                                      ppr_diff=self.config.ppr_diff))
        if self.config.layers >= 6:
            self.layer_norm = LayerNorm(self.config.hidden_dim)
        else:
            self.layer_norm = Identity()
        self.classifier = nn.Linear(in_features=self.config.hidden_dim, out_features=self.config.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.classifier.weight, gain=gain)
        if self.feature_map:
            nn.init.xavier_normal_(self.feature_map.weight, gain=small_init_gain_v2(d_in=self.config.node_emb_dim,
                                                                                    d_out=self.config.degree_emb_dim))

    def forward(self, graph, inputs: Tensor):
        if self.central_emb_layer:
            h = self.feature_map(inputs) + self.central_emb_layer(graph.in_degrees().long())
        else:
            h = inputs
        for l in range(self.config.layers):
            h = self.graph_encoder[l](graph, h)
        logits = self.classifier(self.layer_norm(h))
        return logits


class GatedGDTEncoder(nn.Module):
    def __init__(self, config):
        super(GatedGDTEncoder, self).__init__()
        self.config = config
        if self.config.central_emb:
            self.central_emb_layer = EmbeddingLayer(num=self.config.max_degree, dim=self.config.degree_emb_dim)
            self.feature_map = nn.Linear(in_features=self.config.node_emb_dim, out_features=self.config.degree_emb_dim)
            input_dim = self.config.degree_emb_dim
        else:
            input_dim = self.config.node_emb_dim
            self.central_emb_layer = None
            self.feature_map = None

        self.graph_encoder = nn.ModuleList()
        self.graph_encoder.append(module=GatedGDTLayer(in_ent_feats=input_dim,
                                                       out_ent_feats=self.config.hidden_dim,
                                                       num_heads=self.config.head_num,
                                                       hop_num=self.config.gnn_hop_num,
                                                       alpha=self.config.alpha,
                                                       layer_num=self.config.layers,
                                                       feat_drop=self.config.feat_drop,
                                                       attn_drop=self.config.attn_drop,
                                                       residual=self.config.residual,
                                                       ppr_diff=self.config.ppr_diff))

        for _ in range(1, self.config.layers):
            self.graph_encoder.append(module=GatedGDTLayer(in_ent_feats=self.config.hidden_dim,
                                                           out_ent_feats=self.config.hidden_dim,
                                                           num_heads=self.config.head_num,
                                                           hop_num=self.config.gnn_hop_num,
                                                           alpha=self.config.alpha,
                                                           layer_num=self.config.layers,
                                                           feat_drop=self.config.feat_drop,
                                                           attn_drop=self.config.attn_drop,
                                                           residual=self.config.residual,
                                                           ppr_diff=self.config.ppr_diff))
        self.layer_norm = LayerNorm(self.config.hidden_dim)
        self.classifier = nn.Linear(in_features=self.config.hidden_dim, out_features=self.config.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.classifier.weight, gain=gain)
        if self.feature_map:
            nn.init.xavier_normal_(self.feature_map.weight, gain=small_init_gain_v2(d_in=self.config.node_emb_dim,
                                                                                    d_out=self.config.degree_emb_dim))

    def forward(self, graph, inputs: Tensor):
        if self.central_emb_layer:
            h = self.feature_map(inputs) + self.central_emb_layer(graph.in_degrees().long())
        else:
            h = inputs
        for l in range(self.config.layers):
            h = self.graph_encoder[l](graph, h)
        logits = self.classifier(self.layer_norm(h))
        return logits

