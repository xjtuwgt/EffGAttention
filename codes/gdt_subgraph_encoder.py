from codes.gdt_layers import GDTLayer, RGDTLayer
from torch import nn
from torch import Tensor
from codes.gnn_utils import EmbeddingLayer
import torch


class GDTEncoder(nn.Module):
    def __init__(self, config):
        super(GDTEncoder, self).__init__()
        self.config = config
        self.ent_ember = EmbeddingLayer(num=self.config.num_entities, dim=self.config.node_emb_dim)
        if self.config.arw_position:
            position_num = self.config.sub_graph_hop_num + 2
            if self.config.node_emb_dim == self.config.arw_pos_emb_dim:
                self.position_embed_layer = EmbeddingLayer(num=position_num,
                                                           dim=self.config.arw_pos_emb_dim)
            else:
                self.position_embed_layer = EmbeddingLayer(num=position_num,
                                                           dim=self.config.arw_pos_emb_dim,
                                                           project_dim=self.config.node_emb_dim)
        self.graph_encoder = nn.ModuleList()
        self.graph_encoder.append(module=GDTLayer(in_ent_feats=self.config.node_emb_dim,
                                                  out_ent_feats=self.config.hidden_dim,
                                                  num_heads=self.config.head_num,
                                                  hop_num=self.config.gnn_hop_num,
                                                  alpha=self.config.alpha,
                                                  layer_idx=1,
                                                  layer_num=self.config.layers,
                                                  feat_drop=self.config.feat_drop,
                                                  attn_drop=self.config.attn_drop,
                                                  edge_drop=self.config.edge_drop,
                                                  residual=self.config.residual,
                                                  rescale_res=self.config.rescale_res,
                                                  ppr_diff=self.config.ppr_diff))
        for _ in range(1, self.config.layers):
            self.graph_encoder.append(module=GDTLayer(in_ent_feats=self.config.hidden_dim,
                                                      out_ent_feats=self.config.hidden_dim,
                                                      num_heads=self.config.head_num,
                                                      hop_num=self.config.gnn_hop_num,
                                                      alpha=self.config.alpha,
                                                      edge_drop=self.config.edge_drop,
                                                      layer_idx=_ + 1,
                                                      layer_num=self.config.layers,
                                                      feat_drop=self.config.feat_drop,
                                                      attn_drop=self.config.attn_drop,
                                                      rescale_res=self.config.rescale_res,
                                                      residual=self.config.residual,
                                                      ppr_diff=self.config.ppr_diff))

    def init_graph_ember(self, ent_emb: Tensor = None, ent_freeze=False):
        if ent_emb is not None:
            self.ent_ember.init_with_tensor(data=ent_emb, freeze=ent_freeze)

    def forward(self, batch_g_pair, cls_or_anchor: str = 'cls'):
        batch_g = batch_g_pair[0]
        ent_ids = batch_g.ndata['nid']
        ent_features = self.ent_ember(ent_ids)
        if self.config.arw_position:
            arw_positions = batch_g.ndata['n_rw_label']
            arw_pos_embed = self.position_embed_layer(arw_positions)
            ent_features = ent_features + arw_pos_embed
        with batch_g.local_scope():
            h = ent_features
            for _ in range(self.config.layers):
                h = self.graph_encoder[_](batch_g, h)
            if cls_or_anchor == 'cls':
                batch_node_ids = batch_g_pair[1]
            elif cls_or_anchor == 'anchor':
                batch_node_ids = batch_g_pair[2]
            else:
                raise '{} is not supported'.format(cls_or_anchor)
            batch_graph_embed = h[batch_node_ids]
            return batch_graph_embed


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

        if self.config.arw_position:
            position_num = self.config.sub_graph_hop_num + 2
            if self.config.node_emb_dim == self.config.arw_pos_emb_dim:
                self.position_embed_layer = EmbeddingLayer(num=position_num,
                                                           dim=self.config.arw_pos_emb_dim)
            else:
                self.position_embed_layer = EmbeddingLayer(num=position_num,
                                                           dim=self.config.arw_pos_emb_dim,
                                                           project_dim=self.config.node_emb_dim)

        ent_in_dim = self.config.node_emb_dim
        self.graph_encoder = nn.ModuleList()
        self.graph_encoder.append(module=RGDTLayer(in_ent_feats=ent_in_dim,
                                                   out_ent_feats=self.config.hidden_dim,
                                                   in_rel_feats=rel_in_dim,
                                                   num_heads=self.config.head_num,
                                                   hop_num=self.config.gnn_hop_num,
                                                   alpha=self.config.alpha,
                                                   layer_idx=1,
                                                   layer_num=self.config.layers,
                                                   feat_drop=self.config.feat_drop,
                                                   attn_drop=self.config.attn_drop,
                                                   edge_drop=self.config.edge_drop,
                                                   rescale_res=self.config.rescale_res,
                                                   residual=self.config.residual,
                                                   ppr_diff=self.config.ppr_diff))
        for _ in range(1, self.config.layers):
            self.graph_encoder.append(module=GDTLayer(in_ent_feats=self.config.hidden_dim,
                                                      out_ent_feats=self.config.hidden_dim,
                                                      num_heads=self.config.head_num,
                                                      hop_num=self.config.gnn_hop_num,
                                                      alpha=self.config.alpha,
                                                      layer_idx=_ + 1,
                                                      layer_num=self.config.layers,
                                                      feat_drop=self.config.feat_drop,
                                                      attn_drop=self.config.attn_drop,
                                                      edge_drop=self.config.edge_drop,
                                                      rescale_res=self.config.rescale_res,
                                                      residual=self.config.residual,
                                                      ppr_diff=self.config.ppr_diff))
        self.dummy_param = nn.Parameter(torch.empty(0))

    def init_graph_ember(self, ent_emb: Tensor = None, rel_emb: Tensor = None, rel_freeze=False, ent_freeze=False):
        if rel_emb is not None:
            self.rel_ember.init_with_tensor(data=rel_emb, freeze=rel_freeze)
        if ent_emb is not None:
            self.ent_ember.init_with_tensor(data=ent_emb, freeze=ent_freeze)

    def forward(self, batch_g_pair, cls_or_anchor: str = 'cls'):
        batch_g = batch_g_pair[0]
        ent_ids = batch_g.ndata['nid']
        rel_ids = batch_g.edata['rid']
        ent_features = self.ent_ember(ent_ids)
        rel_features = self.rel_ember(rel_ids)
        if self.config.arw_position:
            arw_positions = batch_g.ndata['n_rw_label']
            arw_pos_embed = self.position_embed_layer(arw_positions)
            ent_features = ent_features + arw_pos_embed
        with batch_g.local_scope():
            h = ent_features
            for _ in range(self.config.layers):
                if _ == 0:
                    h = self.graph_encoder[_](batch_g, h, rel_features)
                else:
                    h = self.graph_encoder[_](batch_g, h)
            if cls_or_anchor == 'cls':
                batch_node_ids = batch_g_pair[1]
            elif cls_or_anchor == 'anchor':
                batch_node_ids = batch_g_pair[2]
            else:
                raise '{} is not supported'.format(cls_or_anchor)
            batch_graph_embed = h[batch_node_ids]
            return batch_graph_embed
