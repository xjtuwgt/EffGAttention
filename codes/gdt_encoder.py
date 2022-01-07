from codes.gdt_layers import GDTLayer
from torch import nn
from torch import Tensor


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
                                                  top_k=self.config.top_k,
                                                  top_p=self.config.top_p,
                                                  sparse_mode=self.config.sparse_mode,
                                                  in_feat_drop=self.config.in_feat_drop,
                                                  feat_drop=self.config.feat_drop,
                                                  attn_drop=self.config.attn_drop,
                                                  residual=self.config.residual,
                                                  ppr_diff=self.config.ppr_diff,
                                                  layer_num=0))

        for _ in range(1, self.config.layers):
            self.graph_encoder.append(module=GDTLayer(in_ent_feats=self.config.hidden_dim,
                                                      out_ent_feats=self.config.hidden_dim,
                                                      num_heads=self.config.head_num,
                                                      hop_num=self.config.gnn_hop_num,
                                                      alpha=self.config.alpha,
                                                      top_k=self.config.top_k,
                                                      top_p=self.config.top_p,
                                                      sparse_mode=self.config.sparse_mode,
                                                      in_feat_drop=self.config.in_feat_drop,
                                                      feat_drop=self.config.feat_drop,
                                                      attn_drop=self.config.attn_drop,
                                                      residual=self.config.residual,
                                                      ppr_diff=self.config.ppr_diff,
                                                      layer_num=_))

        self.classifier = nn.Linear(in_features=self.config.hidden_dim, out_features=self.config.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.classifier.weight, gain=gain)

    def forward(self, graph, inputs: Tensor):
        h = inputs
        for l in range(self.config.layers):
            h = self.graph_encoder[l](graph, h)
        logits = self.classifier(h)
        return logits
