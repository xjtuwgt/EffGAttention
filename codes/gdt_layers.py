import torch
import math
import torch.nn as nn
from dgl.nn.pytorch.utils import Identity
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from torch.nn import LayerNorm as layerNorm
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from codes.gnn_utils import PositionwiseFeedForward, small_init_gain
from codes.gnn_utils import top_kp_attention, top_kp_attn_normalization, relu_edge_normalization


class GDTLayer(nn.Module):
    def __init__(self,
                 in_ent_feats: int,
                 out_ent_feats: int,
                 num_heads: int,
                 hop_num: int = 5,
                 alpha: float = 0.15,
                 top_k: int = 5,
                 top_p: float = 0.75,
                 sparse_mode: str = 'top_k',
                 feat_drop: float = 0.1,
                 attn_drop: float = 0.1,
                 negative_slope: float = 0.2,
                 residual=True,
                 ppr_diff=True):
        super(GDTLayer, self).__init__()

        self.sparse_mode = sparse_mode
        self._top_k, self._top_p = top_k, top_p
        assert self.sparse_mode in {'top_k', 'top_p', 'no_sparse'}

        self._hop_num = hop_num
        self._alpha = alpha
        self._num_heads = num_heads
        self._in_ent_feats = in_ent_feats
        self._in_head_feats, self._in_tail_feats = expand_as_pair(in_ent_feats)
        self._out_feats = out_ent_feats
        self._head_dim = out_ent_feats // num_heads
        self.fc_head = nn.Linear(self._in_head_feats, self._head_dim * self._num_heads, bias=False)
        self.fc_tail = nn.Linear(self._in_tail_feats, self._head_dim * self._num_heads, bias=False)
        self.fc_ent = nn.Linear(self._in_ent_feats, self._head_dim * self._num_heads, bias=False)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, self._num_heads, self._head_dim)))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_activation = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_tail_feats != self._out_feats:
                self.res_fc = nn.Linear(self._in_tail_feats, self._out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

        self.graph_layer_norm = layerNorm(self._in_ent_feats)
        self.ff_layer_norm = layerNorm(self._out_feats)
        self.feed_forward_layer = PositionwiseFeedForward(model_dim=self._out_feats, d_hidden=4 * self._out_feats)
        self.ppr_diff = ppr_diff
        self.reset_parameters()

    def reset_parameters(self):
        gain = 1. / math.sqrt(self._out_feats)
        # gain = small_init_gain(d_in=self._in_ent_feats, d_out=self._out_feats)
        nn.init.xavier_normal_(self.fc_head.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_tail.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_ent.weight, gain=gain)
        nn.init.xavier_normal_(self.attn, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')
            in_head = in_dst = self.feat_drop(self.graph_layer_norm(feat))
            feat_head = self.fc_head(in_head).view(-1, self._num_heads, self._head_dim)
            feat_tail = self.fc_tail(in_dst).view(-1, self._num_heads, self._head_dim)
            feat_enti = self.fc_ent(in_head).view(-1, self._num_heads, self._head_dim)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            graph.srcdata.update({'eh': feat_head, 'ft': feat_enti})  # (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'et': feat_tail})
            graph.apply_edges(fn.u_mul_v('eh', 'et', 'e'))
            e = self.attn_activation(graph.edata.pop('e'))  # (num_src_edge, num_heads, out_dim)
            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)  # (num_edge, num_heads, 1)
            graph.edata.update({'e': e})
            graph.apply_edges(fn.e_mul_v('e', 'log_in', 'e'))
            e = (graph.edata.pop('e')/math.sqrt(self._head_dim))
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if self.sparse_mode != 'no_sparse':
                a_score = edge_softmax(graph, e)
                a_mask, a_top_sum = top_kp_attention(graph=graph, attn_scores=a_score, k=self._top_k, p=self._top_p,
                                                     sparse_mode=self.sparse_mode)
                a_n = top_kp_attn_normalization(graph=graph, attn_scores=a_score.clone(), attn_mask=a_mask,
                                                top_k_sum=a_top_sum)
                if self.ppr_diff:
                    graph.edata['a'] = a_n
                    rst = self.ppr_estimation(graph=graph)
                else:
                    graph.edata['a'] = self.attn_drop(a_n)
                    graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
                    rst = graph.dstdata.pop('ft')
            else:
                # compute softmax
                if self.ppr_diff:
                    # graph.edata['a'] = edge_softmax(graph, e)
                    graph.edata['a'] = relu_edge_normalization(graph, e)
                    rst = self.ppr_estimation(graph=graph)
                else:
                    graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))  # (num_edge, num_heads)
                    # # message passing
                    graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
                    rst = graph.dstdata.pop('ft')

            # residual
            if self.res_fc is not None:
                # this part uses feat (very important to prevent over-smoothing)
                resval = self.res_fc(feat).view(feat.shape[0], -1, self._head_dim)
                rst = self.feat_drop(rst) + resval

            rst = rst.flatten(1)
            ff_rst = self.feed_forward_layer(self.feat_drop(self.ff_layer_norm(rst)))
            rst = self.feat_drop(ff_rst) + rst  # residual

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst

    def ppr_estimation(self, graph):
        with graph.local_scope():
            graph = graph.local_var()
            feat_0 = graph.srcdata.pop('ft')
            feat = feat_0.clone()
            attentions = graph.edata.pop('a')
            for _ in range(self._hop_num):
                graph.srcdata['h'] = self.feat_drop(feat)
                graph.edata['a_temp'] = self.attn_drop(attentions)
                graph.update_all(fn.u_mul_e('h', 'a_temp', 'm'), fn.sum('m', 'h'))
                feat = graph.dstdata.pop('h')
                feat = (1.0 - self._alpha) * self.feat_drop(feat) + self._alpha * feat_0
            return feat
