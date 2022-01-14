import torch
import math
import torch.nn as nn
from dgl.nn.pytorch.utils import Identity
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from torch.nn import LayerNorm as layerNorm
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from codes.gnn_utils import PositionWiseFeedForward, small_init_gain, small_init_gain_v2
from torch import Tensor


class GDTLayer(nn.Module):
    def __init__(self,
                 in_ent_feats: int,
                 out_ent_feats: int,
                 num_heads: int,
                 hop_num: int = 5,
                 alpha: float = 0.1,
                 feat_drop: float = 0.1,
                 attn_drop: float = 0.1,
                 edge_drop: float = 0.1,
                 layer_num: int = 1,
                 residual: bool = True,
                 ppr_diff: bool = True):
        super(GDTLayer, self).__init__()

        self.layer_num = layer_num
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

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        if residual:
            if self._in_tail_feats != self._out_feats:
                self.res_fc = nn.Linear(self._in_tail_feats, self._out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

        self.graph_layer_norm = layerNorm(self._in_ent_feats)
        self.ff_layer_norm = layerNorm(self._out_feats)
        self.feed_forward_layer = PositionWiseFeedForward(model_dim=self._out_feats, d_hidden=4 * self._out_feats)
        self.ppr_diff = ppr_diff
        self.reset_parameters()

    def reset_parameters(self):
        gain = small_init_gain(d_in=self._in_ent_feats, d_out=self._out_feats) / math.sqrt(self.layer_num)
        nn.init.xavier_normal_(self.fc_head.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_tail.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_ent.weight, gain=gain)
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
            in_feat_norm = self.graph_layer_norm(feat)
            feat_head = self.fc_head(self.feat_drop(in_feat_norm)).view(-1, self._num_heads, self._head_dim)
            feat_tail = self.fc_tail(self.feat_drop(in_feat_norm)).view(-1, self._num_heads, self._head_dim)
            feat_enti = self.fc_ent(self.feat_drop(in_feat_norm)).view(-1, self._num_heads, self._head_dim)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            graph.srcdata.update({'eh': feat_head, 'ft': feat_enti})  # (num_src_edge, num_heads, head_dim)
            graph.dstdata.update({'et': feat_tail})
            graph.apply_edges(fn.u_dot_v('eh', 'et', 'e'))
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            graph.apply_edges(fn.e_mul_v('e', 'log_in', 'e'))
            e = (graph.edata.pop('e')/math.sqrt(self._head_dim))
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                a_value = torch.zeros_like(e)
                a_value[eids] = edge_softmax(graph, e[eids], eids=eids)
            else:
                a_value = edge_softmax(graph, e)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # compute softmax
            if self.ppr_diff:
                graph.edata['a'] = a_value
                rst = self.ppr_estimation(graph=graph)
            else:
                graph.edata['a'] = self.attn_drop(a_value)
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
                feat = (1.0 - self._alpha) * feat + self._alpha * feat_0
            return feat


class RGDTLayer(nn.Module):
    """
    Heterogeneous graph neural network (first layer) with different edge type
    """
    def __init__(self,
                 in_ent_feats: int,
                 in_rel_feats: int,
                 out_ent_feats: int,
                 num_heads: int,
                 hop_num: int,
                 alpha: float = 0.1,
                 feat_drop: float = 0.1,
                 attn_drop: float = 0.1,
                 edge_drop: float = 0.1,
                 layer_num: int = 1,
                 residual=True,
                 ppr_diff=True):
        super(RGDTLayer, self).__init__()

        self.layer_num = layer_num
        self._in_ent_feats = in_ent_feats
        self._in_head_feats, self._in_tail_feats = expand_as_pair(in_ent_feats)
        self._out_ent_feats = out_ent_feats
        self._in_rel_feats = in_rel_feats
        self._num_heads = num_heads
        self._hop_num = hop_num
        self._alpha = alpha

        assert self._out_ent_feats % self._num_heads == 0
        self._head_dim = self._out_ent_feats // self._num_heads

        self.fc_head = nn.Linear(self._in_head_feats, self._head_dim * self._num_heads, bias=False)
        self.fc_tail = nn.Linear(self._in_tail_feats, self._head_dim * self._num_heads, bias=False)
        self.fc_ent = nn.Linear(self._in_ent_feats, self._head_dim * self._num_heads, bias=False)
        self.fc_rel = nn.Linear(self._in_rel_feats, self._head_dim * self._num_heads, bias=False)
        self.attn_r = nn.Parameter(torch.FloatTensor(1, self._num_heads, self._head_dim), requires_grad=True)

        if residual:
            if in_ent_feats != out_ent_feats:
                self.res_fc = nn.Linear(in_ent_feats, self._num_heads * self._head_dim, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc_ent', None)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop

        self.graph_layer_ent_norm = layerNorm(self._in_ent_feats)
        self.graph_layer_rel_norm = layerNorm(self._in_rel_feats)
        self.ff_layer_norm = layerNorm(self._out_ent_feats)
        self.feed_forward_layer = PositionWiseFeedForward(model_dim=self._num_heads * self._head_dim,
                                                          d_hidden=4 * self._num_heads * self._head_dim)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.ppr_diff = ppr_diff
        self.reset_parameters()

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = small_init_gain(d_in=self._in_ent_feats, d_out=self._out_ent_feats) / math.sqrt(self.layer_num)
        nn.init.xavier_normal_(self.fc_head.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_tail.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_ent.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_rel.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, ent_feat: Tensor, rel_feat: Tensor, get_attention=False):
        with graph.local_scope():
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               ' Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')
            in_feat_norm = self.graph_layer_ent_norm(ent_feat)
            feat_head = self.fc_head(self.feat_drop(in_feat_norm)).view(-1, self._num_heads, self._head_dim)
            feat_tail = self.fc_tail(self.feat_drop(in_feat_norm)).view(-1, self._num_heads, self._head_dim)
            feat_enti = self.fc_ent(self.feat_drop(in_feat_norm)).view(-1, self._num_heads, self._head_dim)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            graph.srcdata.update({'eh': feat_head, 'ft': feat_enti})  # (num_src_edge, num_heads, head_dim)
            graph.dstdata.update({'et': feat_tail})
            graph.apply_edges(fn.u_dot_v('eh', 'et', 'e'))
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            feat_rel_norm = self.graph_layer_rel_norm(rel_feat)
            feat_rel = self.fc_rel(self.feat_drop(feat_rel_norm)).view(-1, self._num_heads, self._head_dim)
            er = (feat_rel * self.attn_r).sum(dim=-1).unsqueeze(-1)
            er = er[graph.edata['rid']]
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            e = graph.edata.pop('e') * er
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            graph.edata.update({'e': e})
            graph.apply_edges(fn.e_mul_v('e', 'log_in', 'e'))
            e = (graph.edata.pop('e')/math.sqrt(self._head_dim))
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                a_value = torch.zeros_like(e)
                a_value[eids] = edge_softmax(graph, e[eids], eids=eids)
            else:
                a_value = edge_softmax(graph, e)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if self.ppr_diff:
                graph.edata['a'] = a_value
                rst = self.ppr_estimation(graph=graph)
            else:
                graph.edata['a'] = self.attn_drop(a_value)
                graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
                rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(ent_feat).view(ent_feat.shape[0], -1, self._head_dim)
                rst = self.feat_drop(rst) + resval

            rst = rst.flatten(1)
            # +++++++++++++++++++++++++++++++++++++++
            ff_rst = self.feed_forward_layer(self.feat_drop(self.ff_layer_norm(rst)))
            rst = self.feat_drop(ff_rst) + rst  # residual
            # +++++++++++++++++++++++++++++++++++++++
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
                feat = (1.0 - self._alpha) * feat + self._alpha * feat_0
            return feat
