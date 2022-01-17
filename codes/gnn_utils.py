import math

from dgl import DGLHeteroGraph
import torch
import dgl.function as fn
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch.utils import Identity
from torch.nn import BatchNorm1d
from torch import Tensor, LongTensor

"""
    If the number of neighbors is large, then we need to select a subset of neighbors for feature aggregation
    codes for mask testing
    def edge_mask_message_func(edges):
        return {'m_a_mask': edges.graph_data['attn_mask']}

    def mask_top_k_reduce_func(nodes):
        edge_attention_mask = nodes.mailbox['m_a_mask']
        res = edge_attention_mask.sum(1)
        print(torch.min(res.squeeze(-1)))
        # print(res.shape)
        return {'k_num': res}

    graph.update_all(edge_mask_message_func, mask_top_k_reduce_func)
"""


def edge_udf_attn_mask_func(edges):
    attention_mask = edges.data['ta'] >= edges.dst['top_a']
    return {'attn_mask': attention_mask}


def relu_edge_normalization(graph: DGLHeteroGraph, attn_scores: Tensor):
    with graph.local_scope():
        graph.edata['ta'] = F.relu(attn_scores)
        graph.update_all(fn.copy_edge('ta', 'm_a'), fn.sum('m_a', 'attn_sum'))
        graph.ndata['attn_sum'] = graph.ndata['attn_sum'] + 1e-6
        graph.apply_edges(fn.e_div_v('ta', 'attn_sum', 'norm_attn'))
        norm_attentions = graph.edata.pop('norm_attn')
        return norm_attentions


def top_k_attention(graph: DGLHeteroGraph, attn_scores: Tensor, k: int = 5):
    """
    :param attn_scores:
    :param graph:
    :param k:
    :return:
    """

    def top_k_reduce_func(nodes):
        edge_attention_score = nodes.mailbox['m_a']
        batch_size, neighbor_num, head_num, _ = edge_attention_score.shape
        if neighbor_num <= k:
            ret_a = torch.empty(batch_size, head_num, 1).fill_(edge_attention_score.min()).to(attn_scores.device)
            ret_a_sum = edge_attention_score.sum(dim=1)
        else:
            top_k_values, _ = torch.topk(edge_attention_score, k=k, dim=1)
            ret_a = top_k_values[:, -1, :, :]
            ret_a_sum = top_k_values.sum(dim=1)
        return {'top_a': ret_a, 'top_as': ret_a_sum}

    with graph.local_scope():
        graph.edata['ta'] = attn_scores
        graph.update_all(fn.copy_edge('ta', 'm_a'), top_k_reduce_func)
        graph.apply_edges(edge_udf_attn_mask_func)
        attn_mask = graph.edata.pop('attn_mask')
        top_k_attn_sum = graph.ndata.pop('top_as')
        return attn_mask, top_k_attn_sum


def top_p_attention(graph: DGLHeteroGraph, attn_scores: Tensor, p: float = 0.75):
    """
    when the distribution attention values are quite imbalanced (especially for peak distribution)
    :param attn_scores:
    :param graph:
    :param p:
    :return:
    """

    def top_p_reduce_func(nodes):
        edge_attention_score = nodes.mailbox['m_a']
        batch_size, neighbor_num, head_num, _ = edge_attention_score.shape
        sorted_attends, sorted_indices = torch.sort(edge_attention_score, descending=True, dim=1)
        cumulative_attends = torch.cumsum(sorted_attends, dim=1)
        neighbors_to_remove = cumulative_attends > p
        remove_neighbor_count = neighbors_to_remove.sum(dim=1)
        keep_neighbor_count = neighbor_num - remove_neighbor_count
        assert keep_neighbor_count.min() >= 0 and keep_neighbor_count.max() <= neighbor_num
        if keep_neighbor_count.min() == 0:
            keep_neighbor_count[keep_neighbor_count == 0] = 1
        top_p_score_idx = (keep_neighbor_count - 1)
        assert top_p_score_idx.max() < neighbor_num
        sorted_attends = sorted_attends.squeeze(-1).transpose(1, 2).contiguous().view(batch_size * head_num,
                                                                                      neighbor_num)
        top_p_score_idx = top_p_score_idx.squeeze(-1).contiguous().view(batch_size * head_num)
        row_arrange_idx = torch.arange(batch_size * head_num)
        ret_a = sorted_attends[row_arrange_idx, top_p_score_idx].view(batch_size, head_num, 1)
        cumulative_attends = cumulative_attends.squeeze(-1).transpose(1, 2).contiguous().view(batch_size * head_num,
                                                                                              neighbor_num)
        ret_a_sum = cumulative_attends[row_arrange_idx, top_p_score_idx].view(batch_size, head_num, 1)
        return {'top_a': ret_a, 'top_as': ret_a_sum}

    with graph.local_scope():
        graph.edata['ta'] = attn_scores
        graph.update_all(fn.copy_edge('ta', 'm_a'), top_p_reduce_func)
        graph.apply_edges(edge_udf_attn_mask_func)
        attn_mask = graph.edata.pop('attn_mask')
        top_k_attn_sum = graph.ndata.pop('top_as')
        return attn_mask, top_k_attn_sum


def top_kp_attention(graph: DGLHeteroGraph, attn_scores: Tensor, k: int = 5, p: float = 0.75, sparse_mode='top_k'):
    if sparse_mode == 'top_k':
        assert k >= 2
        attn_mask, top_k_attn_sum = top_k_attention(graph=graph, attn_scores=attn_scores, k=k)
    else:
        assert 0.0 < p <= 1.0
        attn_mask, top_k_attn_sum = top_p_attention(graph=graph, attn_scores=attn_scores, p=p)
    return attn_mask, top_k_attn_sum


def top_kp_attn_normalization(graph: DGLHeteroGraph, attn_scores: Tensor, attn_mask: Tensor, top_k_sum: Tensor):
    with graph.local_scope():
        graph.edata['ta'] = attn_scores
        graph.edata['ta'][~attn_mask] = 0.0
        graph.dstdata['attn_sum'] = top_k_sum
        graph.apply_edges(fn.e_div_v('ta', 'attn_sum', 'norm_attn'))
        norm_attentions = graph.edata.pop('norm_attn')
        return norm_attentions


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class PositionWiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, model_dim, d_hidden, model_out_dim, layer_num=1, dropout=0.25):
        super(PositionWiseFeedForward, self).__init__()
        self.model_dim = model_dim
        self.hidden_dim = d_hidden
        self.model_out_dim = model_out_dim
        self.w_1 = nn.Linear(model_dim, d_hidden)
        self.batch_norm = BatchNorm1d(num_features=d_hidden)
        self.w_2 = nn.Linear(d_hidden, model_out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_num = layer_num
        self.init()

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.batch_norm(self.w_1(x)))))

    def init(self):
        gain = small_init_gain(d_in=self.model_dim, d_out=self.hidden_dim) / math.sqrt(self.layer_num)
        nn.init.xavier_normal_(self.w_1.weight, gain=gain)
        gain = small_init_gain(d_in=self.model_dim, d_out=self.model_out_dim) / math.sqrt(self.layer_num)
        nn.init.xavier_normal_(self.w_2.weight, gain=gain)


class LBRLayer(nn.Module):
    def __init__(self, model_dim, model_out_dim, layer_num=1, dropout: float = None):
        super(LBRLayer, self).__init__()
        self.model_dim = model_dim
        self.model_out_dim = model_out_dim
        self.batch_norm = BatchNorm1d(num_features=model_out_dim)
        self.w_1 = nn.Linear(model_dim, model_out_dim)
        self.layer_num = layer_num
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = Identity()
        self.init()

    def init(self):
        gain = small_init_gain(d_in=self.model_dim, d_out=self.model_out_dim) / math.sqrt(self.layer_num)
        nn.init.xavier_normal_(self.w_1.weight, gain=gain)

    def forward(self, x):
        return self.dropout(F.relu(self.batch_norm(self.w_1(x))))


class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, L=2, dropout: float = 0.1):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.dropout = nn.Dropout(dropout)
        self.L = L
        self.init()

    def init(self):
        gain = nn.init.calculate_gain('relu')
        for _ in self.FC_layers:
            nn.init.xavier_normal_(_.weight, gain=gain)

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = self.dropout(F.relu(y))
        y = self.FC_layers[self.L](y)
        return y


def small_init_gain(d_in, d_out):
    return math.sqrt(2.0 / (d_in + 4.0 * d_out))


def small_init_gain_v2(d_in, d_out):
    return math.sqrt(2.0 / (d_in + d_out))


class EmbeddingLayer(nn.Module):
    def __init__(self, num: int, dim: int, drop_out=0.1, project_dim: int = None):
        super(EmbeddingLayer, self).__init__()
        self.num = num
        self.dim = dim
        self.proj_dim = project_dim
        self.dropout = nn.Dropout(p=drop_out)
        self.embedding = nn.Embedding(num_embeddings=num, embedding_dim=dim)
        if self.proj_dim is not None and self.proj_dim > 0:
            self.projection = torch.nn.Linear(self.dim, self.proj_dim, bias=False)
        else:
            self.projection = Identity()
        self.init()

    def init_with_tensor(self, data: Tensor, freeze=False):
        self.embedding = nn.Embedding.from_pretrained(embeddings=data, freeze=freeze)

    def init(self):
        """Initializing the embeddings.
        """
        gain = small_init_gain(d_in=self.dim, d_out=self.dim)
        nn.init.xavier_normal_(self.embedding.weight, gain=gain)
        if isinstance(self.projection, nn.Linear):
            gain = small_init_gain(d_in=self.dim, d_out=self.proj_dim)
            nn.init.xavier_normal_(self.projection.weight, gain=gain)

    def _embed(self, embeddings):
        if isinstance(self.projection, nn.Linear):
            embeddings = self.projection(self.dropout(embeddings))
        else:
            embeddings = self.projection(embeddings)
        return embeddings

    def forward(self, indexes: LongTensor):
        embed_data = self._embed(self.embedding(indexes))
        return embed_data


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def attention_computation(query: Tensor, key: Tensor):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    return p_attn


def neighbor_interaction_computation(graph: DGLHeteroGraph, attn_drop=None):
    def attention_message_function(edges):
        query, neighbors_key, neighbors_v = edges.src['q'], edges.src['k'], edges.src['v']
        return {'m_k': neighbors_key, 'm_q': query, 'm_v': neighbors_v}

    def neighbor_attention_reduce_function(nodes):
        messages = nodes.mailbox['m_q'], nodes.mailbox['m_k'], nodes.mailbox['m_v']
        query, key, value = [_.transpose(1, 2) for _ in messages]
        p_attn = attention_computation(query=query, key=key)
        if attn_drop is not None:
            p_attn = attn_drop(p_attn)
        rv = torch.matmul(p_attn, value)
        rv = rv.transpose(1, 2).contiguous().mean(dim=1)
        return {'rv': rv}

    with graph.local_scope():
        graph.update_all(attention_message_function, neighbor_attention_reduce_function)
        neighbor_value = graph.ndata.pop('rv')
        return neighbor_value
