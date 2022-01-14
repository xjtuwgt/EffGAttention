from torch.nn.modules.instancenorm import _InstanceNorm
import torch.nn.functional as F
from torch import Tensor
import torch

class InstanceNorm(_InstanceNorm):
    r"""Applies instance normalization over each individual example in a batch
    of node features as described in the `"Instance Normalization: The Missing
    Ingredient for Fast Stylization" <https://arxiv.org/abs/1607.08022>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    The mean and standard-deviation are calculated per-dimension separately for
    each object in a mini-batch.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        momentum (float, optional): The value used for the running mean and
            running variance computation. (default: :obj:`0.1`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`False`)
        track_running_stats (bool, optional): If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and always
            uses instance statistics in both training and eval modes.
            (default: :obj:`False`)
    """
    def __init__(self, in_channels, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False):
        super().__init__(in_channels, eps, momentum, affine,
                         track_running_stats)

    def forward(self, x: Tensor) -> Tensor:
        out = F.instance_norm(
            x.t().unsqueeze(0), self.running_mean, self.running_var,
            self.weight, self.bias, self.training
            or not self.track_running_stats, self.momentum, self.eps)
        return out.squeeze(0).t()


# from torch.nn import LayerNorm
# import torch
# from codes.utils import seed_everything
# print()
# seed_everything(seed=42)
# x = torch.randint(0, 100, (5,3)).float()
# insnorm = InstanceNorm(3)
# layerNorm = LayerNorm(3)
#
# y = insnorm(x)
#
# print(layerNorm(x))
#
# print(x)
# print(insnorm(layerNorm(x)))
#
# # x_mean = torch.mean(x, dim=2, keepdim=True)
# # # print(x_mean)
# # x_var = torch.var(x, dim=2, keepdim=True)
# # # print(x_var)
# #
# # x_norm = (x - x_mean)/torch.sqrt(x_var + 1e-6)
# #
# # print(x_norm)
# print(x)
# print(insnorm(x))
from kge_codes.kge_utils import graph_to_triples, triple_train_valid_split
from graph_data.citation_graph_data import citation_graph_reconstruction
from codes.gnn_utils import neighbor_interaction_computation

import dgl
edges = torch.tensor([0, 1, 2, 1, 2, 0]), torch.tensor([1, 2, 0, 0, 1, 2])  # 边：2->3, 5->5, 3->0
g = dgl.graph(edges)
print(g.number_of_nodes())

# g.edata['r_id'] = torch.randint(1, 9, (2,))

g.srcdata.update({'k': torch.rand((3, 2, 8)), 'v': torch.randn((3, 2, 8))})
g.dstdata.update({'q': torch.randn((3, 2, 8))})

# print(g.ndata)

neighbor_value = neighbor_interaction_computation(graph=g)
print(neighbor_value)
#
# x, y, z = g.edges(form='all')
# print(x)
# print(y)
# print(z)
#
# a, b, c, d = graph_to_triples(graph=g, edge_rel_name='r_id')
# print(a)
# print(b)
# print(c)
# print(d)

# graph, n_entities, n_relations, n_classes, n_feats = citation_graph_reconstruction(dataset='cora')
# # print(graph)
# triples, _, _, relation2id = graph_to_triples(graph=graph, edge_rel_name='rid')
# print(triples.shape)
#
# triple_train_valid_split(triples=triples)

# graph_to_triples()
# g_2 = dgl.transform.khop_graph(g, 2)
# print(g_2.edges())
# g_2 = dgl.khop_graph(g=g, k=3)
# print(g_2.number_of_edges())
# print(g_2.edges())

import torch.nn as nn
import torch.nn.functional as F

"""
    MLP Layer used after graph vector representation
"""

x = torch.rand((3,4))
y = torch.rand((3, 1))
x
# class MLPReadout(nn.Module):
#     def __init__(self, input_dim, output_dim, L=2, dropout: float=0.1):  # L=nb_hidden_layers
#         super().__init__()
#         list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
#         list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
#         self.FC_layers = nn.ModuleList(list_FC_layers)
#         self.dropout = nn.Dropout(dropout)
#         self.L = L
#
#     def forward(self, x):
#         y = x
#         for l in range(self.L):
#             y = self.FC_layers[l](y)
#             y = self.dropout(F.relu(y))
#         y = self.FC_layers[self.L](y)
#         return y
#
# f = MLPReadout(input_dim=256, output_dim=6)
# print(f)