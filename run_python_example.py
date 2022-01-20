from time import time
from evens import PROJECT_FOLDER
# from torch.nn.modules.instancenorm import _InstanceNorm
# import torch.nn.functional as F
# from torch import Tensor
# import torch
#
# class InstanceNorm(_InstanceNorm):
#     r"""Applies instance normalization over each individual example in a batch
#     of node features as described in the `"Instance Normalization: The Missing
#     Ingredient for Fast Stylization" <https://arxiv.org/abs/1607.08022>`_
#     paper
#
#     .. math::
#         \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
#         \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
#         \odot \gamma + \beta
#
#     The mean and standard-deviation are calculated per-dimension separately for
#     each object in a mini-batch.
#
#     Args:
#         in_channels (int): Size of each input sample.
#         eps (float, optional): A value added to the denominator for numerical
#             stability. (default: :obj:`1e-5`)
#         momentum (float, optional): The value used for the running mean and
#             running variance computation. (default: :obj:`0.1`)
#         affine (bool, optional): If set to :obj:`True`, this module has
#             learnable affine parameters :math:`\gamma` and :math:`\beta`.
#             (default: :obj:`False`)
#         track_running_stats (bool, optional): If set to :obj:`True`, this
#             module tracks the running mean and variance, and when set to
#             :obj:`False`, this module does not track such statistics and always
#             uses instance statistics in both training and eval modes.
#             (default: :obj:`False`)
#     """
#     def __init__(self, in_channels, eps=1e-5, momentum=0.1, affine=False,
#                  track_running_stats=False):
#         super().__init__(in_channels, eps, momentum, affine,
#                          track_running_stats)
#
#     def forward(self, x: Tensor) -> Tensor:
#         out = F.instance_norm(
#             x.t().unsqueeze(0), self.running_mean, self.running_var,
#             self.weight, self.bias, self.training
#             or not self.track_running_stats, self.momentum, self.eps)
#         return out.squeeze(0).t()
#
#
# # from torch.nn import LayerNorm
# # import torch
# # from codes.utils import seed_everything
# # print()
# # seed_everything(seed=42)
# # x = torch.randint(0, 100, (5,3)).float()
# # insnorm = InstanceNorm(3)
# # layerNorm = LayerNorm(3)
# #
# # y = insnorm(x)
# #
# # print(layerNorm(x))
# #
# # print(x)
# # print(insnorm(layerNorm(x)))
# #
# # # x_mean = torch.mean(x, dim=2, keepdim=True)
# # # # print(x_mean)
# # # x_var = torch.var(x, dim=2, keepdim=True)
# # # # print(x_var)
# # #
# # # x_norm = (x - x_mean)/torch.sqrt(x_var + 1e-6)
# # #
# # # print(x_norm)
# # print(x)
# # print(insnorm(x))
# from kge_codes.kge_utils import graph_to_triples, triple_train_valid_split
# from graph_data.citation_graph_data import citation_graph_reconstruction
# from codes.gnn_utils import neighbor_interaction_computation
# from codes.graph_utils import construct_special_graph_dictionary, anchor_node_sub_graph_extractor
#
#
import torch
import dgl
from codes.fast_ppr import pagerank, pagerank_power
import math
restart_prob = 0.85
edges = torch.tensor([0, 1, 2, 1, 2, 0, 2, 3, 3, 4, 5, 4, 5, 3]), \
        torch.tensor([1, 2, 0, 0, 1, 2, 3, 2, 4, 5, 3, 3, 4, 5])  # 边：2->3, 5->5, 3->0
g = dgl.graph(edges)
g.ndata['n_id'] = torch.arange(0, 6)

g1 = dgl.edge_subgraph(graph=g, edges=[0, 2, 4])

# print(g1.ndata['n_id'])
#
# g1 = dgl.edge_subgraph(graph=g, edges=[0, 4, 2])
#
# print(g1.ndata['n_id'])
#
# g1 = dgl.edge_subgraph(graph=g, edges=[4, 0, 2])
#
# print(g1.ndata['n_id'])
#
# edge_set = [4, 0, 4, 6, 2]
# y = sorted(set(edge_set), key=edge_set.index)
#
# print(y)
# y = g.adj(scipy_fmt='coo')
#
# # z = pagerank(A=y)
# # print(z)
#
# z1 = pagerank_power(A=y)
# print(z1)

# print(y)
#
# g1 = dgl.from_scipy(y)
# print(g1)


# max_nodes_for_seed = max(64,
#                          int((g.out_degree(1) * math.e / (math.e - 1) / restart_prob) + 0.5))
#
# for _ in range(1):
#     trace, types = dgl.sampling.random_walk(g=g, nodes=[0] * (max_nodes_for_seed * 5), length=5,
#                                             restart_prob=restart_prob)
#     print(trace.shape)
#     print(trace)
#     x = (trace >= 0).sum(dim=1) > 1
#     print(sum(x))
#     y = trace[x]
#     print(y.shape)
#     print(y)
    # trace = trace[trace >=0]
    # print(trace)
    # rwr_hop_dict = {}
    # for hop in range(2, trace.shape[1]):
    #     trace_i = trace[:,hop]
    #     trace_i = trace_i[trace_i >= 0]
    #     neighbors_i = torch.unique(trace_i).tolist()
    #     rwr_hop_dict[hop] = neighbors_i
    #
    #     # print(torch.unique(trace_i).tolist())
    # print(rwr_hop_dict)
    # print(trace.shape)
    # # trace_i =
    # trace = trace[trace >=0]
    # subv = torch.unique(trace).tolist()
    # print(subv)
    # trace = trace[trace >= 0]
    # # print(types.shape)
    # print(trace.shape)
    # subv = torch.unique(trace).tolist()
    # print(subv)
    # print(trace.shape)
    # print(trace[trace >=0].shape)


# print(max_nodes_for_seed)
#
# # print(x[0])
#
# y = int((34 * math.e / (math.e - 1) / restart_prob) + 0.5)
# print(y)
#
# print(x)

# dgl.contrib.sampling.random_walk_with_restart()

# print(g.number_of_nodes())
# g.ndata['nid'] = torch.arange(0, g.number_of_nodes(), dtype=torch.long)
# g.edata['rid'] = torch.zeros(g.number_of_edges(), dtype=torch.long)
#
# graph, number_of_relations, special_node_dict, special_relation_dict = \
#     construct_special_graph_dictionary(graph=g, n_relations=1, hop_num=5)
#
# subgraph, parent2sub_dict = anchor_node_sub_graph_extractor(graph=graph, anchor_node_ids=torch.LongTensor([0]),
#                                 cls_node_ids=torch.LongTensor([special_node_dict['cls']]),
#                                 fanouts=[-1,-1],
#                                 edge_dir='in', cls=False,
#                                 special_relation2id=special_relation_dict)
#
# # print(number_of_relations)
# # print(special_node_dict)
# # print(special_relation_dict)
# # print(graph)
# print(graph.edges())
#
# # print(subgraph)
# print(subgraph.ndata['n_rw_pos'])
# print(subgraph.edges())
# print(parent2sub_dict)
# print('*' * 50)
#
# subgraph, parent2sub_dict = anchor_node_sub_graph_extractor(graph=graph, anchor_node_ids=torch.LongTensor([1]),
#                                 cls_node_ids=torch.LongTensor([special_node_dict['cls']]),
#                                 fanouts=[-1,-1,-1],
#                                 edge_dir='in', cls=False,
#                                 special_relation2id=special_relation_dict)
#
# # print(subgraph)
# print(subgraph.ndata['n_rw_pos'])
# print(subgraph.edges())
# print(parent2sub_dict)
#
# # g.edata['r_id'] = torch.randint(1, 9, (2,))
#
# # g.srcdata.update({'k': torch.rand((3, 2, 8)), 'v': torch.randn((3, 2, 8))})
# # g.dstdata.update({'q': torch.randn((3, 2, 8))})
#
# # # print(g.ndata)
# #
# # neighbor_value = neighbor_interaction_computation(graph=g)
# # print(neighbor_value)
# #
# # x, y, z = g.edges(form='all')
# # print(x)
# # print(y)
# # print(z)
# #
# # a, b, c, d = graph_to_triples(graph=g, edge_rel_name='r_id')
# # print(a)
# # print(b)
# # print(c)
# # print(d)
#
# # graph, n_entities, n_relations, n_classes, n_feats = citation_graph_reconstruction(dataset='cora')
# # # print(graph)
# # triples, _, _, relation2id = graph_to_triples(graph=graph, edge_rel_name='rid')
# # print(triples.shape)
# #
# # triple_train_valid_split(triples=triples)
#
# # graph_to_triples()
# # g_2 = dgl.transform.khop_graph(g, 2)
# # print(g_2.edges())
# # g_2 = dgl.khop_graph(g=g, k=3)
# # print(g_2.number_of_edges())
# # print(g_2.edges())
#
# # import torch.nn as nn
# # import torch.nn.functional as F
# #
# # """
# #     MLP Layer used after graph vector representation

from codes.default_argparser import default_parser, complete_default_parser
from graph_data.graph_dataloader import NodeClassificationDataHelper, SelfSupervisedNodeDataHelper
from codes.utils import seed_everything
from codes.simsiam_networks import SimSiamNodeClassification

args = default_parser().parse_args()
args = complete_default_parser(args=args)

seed_everything(seed=args.seed)

dataHelper = SelfSupervisedNodeDataHelper(config=args)
ssl_train_data = dataHelper.data_loader()
start_time = time()
for batch_idx, batch in enumerate(ssl_train_data):
    if batch_idx % 200 == 0:
        print(batch_idx)
print('Runtime = {}'.format(time() - start_time))
# datahelper = node_classification_data_helper(config=args)
# args.num_classes = datahelper.num_class
# args.node_emb_dim = datahelper.n_feats
# node_features = datahelper.graph.ndata.pop('feat')
# print(node_features.shape)
#
# print(datahelper.graph.device)
#
# train_data = datahelper.data_loader(data_type='train')

# simsiam_classifier = SimSiam_NodeClassification(config=args)
# print(simsiam_classifier)

# simsiam_model = SimSiam_Model_Builder(config=args)
# simsiam_model.graph_encoder.init_graph_ember(ent_emb=node_features, ent_freeze=True)
# #
# print(simsiam_model)
# start_time = time()
#
# for batch_idx, batch in enumerate(train_data):
#     print(batch_idx)
#     # batch_graph = batch['batch_graph']
#     # print(batch_graph[0].ndata['nid'])
#     # cls_idx = batch_graph[1]
#     # print(batch_graph[0].ndata['nid'][cls_idx])
#     # print(batch_graph[2])
#     # simsiam_model.encode(batch_graph)
#
# print('runtime = {}'.format(time() - start_time))
