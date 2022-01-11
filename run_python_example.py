# from torch.nn import InstanceNorm1d
# import torch
# from codes.utils import seed_everything
# insnorm = InstanceNorm1d(1)
# print()
# seed_everything(seed=42)
# x = torch.randn((1, 1, 100))
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

import dgl
g = dgl.graph(([0, 1], [1, 2]))
g_2 = dgl.transform.khop_graph(g, 2)
print(g_2.edges())
g_2 = dgl.khop_graph(g=g, k=3)
print(g_2.number_of_edges())
print(g_2.edges())