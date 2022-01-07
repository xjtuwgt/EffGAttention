import dgl
import numpy as np
import torch
from numpy import random
from dgl.sampling import sample_neighbors
from torch import Tensor
from time import time
import copy


def construct_special_graph_dictionary(graph, hop_num: int, n_relations: int):
    """
    :param graph:
    :param hop_num: number of hops to generate special relations
    :param n_relations: number of relations in graph
    :return:
    """
    special_relation_dict = {'loop_r': n_relations - 1}
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for hop in range(1, hop_num):
        special_relation_dict['in_hop_{}_r'.format(hop + 1)] = n_relations + (2 * (hop - 1))
        special_relation_dict['out_hop_{}_r'.format(hop + 1)] = n_relations + (2 * hop - 1)
    n_relations = n_relations + 2 * (hop_num - 1)
    number_of_relations = n_relations
    return graph, number_of_relations, special_relation_dict


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def add_relation_ids_to_graph(graph, edge_type_ids: Tensor):
    """
    :param graph:
    :param edge_type_ids: add 'rid' to graph edge data --> type id
    :return:
    """
    graph.edata['rid'] = edge_type_ids
    return graph


def add_self_loop_in_graph(graph, self_loop_r: int):
    """
    :param graph:
    :param self_loop_r:
    :return:
    """
    g = copy.deepcopy(graph)
    number_of_nodes = g.number_of_nodes()
    self_loop_r_array = torch.full((number_of_nodes,), self_loop_r, dtype=torch.long)
    node_ids = torch.arange(number_of_nodes)
    g.add_edges(node_ids, node_ids, {'rid': self_loop_r_array})
    return g
