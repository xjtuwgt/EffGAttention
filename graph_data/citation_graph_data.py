import torch
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import copy
from dgl import DGLHeteroGraph
import dgl
from codes.graph_utils import construct_special_graph_dictionary, add_relation_ids_to_graph, add_self_loop_in_graph
import numpy as np


def citation_graph_reconstruction(dataset: str):
    if dataset == 'cora':
        data = CoraGraphDataset()
    elif dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif dataset == 'pubmed':
        data = PubmedGraphDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
    graph = data[0]
    n_classes = data.num_labels
    node_features = graph.ndata['feat']
    n_feats = node_features.shape[1]
    graph = dgl.remove_self_loop(g=graph)
    number_of_edges = graph.number_of_edges()
    edge_type_ids = torch.zeros(number_of_edges, dtype=torch.long)
    graph = add_relation_ids_to_graph(graph=graph, edge_type_ids=edge_type_ids)
    graph = add_self_loop_in_graph(graph=graph, self_loop_r=1)
    n_entities = graph.number_of_nodes()
    n_relations = 2
    in_degrees = graph.in_degrees()
    assert in_degrees.min() >= 1
    graph.ndata['log_in'] = torch.log2(in_degrees.float())
    return graph, n_entities, n_relations, n_classes, n_feats


def citation_train_valid_test(graph, data_type: str):
    if data_type == 'train':
        data_mask = graph.ndata['train_mask']
    elif data_type == 'valid':
        data_mask = graph.ndata['val_mask']
    elif data_type == 'test':
        data_mask = graph.ndata['test_mask']
    else:
        raise 'Data type = {} is not supported'.format(data_type)
    data_len = data_mask.int().sum().item()
    data_node_ids = data_mask.nonzero().squeeze()
    return data_len, data_node_ids


def citation_graph_rand_split_construction(dataset: str):
    graph, n_entities, n_relations, n_classes, n_feats = citation_graph_reconstruction(dataset=dataset)
    train_mask = graph.ndata['train_mask']
    valid_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    test_node_num, valid_node_num, train_node_num = sum(test_mask), sum(valid_mask), sum(train_mask)
    graph_mask = torch.logical_or(train_mask, valid_mask)
    graph_mask = torch.logical_or(graph_mask, test_mask)
    graph_node_ids = graph_mask.nonzero().squeeze()
    in_degrees = graph.in_degrees()
    graph_in_degrees = in_degrees[graph_node_ids]

    graph_node_ids = graph_node_ids.tolist()
    graph_in_degrees = graph_in_degrees.tolist()
    graph_node_degree_pairs = list(zip(graph_node_ids, graph_in_degrees))
    graph_node_degree_pairs.sort(key=lambda x: x[1])
    sorted_graph_node_ids = [_[0] for _ in graph_node_degree_pairs]
    train_node_ids = sorted_graph_node_ids[:train_node_num]
    valid_node_ids = sorted_graph_node_ids[train_node_num:(train_node_num + valid_node_num)]
    test_node_ids = sorted_graph_node_ids[(train_node_num + valid_node_num):]
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    new_graph = copy.deepcopy(graph)
    new_train_mask = torch.as_tensor(train_mask).fill_(False)
    new_valid_mask = torch.as_tensor(valid_mask).fill_(False)
    new_test_mask = torch.as_tensor(test_mask).fill_(False)
    # print(sum(new_test_mask), sum(new_valid_mask), sum(new_train_mask))
    new_train_mask[train_node_ids] = True
    new_valid_mask[valid_node_ids] = True
    new_test_mask[test_node_ids] = True
    # print(sum(new_test_mask), sum(new_valid_mask), sum(new_train_mask))
    new_graph.ndata.update({'train_mask': new_train_mask, 'val_mask': valid_mask, 'test_mask': test_mask})
    return new_graph, n_entities, n_relations, n_classes, n_feats


def citation_k_hop_graph_reconstruction(dataset: str, hop_num=5, rand_split=False):
    print('Bi-directional homogeneous graph: {}'.format(dataset))
    if rand_split:
        graph, n_entities, n_relations, n_classes, n_feats = \
            citation_graph_rand_split_construction(dataset=dataset)
    else:
        graph, n_entities, n_relations, n_classes, n_feats = \
            citation_graph_reconstruction(dataset=dataset)
    graph, number_of_relations, special_relation_dict = construct_special_graph_dictionary(graph=graph,
                                                                                           n_relations=n_relations,
                                                                                           hop_num=hop_num)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    number_of_nodes = graph.number_of_nodes()
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    graph.ndata.update({'nid': torch.arange(0, number_of_nodes, dtype=torch.long)})
    return graph, number_of_nodes, number_of_relations, n_classes, n_feats, special_relation_dict


def k_hop_graph_edge_collection(graph: DGLHeteroGraph, hop_num: int = 5):
    k_hop_graph_edge_dict = {}
    copy_graph = copy.deepcopy(graph)
    copy_graph = dgl.remove_self_loop(g=copy_graph)
    # number_of_nodes = copy_graph.number_of_nodes()
    # one_hop_head_nodes,  one_hop_tail_nodes = copy_graph.edges()
    # one_hop_edges = number_of_nodes * one_hop_head_nodes + one_hop_tail_nodes
    # one_hop_edge_set = set(one_hop_edges.tolist())
    # k_hop_graph_edge_dict['1_hop'] = one_hop_edge_set
    for k in range(2, hop_num+1):
        k_hop_graph = dgl.khop_graph(copy_graph, k=k)
        if k_hop_graph.number_of_edges() > 0:
            head_nodes, tail_nodes = k_hop_graph.edges()
            k_hop_graph_edge_dict['{}_hop'.format(k)] = (head_nodes, tail_nodes)
        else:
            break
    return k_hop_graph_edge_dict


def label_mask_drop(train_mask, drop_ratio: float = 0.05):
    train_idxs = train_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    train_size = len(train_idxs)
    drop_idxes = np.random.choice(train_idxs, int(train_size * drop_ratio), replace=False)
    train_mask_clone = train_mask.clone()
    train_mask_clone[drop_idxes] = False
    return train_mask_clone

