from ogb.nodeproppred import DglNodePropPredDataset
from evens import HOME_DATA_FOLDER as ogb_root
import torch
from codes.graph_utils import OON_Initialization
from codes.graph_utils import special_graph_dictionary_construction
from codes.graph_utils import add_relation_ids_to_graph
from codes.utils import IGNORE_IDX


def ogb_nodeprop_graph_reconstruction(dataset: str):
    """
    :param dataset:
    'undirected':
       'ogbn-products': an undirected and unweighted graph (Amazon product)
       'ogbn-proteins': dataset is an undirected, weighted, and typed (according to species) graph
    'directed':
       'ogbn-arxiv': is a directed graph, representing the citation network (MAG) - Microsoft Academic Graph
       'ogbn-papers100M': dataset is a directed citation graph (MAG)
    'heterogeneous':
       'ogbn-mag' dataset is a heterogeneous network composed of a subset of the Microsoft Academic Graph
       directed relations
    :return:
    """
    data = DglNodePropPredDataset(name=dataset, root=ogb_root)
    node_split_idx = data.get_idx_split()
    graph, labels = data[0]
    # +++++++++++++++++++++++++++++++++
    graph.ndata['label'] = labels
    # +++++++++++++++++++++++++++++++++
    n_classes = labels.max().data.item()
    node_features = graph.ndata['feat']
    n_feats = node_features.shape[1]
    if dataset in {'ogbn-products'}:  # 'ogbn-proteins'
        number_of_edges = graph.number_of_edges()
        edge_type_ids = torch.zeros(number_of_edges, dtype=torch.long)
        graph = add_relation_ids_to_graph(graph=graph, edge_type_ids=edge_type_ids)
        nentities, nrelations = graph.number_of_nodes(), 1
    elif dataset in {'ogbn-arxiv', 'ogbn-papers100M'}:
        number_of_edges = graph.number_of_edges()
        edge_type_ids = torch.zeros(number_of_edges, dtype=torch.long)
        graph = add_relation_ids_to_graph(graph=graph, edge_type_ids=edge_type_ids)
        src_nodes, dst_nodes = graph.edges()
        graph.add_edges(dst_nodes, src_nodes, {'rid': edge_type_ids + 1})
        nentities, nrelations = graph.number_of_nodes(), 2
    else:
        raise 'Dataset {} is not supported'.format(dataset)
    print('Number of nodes = {}'.format(graph.number_of_nodes()))
    print('Number of edges = {}'.format(graph.number_of_edges()))
    print('Number of features = {}'.format(n_feats))
    print('Number of classes = {}'.format(n_classes))
    print('Number of train = {}'.format(node_split_idx['train'].shape[0]))
    print('Number of valid = {}'.format(node_split_idx['valid'].shape[0]))
    print('Number of test = {}'.format(node_split_idx['test'].shape[0]))
    return graph, node_split_idx, node_features, nentities, nrelations, n_classes, n_feats


def ogb_train_valid_test(node_split_idx: dict, data_type: str):
    data_node_ids = node_split_idx[data_type]
    data_len = data_node_ids.shape[0]
    return data_len, data_node_ids


def ogb_k_hop_graph_reconstruction(dataset: str, hop_num=5, oon='zero', cls: bool = True):
    assert oon in {'zero', 'one', 'rand'}
    graph, node_split_idx, node_features, n_entities, n_relations, n_classes, n_feats = \
        ogb_nodeprop_graph_reconstruction(dataset=dataset)
    graph, number_of_relations, special_node_dict, \
    special_relation_dict = special_graph_dictionary_construction(graph=graph, n_relations=n_relations,
                                                                  hop_num=hop_num, cls=cls)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    graph.ndata['label'][-2:] = -IGNORE_IDX
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    number_of_added_nodes = graph.number_of_nodes() - n_entities
    print('Added number of nodes = {}'.format(number_of_added_nodes))
    print('*' * 75)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if number_of_added_nodes > 0:
        node_features = graph.ndata['feat']
        added_node_features = OON_Initialization(oon_num=number_of_added_nodes, num_feats=node_features.shape[1],
                                                 oon=oon)
        graph.ndata['feat'][-2:] = added_node_features
    number_of_nodes = graph.number_of_nodes()
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    graph.ndata.update({'nid': torch.arange(0, number_of_nodes, dtype=torch.long)})
    if cls:
        assert graph.ndata['nid'][-1] == special_node_dict['cls'] and special_node_dict['cls'] == number_of_nodes - 1
    return graph, number_of_nodes, number_of_relations, n_classes, n_feats, special_node_dict, special_relation_dict, \
           node_split_idx
