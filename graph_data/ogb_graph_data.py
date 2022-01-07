from ogb.nodeproppred import DglNodePropPredDataset
from evens import HOME_DATA_FOLDER as ogb_root

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
    node_features = graph.ndata.pop('feat')
    n_feats = node_features.shape[1]
    if dataset in {'ogbn-products'}:  # 'ogbn-proteins'
        nentities, nrelations = graph.number_of_nodes(), 1
    elif dataset in {'ogbn-arxiv', 'ogbn-papers100M'}:
        nentities, nrelations = graph.number_of_nodes(), 2
    else:
        raise 'Dataset {} is not supported'.format(dataset)
    return graph, node_split_idx, node_features, nentities, nrelations, n_classes, n_feats


def ogb_train_valid_test(node_split_idx: dict, data_type: str):
    data_node_ids = node_split_idx[data_type]
    data_len = data_node_ids.shape[0]
    return data_len, data_node_ids

