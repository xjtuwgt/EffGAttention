from dgl import DGLHeteroGraph
import torch
import dgl
import copy
from numpy import random
from torch.utils.data import Dataset
from graph_data.citation_graph_data import citation_train_valid_test
from graph_data.citation_graph_data import citation_k_hop_graph_reconstruction
from graph_data.ogb_graph_data import ogb_k_hop_graph_reconstruction
from graph_data.ogb_graph_data import ogb_train_valid_test
from codes.graph_utils import anchor_node_sub_graph_extractor
from torch.utils.data import DataLoader
from evens import PROJECT_FOLDER


class SubGraphPairDataset(Dataset):
    def __init__(self, graph: DGLHeteroGraph, nentity: int, nrelation: int, fanouts: list, special_entity2id: dict,
                 special_relation2id: dict, graph_type: str, restart_prob: float = 0.8, bi_directed: bool = True,
                 self_loop: bool = False, edge_dir: str = 'in', cls: bool = True):
        assert len(fanouts) > 0 and graph_type in {'citation', 'ogb'}
        self.fanouts = fanouts  # list of int == number of hops for sampling
        self.hop_num = len(fanouts)
        assert self.hop_num >= 2
        self.g = graph
        self.cls = cls
        self.nentity, self.nrelation = nentity, nrelation
        self.bi_directed = bi_directed
        self.edge_dir = edge_dir  # "in", "out"
        self.self_loop = self_loop
        self.special_entity2id, self.special_relation2id = special_entity2id, special_relation2id
        self.restart_prob = restart_prob
        self.sample_types = ['ns', 'rwr']
        if self.g.ndata['nid'][-1] == special_entity2id['cls']:
            self.len = self.g.number_of_nodes() - 1
        else:
            self.len = self.g.number_of_nodes()
        self.data_node_ids = torch.arange(0, self.len, 1).to(self.g.device)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        node_idx = self.data_node_ids[idx]
        anchor_node_ids = torch.LongTensor([node_idx])
        samp_hop_num = random.randint(2, self.hop_num + 1)  # sample sub-graph from 2-hop to k-hop (k >= 2)
        samp_fanouts = self.fanouts[:samp_hop_num]
        cls_node_ids = torch.LongTensor([self.special_entity2id['cls']])
        sample_type = random.choice(self.sample_types, size=1)
        subgraph, parent2sub_dict, _ = \
            anchor_node_sub_graph_extractor(graph=self.g,
                                            anchor_node_ids=anchor_node_ids,
                                            cls_node_ids=cls_node_ids,
                                            fanouts=samp_fanouts,
                                            edge_dir=self.edge_dir,
                                            special_relation2id=self.special_relation2id,
                                            self_loop=self.self_loop,
                                            bi_directed=self.bi_directed,
                                            restart_prob=self.restart_prob,
                                            samp_type=sample_type,
                                            cls=self.cls,
                                            debug=False)
        sub_anchor_id = parent2sub_dict[node_idx.data.item()]
        aug_samp_type = random.choice(self.sample_types, size=1)
        aug_subgraph = copy.deepcopy(subgraph)
        assert subgraph.number_of_nodes() == aug_subgraph.number_of_nodes()
        return subgraph, aug_subgraph, sub_anchor_id

    @staticmethod
    def collate_fn(data):
        assert len(data[0]) == 3
        batch_graphs_1 = dgl.batch([_[0] for _ in data])
        batch_graphs_2 = dgl.batch([_[1] for _ in data])

        batch_graph_cls = torch.as_tensor([_[0].number_of_nodes() for _ in data], dtype=torch.long)
        batch_graph_cls = torch.cumsum(batch_graph_cls, dim=0) - 1
        # ++++++++++++++++++++++++++++++++++++++++
        batch_anchor_id = torch.zeros(len(data), dtype=torch.long)
        for idx, _ in enumerate(data):
            if idx == 0:
                batch_anchor_id[idx] = _[2]
            else:
                batch_anchor_id[idx] = _[2] + batch_graph_cls[idx - 1].data.item() + 1
        # +++++++++++++++++++++++++++++++++++++++
        return {'batch_graph_1': (batch_graphs_1, batch_graph_cls, batch_anchor_id),
                'batch_graph_2': (batch_graphs_2, batch_graph_cls, batch_anchor_id)}


class SelfSupervisedNodeDataHelper(object):
    def __init__(self, config):
        self.config = config
        self.graph_type = self.config.graph_type
        if self.graph_type == 'citation':
            graph, number_of_nodes, number_of_relations, n_classes, n_feats, special_node_dict, special_relation_dict = \
                citation_k_hop_graph_reconstruction(dataset=self.config.citation_node_name,
                                                    hop_num=self.config.sub_graph_hop_num, rand_split=False)
            self.node_split_idx = None
        else:
            graph, number_of_nodes, number_of_relations, n_classes, n_feats, special_node_dict, \
            special_relation_dict, node_split_idx = ogb_k_hop_graph_reconstruction(dataset=self.config.ogb_node_name,
                                                                                   hop_num=self.config.sub_graph_hop_num)
            self.node_split_idx = node_split_idx
        graph = dgl.remove_self_loop(g=graph)
        self.graph = graph
        self.number_of_nodes = number_of_nodes
        self.number_of_relations = number_of_relations
        self.n_feats = n_feats
        self.special_entity_dict = special_node_dict
        self.special_relation_dict = special_relation_dict
        self.train_batch_size = self.config.train_batch_size
        self.val_batch_size = self.config.eval_batch_size
        self.edge_dir = self.config.sub_graph_edge_dir
        self.self_loop = self.config.sub_graph_self_loop
        self.fanouts = [int(_) for _ in self.config.sub_graph_fanouts.split(',')]
        self.restart_prob = self.config.rand_walk_restart_prob

    def data_loader(self):
        dataset = SubGraphPairDataset(graph=self.graph, nentity=self.number_of_nodes,
                                      nrelation=self.number_of_relations,
                                      restart_prob=self.restart_prob,
                                      special_entity2id=self.special_entity_dict,
                                      special_relation2id=self.special_relation_dict,
                                      graph_type=self.graph_type,
                                      edge_dir=self.edge_dir, self_loop=self.self_loop,
                                      fanouts=self.fanouts)
        batch_size = self.train_batch_size
        shuffle = True
        if PROJECT_FOLDER == '/Users/wangguangtao/PycharmProjects/EffAttnGNN':
            data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
                                     collate_fn=SubGraphPairDataset.collate_fn, num_workers=0)
        else:
            data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
                                     collate_fn=SubGraphPairDataset.collate_fn, num_workers=self.config.cpu_num)
        return data_loader


class NodeClassificationSubGraphDataset(Dataset):
    """
    Graph representation learning with node labels
    """

    def __init__(self, graph: DGLHeteroGraph, nentity: int, nrelation: int, fanouts: list,
                 special_entity2id: dict, special_relation2id: dict, data_type: str, graph_type: str,
                 bi_directed: bool = True, self_loop: bool = False, edge_dir: str = 'in',
                 node_split_idx: dict = None, cls: bool = True):
        assert len(fanouts) > 0 and (data_type in {'train', 'valid', 'test'})
        assert graph_type in {'citation', 'ogb'}
        self.fanouts = fanouts  # list of int == number of hops for sampling
        self.hop_num = len(fanouts)
        assert self.hop_num >= 2
        self.g = graph
        self.cls = cls
        #####################
        if graph_type == 'ogb':
            assert node_split_idx is not None
            self.len, self.data_node_ids = ogb_train_valid_test(node_split_idx=node_split_idx, data_type=data_type)
        elif graph_type == 'citation':
            self.len, self.data_node_ids = citation_train_valid_test(graph=graph, data_type=data_type)
        else:
            raise 'Graph type = {} is not supported'.format(graph_type)
        assert self.len > 0
        #####################
        self.nentity, self.nrelation = nentity, nrelation
        self.bi_directed = bi_directed
        self.edge_dir = edge_dir  # "in", "out"
        self.self_loop = self_loop
        self.special_entity2id, self.special_relation2id = special_entity2id, special_relation2id

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        node_idx = self.data_node_ids[idx]
        anchor_node_ids = torch.LongTensor([node_idx])
        samp_hop_num = random.randint(2, self.hop_num + 1)  ## sample sub-graph from 2-hop to k-hop (k >= 2)
        samp_fanouts = self.fanouts[:samp_hop_num]
        cls_node_ids = torch.LongTensor([self.special_entity2id['cls']])
        subgraph, parent2sub_dict, _ = \
            anchor_node_sub_graph_extractor(graph=self.g,
                                            anchor_node_ids=anchor_node_ids,
                                            cls_node_ids=cls_node_ids,
                                            fanouts=samp_fanouts,
                                            edge_dir=self.edge_dir,
                                            special_relation2id=self.special_relation2id,
                                            self_loop=self.self_loop,
                                            bi_directed=self.bi_directed,
                                            cls=self.cls)
        sub_anchor_id = parent2sub_dict[node_idx.data.item()]
        class_label = self.g.ndata['label'][node_idx]
        return subgraph, class_label, sub_anchor_id

    @staticmethod
    def collate_fn(data):
        assert len(data[0]) == 3
        batch_graph_cls = torch.as_tensor([_[0].number_of_nodes() for _ in data], dtype=torch.long)
        batch_graph_cls = torch.cumsum(batch_graph_cls, dim=0) - 1
        batch_graphs = dgl.batch([_[0] for _ in data])
        batch_label = torch.as_tensor([_[1].data.item() for _ in data], dtype=torch.long)
        # ++++++++++++++++++++++++++++++++++++++++
        batch_anchor_id = torch.zeros(len(data), dtype=torch.long)
        for idx, _ in enumerate(data):
            if idx == 0:
                batch_anchor_id[idx] = _[2]
            else:
                batch_anchor_id[idx] = _[2] + batch_graph_cls[idx - 1].data.item() + 1
        # +++++++++++++++++++++++++++++++++++++++
        return {'batch_graph': (batch_graphs, batch_graph_cls, batch_anchor_id), 'batch_label': batch_label}


class NodeClassificationDataHelper(object):
    def __init__(self, config):
        self.config = config
        self.graph_type = self.config.graph_type
        if self.graph_type == 'citation':
            graph, number_of_nodes, number_of_relations, n_classes, n_feats, special_node_dict, special_relation_dict = \
                citation_k_hop_graph_reconstruction(dataset=self.config.citation_node_name,
                                                    hop_num=self.config.sub_graph_hop_num, rand_split=False,
                                                    oon=self.config.oon_type, cls=True)
            self.node_split_idx = None
        else:
            graph, number_of_nodes, number_of_relations, n_classes, n_feats, special_node_dict, \
            special_relation_dict, node_split_idx = ogb_k_hop_graph_reconstruction(dataset=self.config.ogb_node_name,
                                                                                   hop_num=self.config.sub_graph_hop_num,
                                                                                   oon=self.config.oon_type, cls=True)
            self.node_split_idx = node_split_idx
        graph = dgl.remove_self_loop(g=graph)
        graph = graph.int().to(self.config.device)
        self.graph = graph
        self.number_of_nodes = number_of_nodes
        self.number_of_relations = number_of_relations
        self.num_class = n_classes
        self.n_feats = n_feats
        self.node_features = self.graph.ndata.pop('feat')
        self.special_entity_dict = special_node_dict
        self.special_relation_dict = special_relation_dict
        self.train_batch_size = self.config.train_batch_size
        self.val_batch_size = self.config.eval_batch_size
        self.edge_dir = self.config.sub_graph_edge_dir
        self.self_loop = self.config.sub_graph_self_loop
        self.fanouts = [int(_) for _ in self.config.sub_graph_fanouts.split(',')]
        self.graph_augmentation = self.config.graph_augmentation

    def data_loader(self, data_type):
        assert data_type in {'train', 'valid', 'test'}
        dataset = NodeClassificationSubGraphDataset(graph=self.graph, nentity=self.number_of_nodes,
                                                    nrelation=self.number_of_relations,
                                                    special_entity2id=self.special_entity_dict,
                                                    special_relation2id=self.special_relation_dict,
                                                    data_type=data_type, graph_type=self.graph_type,
                                                    edge_dir=self.edge_dir, self_loop=self.self_loop,
                                                    fanouts=self.fanouts, cls=True,
                                                    node_split_idx=self.node_split_idx)
        if data_type in {'train'}:
            batch_size = self.train_batch_size
            shuffle = True
        else:
            batch_size = self.val_batch_size
            shuffle = False
        if PROJECT_FOLDER == '/Users/wangguangtao/PycharmProjects/EffAttnGNN':
            data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
                                     collate_fn=NodeClassificationSubGraphDataset.collate_fn)
        else:
            data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
                                     collate_fn=NodeClassificationSubGraphDataset.collate_fn,
                                     num_workers=self.config.cpu_num)
        return data_loader


class NodeClassificationSubGraphAugmentationDataset(Dataset):
    """
    Graph representation learning with node labels with data augmentation
    """

    def __init__(self, graph: DGLHeteroGraph, nentity: int, nrelation: int, fanouts: list,
                 special_entity2id: dict, special_relation2id: dict, data_type: str, graph_type: str,
                 bi_directed: bool = True, self_loop: bool = False, edge_dir: str = 'in',
                 node_split_idx: dict = None, cls: bool = True):
        assert len(fanouts) > 0 and (data_type in {'train', 'valid', 'test'})
        assert graph_type in {'citation', 'ogb'}
        self.fanouts = fanouts  # list of int == number of hops for sampling
        self.hop_num = len(fanouts)
        assert self.hop_num >= 2
        self.g = graph
        self.cls = cls
        #####################
        if graph_type == 'ogb':
            assert node_split_idx is not None
            self.len, self.data_node_ids = ogb_train_valid_test(node_split_idx=node_split_idx, data_type=data_type)
        elif graph_type == 'citation':
            self.len, self.data_node_ids = citation_train_valid_test(graph=graph, data_type=data_type)
        else:
            raise 'Graph type = {} is not supported'.format(graph_type)
        assert self.len > 0
        #####################
        self.nentity, self.nrelation = nentity, nrelation
        self.bi_directed = bi_directed
        self.edge_dir = edge_dir  # "in", "out"
        self.self_loop = self_loop
        self.special_entity2id, self.special_relation2id = special_entity2id, special_relation2id

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        node_idx = self.data_node_ids[idx]
        anchor_node_ids = torch.LongTensor([node_idx])
        samp_hop_num = random.randint(2, self.hop_num + 1)
        samp_fanouts = self.fanouts[:samp_hop_num]
        cls_node_ids = torch.LongTensor([self.special_entity2id['cls']])
        subgraph, parent2sub_dict, _ = \
            anchor_node_sub_graph_extractor(graph=self.g,
                                            anchor_node_ids=anchor_node_ids,
                                            cls_node_ids=cls_node_ids,
                                            fanouts=samp_fanouts,
                                            edge_dir=self.edge_dir,
                                            special_relation2id=self.special_relation2id,
                                            self_loop=self.self_loop,
                                            bi_directed=self.bi_directed,
                                            cls=self.cls)
        sub_anchor_id = parent2sub_dict[node_idx.data.item()]
        class_label = self.g.ndata['label'][node_idx]
        return subgraph, class_label, sub_anchor_id

    @staticmethod
    def collate_fn(data):
        assert len(data[0]) == 3
        batch_graph_cls = torch.as_tensor([_[0].number_of_nodes() for _ in data], dtype=torch.long)
        batch_graph_cls = torch.cumsum(batch_graph_cls, dim=0) - 1
        batch_graphs = dgl.batch([_[0] for _ in data])
        batch_label = torch.as_tensor([_[1].data.item() for _ in data], dtype=torch.long)
        # ++++++++++++++++++++++++++++++++++++++++
        batch_anchor_id = torch.zeros(len(data), dtype=torch.long)
        for idx, _ in enumerate(data):
            if idx == 0:
                batch_anchor_id[idx] = _[2]
            else:
                batch_anchor_id[idx] = _[2] + batch_graph_cls[idx - 1].data.item() + 1
        # +++++++++++++++++++++++++++++++++++++++
        return {'batch_graph': (batch_graphs, batch_graph_cls, batch_anchor_id), 'batch_label': batch_label}

