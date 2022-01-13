import logging
import os
import torch
import json
import numpy as np
from numpy import ndarray
from dgl import DGLHeteroGraph


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'),
        entity_embedding
    )

    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'),
        relation_embedding
    )


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def graph_to_triples(graph: DGLHeteroGraph, edge_rel_name: str = None):
    head_nodes, tail_nodes, edge_ids = graph.edges(form='all')
    relation2id = {}
    if edge_rel_name:
        relations = graph.edata[edge_rel_name][edge_ids]
        r_uniques = torch.unique(relations).tolist()
        relation2id = dict([(_[1], _[0]) for _ in enumerate(r_uniques)])
        relations = relations.tolist()
        relations = [relation2id[_] for _ in relations]
        n_relations = len(relation2id)
    else:
        relations = torch.zeros(edge_ids.shape, dtype=torch.long)
        relations = relations.tolist()
        relation2id[0] = 0
        n_relations = 1
    head_nodes = head_nodes.tolist()
    tail_nodes = tail_nodes.tolist()
    triples = list(zip(head_nodes, relations, tail_nodes))
    n_entities = graph.number_of_nodes()
    triples = np.array(triples)
    return triples, n_entities, n_relations, relation2id


def triple_train_valid_split(triples: ndarray, k=5, valid_ratio: float = 0.2):
    uniqueEntities, indicesList, indegrees = np.unique(triples[:, 2], return_index=True, return_counts=True)
    large_uniqueEntities = uniqueEntities[np.where(indegrees >= k, True, False)]
    # Get indices of poorly connected destination enttiies (entities with >= k edges)
    print(large_uniqueEntities.shape)
    large_triple_indices = np.in1d(triples[:, 2], large_uniqueEntities)
    print(large_triple_indices.shape)
    # train_split = all_ctups[large_triple_indices]
    # # Get length of canonical tuples after dropping poorly connected rows
    # all_ctups_len_after = len(all_ctups)
    # print(long_indices.shape)


def log_metrics(mode, step, metrics):
    """
    Print the evaluation logs
    """
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
