import random
import os
import numpy as np
import torch
import dgl
IGNORE_IDX = -100


def seed_everything(seed: int) -> int:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    dgl.random.seed(seed)
    dgl.seed(seed)
    return seed


def rand_search_parameter(space: dict):
    para_type = space['type']
    if para_type == 'fixed':
        return space['value']
    if para_type == 'choice':
        candidates = space['values']
        value = np.random.choice(candidates, 1).tolist()[0]
        return value
    if para_type == 'range':
        log_scale = space.get('log_scale', False)
        low, high = space['bounds']
        if log_scale:
            value = np.random.uniform(low=np.log(low), high=np.log(high), size=1)[0]
            value = np.exp(value)
        else:
            value = np.random.uniform(low=low, high=high, size=1)[0]
        return value
    else:
        raise ValueError('Training batch mode %s not supported' % para_type)


def Citation_HypeParameterSpace():
    learning_rate = {'name': 'learning_rate', 'type': 'choice', 'values': [5e-4, 1e-3]}
    attn_drop_ratio = {'name': 'attn_drop_ratio', 'type': 'choice', 'values': [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]}
    feat_drop_ratio = {'name': 'feat_drop_ratio', 'type': 'choice', 'values': [0.25, 0.35, 0.45]}
    edge_drop_ratio = {'name': 'edge_drop_ratio', 'type': 'choice', 'values': [0.0, 0.05, 0.1, 0.15]}
    hop_num = {'name': 'hop_num', 'type': 'choice', 'values': [4, 6, 8, 10]}
    alpha = {'name': 'alpha', 'type': 'choice', 'values': [0.05, 0.1, 0.15]}
    hidden_dim = {'name': 'hidden_dim', 'type': 'choice', 'values': [256]}
    layer_num = {'name': 'layer_num', 'type': 'choice', 'values': [3]}
    # ++++++++++++++++++++++++++++++++++
    search_space = [learning_rate, attn_drop_ratio, feat_drop_ratio, edge_drop_ratio, hidden_dim, hop_num, alpha,
                    layer_num]
    search_space = dict((x['name'], x) for x in search_space)
    return search_space


def single_task_trial(search_space: dict, rand_seed: int):
    seed_everything(seed=rand_seed)
    parameter_dict = {}
    for key, value in search_space.items():
        parameter_dict[key] = rand_search_parameter(value)
    parameter_dict['seed'] = rand_seed
    return parameter_dict


def citation_random_search_hyper_tunner(args, search_space: dict, seed: int):
    parameter_dict = single_task_trial(search_space=search_space, rand_seed=seed)
    args.learning_rate = parameter_dict['learning_rate']
    args.feat_drop = parameter_dict['feat_drop_ratio']
    args.attn_drop = parameter_dict['attn_drop_ratio']
    args.edge_drop = parameter_dict['edge_drop_ratio']
    args.layers = parameter_dict['layer_num']
    args.gnn_hop_num = parameter_dict['hop_num']
    args.alpha = parameter_dict['alpha']
    args.hidden_dim = parameter_dict['hidden_dim']
    args.seed = parameter_dict['seed']
    return args, parameter_dict