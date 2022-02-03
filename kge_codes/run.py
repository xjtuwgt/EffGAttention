#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random
from kge_codes.default_kge_argparser import override_config, parse_args
from kge_codes.kge_utils import set_logger

import numpy as np
import torch

from torch.utils.data import DataLoader
from model import KGEModel
from kge_dataloader import TrainDataset
from kge_dataloader import BidirectionalOneShotIterator

        
def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Write logs to checkpoint and console
    set_logger(args)
    
#     with open(os.path.join(args.data_path, 'entities.dict')) as fin:
#         entity2id = dict()
#         for line in fin:
#             eid, entity = line.strip().split('\t')
#             entity2id[entity] = int(eid)
#
#     with open(os.path.join(args.data_path, 'relations.dict')) as fin:
#         relation2id = dict()
#         for line in fin:
#             rid, relation = line.strip().split('\t')
#             relation2id[relation] = int(rid)
#
#     # Read regions for Countries S* datasets
#     if args.countries:
#         regions = list()
#         with open(os.path.join(args.data_path, 'regions.list')) as fin:
#             for line in fin:
#                 region = line.strip()
#                 regions.append(entity2id[region])
#         args.regions = regions
#
#     nentity = len(entity2id)
#     nrelation = len(relation2id)
#
#     args.nentity = nentity
#     args.nrelation = nrelation
#
#     logging.info('Model: %s' % args.model)
#     logging.info('Data Path: %s' % args.data_path)
#     logging.info('#entity: %d' % nentity)
#     logging.info('#relation: %d' % nrelation)
#
#     train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
#     logging.info('#train: %d' % len(train_triples))
#     valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
#     logging.info('#valid: %d' % len(valid_triples))
#     test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
#     logging.info('#test: %d' % len(test_triples))
#
#     #All true triples
#     all_true_triples = train_triples + valid_triples + test_triples
#
#     kge_model = KGEModel(
#         model_name=args.model,
#         nentity=nentity,
#         nrelation=nrelation,
#         hidden_dim=args.hidden_dim,
#         gamma=args.gamma,
#         double_entity_embedding=args.double_entity_embedding,
#         double_relation_embedding=args.double_relation_embedding
#     )
#
#     logging.info('Model Parameter Configuration:')
#     for name, param in kge_model.named_parameters():
#         logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
#
#     if args.cuda:
#         kge_model = kge_model.cuda()
#
#     if args.do_train:
#         # Set training dataloader iterator
#         train_dataloader_head = DataLoader(
#             TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'),
#             batch_size=args.batch_size,
#             shuffle=True,
#             num_workers=max(1, args.cpu_num//2),
#             collate_fn=TrainDataset.collate_fn
#         )
#
#         train_dataloader_tail = DataLoader(
#             TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'),
#             batch_size=args.batch_size,
#             shuffle=True,
#             num_workers=max(1, args.cpu_num//2),
#             collate_fn=TrainDataset.collate_fn
#         )
#
#         train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
#
#         # Set training configuration
#         current_learning_rate = args.learning_rate
#         optimizer = torch.optim.Adam(
#             filter(lambda p: p.requires_grad, kge_model.parameters()),
#             lr=current_learning_rate
#         )
#         if args.warm_up_steps:
#             warm_up_steps = args.warm_up_steps
#         else:
#             warm_up_steps = args.max_steps // 2
#
#     if args.init_checkpoint:
#         # Restore model from checkpoint directory
#         logging.info('Loading checkpoint %s...' % args.init_checkpoint)
#         checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
#         init_step = checkpoint['step']
#         kge_model.load_state_dict(checkpoint['model_state_dict'])
#         if args.do_train:
#             current_learning_rate = checkpoint['current_learning_rate']
#             warm_up_steps = checkpoint['warm_up_steps']
#             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     else:
#         logging.info('Ramdomly Initializing %s Model...' % args.model)
#         init_step = 0
#
#     step = init_step
#
#     logging.info('Start Training...')
#     logging.info('init_step = %d' % init_step)
#     logging.info('batch_size = %d' % args.batch_size)
#     logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
#     logging.info('hidden_dim = %d' % args.hidden_dim)
#     logging.info('gamma = %f' % args.gamma)
#     logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
#     if args.negative_adversarial_sampling:
#         logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
#
#     # Set valid dataloader as it would be evaluated during training
#
#     if args.do_train:
#         logging.info('learning_rate = %d' % current_learning_rate)
#
#         training_logs = []
#
#         #Training Loop
#         for step in range(init_step, args.max_steps):
#
#             log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
#
#             training_logs.append(log)
#
#             if step >= warm_up_steps:
#                 current_learning_rate = current_learning_rate / 10
#                 logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
#                 optimizer = torch.optim.Adam(
#                     filter(lambda p: p.requires_grad, kge_model.parameters()),
#                     lr=current_learning_rate
#                 )
#                 warm_up_steps = warm_up_steps * 3
#
#             if step % args.save_checkpoint_steps == 0:
#                 save_variable_list = {
#                     'step': step,
#                     'current_learning_rate': current_learning_rate,
#                     'warm_up_steps': warm_up_steps
#                 }
#                 save_model(kge_model, optimizer, save_variable_list, args)
#
#             if step % args.log_steps == 0:
#                 metrics = {}
#                 for metric in training_logs[0].keys():
#                     metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
#                 log_metrics('Training average', step, metrics)
#                 training_logs = []
#
#             if args.do_valid and step % args.valid_steps == 0:
#                 logging.info('Evaluating on Valid Dataset...')
#                 metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
#                 log_metrics('Valid', step, metrics)
#
#         save_variable_list = {
#             'step': step,
#             'current_learning_rate': current_learning_rate,
#             'warm_up_steps': warm_up_steps
#         }
#         save_model(kge_model, optimizer, save_variable_list, args)
#
#     if args.do_valid:
#         logging.info('Evaluating on Valid Dataset...')
#         metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
#         log_metrics('Valid', step, metrics)
#
#     if args.do_test:
#         logging.info('Evaluating on Test Dataset...')
#         metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
#         log_metrics('Test', step, metrics)
#
#     if args.evaluate_train:
#         logging.info('Evaluating on Training Dataset...')
#         metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
#         log_metrics('Test', step, metrics)
#
# if __name__ == '__main__':
#     main(parse_args())
