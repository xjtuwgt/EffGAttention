import numpy as np
import time
import torch
from torch.optim import Adam
from tqdm import tqdm, trange
from codes.default_argparser import default_parser, complete_default_parser
from transformers.optimization import get_cosine_schedule_with_warmup
from graph_data.graph_dataloader import NodeClassificationSubGraphDataHelper
from codes.simsiam_networks import SimSiamNodeClassification
import logging
from codes.utils import seed_everything
from codes.utils import citation_hyper_parameter_space, citation_random_search_hyper_tunner

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(model, data_helper, data_type, args):
    data_loader = data_helper.data_loader(data_type=data_type)
    epoch_iterator = tqdm(data_loader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    model.eval()
    correct_pred = 0.0
    total_examples = 0.0
    for batch_idx, batch in enumerate(epoch_iterator):
        batch = subgraph_batch_to_device(batch=batch, device=args.device)
        batch_graphs, batch_labels = batch['batch_graph'], batch['batch_label']
        batch_logits = model(batch_graphs)
        batch_size = batch_logits.shape[0]
        total_examples = total_examples + batch_size
        _, indices = torch.max(batch_logits, dim=1)
        correct = torch.sum(indices == batch_labels).data.item()
        correct_pred = correct_pred + correct
    return correct_pred * 1.0 / total_examples


def subgraph_batch_to_device(batch: dict, device):
    batch_graphs = batch['batch_graph']
    batch_labels = batch['batch_label']
    batch_graphs = [_.to(device) for _ in batch_graphs]
    batch_labels = batch_labels.to(device)
    batch_to = {'batch_graph': batch_graphs, 'batch_label': batch_labels}
    return batch_to


def model_train(model, data_helper, optimizer, scheduler, args):
    dur = []
    best_val_acc = 0.0
    best_test_acc = 0.0
    t0 = time.time()
    patience_count = 0
    loss_fcn = torch.nn.CrossEntropyLoss()
    loss = 0.0
    torch.autograd.set_detect_anomaly(True)
    # **********************************************************************************
    start_epoch = 0
    train_data_loader = data_helper.data_loader(data_type='train')
    logging.info('Loading training data = {} completed'.format(len(train_data_loader)))
    # **********************************************************************************
    logging.info('Starting training the model...')
    train_iterator = trange(start_epoch, start_epoch + int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    logging.info('*' * 75)
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_data_loader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = subgraph_batch_to_device(batch=batch, device=args.device)
            batch_graphs, batch_labels = batch['batch_graph'], batch['batch_label']

            batch_logits = model(batch_graphs)
            loss = loss_fcn(batch_logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
        val_acc = evaluate(model=model, args=args, data_type='valid', data_helper=data_helper)
        logging.info('validation accuracy = {}, Train loss = {} after Epoch'.format(val_acc, loss, epoch))
    return best_val_acc, best_test_acc


def main(args):
    args = complete_default_parser(args=args)
    data_helper = NodeClassificationSubGraphDataHelper(config=args)
    args.num_classes = data_helper.num_class
    node_features = data_helper.node_features  #
    args.num_entities = data_helper.number_of_nodes
    args.node_emb_dim = data_helper.n_feats
    args.num_relations = data_helper.number_of_relations
    num_of_experiments = args.exp_number
    hyper_search_space = citation_hyper_parameter_space()
    acc_list = []
    search_best_test_acc = 0.0
    search_best_val_acc = 0.0
    search_best_settings = None

    for _ in range(num_of_experiments):
        # args, hyper_setting_i = citation_random_search_hyper_tunner(args=args, search_space=hyper_search_space,
        #                                                             seed=args.seed + 1)
        logging.info('Model Hyper-Parameter Configuration:')
        for key, value in vars(args).items():
            logging.info('Hyper-Para {}: {}'.format(key, value))
        logging.info('*' * 75)
        seed_everything(seed=args.seed)
        # create model
        model = SimSiamNodeClassification(config=args)
        model.init_graph_ember(ent_emb=node_features, ent_freeze=True)
        model.to(args.device)
        # ++++++++++++++++++++++++++++++++++++
        logging.info('Model Parameter Configuration:')
        for name, param in model.named_parameters():
            logging.info('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()),
                                                                      str(param.requires_grad)))
        logging.info('*' * 75)
        # create optimizer and scheduler
        optimizer = Adam(params=model.parameters(), lr=args.fine_tuned_learning_rate,
                         weight_decay=args.fine_tuned_weight_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=10,
                                                    num_training_steps=args.num_train_epochs)
        # start model training
        model_train(model=model, data_helper=data_helper, optimizer=optimizer, scheduler=scheduler, args=args)
    #     acc_list.append((hyper_setting_i, test_acc, best_val_acc, best_test_acc))
    #     logger.info('*' * 50)
    #     logger.info('{}\t{:.4f}\t{:.4f}\t{:.4f}'.format(hyper_setting_i, test_acc, best_val_acc, best_test_acc))
    #     logger.info('*' * 50)
    #     if search_best_val_acc < best_val_acc:
    #         search_best_val_acc = best_val_acc
    #         search_best_test_acc = best_test_acc
    #         search_best_settings = hyper_setting_i
    #     logger.info('Current best testing acc = {:.4f} and best dev acc = {}'.format(search_best_test_acc,
    #                                                                                  search_best_val_acc))
    #     logger.info('*' * 30)
    # for _, setting_acc in enumerate(acc_list):
    #     print(_, setting_acc)
    # print(search_best_test_acc)
    # print(search_best_settings)


if __name__ == '__main__':
    main(default_parser().parse_args())
