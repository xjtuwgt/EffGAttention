import numpy as np
import time
import torch
from torch.optim import Adam
from tqdm import tqdm, trange
from codes.default_argparser import default_parser, complete_default_parser
from graph_data.citation_graph_data import citation_k_hop_graph_reconstruction
from transformers.optimization import get_cosine_schedule_with_warmup
from graph_data.graph_dataloader import NodeClassificationDataHelper
from codes.simsiam_networks import SimSiamNodeClassification
import logging
from codes.utils import seed_everything
from codes.utils import citation_hyper_parameter_space, citation_random_search_hyper_tunner

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(logits, labels, debug=False):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    if debug:
        return correct.item() * 1.0 / len(labels), indices, labels
    return correct.item() * 1.0 / len(labels)


def evaluate(model, data_helper, data_type, debug=False, loss=False):
    return


def model_train(model, data_helper, optimizer, scheduler, args):
    dur = []
    best_val_acc = 0.0
    best_test_acc = 0.0
    t0 = time.time()
    patience_count = 0
    loss_fcn = torch.nn.CrossEntropyLoss()
    torch.autograd.set_detect_anomaly(True)

    # **********************************************************************************
    start_epoch = 0
    args.num_train_epochs = 1
    # **********************************************************************************
    logging.info('Starting training the model...')
    train_iterator = trange(start_epoch, start_epoch + int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    train_data_loader = data_helper.data_loader(data_type='train')
    logging.info('Loading training data = {} completed'.format(len(train_data_loader)))
    logging.info('*' * 75)

    for epoch in range(args.num_train_epochs):
        epoch_iterator = tqdm(train_data_loader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch_graphs, batch_labels = batch['batch_graph'], batch['batch_label']

            print(type(batch_graphs[0]), type(batch_graphs[1]), type(batch_graphs[2]))
            print(type(batch))
            print(batch.keys())
        # forward

    return best_val_acc, best_test_acc


def main(args):
    args = complete_default_parser(args=args)
    data_helper = NodeClassificationDataHelper(config=args)
    args.num_classes = data_helper.num_class
    node_features = data_helper.node_features
    args.num_entities = data_helper.number_of_nodes
    args.num_relations = data_helper.number_of_relations
    num_of_experiments = args.exp_number
    hyper_search_space = citation_hyper_parameter_space()
    acc_list = []
    search_best_test_acc = 0.0
    search_best_val_acc = 0.0
    search_best_settings = None
    for _ in range(num_of_experiments):
        args, hyper_setting_i = citation_random_search_hyper_tunner(args=args, search_space=hyper_search_space,
                                                                    seed=args.seed + 1)
        logging.info('Model Hyper-Parameter Configuration:')
        for key, value in vars(args).items():
            logging.info('Hyper-Para {}: {}'.format(key, value))
        logging.info('*' * 75)
        seed_everything(seed=args.seed)
        # create model
        model = SimSiamNodeClassification(config=args)
        model.to(args.device)
        if args.relation_encoder:
            model.init_graph_ember(ent_emb=node_features, ent_freeze=True)
        # ++++++++++++++++++++++++++++++++++++
        logging.info('Model Parameter Configuration:')
        for name, param in model.named_parameters():
            logging.info('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()),
                                                                      str(param.requires_grad)))
        logging.info('*' * 75)
        # create optimizer and scheduler
        optimizer = Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
