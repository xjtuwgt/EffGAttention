import numpy as np
import time
import torch
from codes.gdt_encoder import GraphNodeClassification as NodeClassifier
from codes.gdt_v2_encoder import GraphNodeClassification as NodeClassifierV2
from torch.optim import Adam
from codes.default_argparser import default_parser, complete_default_parser
from graph_data.citation_graph_data import citation_k_hop_graph_reconstruction
from transformers.optimization import get_cosine_schedule_with_warmup
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


def evaluate(graph, model, features, labels, mask, debug=False, loss=False):
    model.eval()
    loss_fcn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        if loss:
            valid_loss = loss_fcn(logits, labels)
            return accuracy(logits, labels, debug=debug), valid_loss
        return accuracy(logits, labels, debug=debug)


def model_train(g, model, features, labels, train_mask, val_mask, test_mask, optimizer, scheduler, args):
    dur = []
    best_val_acc = 0.0
    best_val_loss = 1e9
    best_test_acc = 0.0
    t0 = time.time()
    train_mask_backup = train_mask.clone()
    patience_count = 0
    n_edges = g.number_of_edges()
    loss_fcn = torch.nn.CrossEntropyLoss()
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.num_train_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(g, features)
        # train_mask = label_mask_drop(train_mask=train_mask_backup, drop_ratio=0.25)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])

        if args.fastmode:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
            if args.model_selection_mode == 'loss':
                val_loss = loss_fcn(logits[val_mask], labels[val_mask])
            test_acc = accuracy(logits[test_mask], labels[test_mask])
        else:
            if args.model_selection_mode == 'loss':
                val_acc, val_loss = evaluate(g, model, features, labels, val_mask, debug=False, loss=True)
            else:
                val_acc = evaluate(g, model, features, labels, val_mask, debug=False, loss=False)
            test_acc = evaluate(g, model, features, labels, test_mask)

        if args.model_selection_mode == 'accuracy':
            if best_val_acc <= val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience_count = 0
            else:
                patience_count = patience_count + 1
                if patience_count >= args.patience:
                    break
        else:
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience_count = 0
            else:
                patience_count = patience_count + 1
                if patience_count >= args.patience:
                    break
        logger.info("Epoch {:04d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
                    " ValAcc {:.4f} | B/ValAcc {:.4f} | B/TestAcc {:.4f} | ETputs (KTEPS) {:.2f}".
                    format(epoch, np.mean(dur), loss.item(), train_acc,
                           val_acc, best_val_acc, best_test_acc, n_edges / np.mean(dur) / 1000))

    logger.info('\n')
    test_acc, test_predictions, test_true_labels = evaluate(g, model, features, labels, test_mask, debug=True)
    logger.info("Final Test Accuracy {:.4f} | Best ValAcc {:.4f} | Best TestAcc {:.4f} |".format(test_acc,
                                                                                                 best_val_acc,
                                                                                                 best_test_acc))
    return test_acc, best_val_acc, best_test_acc


def main(args):
    args = complete_default_parser(args=args)
    g, _, n_relations, n_classes, _, _, special_relation_dict = \
        citation_k_hop_graph_reconstruction(dataset=args.citation_name, hop_num=5, rand_split=False)
    logger.info("Number of relations = {}".format(n_relations))
    args.num_classes = n_classes
    args.node_emb_dim = g.ndata['feat'].shape[1]
    g = g.int().to(args.device)
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    n_edges = g.number_of_edges()
    logger.info("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

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
        # create model
        seed_everything(seed=args.seed)
        if args.encoder_v2:
            model = NodeClassifierV2(config=args)
        else:
            model = NodeClassifier(config=args)
        model.to(args.device)
        if args.relation_encoder:
            model.init_graph_ember(ent_emb=features, ent_freeze=True)
        # ++++++++++++++++++++++++++++++++++++
        logging.info('Model Parameter Configuration:')
        for name, param in model.named_parameters():
            logging.info('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()),
                                                                      str(param.requires_grad)))
        logging.info('*' * 75)
        # ++++++++++++++++++++++++++++++++++++
        optimizer = Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=10,
                                                    num_training_steps=args.num_train_epochs)

        test_acc, best_val_acc, best_test_acc = model_train(g=g, model=model, train_mask=train_mask,
                                                            val_mask=val_mask, test_mask=test_mask,
                                                            features=features, labels=labels,
                                                            optimizer=optimizer, scheduler=scheduler,
                                                            args=args)
        acc_list.append((hyper_setting_i, test_acc, best_val_acc, best_test_acc))
        logger.info('*' * 50)
        logger.info('{}\t{:.4f}\t{:.4f}\t{:.4f}'.format(hyper_setting_i, test_acc, best_val_acc, best_test_acc))
        logger.info('*' * 50)
        if search_best_val_acc < best_val_acc:
            search_best_val_acc = best_val_acc
            search_best_test_acc = best_test_acc
            search_best_settings = hyper_setting_i
        logger.info('Current best testing acc = {:.4f} and best dev acc = {}'.format(search_best_test_acc,
                                                                                     search_best_val_acc))
        logger.info('*' * 30)
    for _, setting_acc in enumerate(acc_list):
        print(_, setting_acc)
    print(search_best_test_acc)
    print(search_best_settings)


if __name__ == '__main__':
    main(default_parser().parse_args())
