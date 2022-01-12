import numpy as np
import time
import torch
from codes.gdt_encoder import GDTEncoder
from torch.optim import Adam
from codes.default_argparser import default_parser, complete_default_parser
from graph_data.citation_graph_data import citation_k_hop_graph_reconstruction, label_mask_drop
from transformers.optimization import get_cosine_schedule_with_warmup
import logging

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


def evaluate(graph, model, features, labels, mask, debug=False):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels, debug=debug)


def model_train(g, model, features, labels, train_mask, val_mask, test_mask, optimizer, scheduler, args):
    dur = []
    best_val_acc = 0.0
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
            test_acc = accuracy(logits[test_mask], labels[test_mask])
        else:
            val_acc = evaluate(g, model, features, labels, val_mask)
            test_acc = evaluate(g, model, features, labels, test_mask)

        if best_val_acc <= val_acc:
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
    g, _, n_relations, n_classes, _, special_relation_dict = \
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

    feat_drop_ratio_list = np.arange(0.3, 0.51, 0.05).tolist()
    attn_drop_ratio_list = np.arange(0.3, 0.51, 0.05).tolist()
    edge_drop_ratio_list = [0.05, 0.1]
    lr_ratio_list = [1e-4, 2e-4, 3e-4, 5e-4, 1e-3]

    acc_list = []
    search_best_val_acc = 0.0
    search_best_test_acc = 0.0
    search_best_settings = None
    for f_dr in feat_drop_ratio_list:
        for a_dr in attn_drop_ratio_list:
            for e_dr in edge_drop_ratio_list:
                for lr in lr_ratio_list:
                    args.learning_rate = lr
                    args.feat_drop = f_dr
                    args.attn_drop = a_dr
                    args.edge_drop = e_dr
                    # create model
                    model = GDTEncoder(config=args)
                    model.to(args.device)
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
                    acc_list.append((f_dr, a_dr, lr, test_acc, best_val_acc, best_test_acc))
                    logger.info('*' * 50)
                    logger.info('{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}'.format(f_dr, a_dr, lr, test_acc, best_val_acc, best_test_acc))
                    logger.info('*' * 50)
                    if search_best_val_acc < best_val_acc:
                        search_best_val_acc = best_val_acc
                        search_best_test_acc = best_test_acc
                        search_best_settings = (f_dr, a_dr, lr, e_dr, test_acc, best_val_acc, best_test_acc)
                    logger.info('Current best testing acc = {:.4f} and best dev acc = {}'.format(search_best_test_acc,
                                                                                                 search_best_val_acc))
                    logger.info('*' * 30)
    for _, setting_acc in enumerate(acc_list):
        print(_, setting_acc)
    print(search_best_test_acc)
    print(search_best_settings)


if __name__ == '__main__':
    main(default_parser().parse_args())
