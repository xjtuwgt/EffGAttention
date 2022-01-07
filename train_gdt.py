import numpy as np
import time
import torch
from codes.gdt_encoder import GDTEncoder
from torch.optim import AdamW
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
    for epoch in range(args.num_train_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(g, features)
        train_mask = label_mask_drop(train_mask=train_mask_backup, drop_ratio=0.1)
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
    logger.info("Final Test Accuracy {:.4f} | Best ValAcc {:.4f} | Best TestAcc {:.4f} |".format(test_acc, best_val_acc,
                                                                                           best_test_acc))
    return


def main(args):
    args = complete_default_parser(args=args)
    g, _, _, n_classes, _, special_relation_dict = \
        citation_k_hop_graph_reconstruction(dataset=args.citation_name, hop_num=5, rand_split=False)
    print(special_relation_dict)
    args.num_classes = n_classes
    args.node_emb_dim = g.ndata['feat'].shape[1]

    # if args.gpu < 0:
    #     cuda = False
    # else:
    #     cuda = True
    #     g = g.int().to(args.gpu)

    g = g.int().to(args.device)

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    n_edges = g.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    # add self loop
    # g = dgl.remove_self_loop(g)
    # g = dgl.add_self_loop(g)
    # create model
    model = GDTEncoder(config=args)
    model.to(args.device)
    print(model)
    # if cuda:
    #     model.cuda()
    # use optimizer
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=10,
                                                num_training_steps=args.num_train_epochs)
    model_train(g=g, model=model, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, features=features,
                labels=labels, optimizer=optimizer, scheduler=scheduler, args=args)


if __name__ == '__main__':
    main(default_parser().parse_args())