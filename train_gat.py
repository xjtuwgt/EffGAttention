"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
import dgl
from codes.gpu_utils import get_single_free_gpu
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

from baselines.gat import GAT
from baselines.utils_gat import EarlyStopping
from codes.utils import seed_everything
from graph_data.citation_graph_data import citation_train_valid_test, citation_graph_reconstruction, \
    citation_graph_rand_split_construction


def accuracy(logits, labels, debug=False):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    if debug:
        return correct.item() * 1.0 / len(labels), indices, labels
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask, debug=False):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels, debug=debug)


def main(args):
    # load and preprocess dataset
    # if args.dataset == 'cora':
    #     graph_data = CoraGraphDataset()
    # elif args.dataset == 'citeseer':
    #     graph_data = CiteseerGraphDataset()
    # elif args.dataset == 'pubmed':
    #     graph_data = PubmedGraphDataset()
    # else:
    #     raise ValueError('Unknown dataset: {}'.format(args.dataset))

    seed_everything(seed=args.rand_seed)
    if torch.cuda.is_available():
        gpu_idx, _ = get_single_free_gpu()
        device = torch.device("cuda:{}".format(gpu_idx) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g, _, _, n_classes, _ = citation_graph_reconstruction(dataset=args.dataset)
    # g, _, _,  n_classes, _ = citation_graph_rand_split_construction(dataset=args.dataset)
    # if args.gpu < 0:
    #     cuda = False
    # else:
    #     cuda = True
    #     g = g.int().to(args.gpu)

    g = g.int().to(device)

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    num_feats = features.shape[1]
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
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)
    print(model)
    model.to(device)


    # if args.early_stop:
    #     stopper = EarlyStopping(patience=100)
    # if cuda:
    #     model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    best_val_acc = 0.0
    best_test_acc = 0.0
    patience_count = 1
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])

        if args.fastmode:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
            test_acc = accuracy(logits[test_mask], labels[test_mask])
        else:
            val_acc = evaluate(model, features, labels, val_mask)
            test_acc = evaluate(model, features, labels, test_mask)

        if best_val_acc <= val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            patience_count = 0
        else:
            patience_count = patience_count + 1
            if patience_count >= args.patience:
                break

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} | Best ValAcc {:.4f} | Best TestAcc {:.4f} | ETputs(KTEPS) {:.2f}".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, best_val_acc, best_test_acc, n_edges / np.mean(dur) / 1000))

    print()
    test_acc, test_predictions, test_true_labels = evaluate(model, features, labels, test_mask, debug=True)
    print("Test Accuracy {:.4f}".format(test_acc))
    # gat_error_analysis(g, model, features, labels, val_mask, test_mask, train_mask)

def gat_error_analysis(graph, model, features, labels, val_mask, test_mask, train_mask):

    def degree_vs_correction(degrees, predictions, true_labels):
        correct_flags = predictions == true_labels
        incorrect_flags = predictions != true_labels
        correct_degrees = degrees[correct_flags]
        incorrect_degrees = degrees[incorrect_flags]
        print(correct_degrees.sort()[0])
        print(incorrect_degrees.sort()[0])

    test_acc, test_predictions, test_true_labels = evaluate(model, features, labels, test_mask, debug=True)
    print("Test Accuracy {:.4f}".format(test_acc))
    valid_acc, valid_predictions, valid_true_labels = evaluate(model, features, labels, val_mask, debug=True)
    print("Valid Accuracy {:.4f}".format(valid_acc))
    train_acc, train_predictions, train_true_labels = evaluate(model, features, labels, train_mask, debug=True)
    print("Train Accuracy {:.4f}".format(train_acc))
    print(len(test_true_labels), len(valid_true_labels), len(train_true_labels))


    in_degrees = graph.in_degrees().float()
    train_num, train_node_idx = citation_train_valid_test(graph=graph, data_type='train')
    valid_num, valid_node_idx = citation_train_valid_test(graph=graph, data_type='valid')
    test_num, test_node_idx = citation_train_valid_test(graph=graph, data_type='test')
    train_in_degrees = in_degrees[train_node_idx]
    valid_in_degrees = in_degrees[valid_node_idx]
    test_in_degrees = in_degrees[test_node_idx]

    # degree_vs_correction(degrees=valid_in_degrees, predictions=valid_predictions, true_labels=valid_true_labels)
    # degree_vs_correction(degrees=train_in_degrees, predictions=train_predictions, true_labels=train_true_labels)
    degree_vs_correction(degrees=test_in_degrees, predictions=test_predictions, true_labels=test_true_labels)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument(
        "--dataset",
        type=str,
        default='citeseer',
        help=
        "The input dataset. Can be cora, citeseer, pubmed, syn(synthetic dataset) or reddit"
    )
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=32,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.25,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.25,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--patience', default=50, type=int, help="patience to stop training")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    parser.add_argument('--rand_seed', type=int, default=42,
                        help="skip re-evaluate the validation set")

    args = parser.parse_args()
    print(args)

    main(args)