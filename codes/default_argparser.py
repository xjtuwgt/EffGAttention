import json
import os
import argparse
import torch
from os.path import join
from codes.utils import seed_everything
from codes.gpu_utils import get_single_free_gpu
from evens import HOME_DATA_FOLDER, OUTPUT_FOLDER


def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'


def json_to_argv(json_file):
    j = json.load(open(json_file))
    argv = []
    for k, v in j.items():
        new_v = str(v) if v is not None else None
        argv.extend(['--' + k, new_v])
    return argv


def set_seed(args):
    # +++++++++++++++++++++++
    random_seed = args.seed + args.local_rank
    # +++++++++++++++++++++++
    seed_everything(seed=random_seed)


def complete_default_parser(args):
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # set n_gpu
    # +++++++++++++++
    if HOME_DATA_FOLDER.startswith('/dfs/scratch0'):
        args.stanford = 'true'
    if args.local_rank == -1:
        if args.stanford:
            if torch.cuda.is_available():
                gpu_idx, _ = get_single_free_gpu()
                device = torch.device("cuda:{}".format(gpu_idx) if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.data_parallel:
            args.n_gpu = torch.cuda.device_count()
        else:
            args.n_gpu = 1
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    # output dir name
    if not args.exp_name:
        args.exp_name = '_'.join(['lr.' + str(args.learning_rate), 'layer.' + str(args.layers)])
    args.exp_name = os.path.join(args.output_dir, args.exp_name)
    set_seed(args)
    os.makedirs(args.exp_name, exist_ok=True)
    torch.save(args, join(args.exp_name, "training_args.bin"))
    return args


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=None,
                        help="If set, this will be used as directory name in OUTOUT folder")
    parser.add_argument("--config_file",
                        type=str,
                        default=None,
                        help="configuration file for command parser")
    parser.add_argument('--output_dir', type=str, default=OUTPUT_FOLDER,
                        help='Directory to save model and summaries')
    parser.add_argument('--data_path', type=str, default=HOME_DATA_FOLDER)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--citation_name', type=str, default='cora')
    parser.add_argument('--ogb_node_name', type=str, default='ogbn-arxiv')
    parser.add_argument('--graph_type', type=str, default='citation', choices=["citation", "ogb"])
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--node_emb_dim', type=int, default=300)
    parser.add_argument('--num_classes', type=int, default=-1)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--in_feat_drop', type=float, default=0.35)
    parser.add_argument('--feat_drop', type=float, default=0.35)
    parser.add_argument('--attn_drop', type=float, default=0.35)
    parser.add_argument('--residual', type=boolean_string, default='true')
    parser.add_argument('--ppr_diff', type=boolean_string, default='true')
    parser.add_argument('--stanford', type=boolean_string, default='false')
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--gnn_hop_num', type=int, default=6)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--head_num', type=int, default=8)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--negative_slope', type=float, default=0.2)
    parser.add_argument('--top_p', type=float, default=0.75)
    parser.add_argument('--top_k', type=int, default=4)
    parser.add_argument('--sparse_mode', type=str, default='no_sparse', choices=['top_k', 'top_p', 'no_sparse'])
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--per_gpu_pretrain_batch_size', type=int, default=16)
    parser.add_argument('--train_batch_size', type=int, default=16)
    # Environment+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--cpu_num', type=int, default=8)
    parser.add_argument("--data_parallel", default='false', type=boolean_string, help="use graph_data parallel or not")
    parser.add_argument("--gpu_id", default=None, type=str, help="GPU id")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--gpu", type=int, default=-1, help="which GPU to use. Set -1 to use CPU.")
    # learning and log ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--seed', type=int, default=45, help="random seed for initialization")
    parser.add_argument("--num_train_epochs", default=300, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=0.002, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=5e-4, type=float,
                        help="Weight deay if we apply some.")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument("--fine_tuned_learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--fine_tuned_weight_decay", default=1e-8, type=float,
                        help="Weight deay if we apply some.")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--total_pretrain_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=40,
                        help="Log every X updates steps.")
    parser.add_argument('--eval_interval_ratio', type=float, default=0.25,
                        help="evaluate every X updates steps.")

    parser.add_argument("--optimizer", type=str, default="Adam", choices=["AdamW", "Adam"],
                        help="Choose the optimizer to use. Default RecAdam.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restart"],
                        help="Choose the optimizer to use. Default RecAdam.")
    parser.add_argument("--debug", type=boolean_string, default='false')
    parser.add_argument('--early-stop', default='true', type=boolean_string,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--patience', default=100, type=int, help="patience to stop training")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    return parser
