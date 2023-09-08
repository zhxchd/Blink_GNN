import torch
import argparse
from itertools import product
import logging
import json
import math
import sys
from datetime import datetime

sys.path.append('../src')
from blink import *
from data import make_dataset

# setup parser
parser = argparse.ArgumentParser(description='Start experiment with specified dataset and model.')
parser.add_argument("variant", type=str, help="Blink variant, one of 'soft' or 'hard' or 'hybrid'")
parser.add_argument("dataset", type=str, help="Dataset, one of 'cora', 'citeseer', 'lastfm', 'facebook' or 'twitch'.")
parser.add_argument("model", type=str, help="Model, 'mlp', 'gcn', 'graphsage' or 'gat'.")
parser.add_argument("--grid_search", action="store_true")
parser.add_argument("--eps", nargs='*', type=str, help="Specify what epsilons to run with, integers or None")
args = parser.parse_args()

if args.variant == 'hard':
    variant = 0
elif args.variant == 'soft':
    variant = 1
elif args.variant == 'hybrid':
    variant = 2
dataset_name = args.dataset
model_name = args.model
grid_search = args.grid_search

# setup logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f"log/blink/{args.variant}_{dataset_name}_{model_name}_"+datetime.now().strftime('%y%m%d_%H%M%S.txt')),
        logging.StreamHandler(sys.stdout)
    ])
logging.info(f"Start experiments with {args}")

# download and load dataset
graph = make_dataset(dataset_name, root="../data")
linkless_graph = graph.clone()
linkless_graph.edge_index = None

eps_list = [None]

if model_name != "mlp":
    eps_list = [i for i in range(1,9)] + eps_list

# argument override default choices
if args.eps == None or len(args.eps) == 0:
    pass
else:
    eps_list = [int(i) if i != "None" else None for i in args.eps]

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')
if device.type == 'cuda':
    logging.info(f"{torch.cuda.get_device_properties(0)}")

if grid_search:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logging.info(f"Grid search. Load hyperparameter space from config.json")
    with open("config.json") as f:
        conf = json.load(f)
    baseline_hparam_space = conf["non_private_hparam_space"]
    hparam_space = conf["hparam_space"]

    baseline_hparams = [dict(zip(baseline_hparam_space.keys(), values)) for values in product(*baseline_hparam_space.values())]
    hparams = [dict(zip(hparam_space.keys(), values)) for values in product(*hparam_space.values())]

    def grid_search(eps):
        logging.info(f"[Blink-{args.variant}: {model_name} on {dataset_name} with eps={eps}] Start grid search for hyperparameter tuning.")
        min_val_loss = math.inf
        best_hp = None
        
        hps = baseline_hparams if eps == None else hparams

        for hp in hps:
            val_loss, _ = run_blink(graph, linkless_graph, model_name, eps, hp, 5, variant=variant)
            if val_loss.mean() < min_val_loss:
                min_val_loss = val_loss.mean()
                best_hp = hp

        logging.info(f"[Blink-{args.variant}: {model_name} on {dataset_name} with eps={eps}] Best hparam is: {best_hp} with validation loss {min_val_loss}")
        logging.info(f"[Blink-{args.variant}: {model_name} on {dataset_name} with eps={eps}] Saving best hp to output/best_hp.json")
        with open("output/best_hp.json") as f:
            best_hp_dict = json.load(f)
        if args.variant not in best_hp_dict:
            best_hp_dict[args.variant] = {}
        if dataset_name not in best_hp_dict[args.variant]:
            best_hp_dict[args.variant][dataset_name] = {}
        if model_name not in best_hp_dict[args.variant][dataset_name]:
            best_hp_dict[args.variant][dataset_name][model_name] = {}
        best_hp_dict[args.variant][dataset_name][model_name][str(eps)] = best_hp
        with open('output/best_hp.json', 'w') as fp:
            json.dump(best_hp_dict, fp, indent=2)

    for eps in eps_list:
        grid_search(eps)
    
    logging.info(f"Grid search done!")

logging.info(f"Run experiments using found hyperparameters in best_hp.json.")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

with open("output/best_hp.json", "r") as f:
    best_hp = json.load(f)

for eps in eps_list:
    hp = best_hp[args.variant][dataset_name][model_name][str(eps)]
    logging.info(f"[Blink-{args.variant}: {model_name} on {dataset_name} with eps={eps}] Run with best hp found: {hp}.")
    _, acc = run_blink(graph, linkless_graph, model_name, eps, hp, 30, variant=variant)
    logging.info(f"[Blink-{args.variant}: {model_name} on {dataset_name} with eps={eps}] Test accuracy is {acc.mean()} ({acc.std()}).")
    logging.info(f"[Blink-{args.variant}: {model_name} on {dataset_name} with eps={eps}] Saving training results to output/results.json")

    with open("output/results.json") as f:
        acc_dict = json.load(f)
    if args.variant not in acc_dict:
        acc_dict[args.variant] = {}
    if dataset_name not in acc_dict[args.variant]:
        acc_dict[args.variant][dataset_name] = {}
    if model_name not in acc_dict[args.variant][dataset_name]:
        acc_dict[args.variant][dataset_name][model_name] = {}
    acc_dict[args.variant][dataset_name][model_name][str(eps)] = [acc.mean(), acc.std()]
    with open('output/results.json', 'w') as fp:
        json.dump(acc_dict, fp, indent=2)