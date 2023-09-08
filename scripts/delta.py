import argparse
import json
import logging
import sys
from datetime import datetime
import torch

from torch_sparse import SparseTensor

sys.path.append('../src')
from blink import *
from data import make_dataset

parser = argparse.ArgumentParser(description='Run experiments on the effect of delta.')
parser.add_argument("dataset", type=str, help="Dataset, one of 'cora', 'citeseer', 'lastfm' or 'facebook.")
parser.add_argument("model", type=str, help="Model name, 'gcn', 'graphsage' or 'gat'.")
parser.add_argument("variant", type=str, help="variant, 'hard', 'soft' or 'hybrid'.")
args = parser.parse_args()

dataset_name = args.dataset
model_name = args.model
variant_name = args.variant
if variant_name == "hard":
    variant = 0
elif variant_name == "soft":
    variant = 1
elif variant_name == "hybrid":
    variant = 2

# setup logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f"log/delta_"+datetime.now().strftime('%y%m%d_%H%M%S.txt')),
        logging.StreamHandler(sys.stdout)
    ])
logging.info(f"Experiments on the effects of delta.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')
if device.type == 'cuda':
    logging.info(f"{torch.cuda.get_device_properties(0)}")

eps_list = [1,2,3,4,5,6,7,8]
delta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results = {}

with open("output/best_hp.json", "r") as f:
    best_hp = json.load(f)

graph = make_dataset(dataset_name, root="../data")
linkless_graph = graph.clone()
linkless_graph.edge_index = None

for eps in eps_list:
    for delta in delta_list:
        hp = best_hp[variant_name][dataset_name][model_name][str(eps)]
        hp['delta'] = delta
        logging.info(f"[{model_name} on {dataset_name} with eps={eps} and delta={delta}] Start running.")
        _, acc = run_blink(graph, linkless_graph, model_name, eps, hp, 30, variant=variant)
        logging.info(f"[{model_name} on {dataset_name} with eps={eps} and delta={delta}] Test accuracy is {acc.mean()} ({acc.std()}).")

        with open("output/delta.json") as f:
            acc_dict = json.load(f)
        if variant_name not in acc_dict:
            acc_dict[variant_name] = {}
        if dataset_name not in acc_dict[variant_name]:
            acc_dict[variant_name][dataset_name] = {}
        if model_name not in acc_dict[variant_name][dataset_name]:
            acc_dict[variant_name][dataset_name][model_name] = {}
        if str(eps) not in acc_dict[variant_name][dataset_name][model_name]:
            acc_dict[variant_name][dataset_name][model_name][str(eps)] = {}
        acc_dict[variant_name][dataset_name][model_name][str(eps)][str(delta)] = [acc.mean(), acc.std()] 
        with open('output/delta.json', 'w') as fp:
            json.dump(acc_dict, fp, indent=2)