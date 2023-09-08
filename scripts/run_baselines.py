# this script runs experiments on specified dataset with specified model
# using baseline mechanisms including RR, LDPGCN (LDP variant of DPGCN, SP 22) and Solitude (TIFS 22).
import sys

import torch
sys.path.append('../src')
import argparse
from datetime import datetime
from itertools import product
import json
import logging
import math
import sys
import ldpgcn
import solitude
import rr
import dprr
import ldpgen
import symrr
from data import make_dataset

# setup parser
parser = argparse.ArgumentParser(description='Start baseline experiment with specified dataset and model.')
parser.add_argument("dataset", type=str, help="Dataset, one of 'cora', 'citeseer','lastfm' or 'facebook'.")
parser.add_argument("model", type=str, help="Model name, 'gcn', 'graphsage' or 'gat'.")
parser.add_argument("--method", nargs='*', type=str, help="Specify what baseline methods to run, list of 'rr', 'ldpgcn', 'solitude', 'dprr', 'ldpgen', 'symrr'.")
parser.add_argument("--grid_search", action="store_true")
parser.add_argument("--use_dense_model", action="store_true")
parser.add_argument("--eps", nargs='*', type=int, help="Specify what epsilons (integers only) to run with.")
args = parser.parse_args()

dataset_name = args.dataset
model_name = args.model
grid_search = args.grid_search
use_dense_model = args.use_dense_model
method = args.method

mechanisms = ["rr", "ldpgcn", "solitude", "dprr", "ldpgen", "symrr"]

# if specified only run specified method
if method == None or len(method) == 0:
    pass
else:
    mechanisms = method

# setup logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f"log/baselines/bl_{dataset_name}_{model_name}_{'_'.join(mechanisms)}_"+datetime.now().strftime('%y%m%d_%H%M%S.txt')),
        logging.StreamHandler(sys.stdout)
    ])
logging.info(f"Start experiments with {args}")

# download and load dataset
graph = make_dataset(dataset_name, root="../data")
linkless_graph = graph.clone()
linkless_graph.edge_index = None

eps_list = [1,2,3,4,5,6,7,8]
if args.eps == None or len(args.eps) == 0:
    pass
else:
    eps_list = args.eps

run_experiment = {
    "rr": rr.run_rr,
    "ldpgcn": ldpgcn.run_ldpgcn,
    "solitude": solitude.run_solitude,
    "dprr": dprr.run_dprr,
    "ldpgen": ldpgen.run_ldpgen,
    "symrr": symrr.run_symrr
}


# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')
if device.type == 'cuda':
    logging.info(f"{torch.cuda.get_device_properties(0)}")

if grid_search:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    logging.info(f"Grid search. Load hyperparameter space from config.json")
    with open("fb_config.json") as f:
        conf = json.load(f)
    
    hp_space = {
        "rr": conf["non_private_hparam_space"],
        "ldpgcn": conf["non_private_hparam_space"],
        "solitude": conf["solitude_hparam_space"],
        "dprr": conf["non_private_hparam_space"],
        "ldpgen": conf["non_private_hparam_space"],
        "symrr": conf["non_private_hparam_space"]
    }

    hp_list = {m:[dict(zip(hp_space[m].keys(), values)) for values in product(*hp_space[m].values())] for m in mechanisms}
    
    for m in mechanisms:
        logging.info(f"[{m}: {model_name} on {dataset_name}] Grid search for hyperparameter tuning on various epsilons.")

        for eps in eps_list:
            logging.info(f"[{m}: {model_name} on {dataset_name} with eps={eps}] Start grid search for hyperparameter tuning.")
            min_val_loss = math.inf
            best_hp = None

            for hp in hp_list[m]:
                val_loss, _ = run_experiment[m](graph, linkless_graph, model_name, eps, hp, 5, use_dense_model=use_dense_model)
                if val_loss.mean() < min_val_loss:
                    min_val_loss = val_loss.mean()
                    best_hp = hp
            
            logging.info(f"[{m}: {model_name} on {dataset_name} with eps={eps}] Best hparam is: {best_hp} with validation loss {min_val_loss}")
            logging.info(f"[{m}: {model_name} on {dataset_name} with eps={eps}] Saving best hp to output/bl_best_hp.json")
            with open("output/bl_best_hp.json") as f:
                best_hp_dict = json.load(f)
            
            if dataset_name not in best_hp_dict:
                best_hp_dict[dataset_name] = {}
            if model_name not in best_hp_dict[dataset_name]:
                best_hp_dict[dataset_name][model_name] = {}
            if m not in best_hp_dict[dataset_name][model_name]:
                best_hp_dict[dataset_name][model_name][m] = {}
            
            best_hp_dict[dataset_name][model_name][m][str(eps)] = best_hp
            with open("output/bl_best_hp.json", "w") as fp:
                json.dump(best_hp_dict, fp, indent=2)
        
        logging.info(f"[{m}: {model_name} on {dataset_name}] Grid search done.")
    logging.info("Grid search done!")

logging.info(f"Run baseline experiments using found hyperparameters in bl_best_hp.json.")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

with open("output/bl_best_hp.json", "r") as f:
    best_hp = json.load(f)

for m in mechanisms:
    logging.info(f"[{m}: {model_name} on {dataset_name}] Start running experiments on various epsilons.")
    for eps in eps_list:
        hp = best_hp[dataset_name][model_name][m][str(eps)]
        logging.info(f"[{m}: {model_name} on {dataset_name} with eps={eps}] Run with best hp found: {hp}.")
        _, acc = run_experiment[m](graph, linkless_graph, model_name, eps, hp, 30, use_dense_model=use_dense_model)
        logging.info(f"[{m}: {model_name} on {dataset_name} with eps={eps}] Test accuracy is {acc.mean()} ({acc.std()}).")
        logging.info(f"[{m}: {model_name} on {dataset_name} with eps={eps}] Saving training results to output/bl_results.json")
        with open("output/bl_results.json") as f:
            acc_dict = json.load(f)
        if dataset_name not in acc_dict:
            acc_dict[dataset_name] = {}
        if model_name not in acc_dict[dataset_name]:
            acc_dict[dataset_name][model_name] = {}
        if m not in acc_dict[dataset_name][model_name]:
            acc_dict[dataset_name][model_name][m] = {}
        acc_dict[dataset_name][model_name][m][str(eps)] = [acc.mean(), acc.std()]
        with open('output/bl_results.json', 'w') as fp:
            json.dump(acc_dict, fp, indent=2)
    logging.info(f"[{m}: {model_name} on {dataset_name}] Experiments done.")

logging.info(f"All baseline experiments done!")
