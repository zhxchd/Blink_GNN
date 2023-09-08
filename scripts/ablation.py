import argparse
import json
import logging
import sys
from datetime import datetime
import numpy as np
import torch

from torch_sparse import SparseTensor

sys.path.append('../src')
# from blink import *
from blink.server_evidence import Server as ServerEvidence
from blink.server_prior import Server as ServerPrior
from blink.server import Server as FullServer
from blink.client import Client
from data import make_dataset

parser = argparse.ArgumentParser(description='Run experiments on ablation studies.')
parser.add_argument("dataset", type=str, help="Dataset, one of 'cora', 'citeseer', 'lastfm' or 'facebook'.")
args = parser.parse_args()

dataset_name = args.dataset

# setup logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f"log/ablation_"+datetime.now().strftime('%y%m%d_%H%M%S.txt')),
        logging.StreamHandler(sys.stdout)
    ])
logging.info(f"Experiments on ablation studies.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')
if device.type == 'cuda':
    logging.info(f"{torch.cuda.get_device_properties(0)}")

eps_list = [1,2,3,4,5,6,7,8]
results = {}

graph = make_dataset(dataset_name, root="../data")
linkless_graph = graph.clone()
linkless_graph.edge_index = None
adj = SparseTensor(row=graph.edge_index[0], col=graph.edge_index[1], sparse_sizes=(graph.num_nodes, graph.num_nodes)).to(device).to_dense()

for eps in eps_list:
    ablation_name = "full"
    # we need to do grid search to find the best delta for the entire model
    delta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    best_delta = None
    best_mae = 10
    best_mae_std = None
    for delta in delta_list:
        mae = np.zeros(30)
        for i in range(30):
            client = Client(eps=eps, delta=delta, data=graph)
            server = FullServer(eps=eps, delta=delta, data=linkless_graph)
            priv_adj, priv_deg = client.AddLDP()
            server.receive(priv_adj, priv_deg)
            server.estimate()
            mae[i] = torch.nn.L1Loss()(server.pij, adj)
        if mae.mean() < best_mae:
            best_mae = mae.mean()
            best_mae_std = mae.std()
            best_delta = delta
    logging.info(f"[{dataset_name} with eps={eps} and entire model]: best delta is {best_delta} with {best_mae} ({best_mae_std})")
    with open("output/ablation.json") as f:
        acc_dict = json.load(f)
    if dataset_name not in acc_dict:
        acc_dict[dataset_name] = {}
    if str(eps) not in acc_dict[dataset_name]:
        acc_dict[dataset_name][str(eps)] = {}
    if ablation_name not in acc_dict[dataset_name][str(eps)]:
        acc_dict[dataset_name][str(eps)][ablation_name] = {}
    acc_dict[dataset_name][str(eps)][ablation_name]["delta"] = best_delta
    acc_dict[dataset_name][str(eps)][ablation_name]["mae"] = [best_mae, best_mae_std]
    with open('output/ablation.json', 'w') as fp:
        json.dump(acc_dict, fp, indent=2)

    for ablation_name in ["prior", "evidence"]:
        mae = np.zeros(30)
        for i in range(30):
            if ablation_name == "prior":
                delta = 1.0
                Server = ServerPrior
            elif ablation_name == "evidence":
                delta = None
                Server = ServerEvidence
            client = Client(eps=eps, delta=delta, data=graph)
            server = Server(eps=eps, delta=delta, data=linkless_graph)
            priv_adj, priv_deg = client.AddLDP()
            server.receive(priv_adj, priv_deg)
            server.estimate()
            mae[i] = torch.nn.L1Loss()(server.pij, adj)

        logging.info(f"[{dataset_name} with eps={eps} and {ablation_name}-only]: {mae.mean()} ({mae.std()})")
        with open("output/ablation.json") as f:
            acc_dict = json.load(f)
        if dataset_name not in acc_dict:
            acc_dict[dataset_name] = {}
        if str(eps) not in acc_dict[dataset_name]:
            acc_dict[dataset_name][str(eps)] = {}
        acc_dict[dataset_name][str(eps)][ablation_name] = [mae.mean(), mae.std()] 
        with open('output/ablation.json', 'w') as fp:
            json.dump(acc_dict, fp, indent=2)