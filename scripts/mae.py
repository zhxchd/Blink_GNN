import argparse
import json
import logging
import sys
from datetime import datetime
import torch

from torch_sparse import SparseTensor

sys.path.append('../src')
import numpy as np
import blink
from data import make_dataset

parser = argparse.ArgumentParser(description='Start experiment with specified dataset.')
parser.add_argument("--dataset", nargs='*', type=str, help="Specify what datasets to run with, cora, citeseer, lastfm or facebook.")
# add a grid_search option which defaults to False
parser.add_argument("--grid_search", action='store_true', help="Whether to run grid search on the hyperparameters, i.e., delta.")
args = parser.parse_args()

if args.dataset == None or len(args.dataset) == 0:
    datasets = ["cora", "citeseer", "lastfm", "facebook"]
else:
    datasets = args.dataset

# setup logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f"log/blink/mae_"+datetime.now().strftime('%y%m%d_%H%M%S.txt')),
        logging.StreamHandler(sys.stdout)
    ])

logging.info(f"Experiments on the MAE between A and P.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')
if device.type == 'cuda':
    logging.info(f"{torch.cuda.get_device_properties(0)}")

eps_list = [1,2,3,4,5,6,7,8]
for data in datasets:
    graph = make_dataset(data, root="../data")
    linkless_graph = graph.clone()
    linkless_graph.edge_index = None
    adj = SparseTensor(row=graph.edge_index[0], col=graph.edge_index[1], sparse_sizes=(graph.num_nodes, graph.num_nodes)).to(device).to_dense()
    # logging.info(f"Density of actual graph {data} is {graph.num_edges}")
    if args.grid_search:
        # 0.01, ..., 1.00
        delta_space = np.linspace(0.01, 1.00, num=100)
    else:
        delta_space = [0.1]
    for eps in eps_list:
        best_mae = [1e10, 0]
        best_delta = None
        best_bound = None
        for delta in delta_space:
            mae = np.zeros(30)
            for i in range(30):
                client = blink.Client(eps=eps, delta=delta, data=graph)
                server = blink.Server(eps=eps, delta=delta, data=linkless_graph)
                priv_adj, priv_deg = client.AddLDP()
                server.receive(priv_adj, priv_deg)
                server.estimate()
                mae[i] = torch.nn.L1Loss()(server.pij, adj)
            eps_a = eps*(1-delta)
            eps_d = eps*delta
            # bound = ((1+3*np.exp(eps_a)/(1+np.exp(eps_a))**2)*adj.sum().item() + 3*np.exp(eps_a)/(1+np.exp(eps_a))**2*graph.num_nodes/eps_d+graph.num_nodes)/graph.num_nodes**2
            bound = (2*adj.sum().item()+graph.num_nodes/2.0/eps_d)/graph.num_nodes**2
            if mae.mean() < best_mae[0]:
                best_mae = [mae.mean(), mae.std()]
                best_delta = delta
                best_bound = bound
        logging.info(f"MAE of P on {data} with eps={eps} (best delta={best_delta}): {best_mae[0]} ({best_mae[1]}), theoretically bounded by {best_bound}")
        logging.info(f"MAE of P on {data} with eps={eps}: Saving result to output/mae.json")
        with open("output/mae_gs.json") as f:
            d = json.load(f)
        if data not in d:
            d[data] = {}
        d[data][str(eps)] = [best_mae[0], best_mae[1], best_bound, best_delta]
        with open('output/mae_gs.json', 'w') as fp:
            json.dump(d, fp, indent=2)