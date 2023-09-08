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
        logging.FileHandler(f"log/density_"+datetime.now().strftime('%y%m%d_%H%M%S.txt')),
        logging.StreamHandler(sys.stdout)
    ])

logging.info(f"Experiments on the density difference between actual graph and estimated graph.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')
if device.type == 'cuda':
    logging.info(f"{torch.cuda.get_device_properties(0)}")

eps_list = [1,2,3,4,5,6,7,8]
results = {}

for data in datasets:
    graph = make_dataset(data, root="../data")
    linkless_graph = graph.clone()
    linkless_graph.edge_index = None
    adj = SparseTensor(row=graph.edge_index[0], col=graph.edge_index[1], sparse_sizes=(graph.num_nodes, graph.num_nodes)).to(device).to_dense()
    logging.info(f"Density of actual graph {data} is {graph.num_edges}")
    for eps in eps_list:
        A_hat_l1 = np.zeros(30)
        tp = np.zeros(30)
        for i in range(30):
            client = blink.Client(eps=eps, delta=0.1, data=graph)
            server = blink.Server(eps=eps, delta=0.1, data=linkless_graph)
            priv_adj, priv_deg = client.AddLDP()
            server.receive(priv_adj, priv_deg)
            server.estimate()
            A_hat = (server.pij > 0.5).float()
            A_hat_l1[i] = A_hat.sum().item()
            tp[i] = (A_hat * adj).sum().item()
            # A_hat_l1[i] = server.est_edge_index.shape[1]
        logging.info(f"Density on {data} with eps={eps}: {A_hat_l1.mean()} ({A_hat_l1.std()})")
        logging.info(f"TP on {data} with eps={eps}: {tp.mean()} ({tp.std()})")
        logging.info(f"Density on {data} with eps={eps}: Saving result to output/density.json")
        with open("output/density.json") as f:
            d = json.load(f)
        if data not in d:
            d[data] = {}
        d[data][str(eps)] = [A_hat_l1.mean(), A_hat_l1.std(), tp.mean(), tp.std()]
        with open('output/density.json', 'w') as fp:
            json.dump(d, fp, indent=2)