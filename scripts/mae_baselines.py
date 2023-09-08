import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

import argparse
import json
import logging
import sys
from datetime import datetime
import torch
from torch_sparse import SparseTensor

sys.path.append('../src')
import numpy as np
import ldpgcn
import rr
from data import make_dataset

parser = argparse.ArgumentParser(description='Start experiment with specified dataset.')
parser.add_argument("--dataset", nargs='*', type=str, help="Specify what datasets to run with, cora, citeseer, lastfm or facebook.")
parser.add_argument("--method", nargs='*', type=str, help="Specify what baseline methods to run, list of 'rr', 'ldpgcn' and 'solitude'.")
parser.add_argument("--eps", nargs='*', type=float, help="Specify what eps to run with, default is 1 to 8.")
args = parser.parse_args()

# if not specified, run all datasets
if args.dataset == None or len(args.dataset) == 0:
    datasets = ["cora", "citeseer", "lastfm", "facebook"]
else:
    datasets = args.dataset

# if not specified, run all methods
if args.method == None or len(args.method) == 0:
    methods = ["rr", "ldpgcn"]
else:
    methods = args.method

# if not specified, run all eps
if args.eps == None or len(args.eps) == 0:
    eps_list = [1,2,3,4,5,6,7,8]
else:
    eps_list = args.eps

make_client = {
    "rr": rr.Client,
    "ldpgcn": ldpgcn.Client
}

make_server = {
    "rr": rr.Server,
    "ldpgcn": ldpgcn.Server
}

# setup logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f"log/baselines/mae_"+datetime.now().strftime('%y%m%d_%H%M%S.txt')),
        logging.StreamHandler(sys.stdout)
    ])

logging.info(f"Experiments on the MAE between A and P for method {methods} on {datasets}.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')
if device.type == 'cuda':
    logging.info(f"{torch.cuda.get_device_properties(0)}")

for data in datasets:
    graph = make_dataset(data, root="../data")
    linkless_graph = graph.clone()
    linkless_graph.edge_index = None
    adj = SparseTensor(row=graph.edge_index[0], col=graph.edge_index[1], sparse_sizes=(graph.num_nodes, graph.num_nodes)).to(device).to_dense()
    # logging.info(f"Density of actual graph {data} is {graph.num_edges}")
    for method in methods:
        for eps in eps_list:
            mae = np.zeros(30)
            for i in range(30):
                client = make_client[method](eps=eps, data=graph)
                # set server to use dense model so that it will estimate the dense matrix
                server = make_server[method](eps=eps, data=linkless_graph, use_dense_model=True)
                priv_adj = client.AddLDP()
                server.receive(priv_adj.float())
                server.estimate()
                mae[i] = torch.nn.L1Loss()(server.pij, adj)
            logging.info(f"{method} MAE of P on {data} with eps={eps}: {mae.mean()} ({mae.std()})")
            logging.info(f"{method} MAE of P on {data} with eps={eps}: Saving result to output/mae_baseline.json")
            with open("output/mae_baseline.json") as f:
                d = json.load(f)
            if method not in d:
                d[method] = {}
            if data not in d[method]:
                d[method][data] = {}
            d[method][data][str(eps)] = [mae.mean(), mae.std()]
            with open('output/mae_baseline.json', 'w') as fp:
                json.dump(d, fp, indent=2)