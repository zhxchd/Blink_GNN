from typing import Tuple

import torch
from torch_sparse import SparseTensor

class Client():
    def __init__(self, eps, data) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data = data.to(device)
        self.eps = eps

    def AddLDP(self) -> Tuple[torch.Tensor]:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        n = self.data.num_nodes
        adj = SparseTensor(row=self.data.edge_index[0], col=self.data.edge_index[1], sparse_sizes=(n, n)).to(device).to_dense()

        # in DPGCN, each entry is added some Laplacian noise
        res = adj + torch.distributions.laplace.Laplace(loc=0, scale=1/self.eps).sample((n,n)).to(device)
        res.fill_diagonal_(0)
        return res
    