import math
from typing import Tuple

import torch
from torch_sparse import SparseTensor

# Simple randomized response, nothing more
class Client():
    def __init__(self, eps, data) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data = data.to(device)
        self.eps = eps

    def AddLDP(self) -> Tuple[torch.Tensor]:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        n = self.data.num_nodes
        adj = SparseTensor(row=self.data.edge_index[0], col=self.data.edge_index[1], sparse_sizes=(n, n)).to(device).to_dense()

        def rr_adj() -> torch.Tensor:
            p = 1.0/(1.0+math.exp(self.eps))
            # return 1 with probability p, but does not flip diagonal edges since no self loop allowed
            res = ((adj + torch.bernoulli(torch.full((n, n), p)).to(device)) % 2).float()
            res.fill_diagonal_(0)
            return res

        return rr_adj()