import math
from typing import Tuple

import torch
from torch_sparse import SparseTensor

# Simple randomized response, nothing more
class Client():
    def __init__(self, eps, data) -> None:
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data = data
        self.eps = eps

    def AddLDP(self) -> Tuple[torch.Tensor]:
        n = self.data.num_nodes
        # put dense matrix on CPU
        adj = SparseTensor(row=self.data.edge_index[0].cpu(), col=self.data.edge_index[1].cpu(), sparse_sizes=(n, n)).to_dense()

        def rr_adj() -> torch.Tensor:
            p = 1.0/(1.0+math.exp(self.eps))
            # return 1 with probability p, but does not flip diagonal edges since no self loop allowed
            res = ((adj + torch.bernoulli(torch.full((n, n), p))) % 2).bool() # reduce the size by using bool tensor
            res.fill_diagonal_(0)
            return res # still on CPU

        return rr_adj()

    