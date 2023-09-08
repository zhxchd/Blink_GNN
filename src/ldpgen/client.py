import math
from typing import Tuple

import torch
from torch_sparse import SparseTensor

class Client():
    def __init__(self, eps, data) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data = data.to(device)
        self.eps_1 = eps * 0.5
        self.eps_2 = eps * 0.5
        self.n = data.num_nodes
        self.adj = SparseTensor(row=self.data.edge_index[0], col=self.data.edge_index[1], sparse_sizes=(self.n, self.n)).to(device).to_dense() # n by n matrix

    def phase_one(self, xi_0):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # for each node, find the number of nodes in each partition that are connected to it, xi_0 is a partition of two clusters
        # n by 2 matrix
        k_0 = 2
        xi_0 = xi_0.to(device) # n by 1 vector
        vector_1_x_n = torch.ones(1,self.n).to(device)
        delta_0 = torch.concat([(((xi_0 == i)*vector_1_x_n).transpose(0,1) * self.adj).sum(1).reshape(self.n, 1) for i in range(k_0)], 1).to(device)
        delta_0 = delta_0 + torch.distributions.laplace.Laplace(loc=0, scale=1/self.eps_1).sample((self.n,2)).to(device)
        return delta_0
    
    def phase_two(self, xi_1, k_1):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        xi_1 = xi_1.to(device) # n by 1 vector
        vector_1_x_n = torch.ones(1,self.n).to(device)
        delta_1 = torch.concat([(((xi_1 == i)*vector_1_x_n).transpose(0,1) * self.adj).sum(1).reshape(self.n, 1) for i in range(k_1)], 1).to(device) # n by k_1 matrix
        delta_1 = delta_1 + torch.distributions.laplace.Laplace(loc=0, scale=1/self.eps_2).sample((self.n,k_1)).to(device)
        return delta_1