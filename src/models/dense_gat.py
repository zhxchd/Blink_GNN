import torch
from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F
import math
from typing import Any
from torch import Tensor
from torch.nn import Parameter

"""
Glorot initialization for weights (copied from torch_geometric).
"""
def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)

"""
Constant initialization of weights.
"""
def constant(value: Any, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)

"""
torch_geometric does not have an implementation of dense GATConv operator.
This is a single headed graph attention convolutional layer using concat and bias.
"""
class DenseGATConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.2, dropout=0.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        self.att_src = Parameter(torch.Tensor(1, out_channels)) # C
        self.att_dst = Parameter(torch.Tensor(1, out_channels)) # C

        self.bias = Parameter(torch.Tensor(out_channels))    # C

        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        constant(self.bias, 0.)
    
    def forward(self, x: Tensor, adj: Tensor, add_self_loops=True):
        # x should be of shape [N, F]
        # adj should be of shape [N, N]
        
        N = len(adj)

        # add self loop
        if add_self_loops:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[idx, idx] = 1

        out = self.lin(x) # [N, C]
        alpha_src = torch.sum(out * self.att_src, dim=-1) # [N]
        alpha_dst = torch.sum(out * self.att_dst, dim=-1) # [N]
        alpha = alpha_src.unsqueeze(1).repeat(1,N) + alpha_dst.unsqueeze(0).repeat(N,1)
        alpha = F.leaky_relu(alpha, self.negative_slope)

        alpha = adj * torch.exp(alpha)      # weighted/masked softmax, if adj=0, result would be zero
        alpha = alpha / alpha.sum(dim=0)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training) # [N, N]

        out = torch.matmul((adj * alpha).transpose(0,1), out)

        if self.bias is not None:
            out = out + self.bias

        return out
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')

class DenseGAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_classes, dropout_p=0.5):
        super().__init__()
        self.p = dropout_p
        # torch.manual_seed(1234567)
        self.conv1 = DenseGATConv(num_features, hidden_channels)
        self.conv2 = DenseGATConv(hidden_channels, num_classes)
        self.is_dense = True

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = x.relu()
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv2(x, adj)
        return x