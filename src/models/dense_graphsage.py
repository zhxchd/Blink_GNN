import torch
from torch_geometric.nn import DenseGraphConv
import torch.nn.functional as F

class DenseGraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_classes, dropout_p=0.5):
        super().__init__()
        self.p = dropout_p
        # torch.manual_seed(1234567)
        self.conv1 = DenseGraphConv(num_features, hidden_channels, aggr="mean")
        self.conv2 = DenseGraphConv(hidden_channels, num_classes, aggr="mean")
        self.is_dense = True

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = x.relu()
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv2(x, adj)
        x = x.reshape([x.shape[1], x.shape[2]])
        return x