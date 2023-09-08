import torch
from torch_geometric.nn import DenseGCNConv
import torch.nn.functional as F

class DenseGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_classes, dropout_p=0.5):
        super().__init__()
        self.p = dropout_p
        # torch.manual_seed(1234567)
        self.conv1 = DenseGCNConv(num_features, hidden_channels)
        self.conv2 = DenseGCNConv(hidden_channels, num_classes)
        self.is_dense = True

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = x.relu()
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv2(x, adj)
        x = x.reshape([x.shape[1], x.shape[2]])
        return x