import os
from data.split import train_val_test_split
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

def load_cora(root):
    dataset = Planetoid(root=os.path.join(root, "Planetoid"), name='Cora', transform=NormalizeFeatures())
    graph = dataset[0]
    graph.num_features, graph.num_classes = dataset.num_features, dataset.num_classes
    graph.train_mask, graph.val_mask, graph.test_mask = train_val_test_split(graph.num_nodes)
    return graph