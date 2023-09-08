import os
from torch_geometric.datasets import FacebookPagePage
from torch_geometric.transforms import NormalizeFeatures
from data.split import train_val_test_split

def load_facebook(root):
    dataset = FacebookPagePage(root=os.path.join(root, "FacebookPagePage"), transform=NormalizeFeatures())
    graph = dataset[0]
    graph.num_features, graph.num_classes = dataset.num_features, dataset.num_classes
    graph.train_mask, graph.val_mask, graph.test_mask = train_val_test_split(graph.num_nodes)
    return graph

