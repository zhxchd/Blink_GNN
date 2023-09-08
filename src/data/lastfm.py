import os
from torch_geometric.datasets import LastFMAsia
from torch_geometric.transforms import NormalizeFeatures
from data.split import train_val_test_split

def load_lastfm(root):
    dataset = LastFMAsia(root=os.path.join(root, "LastFMAsia"), transform=NormalizeFeatures())
    graph = dataset[0]
    graph.num_features, graph.num_classes = dataset.num_features, dataset.num_classes
    graph.train_mask, graph.val_mask, graph.test_mask = train_val_test_split(graph.num_nodes)
    return graph

