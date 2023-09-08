from models.dense_gat import DenseGAT
from models.dense_gcn import DenseGCN
from models.dense_graphsage import DenseGraphSAGE
from models.gat import GAT
from models.gcn import GCN
from models.graphsage import GraphSAGE
from models.mlp import MLP

def make_model(model_type, hidden_channels, num_features, num_classes, dropout_p=0.5, hard=True):
    if model_type == "gcn":
        if hard:
            return GCN(hidden_channels=hidden_channels, num_features=num_features, num_classes=num_classes, dropout_p=dropout_p)
        else:
            return DenseGCN(hidden_channels=hidden_channels, num_features=num_features, num_classes=num_classes, dropout_p=dropout_p)
    elif model_type == "mlp":
        return MLP(hidden_channels=hidden_channels, num_features=num_features, num_classes=num_classes, dropout_p=dropout_p)
    elif model_type == "graphsage":
        if hard:
            return GraphSAGE(hidden_channels=hidden_channels, num_features=num_features, num_classes=num_classes, dropout_p=dropout_p)
        else:
            return DenseGraphSAGE(hidden_channels=hidden_channels, num_features=num_features, num_classes=num_classes, dropout_p=dropout_p)
    elif model_type == "gat":
        if hard:
            return GAT(hidden_channels=hidden_channels, num_features=num_features, num_classes=num_classes, dropout_p=dropout_p)
        else:
            return DenseGAT(hidden_channels=hidden_channels, num_features=num_features, num_classes=num_classes, dropout_p=dropout_p)