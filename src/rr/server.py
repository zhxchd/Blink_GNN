import numpy as np
import torch
from models import make_model

class Server:
    def __init__(self, eps, data, use_dense_model=False) -> None:
        self.eps = eps
        self.data = data
        self.n = data.num_nodes
        self.use_dense_model = use_dense_model

    def receive(self, priv_adj):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.priv_adj = priv_adj # RR result, on CPU

    def estimate(self):
        # no actual estimation is done here
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.use_dense_model:
            self.pij = self.priv_adj.to(device)
        else:
            self.est_edge_index = self.priv_adj.to_sparse().coalesce().indices().to(device) #sparse tensor move to GPU

    def fit(self, model, hparam, iter=200):
        log = np.zeros((iter, 3))

        # we train the model on GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.use_dense_model:
            model = make_model(model_type=model, hidden_channels=16, num_features=self.data.num_features, num_classes=self.data.num_classes, dropout_p=hparam["do"], hard=False).to(device)
        else:
            model = make_model(model_type=model, hidden_channels=16, num_features=self.data.num_features, num_classes=self.data.num_classes, dropout_p=hparam["do"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hparam["lr"], weight_decay=hparam["wd"])
        criterion = torch.nn.CrossEntropyLoss()

        self.data = self.data.to(device) # data on GPU
        if self.use_dense_model:
            self.pij = self.pij.float().to(device)
        else:
            self.est_edge_index = self.est_edge_index.to(device)

        def train():
            model.train()
            optimizer.zero_grad()
            if self.use_dense_model:
                out = model(self.data.x, self.pij)
            else:
                out = model(self.data.x, self.est_edge_index)
            loss = criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()
            return float(loss)
        
        def validate():
            model.eval()
            if self.use_dense_model:
                out = model(self.data.x, self.pij)
            else:
                out = model(self.data.x, self.est_edge_index)
            loss = criterion(out[self.data.val_mask], self.data.y[self.data.val_mask])
            return float(loss)
        
        def test():
            model.eval()
            if self.use_dense_model:
                out = model(self.data.x, self.pij)
            else:
                out = model(self.data.x, self.est_edge_index)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            test_correct = pred[self.data.test_mask] == self.data.y[self.data.test_mask]
            test_acc = int(test_correct.sum()) / int(self.data.test_mask.sum())
            return test_acc
        
        for epoch in range(1, iter+1):
            loss = train()
            val_loss = validate()
            test_acc = test()
            log[epoch-1] = [loss, val_loss, test_acc]
        
        return log