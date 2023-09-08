import math
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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.priv_adj = priv_adj.to(device)

    def estimate(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # number of edges
        e = torch.round(self.priv_adj.sum()/2).long().item()
        # the following are (very very rare) corner cases
        # at least 0 edges, at most n*(n-1)/2 edges
        if e <= 0:
            # no edge to keep at all
            if self.use_dense_model:
                self.pij = torch.zeros(self.n, self.n).to(device)
            else:
                self.est_edge_index = torch.zeros(self.n, self.n).to_sparse().to(device).coalesce().indices()
            return
        elif e >= self.n * (self.n-1)/2:
            # keep all edges, complete graph
            if self.use_dense_model:
                self.pij = torch.ones(self.n, self.n).to(device)
            else:
                self.est_edge_index = torch.ones(self.n, self.n).to_sparse().to(device).coalesce().indices()
            return

        # upper triangular matrix, consider both a_ij and a_ji
        atr = torch.triu(self.priv_adj, 1) + torch.triu(self.priv_adj.transpose(0,1), 1)
        triu_mask = (torch.triu(torch.ones(self.n, self.n), 1)==1).to(device)
        # get the e largest entris in atr (only count upper triangular part)
        e_th = torch.topk(atr[triu_mask].flatten(), e).values.min()
        atr[atr < e_th] = 0
        atr[atr > 0] = 1
        atr[~triu_mask] = 0
        triu_mask = None
        if self.use_dense_model:
            self.pij = atr + atr.transpose(0,1)
        else:
            self.est_edge_index = (atr + atr.transpose(0,1)).to_sparse().coalesce().indices()

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

        self.data = self.data.to(device)

        if self.use_dense_model:
            self.pij = self.pij.to(device)
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