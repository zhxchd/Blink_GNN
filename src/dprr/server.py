import math
import numpy as np
import torch
from models import make_model

class Server:
    def __init__(self, eps, data, use_dense_model=False) -> None:
        self.priv_deg = True
        self.eps_1 = eps * 0.1
        self.eps_2 = eps * 0.9
        self.data = data
        self.n = data.num_nodes
        self.use_dense_model = use_dense_model

    def receive(self, priv_adj, priv_deg):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.priv_adj = priv_adj.to(device)
        self.priv_deg = priv_deg.to(device)

    def estimate(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        p = math.exp(self.eps_2)/(1+math.exp(self.eps_2)) # probability of NOT flipping an edge
        q = (self.priv_deg / (self.priv_deg * (2*p-1) + (self.n - 1)*(1-p))).to(device) # sampling probability of each edge
        # clip q into 0 to 1
        q = torch.clamp(q, 0, 1) # n by 1 vector
        # expand q into n by n matrix
        self.pij = self.priv_adj * torch.bernoulli(q.matmul(torch.ones(1, self.n).to(device))) # n by n matrix
        # get sparse representation of pij
        self.est_edge_index = self.pij.to_sparse().coalesce().indices().to(device)
        torch.cuda.empty_cache()

    def fit(self, model, hparam, iter=300):
        log = np.zeros((iter, 3))

        # we train the model on GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.use_dense_model:
            model = make_model(model_type=model, hidden_channels=16, num_features=self.data.num_features, num_classes=self.data.num_classes, dropout_p=hparam["do"], hard=False).to(device)
        else:
            model = make_model(model_type=model, hidden_channels=16, num_features=self.data.num_features, num_classes=self.data.num_classes, dropout_p=hparam["do"]).to(device)
        

        if self.use_dense_model:
            self.pij = self.pij.to(device)
        else:
            self.est_edge_index = self.est_edge_index.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=hparam["lr"], weight_decay=hparam["wd"])
        criterion = torch.nn.CrossEntropyLoss()

        self.data = self.data.to(device)

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