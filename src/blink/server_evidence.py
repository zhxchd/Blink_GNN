import math
import numpy as np
import torch
from models import make_model

class Server:
    def __init__(self, eps, delta, data, variant=0) -> None:
        # no privacy
        if eps == None:
            self.priv = False
        else:
            self.priv = True
            self.variant = variant
            if delta == None:
                # do not privatize degree sequence
                self.priv_deg = False
                self.eps_a = eps
                self.eps_d = None
            else:
                self.priv_deg = True
                self.eps_d = eps * delta
                self.eps_a = eps * (1-delta)
        self.data = data
        self.n = data.num_nodes

    def receive(self, priv_adj, priv_deg):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.priv_adj = priv_adj.to(device)

    def estimate(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # store 1 vectors to save RAM
        ones_1xn = torch.ones(1,self.n).to(device)
        ones_nx1 = torch.ones(self.n,1).to(device)

        def estimate_prior():
            prior = 0.5*torch.ones(self.n, self.n).to(device)
            prior.fill_diagonal_(0)
            return prior
        
        def estimate_posterior(prior):
            p = 1.0/(1.0+np.exp(self.eps_a))
            priv_adj_t = self.priv_adj.transpose(0,1)
            x = self.priv_adj + priv_adj_t
            pr_y_edge = 0.5*(x-1)*(x-2)*p*p + 0.5*x*(x-1)*(1-p)*(1-p) - 1*x*(x-2)*p*(1-p)
            pr_y_no_edge = 0.5*(x-1)*(x-2)*(1-p)*(1-p) + 0.5*x*(x-1)*p*p - 1*x*(x-2)*p*(1-p)
            pij = pr_y_edge * prior / (pr_y_edge * prior + pr_y_no_edge * (1 - prior))
            return pij
        
        self.pij = estimate_posterior(estimate_prior())
        # this is to choose the top degree edges of each node, and also add weights
        # dth_max = pij.sort(dim=1, descending=True).values.gather(1, self.priv_deg.long() - 1)
        # weighted_edges = (pij * (pij >= dth_max)).float().to_sparse().coalesce()
        # self.est_edge_index = weighted_edges.indices()
        # self.est_edge_value = weighted_edges.values()

        # if self.hard:
        #     # hard threshold of 0.5
        #     self.est_edge_index = (self.pij > 0.5).float().to_sparse().coalesce().indices()
        # else:
        #     self.adj = self.pij
        del ones_1xn # reset variable so that it's no longer used and VRAM can be freed
        del ones_nx1
        torch.cuda.empty_cache()
        # take random graph based on pij
        # self.est_edge_index = torch.bernoulli(pij).to_sparse().coalesce().indices()

    def fit(self, model, hparam, iter=300):
        log = np.zeros((iter, 3))

        # we train the model on GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = make_model(model_type=model, hidden_channels=16, num_features=self.data.num_features, num_classes=self.data.num_classes, dropout_p=hparam["do"], hard=self.variant==0).to(device) # hard/soft version uses different models
        
        optimizer = torch.optim.Adam(model.parameters(), lr=hparam["lr"], weight_decay=hparam["wd"])
        criterion = torch.nn.CrossEntropyLoss()

        self.data = self.data.to(device)
        if self.priv:
            if self.variant == 0: # hard version has no edge weights and must be sparse GNNs
                self.est_edge_index = (self.pij > 0.5).float().to_sparse().coalesce().indices()
                self.est_edge_value = None
            elif self.variant == 1: # soft version and model is dense version, we feed in a adjacency matrix
                self.adj = self.pij
            else:
                # hybrid version, thresholding pij first
                kth_p = torch.topk(self.pij.flatten(), k=self.pij.sum().long().item()).values[-1].item()
                self.adj = self.pij * (self.pij >= kth_p).float()
        else:
            self.est_edge_index = self.data.edge_index
            self.est_edge_value = None

        def train():
            model.train()
            optimizer.zero_grad()
            out = model(self.data.x, self.adj) if model.is_dense else model(self.data.x, self.est_edge_index, self.est_edge_value)
            loss = criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()
            return float(loss)
        
        def validate():
            model.eval()
            out = model(self.data.x, self.adj) if model.is_dense else model(self.data.x, self.est_edge_index, self.est_edge_value)
            loss = criterion(out[self.data.val_mask], self.data.y[self.data.val_mask])
            return float(loss)
        
        def test():
            model.eval()
            out = model(self.data.x, self.adj) if model.is_dense else model(self.data.x, self.est_edge_index, self.est_edge_value)
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