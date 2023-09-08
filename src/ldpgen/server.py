from collections import Counter
import math
import numpy as np
import torch
from models import make_model
from sklearn.cluster import KMeans

class Server:
    def __init__(self, eps, data, use_dense_model=False) -> None:
        self.priv_deg = True
        self.eps_1 = eps * 0.5
        self.eps_2 = eps * 0.5
        self.data = data
        self.n = data.num_nodes
        self.use_dense_model = use_dense_model
    
    def initial_partition(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.k_0 = 2 # random partition into 2 clusters
        self.xi_0 = torch.bernoulli(torch.full((self.n, 1), 0.5)).int().to(device) # n by 1 vector, each entry is 0 or 1
        return self.xi_0
    
    def phase_one(self, delta_0):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.delta_0 = delta_0.to(device) # n by 2 matrix
        self.eta = self.delta_0.sum(1).reshape(self.n, 1).to(device) # n by 1 vector, server's estimate of the number of edges connected to each node
        self.eta = torch.clamp(torch.round(self.eta), 1, self.n-1).int().to(device) # round and clip eta to 1 to n-1
        # now we need to determine the number of clusters, k_1, for the next phase
        k_1_d = lambda d: d+(d*d-2*(1+math.sqrt(5))*d+1)/self.eps_2
        unique_eta, eta_counts = torch.unique(self.eta, return_counts=True)
        self.k_1 = torch.clamp(torch.ceil((k_1_d(unique_eta/2) * eta_counts/self.n).sum()), 1, self.n).int().item() # number of clusters for the next phase, round to nearest integer and clip to 1 to n

        # k-means based on self.delta, each node has two coordinates, (d_0, d_1), number of clusters = k_1
        # we use sklearn.cluster.KMeans
        # prepare X as numpy array
        X = self.delta_0.cpu().numpy() # n by k_0=2 matrix
        kmeans = KMeans(n_clusters=self.k_1, n_init="auto").fit(X)
        self.xi_1 = torch.tensor(kmeans.labels_).reshape(self.n, 1).long().to(device) # n by 1 vector, each entry is 0 to k_1-1
        return self.xi_1, self.k_1
    
    def phase_two(self, delta_1):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.delta_1 = delta_1.to(device) 
        # delta_1 is n by k_1 matrix, use it to cluster the nodes into k_1 clusters
        # we use sklearn.cluster.KMeans
        # prepare X as numpy array
        X = self.delta_1.cpu().numpy() # n by k_1 matrix
        kmeans = KMeans(n_clusters=self.k_1, n_init="auto").fit(X)
        self.xi_2 = torch.tensor(kmeans.labels_).reshape(self.n, 1).long().to(device) # n by 1 vector, each entry is 0 to k_1-1
    
    # I just use nested for loops to calculate the estimate, it's not efficient but it's easy to understand
    def phase_three(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # first estimate degree vectors for partition xi_2
        self.delta_1 = torch.clamp(torch.round(self.delta_1), 0, self.n-1).int().to(device) # n by 2 matrix
        self.delta_hat = torch.zeros(self.n, self.k_1).to(device) # n by k_1 matrix
        for i in range(self.k_1): # for each cluster i
            for j in range(self.k_1): # for each cluster j in partition xi_1
                self.delta_hat[:,i] += self.delta_1[:,j] / (self.xi_1 == j).sum() * ((self.xi_1 == j) * (self.xi_2 == i)).sum()

        # now we estimate the adjacency matrix
        self.pij_k_by_k = torch.zeros(self.k_1, self.k_1).to(device) # k_1 by k_1 lists
        for i in range(self.k_1): # for each cluster i
            for j in range(self.k_1): # for each cluster j
                self.pij_k_by_k[i][j] = self.delta_hat[:,j].sum()/(self.xi_2==j).sum() / (self.delta_hat[:,j].sum() + (self.delta_hat[:,j]*(self.xi_2==i).reshape(self.n)).sum())
        
        self.pij = self.delta_hat[:, self.xi_2.reshape(self.n)] * self.pij_k_by_k[self.xi_2, self.xi_2.transpose(0,1)]
        # clip to [0,1]
        self.pij = torch.clamp(self.pij, 0, 1)

        # remove self loops
        self.pij[torch.arange(self.n), torch.arange(self.n)] = 0
    
        self.pij = torch.bernoulli(self.pij).to(device) # n by n matrix
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