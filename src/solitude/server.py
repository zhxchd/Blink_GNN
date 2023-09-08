import numpy as np
import torch
from models import make_model

class Server:
    def __init__(self, eps, data) -> None:
        self.eps = eps
        self.data = data
        self.n = data.num_nodes

    def receive(self, priv_adj):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.priv_adj = priv_adj.to(device)

    # in solitude there's no estimation, essentially the mechanism "learns" the ground truth graph
    # that minimize the final loss
    def estimate(self):
        pass

    def fit(self, model, hparam, iter=200):
        log = np.zeros((iter, 3))

        # we train the model on GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # estemated adjacency list, to be optimized, initialized to noisy adj
        self.est_adj = self.priv_adj.clone().detach().requires_grad_(True).to(device)
        # self.est_adj = torch.zeros(self.n, self.n, device=device, requires_grad=True)

        # we need dense model
        model = make_model(model_type=model, hidden_channels=16, num_features=self.data.num_features, num_classes=self.data.num_classes, dropout_p=hparam["do"], hard=False).to(device)
        theta_optimizer = torch.optim.Adam(model.parameters(), lr=hparam["lr_theta"], weight_decay=hparam["wd"]) # optimize model paramters
        adj_optimizer = torch.optim.Adam([self.est_adj], lr=hparam["lr_adj"])             # optimize adjacency matrix, no weight decay because the l1 norm of est_adj is in the loss function
        criterion = torch.nn.CrossEntropyLoss()
        assert model.is_dense, "The model in solitude must be dense."

        self.data = self.data.to(device)

        def train(theta_step):
            model.train()
            theta_optimizer.zero_grad()
            adj_optimizer.zero_grad()
            out = model(self.data.x, self.est_adj)
            # loss is to regularize the estimated adjacency matrix so it is close to received one and also sparse
            loss = criterion(out[self.data.train_mask], self.data.y[self.data.train_mask]) \
                + hparam["lam1"] * torch.frobenius_norm(self.est_adj - self.priv_adj) \
                + hparam["lam2"] * torch.norm(self.est_adj, p=1)
            loss.backward()
            # as per the paper, they use alternating optimization
            if theta_step:
                theta_optimizer.step()
            else:
                adj_optimizer.step()
            return float(loss)
        
        def validate():
            model.eval()
            out = model(self.data.x, self.est_adj)
            # for validation loss, we no longer take the regularization terms
            loss = criterion(out[self.data.val_mask], self.data.y[self.data.val_mask])
            return float(loss)
        
        def test():
            model.eval()
            out = model(self.data.x, self.est_adj)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            test_correct = pred[self.data.test_mask] == self.data.y[self.data.test_mask]
            test_acc = int(test_correct.sum()) / int(self.data.test_mask.sum())
            return test_acc
        
        for epoch in range(1, iter+1):
            loss = train(epoch % 2 == 1)
            val_loss = validate()
            test_acc = test()
            log[epoch-1] = [loss, val_loss, test_acc]
        
        return log