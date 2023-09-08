import numpy as np
import solitude

def run_solitude(graph, linkless_graph, model_name, eps, hp, num_trials, use_dense_model=True):
    val_loss = np.zeros(num_trials)
    test_acc = np.zeros(num_trials)
    for i in range(num_trials):
        client = solitude.Client(eps=eps, data=graph)
        server = solitude.Server(eps=eps, data=linkless_graph)

        priv_adj = client.AddLDP()
        server.receive(priv_adj)
        server.estimate() # this line of code does no use
        log = server.fit(model_name, hparam=hp)
        val_loss[i] = log[:,1].min() # this val loss is without regularization terms
        test_acc[i] = log[np.argmin(log[:,1])][2]
    return val_loss, test_acc