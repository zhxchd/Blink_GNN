import numpy as np
import ldpgcn

def run_ldpgcn(graph, linkless_graph, model_name, eps, hp, num_trials, use_dense_model=False):
    val_loss = np.zeros(num_trials)
    test_acc = np.zeros(num_trials)
    for i in range(num_trials):
        client = ldpgcn.Client(eps=eps, data=graph)
        server = ldpgcn.Server(eps=eps, data=linkless_graph, use_dense_model=use_dense_model)

        priv_adj = client.AddLDP()
        server.receive(priv_adj)
        server.estimate()
        log = server.fit(model_name, hparam=hp)
        val_loss[i] = log[:,1].min()
        test_acc[i] = log[np.argmin(log[:,1])][2]
    return val_loss, test_acc