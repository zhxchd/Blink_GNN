import numpy as np
import dprr

def run_dprr(graph, linkless_graph, model_name, eps, hp, num_trials, use_dense_model=False):
    val_loss = np.zeros(num_trials)
    test_acc = np.zeros(num_trials)

    for i in range(num_trials):
        client = dprr.Client(eps=eps, data=graph)
        server = dprr.Server(eps=eps, data=linkless_graph, use_dense_model=use_dense_model)

        priv_adj, priv_deg = client.AddLDP()
        server.receive(priv_adj, priv_deg)
        server.estimate()
        log = server.fit(model_name, hparam=hp)
        val_loss[i] = log[:,1].min()
        test_acc[i] = log[np.argmin(log[:,1])][2]
    return val_loss, test_acc