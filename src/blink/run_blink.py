import numpy as np
import blink

def run_blink(graph, linkless_graph, model_name, eps, hp, num_trials, variant=0):
    val_loss = np.zeros(num_trials)
    test_acc = np.zeros(num_trials)
    # non private, there's no client
    if eps == None:
        for i in range(num_trials):
            server = blink.Server(None, None, graph)
            log = server.fit(model_name, hparam=hp)
            val_loss[i] = log[:,1].min()
            test_acc[i] = log[np.argmin(log[:,1]),2]
    # link LDP with blink
    else:
        for i in range(num_trials):
            client = blink.Client(eps=eps, delta=hp["delta"], data=graph)
            server = blink.Server(eps=eps, delta=hp["delta"], data=linkless_graph, variant=variant)

            priv_adj, priv_deg = client.AddLDP()
            server.receive(priv_adj, priv_deg)
            server.estimate()
            log = server.fit(model_name, hparam=hp)
            val_loss[i] = log[:,1].min()
            test_acc[i] = log[np.argmin(log[:,1])][2]
    return val_loss, test_acc