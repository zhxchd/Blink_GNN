import numpy as np
import ldpgen

def run_ldpgen(graph, linkless_graph, model_name, eps, hp, num_trials, use_dense_model=False):
    val_loss = np.zeros(num_trials)
    test_acc = np.zeros(num_trials)
    for i in range(num_trials):
        client = ldpgen.Client(eps=eps, data=graph)
        server = ldpgen.Server(eps=eps, data=linkless_graph, use_dense_model=use_dense_model)

        xi_0 = server.initial_partition()
        delta_0 = client.phase_one(xi_0)
        xi_1, k_1 = server.phase_one(delta_0)
        delta_1 = client.phase_two(xi_1, k_1)
        server.phase_two(delta_1)
        server.phase_three()

        log = server.fit(model_name, hparam=hp)
        val_loss[i] = log[:,1].min()
        test_acc[i] = log[np.argmin(log[:,1])][2]
    return val_loss, test_acc