# L-DPGCN

This is the local DP variant of the DPGCN mechanism from [LinkTeller](https://www.computer.org/csdl/proceedings-article/sp/2022/131600a522/1FlQypPVMis).

In DPGCN, the server knows the ground truth adjacency matrix and edge count so it can add Laplacian noise to both of them. In a LDP variant, the server receives noisy adjacency lists and edge counts from nodes. Then the server selects the top links as estimated links and proceeds with GNN training and inference. 