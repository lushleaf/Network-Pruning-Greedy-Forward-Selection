# Good Subnetworks Provably Exist: Pruning via Greedy Forward Selection (ICML 2020)

Most existing methods of network pruning are empirical and heuristic, leaving it open whether good subnetworks provably exist, how to find them efficiently, and if network pruning can be
provably better than direct training using gradient descent.

We answer these problems positively by proposing a simple greedy selection approach for finding good subnetworks, which starts from an empty network and greedily adds 
important neurons from the large network. This differs from the existing methods based on backward elimination, which remove redundant neurons from the large network.

Theoretically, applying the greedy selection strategy on sufficiently large pre-trained networks guarantees to
find small subnetworks with lower loss than networks directly trained with gradient descent. Our
results also apply to pruning randomly weighted networks.

Our theory and empirical results on MobileNet suggest that we should fine-tune the pruned subnetworks to leverage the information from the large model, instead
of re-training from new random initialization.
