Sorting is a necessary tool for machine learning, to create algorithms (k-NN) or test-time metrics like top-k classification accuracy or losses based on the rank.
Nevertheless it seems to be a difficult task for automatically differentiable pipelines in DL.
Sorting gives us two vectors, this application is not differentiable as we are working with integer-valued permutation. In the paper they aim to implement a differentiable proxy of the basic approach. 

The article conceive this proxy by thinking of an optimal assignment problem. 
We sort n values by matching them to a probability measure
supported on any increasing family
of n target values. 
Therefore we are considering Optimal Transport (OT) as a relaxation of the basic problem allowing us to extend rank and sort operators using probability measures.
The auxiliary measure will be supported on m increasing values with m != n. 
Introducing regularization with an entropic penalty and applying Sinkhorn iterations will allow to gain back differentiable operators.
The smooth approximation of rank and sort allow to use the 0/1 loss and the quantile regression loss.
Using numpy we implemented this version of Differentiable Ranks and Sorting using Optimal Transport by Marco Cuturi Olivier Teboul Jean-Philippe Vert.

