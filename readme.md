# Investigations into the information content and causal impact of CLS tokens

This work is motivated by an observation in my Test-Time-Training experiments: information in the CLS tokens of vision transformer models is not used by later layers. This can be seen by shuffling the CLS tokens in large test-set batches, which has no effect on the accuracy of classification.


## Results along the way
1) Decodable vs causal information in the CLS tokens of ViTs https://lrast.github.io/science/2026/01/20/lab_notes.html
2) Internal layers are sensitive to CLS randomization, and CLS tokens are not optimized for readout. https://lrast.github.io/science/2026/02/13/lab_notes.html




## Reproduction notes:
Reproducing previous results with the current code:

0) Notebooks starting with 0.* use the 'analysis' tools, with no reliance on the readout features of the model harness
1) Notebook 1.0 does not have a script to reproduce these results. Nor does notebook 1.1 given changes to the model harness. These are small informative results, but careful reproduction would require more code.


