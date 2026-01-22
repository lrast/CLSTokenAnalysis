# Investigations into the information content and causal impact of CLS tokens

This work is motivated by an observation in my Test-Time-Training experiments: information in the CLS tokens of vision transformer models is not used by later layers. This can be seen by shuffling the CLS tokens in large test-set batches, which has no effect on the accuracy of classification.


## Part 1:
Decodable vs cauasal information in the CLS tokens of ViTs https://lrast.github.io/science/2026/01/20/lab_notes.html
