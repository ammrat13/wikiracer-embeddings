# Link Prediction Task

This folder contains code for training and evaluating different models for link
prediction. It samples random edges from the graph as positive examples, and
samples random pairs of unconnected nodes for negative examples. All the models
are logistic regression with different ways of combining the two node
embeddings, namely:
 * concatenation
 * element-wise multiplication (a.k.a. Hadamard product)
 * (flattened) outer product

## Results

Element-wise multiplication seems to perform the best. It's able to get an AUC
of 95.7% on a training set of just 1K points, wheras concatenation gets an AUC
of 85.3% on a training set of 100K points. It sees similarly improved accuracy
and F1 scores. Outer product seems to be prone to overfitting. On 10K samples,
it gets 99.9% training-set accuracy but just 93.1% test-set accuracy. The effect
is smaller on 100K samples, but it's still there. Even though it did better than
element-wise multiplication on the test set, I want to avoid overfitting for
other tasks.

The training set size seems to have somewhat significant impact on performance.
For element-wise multiplication, a training set size of 30K seems to be enough
to get similar performance to a training set size of 100K, at least on this
task.

Finally, the length of the embedding vector seems to have a small impact on the
loss and AUC, but no impact on the accuracy or F1 score. Vectors of size 256
should be good enough.
