# Distance Estimation Task

This folder contains code for training and evaluating models for distance
estimation. The training data is mostly BFS, with landmark nodes selected
randomly according to their PageRank. However, additional edges are added to
compensate for the low representation of adjacent nodes. On the dataset, two
types of models can be built:
  * categorical models to predict a distribution of a node pair's true distance
   given their embedding vectors, and
  * regression models to predict the distance directly.

## Creating Models

Unlike with link prediction, no internal structure is imposed on the models
themselves. They are simply defined as classes in the `models/` folder and
placed in `models/registry.py`. Each model has an `IModelMetadata` to help in
instantiating and evaluating the model, and an `IModel` can be queried from the
metadata.

## Results

The best hyperparameter configuration was found to be:
  * concatenated input features
  * 3 hidden layers
  * 192D hidden lengths
  * linear output
