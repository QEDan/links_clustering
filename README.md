# Links Online Clustering

Implementation of the Links Online Clustering algorithm: 
https://arxiv.org/abs/1801.10123

## Overview
This is a clustering algorithm for online data. That is, it will predict 
cluster membership for vectors that it is shown one-by-one. It does not 
require examining the entire dataset to predict cluster membership.

It works by maintaining a two-level hierarchy of clusters and subclusters.
Each subcluster has a centroid that is compared with new vector for prediction
using cosine similarity. Depending on the previous data that has been seen, 
the new data point can be assigned to an existing cluster/subcluster, 
assigned to a new subcluster within an existing cluster, or 
assigned to a new subcluster and cluster. 

Instantiating the class requires 3 hyperparameters:
* cluster_similarity_threshold
* subcluster_similarity_threshold
* pair_similarity_maximum

These details are best understood by reading the paper.

## Installation

`pip install -r requirements.txt`

## Usage example

```python
from links_cluster import LinksCluster

...
links_cluster = LinksCluster(cluster_similarity_threshold, subcluster_similarity_threshold, pair_similarity_maximum)
for vector in data:
    predicted_cluster = links_cluster.predict(vector)

```

For more usage examples, see the `tests`.


