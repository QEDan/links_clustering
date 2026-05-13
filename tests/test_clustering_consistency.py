import numpy as np
import pytest
from links_cluster import LinksCluster, Subcluster

def test_subcluster_merge_natural_consistency():
    """
    Reproduce the ValueError by using the `predict()` method to create
    an environment where cluster merging is triggered naturally.
    """
    lc = LinksCluster(
        cluster_similarity_threshold=0.5,
        subcluster_similarity_threshold=0.6,
        pair_similarity_maximum=0.7
    )

    # 1. Create three subclusters that are all connected to each other
    # by adding vectors that are close to the existing subclusters.
    # This ensures a high level of connectivity.
    lc.predict(np.array([1.0, 0.0]))  # SC1
    lc.predict(np.array([0.9, 0.1]))  # SC2 (connected to SC1)
    lc.predict(np.array([0.8, 0.2]))  # SC3 (connected to SC2)
    lc.predict(np.array([0.85, 0.15])) # Trigger connectivity between SC1 and SC3

    # 2. Add a vector that forces a re-merge of the existing subclusters.
    # This should trigger the recursive call to update_cluster.
    lc.predict(np.array([0.92, 0.08]))
