import numpy as np
import pytest
import os
from links_cluster import LinksCluster

def test_model_persistence(tmp_path):
    """
    Verify that a clustering model can be saved and reloaded 
    without losing state or consistency.
    """
    lc = LinksCluster(
        cluster_similarity_threshold=0.5,
        subcluster_similarity_threshold=0.6,
        pair_similarity_maximum=0.7
    )

    # Train model with some vectors
    vectors = [
        np.array([1.0, 0.0]),
        np.array([0.9, 0.1]),
        np.array([0.8, 0.2])
    ]
    for v in vectors:
        lc.predict(v)
    
    save_path = tmp_path / "model.pkl"
    lc.save(str(save_path))
    
    # Reload model
    lc_loaded = LinksCluster.load(str(save_path))
    
    # Check that model parameters were preserved
    assert lc_loaded.cluster_similarity_threshold == 0.5
    assert len(lc_loaded.clusters) == len(lc.clusters)
    
    # Verify that the loaded model still works as expected
    lc_loaded.predict(np.array([0.95, 0.05]))
    assert len(lc_loaded.clusters) > 0
