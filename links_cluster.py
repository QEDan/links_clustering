"""Links online clustering algorithm.

Reference: https://arxiv.org/abs/1801.10123
"""
import logging

import numpy as np
from scipy.spatial.distance import cosine


class Subcluster:
    """Class for subclusters and edges between subclusters."""
    def __init__(self, initial_vector: np.ndarray, store_vectors: bool = False):
        self.input_vectors = [initial_vector]
        self.centroid = initial_vector
        self.n_vectors = 1
        self.store_vectors = store_vectors
        self.connected_subclusters = set()

    def add(self, vector: np.ndarray):
        """Add a new vector to the subcluster, update the centroid."""
        if self.store_vectors:
            self.input_vectors.append(vector)
        self.n_vectors += 1
        if self.centroid is None:
            self.centroid = vector
        else:
            self.centroid = (self.n_vectors - 1) / \
                            self.n_vectors * self.centroid \
                            + vector / self.n_vectors

    def merge(self,
              subcluster_merge: 'Subcluster',
              delete_merged: bool = True):
        """Merge subcluster_merge into self. Update centroids."""
        if self.store_vectors:
            self.input_vectors += subcluster_merge.input_vectors

        # Update centroid and n_vectors
        self.centroid = self.n_vectors * self.centroid \
            + subcluster_merge.n_vectors \
            * subcluster_merge.centroid
        self.centroid /= self.n_vectors + subcluster_merge.n_vectors
        self.n_vectors += subcluster_merge.n_vectors
        try:
            subcluster_merge.connected_subclusters.remove(self)
            self.connected_subclusters.remove(subcluster_merge)
        except KeyError:
            logging.warning("Attempted to merge unconnected subclusters. "
                            "Merging anyway.")
        for sc in subcluster_merge.connected_subclusters:
            sc.connected_subclusters.remove(subcluster_merge)
            if self not in sc.connected_subclusters and sc != self:
                sc.connected_subclusters.update({self})
        self.connected_subclusters.update(subcluster_merge.connected_subclusters)
        if delete_merged:
            del subcluster_merge


class LinksCluster:
    """An online clustering algorithm."""
    def __init__(self,
                 cluster_similarity_threshold: float,
                 subcluster_similarity_threshold: float,
                 pair_similarity_maximum: float,
                 store_vectors=False
                 ):
        self.clusters = []
        self.cluster_similarity_threshold = cluster_similarity_threshold
        self.subcluster_similarity_threshold = subcluster_similarity_threshold
        self.pair_similarity_maximum = pair_similarity_maximum
        self.store_vectors = store_vectors

    def predict(self, new_vector: np.ndarray) -> int:
        """Predict a cluster id for new_vector."""
        if len(self.clusters) == 0:
            # Handle first vector
            self.clusters.append([Subcluster(new_vector, store_vectors=self.store_vectors)])
            return 0

        best_subcluster = None
        best_similarity = -np.inf
        best_subcluster_cluster_id = None
        best_subcluster_id = None
        for cl_idx, cl in enumerate(self.clusters):
            for sc_idx, sc in enumerate(cl):
                cossim = 1.0 - cosine(new_vector, sc.centroid)
                if cossim > best_similarity:
                    best_subcluster = sc
                    best_similarity = cossim
                    best_subcluster_cluster_id = cl_idx
                    best_subcluster_id = sc_idx
        if best_similarity >= self.subcluster_similarity_threshold:  # eq. (20)
            # Add to existing subcluster
            best_subcluster.add(new_vector)
            self.update_cluster(best_subcluster_cluster_id, best_subcluster_id)
            assigned_cluster = best_subcluster_cluster_id
        else:
            # Create new subcluster
            new_subcluster = Subcluster(new_vector, store_vectors=self.store_vectors)
            cossim = 1.0 - cosine(new_subcluster.centroid, best_subcluster.centroid)
            if cossim >= self.sim_threshold(best_subcluster.n_vectors, 1):  # eq. (21)
                # New subcluster is part of existing cluster
                self.add_edge(best_subcluster, new_subcluster)
                self.clusters[best_subcluster_cluster_id].append(new_subcluster)
                assigned_cluster = best_subcluster_cluster_id
            else:
                # New subcluster is a new cluster
                self.clusters.append([new_subcluster])
                assigned_cluster = len(self.clusters) - 1
        return assigned_cluster

    @staticmethod
    def add_edge(sc1: Subcluster, sc2: Subcluster):
        """Add an edge between subclusters sc1, and sc2."""
        sc1.connected_subclusters.add(sc2)
        sc2.connected_subclusters.add(sc1)

    def update_edge(self, sc1: Subcluster, sc2: Subcluster):
        """Compare subclusters sc1 and sc2, remove or add an edge depending on cosine similarity.

        Args:
            sc1: Subcluster
                First subcluster to compare
            sc2: Subcluster
                Second subcluster to compare

        Returns:
            bool
                True if the edge is valid
                False if the edge is not valid
        """
        cossim = 1.0 - cosine(sc1.centroid, sc2.centroid)
        threshold = self.sim_threshold(sc1.n_vectors, sc2.n_vectors)
        if cossim < threshold:
            try:
                sc1.connected_subclusters.remove(sc2)
                sc2.connected_subclusters.remove(sc1)
            except KeyError:
                logging.warning("Attempted to update an invalid edge that didn't exist. "
                                "Edge remains nonexistant.")
            return False
        else:
            sc1.connected_subclusters.add(sc2)
            sc2.connected_subclusters.add(sc1)
            return True

    def merge_subclusters(self, cl_idx, sc_idx1, sc_idx2):
        """Merge subclusters with id's sc_idx1 and sc_idx2 of cluster with id cl_idx."""
        sc2 = self.clusters[cl_idx][sc_idx2]
        self.clusters[cl_idx][sc_idx1].merge(sc2)
        self.update_cluster(cl_idx, sc_idx1)
        self.clusters[cl_idx] = self.clusters[cl_idx][:sc_idx2] \
            + self.clusters[cl_idx][sc_idx2 + 1:]
        for sc in self.clusters[cl_idx]:
            if sc2 in sc.connected_subclusters:
                sc.connected_subclusters.remove(sc2)

    def update_cluster(self, cl_idx: int, sc_idx: int):
        """Update cluster

        Subcluster with id sc_idx has been changed, and we want to
        update the parent cluster according to the discussion in
        section 3.4 of the paper.

        Args:
            cl_idx: int
                The index of the cluster to update
            sc_idx: int
                The index of the subcluster that has been changed

        Returns:
            None

        """
        updated_sc = self.clusters[cl_idx][sc_idx]
        severed_subclusters = []
        connected_scs = set(updated_sc.connected_subclusters)
        for connected_sc in connected_scs:
            connected_sc_idx = None
            for c_sc_idx, sc in enumerate(self.clusters[cl_idx]):
                if sc == connected_sc:
                    connected_sc_idx = c_sc_idx
            if connected_sc_idx is None:
                raise ValueError(f"Connected subcluster of {sc_idx} "
                                 f"was not found in cluster list of {cl_idx}.")
            cossim = 1.0 - cosine(updated_sc.centroid, connected_sc.centroid)
            if cossim >= self.subcluster_similarity_threshold:
                self.merge_subclusters(cl_idx, sc_idx, connected_sc_idx)
            else:
                are_connected = self.update_edge(updated_sc, connected_sc)
                if not are_connected:
                    severed_subclusters.append(connected_sc_idx)
        for severed_sc_id in severed_subclusters:
            severed_sc = self.clusters[cl_idx][severed_sc_id]
            if len(severed_sc.connected_subclusters) == 0:
                for cluster_sc in self.clusters[cl_idx]:
                    if cluster_sc != severed_sc:
                        cossim = 1.0 - cosine(cluster_sc.centroid,
                                              severed_sc.centroid)
                        if cossim >= self.sim_threshold(cluster_sc.n_vectors,
                                                        severed_sc.n_vectors):
                            self.add_edge(cluster_sc, severed_sc)
            if len(severed_sc.connected_subclusters) == 0:
                self.clusters[cl_idx] = self.clusters[cl_idx][:severed_sc_id] \
                    + self.clusters[cl_idx][severed_sc_id + 1:]
                self.clusters.append([severed_sc])

    def get_all_vectors(self):
        """Return all stored vectors from entire history.

        Returns:
            list
                list of vectors

        Raises:
            RuntimeError
                if self.store_vectors is False (i.e. there are no stored vectors)
        """
        if not self.store_vectors:
            raise RuntimeError("Vectors were not stored, so can't be collected")
        all_vectors = []
        for cl in self.clusters:
            for scl in cl:
                all_vectors += scl.input_vectors
        return all_vectors

    def sim_threshold(self, k: int, kp: int) -> float:
        """Compute the similarity threshold.

        This is based on equations (16) and (24) of the paper.

        Args:
            k: int
                The number of vectors in a cluster or subcluster
            kp: int
                k-prime in the paper, the number of vectors in another
                cluster or subcluster

        Returns:
            float
                The similarity threshold for inclusion in a cluster or subcluster.
        """
        s = (1.0 + 1.0 / k * (1.0 / self.cluster_similarity_threshold ** 2 - 1.0))
        s *= (1.0 + 1.0 / kp * (1.0 / self.cluster_similarity_threshold ** 2 - 1.0))
        s = 1.0 / np.sqrt(s)  # eq. (16)
        s = self.cluster_similarity_threshold ** 2 \
            + (self.pair_similarity_maximum - self.cluster_similarity_threshold ** 2) \
            / (1.0 - self.cluster_similarity_threshold ** 2) \
            * (s - self.cluster_similarity_threshold ** 2)  # eq. (24)
        return s
