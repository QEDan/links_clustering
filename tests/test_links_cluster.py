"""Tests for LinksCluster and LinksSubcluster classes."""
from links_cluster import LinksCluster, Subcluster
import numpy as np


class TestLinksCluster:
    def setup_method(self):
        self.cluster_similarity_threshold = 0.3
        self.subcluster_similarity_threshold = 0.2
        self.pair_similarity_maximum = 1.0
        self.cluster = LinksCluster(self.cluster_similarity_threshold,
                                    self.subcluster_similarity_threshold,
                                    self.pair_similarity_maximum,
                                    store_vectors=True)
        self.vector_dim = 256

    def test_init(self):
        assert isinstance(self.cluster, LinksCluster)

    def test_predict_once(self):
        vector = np.random.random((self.vector_dim, ))
        prediction = self.cluster.predict(vector)
        assert prediction == 0
        assert vector in self.cluster.get_all_vectors()

    def test_predict_many(self):
        how_many = 100
        predictions = []
        vectors = []
        for _ in range(how_many):
            vector = np.random.uniform(0, 1, self.vector_dim)
            if np.random.uniform(0, 1) < 0.5:
                vector[0] += 1e6
            prediction = self.cluster.predict(vector)
            vectors.append(vector)
            predictions.append(prediction)
        assert len(predictions) == how_many
        assert not all([p == 0 for p in predictions])
        assert len(vectors) == len(self.cluster.get_all_vectors())

class TestSubcluster:
    def setup_method(self):
        self.vector_dim = 256
        self.initial_vector = self.random_vec()
        self.subcluster = Subcluster(self.initial_vector, store_vectors=True)

    def random_vec(self):
        """Generate random vector of the correct shape."""
        return np.random.random((self.vector_dim))

    def test_init(self):
        """Test basic class initialization properties."""
        assert isinstance(self.subcluster, Subcluster)
        assert self.subcluster.n_vectors == 1
        assert len(self.subcluster.input_vectors) == 1

    def test_store_vector(self):
        """Test that the input vectors are stored."""
        assert np.array_equal(self.initial_vector, self.subcluster.input_vectors[0])

    def test_add_vector(self):
        """Test that we can a new vector."""
        new_vector = self.random_vec()
        self.subcluster.add(new_vector)
        assert self.subcluster.n_vectors == 2
        assert len(self.subcluster.input_vectors) == 2
        assert np.array_equal(self.subcluster.centroid, np.mean([self.initial_vector, new_vector], axis=0))

    def test_add_multiple_vectors(self):
        """Test that we can add multiple vectors."""
        how_many = 10
        new_vectors = np.random.random((how_many, self.vector_dim))
        for i in range(how_many):
            new_vector = new_vectors[i]
            self.subcluster.add(new_vector)
        expected_centroid = np.mean(
            np.concatenate(
                [np.expand_dims(self.initial_vector, axis=0),
                 new_vectors],
                axis=0
            ),
            axis=0
        )
        assert self.subcluster.n_vectors == how_many + 1
        assert len(self.subcluster.input_vectors) == how_many + 1
        np.testing.assert_array_almost_equal(
            self.subcluster.centroid,
            expected_centroid,
            decimal=12)

    def test_merge(self):
        """Test that subclusters can be merged."""
        new_vector = self.random_vec()
        new_subcluster = Subcluster(new_vector, store_vectors=True)
        self.subcluster.connected_subclusters.add(new_subcluster)
        new_subcluster.connected_subclusters.add(self.subcluster)
        self.subcluster.merge(new_subcluster)
        assert self.subcluster.n_vectors == 2
        assert len(self.subcluster.input_vectors) == 2

    def test_merge_connections(self):
        """Test that we can merge subclusters that have external edges."""
        new_vector_1 = self.random_vec()
        new_subcluster_1 = Subcluster(new_vector_1, store_vectors=True)
        new_vector_2 = self.random_vec()
        new_subcluster_2 = Subcluster(new_vector_2, store_vectors=True)
        new_subcluster_1.connected_subclusters.update({new_subcluster_2, self.subcluster})
        new_subcluster_2.connected_subclusters.update({new_subcluster_1, self.subcluster})
        self.subcluster.connected_subclusters.update({new_subcluster_1, new_subcluster_2})
        self.subcluster.merge(new_subcluster_2)
        assert self.subcluster.n_vectors == 2
        assert len(self.subcluster.input_vectors) == 2
        assert len(self.subcluster.connected_subclusters) == 1
        assert self.subcluster.connected_subclusters == {new_subcluster_1}