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
        return np.random.random((self.vector_dim))

    def test_init(self):
        assert isinstance(self.subcluster, Subcluster)
        assert self.subcluster.n_vectors == 1
        assert len(self.subcluster.input_vectors) == 1

    def test_store_vector(self):
        assert np.array_equal(self.initial_vector, self.subcluster.input_vectors[0])

    def test_add_vector(self):
        new_vector = self.random_vec()
        self.subcluster.add(new_vector)
        assert self.subcluster.n_vectors == 2
        assert len(self.subcluster.input_vectors) == 2