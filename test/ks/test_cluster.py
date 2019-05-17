from unittest import TestCase
from KS.cluster import Cluster


class TestCluster(TestCase):
    def test_estimate_optimum_distance(self):
        cluster = Cluster('test')
        cluster.train_len = 6
        cluster.compute_stats([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])
        print(cluster.cost_threshold, cluster.precision, cluster.recall)
        self.assertEqual(cluster.cost_threshold, 5, 'yes')
