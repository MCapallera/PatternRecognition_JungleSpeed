from unittest import TestCase
from KS.cluster import Cluster


class TestCluster(TestCase):
    def test_estimate_optimum_distance(self):
        cluster = Cluster('test')
        cluster.train_len = 5
        cluster.calculate_stats([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])
        print(cluster.estimated_cost_barrier, cluster.precision, cluster.recall)
        self.assertEqual(cluster.estimated_cost_barrier, 5, 'yes')
