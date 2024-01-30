from typing import List, Tuple, Union

import numpy as np

from .cluster import Cluster
from .linkmethodabc import LinkageMethod
from .metric import HCAMetric
from .ward import ScipyWard as Ward  # noqa: set ScipyWard as default Ward method


class Centroid(LinkageMethod):
    def __init__(self, metric: HCAMetric) -> None:
        """
        Initialize the Centroid linkage method.

        Parameters:
        - metric (HCAMetric): The metric to be used for distance calculations.

        NOTE: In Centroid method, Euclidean distance must generally be used.
        If this class is used with a metric other than Euclidean distance,
        the centroid is defined by the Euclidean distance, but the distance is
        calculated with the specified metric.
        """
        self.__metric = metric

    def cluster_distance_multi(self, single_cluster: Cluster,
                               multi_clusters: List[Union[Cluster, None]],
                               dist_matrix: np.ndarray,
                               linked_pair: Tuple[int, int]) -> np.ndarray:
        centroid_distances = self.centroid_distances(
            single_cluster, multi_clusters)
        return self.full_distance_vector(multi_clusters, centroid_distances)
        # NOTE: distance to single_cluster itself is zero

    def centroid(self, cluster: Cluster) -> np.ndarray:
        return np.array(np.average(cluster.member_vectors, axis=0))

    def centroid_distances(self, single_cluster: Cluster,
                           multi_clusters: List[Union[Cluster, None]]
                           ) -> np.ndarray:
        centroid_single = self.centroid(single_cluster)
        centroids_multi = np.array([self.centroid(cluster)
                                    for cluster in multi_clusters if cluster])
        return self.__metric.distance_matrix(centroid_single, centroids_multi)
