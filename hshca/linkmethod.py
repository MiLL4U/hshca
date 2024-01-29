from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

from .cluster import Cluster
from .metric import HCAMetric


class LinkageMethod(ABC):
    @abstractmethod
    def __init__(self, metric: HCAMetric) -> None:
        """
        Abstract constructor for LinkageMethod.

        Parameters:
        - metric (HCAMetric): The metric to be used for distance calculations.

        Raises:
        - NotImplementedError: If called directly on the abstract class.
        """
        raise NotImplementedError

    @abstractmethod
    def cluster_distance_multi(self, single_cluster: Cluster,
                               multi_clusters: List[Union[Cluster, None]]
                               ) -> np.ndarray:
        """Returns the distances between a single cluster and multiple clusters.
        NOTE: For distance between a cluster and None, np.inf is returned.

        Args:
            single_cluster (Cluster): Single cluster
            multi_clusters (List[Cluster]): Array of multiple clusters

        Returns:
            np.ndarray: Array of distances between clusters
        """
        raise NotImplementedError

    def full_distance_vector(self, clusters: List[Union[Cluster, None]],
                             distances: Union[np.ndarray, List[float]]
                             ) -> np.ndarray:
        """Generate full-size distance vector (distances between a cluster and
        None is filled with np.inf)

        Args:
            clusters (List[Union[Cluster, None]]): Array of multiple clusters
            distances (np.ndarray): Array of distances between existing clusters

        Returns:
            np.ndarray: Array of distances between clusters
                        (filled with np.inf for None)
        """
        exist_cluster_idx = np.array([i for i in range(len(clusters))
                                      if clusters[i]])
        res = np.full(len(clusters), np.inf)
        res[exist_cluster_idx] = distances
        return res


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
                               multi_clusters: List[Union[Cluster, None]]
                               ) -> np.ndarray:
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


# class Ward(LinkageMethod):
#     def __init__(self, metric: HCAMetric) -> None:
#         """
#         Initialize the Ward linkage method.

#         Args:
#         - metric (HCAMetric): The metric to be used for distance calculations.

#         NOTE: In Ward's method, Euclidean distance must generally be used.
#         If this class is used with a metric other than Euclidean distance,
#         the centroid is defined by the Euclidean distance, but the variance is
#         calculated with the specified metric.

#         NOTE: This class was implemented based on the following web page,
#         but the algorithm may be incorrect.
#         https://ja.wikipedia.org/wiki/%E3%82%A6%E3%82%A9%E3%83%BC%E3%83%89%E6%B3%95
#         """
#         self.__metric = metric

#     def cluster_distance_multi(self, single_cluster: Cluster,
#                                multi_clusters: List[Cluster | None]
#                                ) -> np.ndarray:
#         var_single = self.__clutster_variance(single_cluster)
#         var_multi = [self.__clutster_variance(cluster)
#                      for cluster in multi_clusters if cluster]

#         merged_clusters = [single_cluster.merged(cluster)
#                            for cluster in multi_clusters if cluster]
#         var_merged = [self.__clutster_variance(cluster)
#                       for cluster in merged_clusters]
#         distances = [var_mer - var_single - var_mul
#                      for var_mer, var_mul in zip(var_merged, var_multi)]
#         return self.full_distance_vector(multi_clusters, distances)

#     def __clutster_variance(self, cluster: Cluster) -> float:
#         # TODO: use cache (bind to Cluster?) to improve performance
#         centroid = np.mean(cluster.member_vectors, axis=0)
#         distances = self.__metric.distance_matrix(
#             centroid, cluster.member_vectors)  # TODO: support 1-dim array
#         return np.sum(np.square(distances))


class Ward(Centroid):
    def __init__(self, metric: HCAMetric) -> None:
        """
        Initialize the Ward linkage method.

        Args:
        - metric (HCAMetric): The metric to be used for distance calculations.

        NOTE: In Ward's method, Euclidean distance must generally be used.
        If this class is used with a metric other than Euclidean distance,
        the centroid is defined by the Euclidean distance, but the variance is
        calculated with the specified metric.

        NOTE: This class was implemented based on the following web page.
        https://www.business-research-lab.com/220512-3/
        """
        self.__metric = metric

    def cluster_distance_multi(self, single_cluster: Cluster,
                               multi_clusters: List[Cluster | None]
                               ) -> np.ndarray:
        return np.array([])
