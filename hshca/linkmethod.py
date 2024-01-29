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
                             distances: np.ndarray) -> np.ndarray:
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
        centroid_single = np.array(
            [np.average(single_cluster.member_vectors, axis=0)])  # 2-dimension
        centroids = np.array([np.average(cluster.member_vectors, axis=0)
                              for cluster in multi_clusters if cluster])

        distances = self.__metric.distance_matrix(centroid_single, centroids)
        return self.full_distance_vector(multi_clusters, distances)
        # NOTE: distance to single_cluster itself is zero


class Ward(LinkageMethod):
    def __init__(self, metric: HCAMetric) -> None:
        """
        Initialize the Ward linkage method.

        Args:
        - metric (HCAMetric): The metric to be used for distance calculations.

        NOTE: In Ward's method, Euclidean distance must generally be used.
        If this class is used with a metric other than Euclidean distance,
        the centroid is defined by the Euclidean distance, but the variance is
        calculated with the specified metric.
        """
        self.__metric = metric

    def cluster_distance_multi(self, single_cluster: Cluster,
                               multi_clusters: List[Cluster | None]
                               ) -> np.ndarray:
        var_single = self.__clutster_variance(single_cluster)
        distances = []
        for cluster in multi_clusters:
            if cluster:
                distances.append(self.__ward_distance(single_cluster, cluster))
            else:
                distances.append(np.inf)
        return np.array(distances)

    def __ward_distance(self, cluster1: Cluster, cluster2: Cluster) -> float:
        """
        Calculate the Ward distance between two clusters.
        """
        merged_data = np.vstack(
            [cluster1.member_vectors, cluster2.member_vectors])
        # Node indices will be recalculated in the merge step
        merged_cluster = Cluster(cluster1.all_vectors, [])

        ssd_merged = self.__sum_of_squares_deviation(merged_data)
        ssd_cluster1 = self.__sum_of_squares_deviation(cluster1.member_vectors)
        ssd_cluster2 = self.__sum_of_squares_deviation(cluster2.member_vectors)

        increase_in_ssd = ssd_merged - (ssd_cluster1 + ssd_cluster2)
        return np.sqrt(increase_in_ssd)

    def __clutster_variance(self, cluster: Cluster) -> float:
        mean = np.mean(cluster.member_vectors, axis=0)
        deaviations = cluster.member_vectors - mean
        squared = np.square(deaviations)
        return np.sum(np.square(deaviations))

    def __sum_of_squares_deviation(self, vectors: np.ndarray) -> float:
        """
        Calculate the sum of squares deviation of vectors from their mean.
        """
        mean_vector = np.mean(vectors, axis=0)
        return np.sum(np.square(vectors - mean_vector))
