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
        """
        self.__metric = metric

    def cluster_distance_multi(self, single_cluster: Cluster,
                               multi_clusters: List[Union[Cluster, None]]
                               ) -> np.ndarray:
        centroid_single = np.array(
            [np.average(single_cluster.vectors, axis=0)])  # 2-dimension
        centroids = np.array([np.average(cluster.vectors, axis=0)
                              for cluster in multi_clusters if cluster])

        distances = self.__metric.distance_matrix(centroid_single, centroids)
        return self.full_distance_vector(multi_clusters, distances)
        # NOTE: distance to single_cluster itself is zero
