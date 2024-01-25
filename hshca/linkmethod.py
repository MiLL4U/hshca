from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

from .cluster import Cluster
from .metric import HCAMetric


class LinkageMethod(ABC):
    @abstractmethod
    def __init__(self, metric: HCAMetric) -> None:
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


class Centroid(LinkageMethod):
    def __init__(self, metric: HCAMetric) -> None:
        self.__metric = metric

    def cluster_distance_multi(self, single_cluster: Cluster,
                               multi_clusters: List[Union[Cluster, None]]
                               ) -> np.ndarray:
        exist_cluster_idx = np.array([i for i in range(len(multi_clusters))
                                      if multi_clusters[i]])

        centroid_single = np.array(
            [np.average(single_cluster.vectors, axis=0)])  # 2-dimension
        centroids = np.array([np.average(cluster.vectors, axis=0)
                              for cluster in multi_clusters if cluster])

        distances = self.__metric.distance_matrix(centroid_single, centroids)
        res = np.full(len(multi_clusters), np.inf)
        res[exist_cluster_idx] = distances

        return res  # NOTE: distance to single_cluster itself is zero
