from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .cluster import Cluster
from .metric import HCAMetric


class LinkageMethod(ABC):
    @abstractmethod
    def __init__(self, metric: HCAMetric) -> None:
        raise NotImplementedError

    # @abstractmethod
    # def cluster_distance(self, cluster_1: Cluster,
    #                      cluster_2: Cluster) -> float:
    #     raise NotImplementedError

    @abstractmethod
    def cluster_distance_multi(self, single_cluster: Cluster,
                               multi_clusters: List[Cluster]) -> np.ndarray:
        raise NotImplementedError


class Centroid(LinkageMethod):
    def __init__(self, metric: HCAMetric) -> None:
        self.__metric = metric

    # def cluster_distance(self, cluster_1: Cluster, cluster_2: Cluster
    #                      ) -> float:
    #     centroid_1 = np.average(cluster_1.vectors, axis=1)
    #     centroid_2 = np.average(cluster_2.vectors, axis=1)
    #     return self.__metric.vector_distance(centroid_1, centroid_2)

    def cluster_distance_multi(self, single_cluster: Cluster,
                               multi_clusters: List[Cluster]) -> np.ndarray:
        centroid_single = np.average(single_cluster.vectors, axis=1)
        centroids = np.array([np.average(cluster.vectors, axis=1) for
                              cluster in multi_clusters])
        return self.__metric.distance_matrix(centroid_single, centroids)
