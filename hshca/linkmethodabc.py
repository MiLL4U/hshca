from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np

from .cluster import Cluster
from .metric import HCAMetric


class LinkageMethod(ABC):
    def __init__(self, metric: HCAMetric) -> None:
        self.__metric = metric

    @property
    def metric(self) -> HCAMetric:
        return self.__metric

    @abstractmethod
    def cluster_distance_multi(self, single_cluster: Cluster,
                               multi_clusters: List[Union[Cluster, None]],
                               dist_matrix: np.ndarray,
                               linked_pair: Tuple[int, int]) -> np.ndarray:
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

    def centroid(self, cluster: Cluster) -> np.ndarray:
        return np.array(np.average(cluster.member_vectors, axis=0))

    def centroid_distances(self, single_cluster: Cluster,
                           multi_clusters: List[Union[Cluster, None]]
                           ) -> np.ndarray:
        centroid_single = self.centroid(single_cluster)
        centroids_multi = np.array([self.centroid(cluster)
                                    for cluster in multi_clusters if cluster])
        return self.metric.distance_matrix(centroid_single, centroids_multi)