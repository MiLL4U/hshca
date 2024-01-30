from abc import ABC, abstractmethod
from typing import List, Tuple, Union

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

        NOTE: This class was implemented based on the following web page.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
        """

    def cluster_distance_multi(self, single_cluster: Cluster,
                               multi_clusters: List[Union[Cluster, None]],
                               dist_matrix: np.ndarray,
                               linked_pair: Tuple[int, int]) -> np.ndarray:
        size_s = single_cluster.previous_size
        size_t = single_cluster.size - single_cluster.previous_size
        size_st = size_s + size_t
        dist_st = dist_matrix[linked_pair[0]][linked_pair[1]]

        dist_vs_horz = dist_matrix[linked_pair[0]]
        dist_vs_vert = dist_matrix[:, linked_pair[0]]
        dist_vs = np.min([dist_vs_horz, dist_vs_vert], axis=0)

        dist_vt_horz = dist_matrix[linked_pair[1]]
        dist_vt_vert = dist_matrix[:, linked_pair[1]]
        dist_vt = np.min([dist_vt_horz, dist_vt_vert], axis=0)

        res = []
        for v_idx, cluster_v in enumerate(multi_clusters):
            if not cluster_v or v_idx == linked_pair[0]:
                res.append(np.inf)
                continue
            size_v = cluster_v.size
            size_T = size_st + size_v
            sum_ = (size_v + size_s) / size_T * dist_vs[v_idx] ** 2 \
                + (size_v + size_t) / size_T * dist_vt[v_idx] ** 2 \
                - (size_v / size_T) * dist_st ** 2
            res.append(np.sqrt(sum_))

        return np.array(res)
