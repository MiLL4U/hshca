from typing import List, Tuple, Union

import numpy as np

from .cluster import Cluster
from .linkmethodabc import LinkageMethod
from .metric import HCAMetric


class ScipyWard(LinkageMethod):
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


class VarianceWard(LinkageMethod):
    def __init__(self, metric: HCAMetric) -> None:
        """
        Initialize the Ward linkage method.

        Args:
        - metric (HCAMetric): The metric to be used for distance calculations.

        NOTE: In Ward's method, Euclidean distance must generally be used.
        If this class is used with a metric other than Euclidean distance,
        the centroid is defined by the Euclidean distance, but the variance is
        calculated with the specified metric.

        NOTE: This class was implemented based on the following web page,
        but the algorithm may be incorrect.
        https://ja.wikipedia.org/wiki/%E3%82%A6%E3%82%A9%E3%83%BC%E3%83%89%E6%B3%95
        """
        super().__init__(metric)

    def cluster_distance_multi(self, single_cluster: Cluster,
                               multi_clusters: List[Union[Cluster, None]],
                               dist_matrix: np.ndarray,
                               linked_pair: Tuple[int, int]) -> np.ndarray:
        var_single = self.__clutster_variance(single_cluster)
        var_multi = [self.__clutster_variance(cluster)
                     for cluster in multi_clusters if cluster]

        merged_clusters = [single_cluster.merged(cluster)
                           for cluster in multi_clusters if cluster]
        var_merged = [self.__clutster_variance(cluster)
                      for cluster in merged_clusters]
        distances = [var_mer - var_single - var_mul
                     for var_mer, var_mul in zip(var_merged, var_multi)]
        return self.full_distance_vector(multi_clusters, distances)

    def __clutster_variance(self, cluster: Cluster) -> float:
        # use cache (bind to Cluster?) to improve performance?
        centroid = np.mean(cluster.member_vectors, axis=0)
        distances = self.metric.distance_matrix(
            centroid, cluster.member_vectors)
        return np.sum(np.square(distances))


class FactorWard(LinkageMethod):
    def __init__(self, metric: HCAMetric) -> None:
        """
        Initialize the Ward linkage method.

        Args:
        - metric (HCAMetric): The metric to be used for distance calculations.

        NOTE: In Ward's method, Euclidean distance must generally be used.
        If this class is used with a metric other than Euclidean distance,
        the centroid is defined by the Euclidean distance, but the variance is
        calculated with the specified metric.

        This class was implemented based on the following reference (p.230).
        https://www.researchgate.net/publication/220695963_Finding_Groups_in_Data_An_Introduction_To_Cluster_Analysis
        """
        super().__init__(metric)

    def cluster_distance_multi(self, single_cluster: Cluster,
                               multi_clusters: List[Union[Cluster, None]],
                               dist_matrix: np.ndarray,
                               linked_pair: Tuple[int, int]) -> np.ndarray:
        centroid_distances = self.centroid_distances(
            single_cluster, multi_clusters)
        factors = [self.ward_distance_factor(single_cluster, cluster)
                   for cluster in multi_clusters if cluster]
        ward_distances = [distance * factor for distance, factor
                          in zip(centroid_distances, factors)]
        return self.full_distance_vector(multi_clusters, ward_distances)

    def ward_distance_factor(self, cluster_1: Cluster,
                             cluster_2: Cluster) -> float:
        size_1 = cluster_1.size
        size_2 = cluster_2.size
        return np.sqrt((size_1 * size_2 * 2) / (size_1 + size_2))
