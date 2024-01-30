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
