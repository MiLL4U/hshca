from __future__ import annotations

from typing import List

from numpy import ndarray


class Cluster:
    def __init__(self, all_vectors: ndarray, node_idxs: List[int]) -> None:
        self.__all_vectors = all_vectors
        self.__node_idxs = node_idxs
        # self.__repr_point = ...
        # TODO: hold representing point to improve performance?

    @property
    def node_idxs(self) -> List[int]:
        return self.__node_idxs

    @property
    def size(self) -> int:
        return len(self.__node_idxs)

    @property
    def all_vectors(self) -> ndarray:
        return self.__all_vectors

    @property
    def member_vectors(self) -> ndarray:
        return self.__all_vectors[self.__node_idxs]

    def has_same_vectors(self, vectors: ndarray) -> bool:
        return True if vectors is self.__all_vectors else False

    def merge(self, other: Cluster) -> Cluster:
        if not other.has_same_vectors(self.__all_vectors):
            raise ValueError("attempted to merge clusters with different data")
        self.__node_idxs.extend(other.node_idxs)
        return self
