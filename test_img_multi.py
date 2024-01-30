from typing import List

import ibwpy as ip
import matplotlib.pyplot as plt
import numpy as np

from hshca import HyperSpectralHCA
from hshca.linkmethod import Centroid, Ward  # noqa
from hshca.metric import CityBlock, Euclidean  # noqa

DATA_PATH = "./test/mHeLa_control_1_smth.ibw"
MAP_SHAPE = (30, 30)

METHOD = Ward
METRIC = Euclidean
SPATIAL_FACTORS = (1e-7, 5e-7, 1e-6, 5e-6, 1e-5,
                   1e-4, 3e-4, 5e-4, 8e-4, 3e-3, 3e-2)
SPATIAL_SCALE = (1.0, 1.0, 1.0)

CLUSTER_NUM = 5

ibw = ip.load(DATA_PATH)
data = ibw.array  # 4D array (x, y, z, r)

results: List[np.ndarray] = []
for factor in SPATIAL_FACTORS:
    print(factor)
    hca = HyperSpectralHCA(
        data, METHOD, METRIC,
        factor, SPATIAL_SCALE,
        show_progress=True)
    hca.compute()
    res = hca.get_cluster_map(CLUSTER_NUM).reshape(MAP_SHAPE).T
    results.append(res)

i = 0
for factor, result in zip(SPATIAL_FACTORS, results):
    name = f"{i}_{factor:.0e}.png"
    plt.imshow(result)
    plt.savefig(name)
    i += 1
