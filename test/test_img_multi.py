from os import makedirs

import matplotlib.pyplot as plt
import numpy as np

from hshca import HyperSpectralHCA
from hshca.linkmethod import Centroid, Ward  # noqa
from hshca.metric import CityBlock, Euclidean  # noqa

DATA_PATH = "./test/test_cell.npy"
DST_PATH = "./dst/"

METHOD = Ward
METRIC = Euclidean
SPATIAL_FACTORS = (1e-7, 5e-7, 1e-6, 5e-6, 1e-5,
                   1e-4, 3e-4, 5e-4, 8e-4, 3e-3, 3e-2)
SPATIAL_SCALE = (1.0, 1.0)

CLUSTER_NUMS = (3, 4, 5, 6, 7, 8)

makedirs(DST_PATH, exist_ok=True)

data = np.load(DATA_PATH)  # 3D array (x, y, r)


for factor_idx, factor in enumerate(SPATIAL_FACTORS):
    print(factor)
    hca = HyperSpectralHCA(
        data, METHOD, METRIC,
        factor, SPATIAL_SCALE,
        show_progress=True)
    hca.compute()

    for cluster_num in CLUSTER_NUMS:
        res = hca.get_cluster_map(cluster_num).reshape(hca.map_shape)
        plt.imshow(res)

        name = f"cls{cluster_num}_{factor_idx}_{factor:.0e}.png"
        plt.savefig(DST_PATH + name)
