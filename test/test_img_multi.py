from os import makedirs

import ibwpy as ip
import matplotlib.pyplot as plt

from hshca import HyperSpectralHCA
from hshca.linkmethod import Centroid, Ward  # noqa
from hshca.metric import CityBlock, Euclidean  # noqa

DATA_PATH = "./test/mHeLa_control_1_smth.ibw"
DST_PATH = "./dst/"
MAP_SHAPE = (30, 30)

METHOD = Ward
METRIC = Euclidean
SPATIAL_FACTORS = (1e-7, 5e-7, 1e-6, 5e-6, 1e-5,
                   1e-4, 3e-4, 5e-4, 8e-4, 3e-3, 3e-2)
SPATIAL_SCALE = (1.0, 1.0, 1.0)

CLUSTER_NUMS = (3, 4, 5, 6, 7, 8)

makedirs(DST_PATH, exist_ok=True)

ibw = ip.load(DATA_PATH)
data = ibw.array  # 4D array (x, y, z, r)


for factor_idx, factor in enumerate(SPATIAL_FACTORS):
    print(factor)
    hca = HyperSpectralHCA(
        data, METHOD, METRIC,
        factor, SPATIAL_SCALE,
        show_progress=True)
    hca.compute()

    for cluster_num in CLUSTER_NUMS:
        res = hca.get_cluster_map(cluster_num).reshape(MAP_SHAPE).T
        plt.imshow(res)

        name = f"cls{cluster_num}_{factor_idx}_{factor:.0e}.png"
        plt.savefig(DST_PATH + name)
