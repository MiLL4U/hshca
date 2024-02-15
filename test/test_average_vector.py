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
SPATIAL_FACTOR = None
SPATIAL_SCALE = (1.0, 1.0)

CLUSTER_NUM = 4

makedirs(DST_PATH, exist_ok=True)

data = np.load(DATA_PATH)  # 3D array (x, y, r)


hca = HyperSpectralHCA(
    data, METHOD, METRIC, SPATIAL_FACTOR, SPATIAL_SCALE, show_progress=True)
print("auto spatial factor:", hca.auto_spatial_factor())
print("spatial factor:", hca.spatial_factor)
hca.print_dist_scales()
hca.compute()
average_spectra = hca.get_average_vectors(CLUSTER_NUM)
for spectrum in average_spectra:
    plt.plot(spectrum)
plt.show()
