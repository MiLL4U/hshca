import numpy as np
import matplotlib.pyplot as plt

from hshca import HyperSpectralHCA
from hshca.linkmethod import Ward
from hshca.metric import Euclidean

image = np.load("./test/test_cell.npy")
hshca = HyperSpectralHCA(
    data=image,  # hyperspectral image to analyze
    method=Ward,  # linkage method
    spectral_metric=Euclidean,  # metric
    spatial_dist_factor=1e-5,  # factor for spatial distance (lambda)
    spatial_scale=(1.0, 1.0),  # scale for each spatial axis (x, y, ...)
    show_progress=True  # set True here if you want to see progress bar
)
hshca.compute()

cluster_map = hshca.get_cluster_map(5)  # number of clusters
average_spectra = hshca.get_average_vectors(5)

fig, ax = plt.subplots(1, 2, layout='constrained')
cluster_map = ax[0].imshow(cluster_map, cmap='Paired')
for cluster_idx, spectra in enumerate(average_spectra):
    ax[1].plot(spectra, label=f"Cluster {cluster_idx}")
ax[1].legend()
fig.colorbar(cluster_map, ax=ax[0], orientation='horizontal',  # type: ignore
             label="Cluster index")
plt.show()
