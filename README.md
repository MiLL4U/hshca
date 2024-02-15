# Hyper Spectral HCA
Hyper Spectral Hierarchical Clustering Analysis (HSHCA) provides [hierarchical clustering analysis (HCA)](https://en.wikipedia.org/wiki/Hierarchical_clustering) specialized for hyperspectral images that can be measured by techniques such as [Raman microscopy](https://raman.oxinst.com/techniques/raman-imaging).

Generally, when applying HCA to hyperspectral images, the numerous spectra composing the image are treated as vectors in independent multidimensional spaces. Therefore, information about spatial coordinates (unit example: μm) at each measurement point of the spectra is lost.

HSHCA efficiently clusters spectra originating from physically adjacent measurement points by using both the distances between spectra ($d_{spectral}$) and the distances in real space between measurement points ($d_{spatial}$) as the definition of distance when performing HCA.

## Principle
In HSHCA, the distance between cluster A and cluster B ( $d(A,B)$ ) is processed in HCA using the distance between spectra ( $d_{spectral}(A,B)$ ) and the distance in real space ( $d_{spatial}(A,B)$ ) as:

$$d(A,B) = d_{spectral}(A,B) + \lambda\cdot d_{spatial}(A,B)$$

Since $d_{spectral}$  and $d_{spatial}$ have different dimensions, $\lambda$ is used as a coefficient to scale these values to the same magnitude. Note that $\lambda$ is a **hyperparameter** that needs to be adjusted according to measurement conditions and other factors.

$d_{spectral}$ is defined by metrics and methods commonly used in traditional HCA algorithms. For example, when using the Euclidean distance, $d_{spectral}$ corrensponds to the square root of the sum of squared differences of two spectra, and when using the cityblock distance, $d_{spectral}$ corresponds to the sum of the absolute differences of two spectra.

$d_{spatial}$ is defined as the Euclidean distance in real space, which is represented by the formula:

$$d_{spatial}(A,B) = \sqrt{(x_A - x_B)^2 + (y_A - y_B)^2}$$

where $(x_A, y_A)$ and $(x_B, y_B)$ are the spatial coordinates of points A and B, respectively.


## Installation
### Install with pip (using Git, recommended)
Install Hyper Spectral HCA with:
```bash
$ python -m pip install git+https://github.com/MiLL4U/hshca.git
```
### Install with pip
1. download a wheel package (*.whl) from [Releases](https://github.com/MiLL4U/hshca/releases)

2. Install Hyper Spectral HCA with pip:
```bash
$ python -m pip install hshca-x.y.z-py3-none-any.whl
```
(replace x.y.z with the version of HSHCA which you downloaded)

### Install with git clone
1. Clone this repository

```bash
$ git clone https://github.com/MiLL4U/hshca.git
```

2. Go into the repository

```bash
$ cd hshca
```

3. Install Hyper Spectral HCA with setup.py

```bash
$ python setup.py install
```

## Usage
Please refer to [this script](https://github.com/MiLL4U/hshca/blob/master/example.py) for an example of execution (need to install [matplotlib](https://github.com/matplotlib/matplotlib)).
### Import HyperSpectralHCA, link method, and metric
```python
from hshca import HyperSpectralHCA
from hshca.linkmethod import Ward
from hshca.metric import Euclidean
```

### Load hyperspectral image as NumPy array
```python
import numpy as np

image = np.load("./test/test_cell.npy")
```

### Run HyperSpectralHCA
```python
hshca = HyperSpectralHCA(
    data=image,  # hyperspectral image to analyze
    method=Ward,  # linkage method
    spectral_metric=Euclidean,  # metric
    spatial_dist_factor=1e-5,  # factor for spatial distance (lambda)
    spatial_scale=(1.0, 1.0),  # scale for each spatial axis (x, y, ...)
    show_progress=True  # set True here if you want to see progress bar
)
hshca.compute()
```

### Get result
```python
cluster_map = hshca.get_cluster_map(5)  # number of clusters
average_spectra = hshca.get_average_vectors(5)
```

### show results (using matplotlib)
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, layout='constrained')
cluster_map = ax[0].imshow(cluster_map, cmap='Paired')
for cluster_idx, spectra in enumerate(average_spectra):
    ax[1].plot(spectra, label=f"Cluster {cluster_idx}")
ax[1].legend()
fig.colorbar(cluster_map, ax=ax[0], orientation='horizontal',  # type: ignore
             label="Cluster index")
plt.show()
```
