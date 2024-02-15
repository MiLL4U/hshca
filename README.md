# Hyper Spectral HCA
Hyper Spectral Hierarchical Clustering Analysis (HSHCA) provides [hierarchical clustering analysis (HCA)](https://en.wikipedia.org/wiki/Hierarchical_clustering) specialized for hyperspectral images that can be measured by techniques such as [Raman microscopy](https://raman.oxinst.com/techniques/raman-imaging).

Generally, when applying HCA to hyperspectral images, the numerous spectra composing the image are treated as vectors in independent multidimensional spaces. Therefore, information about spatial coordinates (unit example: μm) at each measurement point of the spectra is lost.

HSHCA efficiently clusters spectra originating from physically adjacent measurement points by using both the distances between spectra ($d_{spectral}$) and the distances in real space between measurement points ($d_{spatial}$) as the definition of distance when performing HCA.

## Principle
In HSHCA, the distance between cluster A and cluster B ($d(A,B)$) is processed in HCA using the distance between spectra ($d_{spectral}(A,B)$) and the distance in real space ($d_{spatial}(A,B)$) as:
$$
d(A,B) = d_{spectral}(A,B) + \lambda\cdot d_{spatial}(A,B)
$$

Since $d_{spectral}$  and $d_{spatial}$ have different dimensions, $\lambda$ is used as a coefficient to scale these values to the same magnitude. Note that $\lambda$ is a **hyperparameter** that needs to be adjusted according to measurement conditions and other factors.

$d_{spectral}$ is defined by metrics and methods commonly used in traditional HCA algorithms. For example, when using the Euclidean distance, $d_{spectral}$ corrensponds to the square root of the sum of squared differences of two spectra, and when using the cityblock distance, $d_{spectral}$ corresponds to the sum of the absolute differences of two spectra.

$d_{spatial}$ is defined as the Euclidean distance in real space, which is represented by the formula:
$$
d_{spatial}(A,B) = \sqrt{(x_A - x_B)^2 + (y_A - y_B)^2}
$$
where $(x_A, y_A)$ and $(x_B, y_B)$ are the spatial coordinates of points A and B, respectively.
