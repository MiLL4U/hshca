# Hyper Spectral HCA
Hyper Spectral Hierarchical Clustering Analysis (HSHCA) provides [hierarchical clustering analysis (HCA)](https://en.wikipedia.org/wiki/Hierarchical_clustering) specialized for hyperspectral images that can be measured by techniques such as [Raman microscopy](https://raman.oxinst.com/techniques/raman-imaging).

Generally, when applying HCA to hyperspectral images, the numerous spectra composing the image are treated as vectors in independent multidimensional spaces. Therefore, information about spatial coordinates (unit example: μm) at each measurement point of the spectra is lost.

HSHCA efficiently clusters spectra originating from physically adjacent measurement points by using both the distances between spectra ($d_{spectral}$) and the distances in real space between measurement points ($d_{spatial}$) as the definition of distance when performing HCA.
