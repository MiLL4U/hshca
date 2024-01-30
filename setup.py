from typing import List
import setuptools


def _requires_from_file(filename: str) -> List[str]:
    return open(filename).read().splitlines()


setuptools.setup(
    name="hshca",
    version="0.1.0",
    install_requires=_requires_from_file('requirements.txt'),
    author="Hiroaki Takahashi",
    author_email="aphiloboe@gmail.com",
    url="https://github.com/MiLL4U/hshca",
    description="Hierarchical cluster analysis for hyperspectral image",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
