# ISMgas

ISMgas is a python module used to analyze absorption line kinematics of Interstellar Medium gas. The module takes a 1D spectra as an input and extracts key kinematic diagnostics (such as velocity centroid of absorption) by fitting the absorption profile with multiple gaussian profiles. 

### Installation and development

```shell
git clone https://github.com/kvgc153/ISMgas.git
cd ISMgas/
conda create --name ISMgas
conda activate ISMgas
conda install python==3.8
pip install -e .
```


This repository is under constant development. Kindly report any issues that you encounter and/or submit a PR. 

### Tutorials
- [Fit a double Gaussian to an absorption profile using ISMgas.fitting](ISMgas/fitting/README.md)
- [Introduction to the ISMgas.AnalyzeSpectra baseclass](ISMgas/spectra/README.md)

### Papers using ISMgas 

- ['Resolved velocity profiles of galactic winds at Cosmic Noon' by Vasan G.C., et al (2022)](https://ui.adsabs.harvard.edu/abs/2022arXiv220905508K/abstract).

- ['Spatially Resolved Galactic Winds at Cosmic Noon: Outflow Kinematics and Mass Loading in a Lensed Star-Forming Galaxy at z=1.87'](https://ui.adsabs.harvard.edu/abs/2024arXiv240200942K/abstract)
