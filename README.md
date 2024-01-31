# Bayesian Estimation of the AGN Structure Function in the case of Poisson Data

This project provides code (python script SF.py) to determine the X-ray Structure Function of a population of Active Galactic Nuclei (AGN) for which two epoch X-ray observations are available and are separated by rest frame time interval $\Delta T$. The calculation of the X-ray structure function follows the Bayesian methodology described in [Georgakakis et al. (2024)](https://arxiv.org/abs/2401.17285). The data used in that paper to estimate the structure function of SDSS DRQ16 QSOs is available on [zenodo](https://zenodo.org/records/10560969).

The sampling of the likelihood uses the python wrapper of the Stan platform (https://mc-stan.org/) for statistical modeling and high-performance statistical computation. Installation of the Stan python wrapper (https://pystan.readthedocs.io/en/latest/) is therefore required to run the code. 