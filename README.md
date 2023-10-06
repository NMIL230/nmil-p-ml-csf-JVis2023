## Citation
"Contrast Response Function Estimation with Nonparametric Bayesian Active Learning"
Dom CP Marticorena, Quinn Wai Wong, Jake Browning, Ken Wilbur, Samyukta Jayakumar, Pinakin Davey, Aaron R. Seitz, Jacob R. Gardner, Dennis L. Barbour
_Journal of Vision_
https://www.medrxiv.org/content/10.1101/2023.05.11.23289869v2

## Description

The code repository recreates the figures published in the reference above and is structured with the following directories
- `data/` stores raw data and results files created from figures
- `utility/` stores the shared methods and classes called across code
- `analysis/` stores all outputs generated by the figure notebooks
- `QuickCSF/` stores the modified quickCSF code used to train qCSF models

## Setup

### Environment

To create a conda environment with the relevant packages, ensure you have conda installed (installation [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)). Then navigate to the root path of the repo and run:

`conda env create -f nmil-p-ml-csf-env-JVis2023.yml`

This will create a conda environment named `ml-csf`. To use the environment:

`conda activate ml-csf`

### Informative Prior

To create your own prior, train your desired GP and save the `state_dict()` and `Xt` of the model in `data/priors/` with file names of `prior_mean_model_state_<NAME_OF_PRIOR>.pth` and `prior_mean_model_Xt_<NAME_OF_PRIOR>.pth` respectively. Doing so ensures that your prior file will work with existing notebook code such as `figure_06.ipynb` and `figure_09.ipynb`. The current informative prior was created using 250 randomly generated points from each canonical textbook phenotype from experiment 1. More details on the parameters used can be found in the json file within `data/priors`.
