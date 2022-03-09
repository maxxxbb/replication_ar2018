# Final Project: Replication of Augenblick&Rabin
### Author

- Max Boehringer (University of Bonn, s6mxboeh@uni-bonn.de)


### About

This repository contains code to replicate the main structural estimates in the following paper:

- Ned Augenblick and Matthew Rabin, 2019, "An Experiment on Time Preference and Misprediction in Unpleasant Tasks", *Review of Economic Studies*, 86(3): 941&ndash;975
	- Estimation Method: Maximum Likelihood
	- Source of Heterogeneity: Implementation Errors.

This project is aimed at applying programming techniques learned in the course "Effective Programming Practices for Economists" to the replication of said paper of Pozzi&Nunnari(2022) (Bocconi University). Their replication is openly available [here](https://github.com/MassimilianoPozzi/python_julia_structural_behavioral_economics/blob/main/README.md).


### Requires
In order to run this project on your local machine you need to have installed Python, an Anaconda distribution and LaTex distribution in order to compile .tex documents.

The project was created on Windows 10 using

- Anaconda 4.11.0
- Python 3.9.10
- MikTex 22.1

1. All necessary python dependencies are contained in environment.yml . To install the virtual environment in a terminal move to the root folder of the repository and type `$ conda env create -f environment.yml` and to activate type  `$ conda activate replication_ar2018`.

2. In order for imports to properly work in a terminal move to the root of the repository and type `$ conda develop .`

3. This project relies on pytask. To run the project once the repository is cloned to your local machine and above steps are completed just type
`$ pytask`

### How Do I Navigate This Repository?

The repository was set up using the [Templates for Reproducible Research Projects in Economics](https://econ-project-templates.readthedocs.io/en/latest/index.html) and follows its structure. All the source code and data is in the /src while all output will be generated into the bld folder. An extensive documentation of the code in source folder will be created by running  `$ pytask`. The /bld folder will contain following subfolders after running all tasks:

- **data**: prepared data
- **documentation**: documentation on the code in /src
- **estimation**: estimation results for generating tables and graphs.
- **figures** :  plots
- **paper** : final pdf
- **tables** : tables of estimates and summary statistics


### Following tools where used in this project:

- Gabler , Janos, 2021: [A Python Tool for the Estimation of (Structural) Econometric Models.](https://github.com/OpenSourceEconomics/estimagic)
- Raabe, Tobias, 2020: [A Python tool for managing scientific workflows.](https://github.com/pytask-dev/pytask)
- von Gaudecker, Hans-Martin , 2019: [Templates for Reproducible Research Projects in Economics](https://doi.org/10.5281/zenodo.2533241),

