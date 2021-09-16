# Leakage Assessment

**Material related to the Master thesis of Chris Berg.**

Master thesis performed as part of an MSc programme in Computer Science. 
Thesis performed under supervision of Dr. [S. Picek](https://www.tudelft.nl/en/staff/s.picek/) at the 
[Artificial Intelligence and Security](https://www.aisylab.com/) lab of the Delft University of Technology.

## Contact
- University mail: [c.c.berg@student.tudelft.nl](mailto:c.c.berg@student.tudelft.nl)
- Other: [la@ccberg.nl](mailto:la@ccberg.nl)

## Setup

1. **Clone repository**: All commands and paths in this readme are relative to the root of the repository. 
1. **Create dir roots**: Create data directory roots (`/data` and `/data/LA`) or update their default values in 
`./src/config.py`.
1. **Fetch data**: 
   - Clone the ASCAD repo in `/data/` and rename it's root dir to `/data/ASCAD`. 
   Extract the fixed and variable key traces as described in their repo. (`~70GiB` of data)
   - **Alternative**: Skip point of interest extraction by downloading preprocessed data 
     [here](https://ccberg.nl/la/preprocessed-data/) (`~634MB` of data). Extract the three directories from this archive
     to the preprocessed data directory root `/data/LA/`. 
1. **Setup conda environment**: Create the `LA` environment using `conda env create -f ./support/conda.yml` and then activate it 
   using `conda activate LA`.

## Getting started

### Project

- `./report/` contains a collection of Jupyter notebooks containing the experiments described in the master 
  thesis report. 
- `./src/` contains a toolbox for powering the experiments in the Jupyter notebooks. 
- `./thesis-report/` contains a copy of the Master thesis report.

### Caching
`./report/**/.cache/` contains pre-processed data from experiments that take a long time to run. As the cache might grow
large (order of `GiB`s) it is not included in this repository. A copy of the original caches can be requested from the 
author.
