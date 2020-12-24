# Leakage Assessment

**Material related to the Master thesis of Chris Berg.**

Master thesis performed as part of an MSc programme in Computer Science. 
Thesis performed under supervision of Dr. [S. Picek](https://www.tudelft.nl/en/staff/s.picek/) at the 
[Artificial Intelligence and Security](https://www.aisylab.com/) lab of the Delft University of Technology.

## Contact
- University mail: [c.c.berg@student.tudelft.nl](c.c.berg@student.tudelft.nl)
- Other: [ccberg.nl/contact](https://ccberg.nl/contact)

## Setup
1. **Clone repository**: All commands and paths in this readme are relative to the root of the repository.
1. **Fetch data**: Clone the ASCAD repo in `/data/` and rename it's root dir to `/data/ASCAD`. 
   Extract the fixed and variable key traces as described in their repo.
1. **Setup conda**: Create the `LA` environment using `conda env create -f ./support/conda.yml` and then activate it 
   using `conda activate LA`.

## Getting started

### Project

- `./books/` contains a collection of Jupyter notebooks which form a large part of the experiments done in this 
  master thesis. 
- `./src/` contains a toolbox for powering the experiments in the Jupyter notebooks. 
- `./report/` contains a copy of the Master thesis report.

### Caching
`./.cache/` contains all results from long-running functions in the report. As the cache might grow large 
(order of `GiB`s) it is not included in this repo. A copy of the original cache can be requested from the author.
   
*The cache's subdirectory structure follows that of `./books/`. Leaf directories in the cache correspond to individual
notebooks.*
