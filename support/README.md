### Some tools and instructions for maintaining this repository.

**All commands are meant to be executed from the root dir of this repository**.

### Submodules
- Add submodule `git submodule add git@github.com:ccberg/LA-report.git ./report/tex`.
- Update git submodules `git submodule update --recursive`.
- Git module config: `./.gitmodules`.

### Conda
- Export the conda env: `conda env export > ./support/conda.yml`
- Update YAML file: `conda env update -f ./support/conda.yml`