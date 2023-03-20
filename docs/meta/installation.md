# Installation
`ratingcurve` is avaiable on PyPI and conda-forge and can be installed by calling

```sh
pip install ratingcurve
```
or conda
```sh
conda install -c conda-forge ratingcurve
```

## Creating a Jupyter Environment
If using `ratingcurve` in a Jupyter Notebook,
consider building it into an `ipykernel`
```sh
# acreate a new environment
conda create -n ratingcurve
conda activate ratingcurve
conda install -c conda-forge ratingcurve
# add environment to jupyter
python -m ipykernel install --user --name=ratingcurve

jupyter lab
```
Now, when opening a new notebook, select the kernel `ratingcurve` in the Jupyter Lab launcher.
