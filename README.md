# ratingcurve
*A Python library for fitting hydrologic rating curves.*

In hydrology, a rating curve is a mathematical relationship between streamflow and water surface elevation (stage).
Because stage is much easier to measure than streamflow, almost all streamflow timeseries are generated from rating curves.
Historically, those ratings were fitted by hand, which can be time consuming and error prone.
`ratingcurve` provides an easy-to-use algorithm for fitting the standard form of rating curve, the segmented power law.

## Installation
Install using pip
```sh
pip install ratingcurve
```
or conda
```sh
conda install -c conda-forge ratingcurve
```

## Getting Started
The [`segmented-power-law-demo.ipynb`](https://github.com/thodson-usgs/ratingcurve/blob/main/notebooks/segmented-power-law-demo.ipynb)
notebook demonstrates basic usage of the package.
Try it locally or in Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thodson-usgs/ratingcurve/blob/master/notebooks/segmented-power-law-demo.ipynb).

```python
from ratingcurve.ratingmodel import PowerLawRating
from ratingcurve import data

# load tutorial data
df = data.load('green channel')

# setup model
segments = 2
powerrating = PowerLawRating(q=df['q'],
                             h=df['stage'], 
                             q_sigma=df['q_sigma'],
                             segments=segments)
                                   
# fit the model
trace = powerrating.fit()
powerrating.plot(trace)
```
![example plot](https://github.com/thodson-usgs/ratingcurve/blob/main/paper/green_example.png?raw=true)

See the [notebooks](https://github.com/thodson-usgs/ratingcurve/tree/main/notebooks) for more examples.

## Environment setup
In practice, many prefer to create a new environment and and install it using `ipykernel`.
```sh
# acreate a new environment
conda create -n ratingcurve
conda activate ratingcurve
conda install -c conda-forge ratingcurve
# add environment to jupyter
python -m ipykernel install --user --name=ratingcurve

jupyter lab
```
Now select the kernel `ratingcurve` in the Jupyter Lab launcher.

## Disclaimer

This software is preliminary or provisional and is subject to revision. 
It is being provided to meet the need for timely best science.
The software has not received final approval by the U.S. Geological Survey (USGS).
No warranty, expressed or implied, is made by the USGS or the U.S. Government as to the functionality of the software and related material nor shall the fact of release constitute any such warranty. 
The software is provided on the condition that neither the USGS nor the U.S. Government shall be held liable for any damages resulting from the authorized or unauthorized use of the software.
