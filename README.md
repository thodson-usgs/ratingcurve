# ratingcurve
*A Python library for fitting stage-discharge rating curves.*

Use `ratingcurve` to fit streamflow ratings with a segmented power law,
which is the the most common model used by USGS.

At this time, the library is for research and is not ready for operation. 

## Installation
Install using pip
```sh
pip install ratingcurve
```
or conda
```sh
# acreate a new environment
conda create -n ratingcurve
conda activate ratingcurve
conda install -c conda-forge ratingcurve
# add environment to jupyter
python -m ipykernel install --user --name=ratingcurve
```

## Getting Started
The [`segmented-power-law-demo.ipynb`](https://github.com/thodson-usgs/ratingcurve/blob/main/notebooks/segmented-power-law-demo.ipynb)
notebook demonstrates basic use of the package.
You can try it out in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thodson-usgs/ratingcurve/blob/master/notebooks/segmented-power-law-demo.ipynb)

or locally using the environment created in the previous step
```sh
conda activate base # or your prefered jupyter lab environment
jupyter lab
```
then open the notebook and select the `ratingcurve` kernel that was installed earlier.

A simple example is given below.

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
                                   
# fit model, then simulate the rating
with powerrating:
    mean_field = pm.fit(n=150_000)
    trace = mean_field.sample(5000)
    
powerrating.plot(trace)
```
![example plot](https://github.com/thodson-usgs/ratingcurve/blob/main/paper/green_example.png?raw=true)

See the [notebooks](https://github.com/thodson-usgs/ratingcurve/tree/main/notebooks) for more examples.

## Disclaimer

This software is preliminary or provisional and is subject to revision. 
It is being provided to meet the need for timely best science.
The software has not received final approval by the U.S. Geological Survey (USGS).
No warranty, expressed or implied, is made by the USGS or the U.S. Government as to the functionality of the software and related material nor shall the fact of release constitute any such warranty. 
The software is provided on the condition that neither the USGS nor the U.S. Government shall be held liable for any damages resulting from the authorized or unauthorized use of the software.
