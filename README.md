# ratingcurve
*A Python package for fitting hydrologic rating curves.*

In hydrology, a rating curve is a mathematical relationship between streamflow and water surface elevation (stage).
Because stage is much easier to measure than streamflow, almost all streamflow timeseries are generated from rating curves.
For the most part, those rating curves are still fit manually by drawing a curve to the data,
which can be time consuming and subjective.
To improve that process, the U.S. Geological Survey (USGS), among others, is evaluating methods for automating that fitting. 
Several automated methods currently exist, but each parameterizes the rating curve slightly differently,
and because of the nature of the problem, those slight differences can greatly affect performance.
To help the community evaluate different parameterizations,
we created the `ratingcurve` package, which implements our best parameterization for others to try.
Furthermore, the implementation uses [PyMC](https://www.pymc.io/welcome.html), a general purpose library for probabilistic modeling, 
which makes it easier for others to modify the model to test different parameterizations or fitting algorithms.
If you can improve upon our parameterization, USGS might use your algorithm to generate streamflow timeseries at thousands of locations around the United States.
The package includes simple demonstrations and test datasets to get you started.

Please report any bugs, suggest enhancements, or ask questions by creating an [issue](https://github.com/thodson-usgs/ratingcurve/issues).
  
## Installation
Install using pip
```sh
pip install ratingcurve
```
or conda
```sh
conda install -c conda-forge ratingcurve
```

## Basic Usage
This [`tutorial`](https://github.com/thodson-usgs/ratingcurve/blob/main/docs/notebooks/segmented-power-law-tutorial.ipynb) demonstrates basic usage of fitting and plottig a rating curve

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thodson-usgs/ratingcurve/blob/main/docs/notebooks/segmented-power-law-tutorial.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/thodson-usgs/ratingcurve/main?labpath=docs%2Fnotebooks%2Fsegmented-power-law-tutorial.ipynb)

```python
from ratingcurve.ratings import PowerLawRating
from ratingcurve import data

# load tutorial data
df = data.load('green channel')

# initialize the model
powerrating = PowerLawRating(segments=2)
                                   
# fit the model
trace = powerrating.fit(q=df['q'],
                        h=df['stage'], 
                        q_sigma=df['q_sigma'])
powerrating.plot()
```
![example plot](https://github.com/thodson-usgs/ratingcurve/blob/main/docs/assets/green-channel-rating-plot.png?raw=true)


Once fit, easily generate a rating table that can be imported into other applications.
```python
powerrating.table()
```

For more, see the [documentation](https://thodson-usgs.github.io/ratingcurve/meta/intro.html).

## Disclaimer

This software is preliminary or provisional and is subject to revision. 
It is being provided to meet the need for timely best science.
The software has not received final approval by the U.S. Geological Survey (USGS).
No warranty, expressed or implied, is made by the USGS or the U.S. Government as to the functionality of the software and related material nor shall the fact of release constitute any such warranty. 
The software is provided on the condition that neither the USGS nor the U.S. Government shall be held liable for any damages resulting from the authorized or unauthorized use of the software.
