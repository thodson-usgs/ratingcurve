# ratingcurve
*A Python library for fitting stage-discharge rating curves.*

Use `ratingcurve` to fit streamflow ratings with a segmented power law,
which is the the most common type rating model used by USGS.

The segmented power law is parameterized as:

$$
    \log(Q) = a + \sum_{i=1}^{n} b_i \log(x - x_{o,i}) H_i(x - x_{o,i})
$$

where
$Q$ is a vector discharge, \
$n$ is the number of breakpoints in the rating, \
$a$ and $b$ are model parameters, \
$x$ is a vector of stage observations, \
$x_o$ is a vector of breakpoints, and \
$H$ is the Heaviside function. 

In a standard linear model $b$ represents the slope of the function with respect the input.
In the segmented power law $b_o$ is the slope and each subsequent $b_i$ are adjustment to the base slope for each segment.

This library is for experimental purposes only.

## Installation

```sh
conda env create -f environment.yaml # use mamba if possible

# add environment to jupyter
conda activate ratingcurve
python -m ipykernel install --user --name=ratingcurve
```

## Getting Started
The [`segmented-power-law-demo.ipynb`](https://github.com/thodson-usgs/ratingcurve/blob/main/notebooks/segmented-power-law-demo.ipynb)
notebook demonstrates basic use of the package.
To run the notebook click the Google Colab badge \
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thodson-usgs/ratingcurve/blob/master/notebooks/segmented-power-law-demo.ipynb)

or to run the notebook locally
```sh
conda activate base # or your jupyter lab environment
jupyter lab
```
then open the notebook and select the `ratingcurve` kernel that was installed earlier.

A simple example is given below.

```python
from ratingcurve.ratingmodel import SegmentedRatingModel

# load tutorial data
df = tutorial.open_dataset('green_channel')


# setup model
segments = 2

powerrating = SegmentedRatingModel(q=df['q'],
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


## Disclaimer

This software is in the public domain because it contains materials that originally came from the U.S. Geological Survey,
an agency of the United States Department of Interior. For more information, see the
[official USGS copyright policy](https://www2.usgs.gov/visual-id/credit_usgs.html#copyright).

Although this software program has been used by the U.S. Geological Survey (USGS),
no warranty, expressed or implied, is made by the USGS or the U.S. Government
as to the accuracy and functioning of the program and related program material nor shall the fact of distribution
constitute any such warranty, and no responsibility is assumed by the USGS in connection therewith.

This software is provided “AS IS.”

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png)](http://creativecommons.org/publicdomain/zero/1.0/)
