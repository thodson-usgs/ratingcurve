# ratingcurve
*A python library for fitting stage-discharge rating curves.*

[![DOI](https://zenodo.org/badge/484096335.svg)](https://zenodo.org/badge/latestdoi/484096335)

Use `ratingcurve` to fit streamflow ratings with a segmented power law,
which is the the most common type rating model used by USGS.

The general form of the equation is:

$$\log(Q) = a + \sum_{i=1}^{n} b_i \log(x - x_{o,i}) H_i(x - x_{o,i})$$

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
h_obs = df['stage'].values.reshape(-1, 1)
q_obs = df['q'].values.reshape(-1, 1)

# setup model
segments = 2
powerrating = SegmentedRatingModel(q_obs, h_obs,  segments=segments,
                                   prior = {'distribution':'uniform'})

# fit model, then simulate the rating
with powerrating:
    mean_field = pm.fit(n=150_000)
    trace = mean_field.sample(5000)
    
powerrating.plot(trace)
```

##  Development
The included notebook demonstrates fitting stage-discharge ratingings using a segmented power law and cubic spline.
Depending on the dataset, one approach may work better than another. 

Because of how USGS applies its rating models in practice, we are free to choose any model $f()$ for fitting a rating

$$q = f(\theta,s)$$

where f is the functional form of the rating: power law, spline, NN, etc.

This is because USGS doesn't use the rating model directly.
Instead the model is discretized to form a stage-rating lookup table

$$d(f(\theta,s)) = \begin{bmatrix} s & \hat q \end{bmatrix}$$

We typically don't care how that table was generated,
we just want to know the predicted discharge for a given stage. 


1. To develop a rating, select a set of observations $(q_1, s_1)$ and weights $w_1$, fit a rating model and  discretize to yield $\hat q_1$.

1. At a later point in time, develop a new rating from another (perhaps overlapping) set of observations $(q_2, s_2, w_2)$ and discretize as $\hat q_2$.

1. As we accrue more ratings, we form a matrix $q_{ij}$, where $i$ is the rating and $j$ is the stage index. Flow at a particular time and stage  $(t, s)$ is estimated by interpolating between elements in this matrix.

1. After many ratings, we can compute shift uncertainty at each stage $q_{,j}$.
