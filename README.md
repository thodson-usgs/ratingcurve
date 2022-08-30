# ratingcurve
*A python library for fitting stage-discharge rating curves.*

[![DOI](https://zenodo.org/badge/484096335.svg)](https://zenodo.org/badge/latestdoi/484096335)

Use `ratingcurve` to fit streamflow ratings with a segmented power law,
which is the the most common type rating model used by USGS.

The general form of the equation is:
```math
log(Q) = a + \sum b_i \log(x - x_{o,i}) H_i(x - x_{o,i})
```
where
$Q$ is discharge,  
$a$ and $b$ are model parameters,  
$x$ is stage,  
$x_{o,i}$ is the $i$-th breakpoint, and  
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

##  Development
The included notebook demonstrates fitting stage-discharge ratingings using a segmented power law and cubic spline.
Depending on the dataset, one approach may work better than another. 

Because of how USGS applies its rating models in practice, we are free to choose any model $f()$ for fitting a rating

\begin{align}
q = f(\theta,s)
\end{align}

where f is the functional form of the rating: power law, spline, NN, etc.

This is because USGS doesn't use the rating model directly.
Instead the model is discretized to form a stage-rating lookup table

\begin{align}
d(f(\theta,s)) = \begin{bmatrix} s & \hat q \end{bmatrix}
\end{align}

We typically don't care how that table was generated,
we just want to know the predicted discharge for a given stage. 


1. To develop a rating, select a set of observations ($q_1$,$s_1$) and weights $w_1$, fit a rating model and  discretize to yield $\hat q_1$.

1. At a later point in time, develop a new rating from another (perhaps overlapping) set of observations ($q_2$, $s_2$, $w_2$) and discretize as $\hat q_2$.

1. As we accrue more ratings, we form a matrix $q_{ij}$, where $i$ is the rating and $j$ is the stage index. Flow at a particular time and stage ($t$,$s$) is estimated by interpolating between elements in this matrix.

1. After many ratings, we can compute shift uncertainty at each stage $q_{,j}$.
