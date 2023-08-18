# Implementation
We implemented our curve-fitting algorithm, along with some additional plotting and evaluation tools, test datasets, and tutorials, as a Python package called `ratingcurve`. 
The algorithm is written using `PyMC` {cite:p}`Salvatier_2016`,
an open-source community-developed library for Bayesian statistical modeling and probabilistic machine learning.
Using a probabilistic programming framework like `PyMC`, the core algorithm can be expressed in several lines of code,
making it relatively easy to extend or modify,
such as by changing the priors, the structure of the rating-curve itself, or switch between different approximation algorithms that trade off speed and accuracy.
We demonstrate one such tradeoff,
switching between faster Automatic Differentiation Variational Inference (ADVI) {cite:p}`Kucukelbir_2017`
and more accurate Hamiltonian Monte Carlo with the No-U-Turn Sampler (NUTS) {cite:p}`Hoffman_2014`.

## Parameterization

Our rating curve model is functionally equivalent to that of {cite:t}`Reitan_2008`
but adapts the parameterization used by {cite:t}`Muggeo_2003` for piecewise regression.

Streamflow or discharge $Q$ is a function of stage $h$ and parameterized as
\begin{align}
    B &= \ln(\max(h - h_s, 0) + h_o) \\
    \ln(Q) &= a + w^T B + \epsilon
\end{align}
where
$h_s$ is a vector containing the stage of each segment breakpoint;
$h_o$ is a vector of offsets, the first is 0 and the rest are 1;
$\max$ is the element-wise maximum, which returns a vector of size $h_s$;
$a$ is a bias parameter;
$w$ is a vector of weights;
and $\epsilon$ is the error.

{cite:t}`Reitan_2008` parameterize the rating curve as separate control segments.
As stage increases, the flow is divided among segments, filling from the bottom up,
similar to how different controls become active as stage increases in the channel.
Once stage rises beyond the range of a particular control, that control is "drowned out" and its flow ceases to increase with stage.
Our parameterization is simpler though less physical.
As stage increases, new segments are activated, but their effect never drowns out:
each successive $w_i$ term makes cumulative adjustments to the base slope $w_1$.

The current priors and optimizer settings are documented in the package;
in general, they do not need to modified.
Besides selecting the number of segments, the user can specify a prior distribution on the breakpoints.
The default assumes the breakpoints are uniformly distributed, which works well for general use.
Alternatively, the user can specify the approximate location of each breakpoint using a normal distribution,
like if they had other evidence that a breakpoint occurred at a particular stage. 
With the normal prior, the user specifies the expected stage for each breakpoint and their uncertainty about it.

Like {cite:t}`Reitan_2008`, we assume $\epsilon$ is normally distributed with mean zero and variance $\sigma^2$, $\epsilon \sim N(0, \sigma^2)$.
That simplification can create unaccounted heteroscedasticity {cite:p}`Petersen_Overleir_2004`
but generally yields a reasonable estimate for the rating and its uncertainty.
Other error distributions could be tested and incorporated in the package;
here, our intent is to define a baseline algorithm equivalent to {cite:t}`Reitan_2008`,
while being relatively simple and easy to extend.
