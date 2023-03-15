# Implementation
We implemented our curve-fitting algorithm, along with some additional plotting and evaluation tools, test datasets, and tutorials, as a Python package called \texttt{ratingcurve}. 
The algorithm is written using \texttt{PyMC} \citep{Salvatier_2016},
an open-source community-developed library for Bayesian statistical modeling and probabilistic machine learning.
Using a probabilistic programming framework like \texttt{PyMC}, the core algorithm can be expressed in several lines of code,
making it relatively easy to extend or modify,
such as by changing the priors, the structure of the rating-curve itself, or switch between different approximation algorithms that trade off speed and accuracy.
We demonstrate one such tradeoff,
switching between faster Automatic Differentiation Variational Inference \citep[ADVI;][]{Kucukelbir_2017}
and more accurate Hamiltonian Monte Carlo with the No-U-Turn Sampler \citep[NUTS;][]{Hoffman_2014}.

## Parameterization

Our rating curve model is functionally equivalent to that of \citet{Reitan_2008}
but adapts the parameterization used by \citet{Muggeo_2003} for piecewise regression.

Streamflow or discharge $Q$ is a function of stage $h$ and parameterized as
\begin{align}
    \label{eq:parameterization}
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

\citet{Reitan_2008} parameterize the rating curve as separate control segments.
As stage increases, the flow is divided among segments, filling from the bottom up,
similar to how different controls become active as stage increases in the channel.
Once stage rises beyond the range of a particular control, that control is "drowned out" and its flow ceases to increase with stage.
Our parameterization is simpler though less physical.
As stage increases, new segments are activated, but their effect never drowns out:
each successive $w_i$ term makes cumulative adjustments to the base slope $w_1$.

Unless we erred in implementing \citet{Reitan_2008},
the less physical parametrization performed better in our testing.
Other parametrizations, priors, or optimization algorithms might be even better,
and we intend to update the package as we learn of them. 

The current priors and optimizer settings are documented in the package;
in general, they do not need to modified.
Besides selecting the number of segments, the user can specify a prior distribution on the breakpoints.
The default assumes the breakpoints are uniformly distributed, which works well for general use.
Alternatively, the user can specify the approximate location of each breakpoint using a normal distribution,
like if they had other evidence that a breakpoint occurred at a particular stage. 
With the normal prior, the user specifies the expected stage for each breakpoint and their uncertainty about it.

Like \citet{Reitan_2008}, we assume $\epsilon$ is normally distributed with mean zero and variance $\sigma^2$, $\epsilon \sim N(0, \sigma^2)$.
That simplification can create unaccounted heteroscedasticity \citep{Petersen_Overleir_2004}
but generally yields a reasonable estimate for the rating and its uncertainty.
Other error distributions could be tested and incorporated in the package;
here, our intent is to define a baseline algorithm equivalent to \citet{Reitan_2008},
while being relatively simple and easy to extend.

## Priors
TODO

## Optimization Settings
TODO