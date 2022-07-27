# ratingcurve
A python library for fitting stage-discharge rating curves.

For the moment, the library only fits segmented power law, 
which is the the most common type of rating curve used by USGS.

The general form of the equation is:

\begin{align}
    log(Q) = a + \sum b_i\log(x - x_{o,i})H_i(x - x_{o,i})
\end{align}
where
$Q$ is discharge,  
$a$ and $b$ are model parameters,  
$x$ is stage,  
$x_{o,i}$ is the $i$th breakpoint, and  
$H$ is the Heaviside function.  
In a standard linear model $b$ represents the slope of the function with respect the input.
In the segmented power law $b_o$ is the slope and each subsequent $b_i$ are adjustment to the base slope for each segment.

This library is for experimental purposes only. To inquire about the project, create an issue.