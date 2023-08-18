# Background
Streamflow timeseries are widely used in hydrologic research, water resource management, and flood forecasting,
but they are difficult to measure directly.
In nearly all timeseries applications, streamflow is estimated from rating curves or ``ratings'' that describe the relationship between streamflow and an easy-to-measure predictor like stage.
The shape of the rating is specific to each streamgage and is governed by channel conditions at or downstream from the gage, referred to as controls.
Section controls, like riffles or weirs, occur downstream of the gage, whereas channel controls, like the geometry of the banks, occur at the gage.
Regardless of the type, the behavior of each control is often well-approximated with standard hydrologic equations that take the general form of a power law with a location parameter
\begin{equation}
    Q = C(h - h_0)^{b}
\end{equation}
where $Q$ is the discharge (streamflow);
$h$ is the height of the water above some datum (stage);
$h_0$ is the stage of zero flow (the location or offset parameter);
$(h-h_0)$ is the hydraulic head {cite:p}`ISO18320_2020`;
$b$ is the slope of the rating curve when plotted in log-log space;
$C$ is a scale factor equal to the discharge when the head is equal to one;
When multiple controls are present, the rating curve is divided into segments
with one power-law corresponding to each control resulting in a multi-segment or compound rating.

Though several automated methods exist, most ratings are still fit by hand using a graphical method of plotting stage and discharge in log-log space.
With the appropriate location parameter, each control can be fit to a straight-line segment in log space {cite:p}`Kennedy_1984, ISO18320_2020`.
Variants of this method have been used for decades,
first with pencil and log paper
and now with computer-aided software, though fitting is still done by manually adjusting parameters until an acceptable fit is achieved.

While single-segment ratings are relatively easy to fit by automated methods {cite:p}`Venetis_1970`,
compound ratings are more challenging, because their solution is multimodal,
meaning it has multiple optima {cite:p}`Reitan_2006`.
As a result, standard optimization algorithms can become stuck in local optima and fail to converge to the global optimum.
General function approximators, such as natural splines {cite:p}`Fenton_2018` or neural networks,
can be easier to fit but their generality comes at a cost.
The form of the power law matches that of the hydrologic equations governing uniform open-channel flow,
like the Manning equation {cite:p}`Manning_1891`.
Due to that physical basis, power laws are potentially more robust than other generic curve-fitting functions:
requiring less data to achieve the same fit and being less prone to overfitting.

This paper describes a basic algorithm for fitting compound ratings and compares its performance against a natural spline.
Similar algorithms already exist, notably those of {cite:t}`Reitan_2008, Le_Coz_2014`,
as well as the so-called generalized power law {cite:p}`Hrafnkelsson_2021`.
Like the algorithm described here, each of these examples is Bayesian, meaning they can utilize prior information to help constrain the solution space and reduce multimodality.
Simple examples of priors include constraining the exponent $b$ to a narrow range around the value of 5/3,
or constraining the number of rating segments,
or constraining the transitions between segments around a particular stage.
Being Bayesian, they also inherently estimate uncertainty in the fitted parameters and discharge,
which is important for many applications.

Our algorithm differentiates itself in two main ways.
First, it minimally reproduces the manual method used by hydrologists for decades while being both robust and fast;
the user needs only to provide stage-discharge pairs and specify the number of segments in the rating, but even that can be inferred from the data.
Second, it is implemented in Python using a popular community-developed open-source probabilistic programming library.
As a result, most of the underlying numerical code is maintained by a broader community of developers,
enabling us to focus on the particular nuances of parameterizing a rating curve model,
as well as making it easier to extend or modify the algorithm
or switch between different approximation algorithms to achieve different tradeoffs of speed and accuracy.

```{bibliography}
```
