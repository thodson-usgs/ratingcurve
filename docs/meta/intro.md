# Motivation
*A simple reference algorithm for fitting multi-segment rating curves*

Streamflow is one of the most important variables in hydrology but is costly to measure.
As a result, nearly all streamflow timeseries are estimated from rating curves
that define a mathematical relationship between streamflow and some easy-to-measure predictor like water surface elevation (stage).
Despite existance of several automated methods,
most ratings are still fit by hand, which can be time consuming and error prone.
To promote more widespread use and testing of automated methods,
we combined elements from a classic algorithm with modern probabilistic machine learning,
and packaged it as an easy-to-use Python library.
In this way, our algorithm is relatively easy to modify or extend and could serve as a benchmark for others. 

```{tableofcontents}
```
