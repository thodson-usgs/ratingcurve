# Troubleshooting
## ADVI
Ocassionally, the multi-segment power law fails to converge with ADVI.
If this occurs, reinitializing the model by reruning
```python
powerrating = PowerLawRating(...)
```
The problem seems to be related to the initialization. Working on a fix.

As the number of observations decreases, ADVI begins to give poorer performance.
One clue that this is happening is if the uncertainty bounds on break points are overlapping.
If this occurs, try using NUTS instead of ADVI.
This problem might be related to how the breakpoint prior is specified,
specifically how the sorting of the breakpoints is handled.

## NUTS
NUTS is a more accurate but slower than ADVI.
It will generally work with fewer observations than ADVI,
but if the observations are too few, NUTS will begin to experience divergences during sampling.
A handful of divergences may not substantially affect the results,
but you can reduce them by increasing the target acceptance rate.
The default is 0.95, which is fairly high to begin with,
but you can try increasing it to 0.99 or 0.999.