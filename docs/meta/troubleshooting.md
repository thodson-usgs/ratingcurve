# Troubleshooting
## ADVI
Ocassionally, the multi-segment power law fails to converge with ADVI.
If this occurs, reinitializing the model by reruning
```python
powerrating = PowerLawRating(...)
```

The problem seems to be with the initialization. Working on a fix.