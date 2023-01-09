"""Data transformations to improve optimization"""

import numpy as np
import patsy


class Transform:
    """Transformation class

    All children of Transform must have transfom and untransform methods
    """
    def __init__(self):
        """Create empty Transform object
        """
        pass


class ZTransform(Transform):
    """Z-transforms data to have zero mean and unit variance
    """

    def __init__(self, x):
        """Create a ZTransform object
        """
        self._mean = np.nanmean(x, axis=0)
        self._std = np.nanstd(x, axis=0)

    def transform(self, x):
        """Transform to z score (standardize x)
        """
        return (x - self._mean)/self._std

    def untransform(self, x):
        """Transform from z score back to original units.
        """
        return x*self._std + self._mean


class UnitTransform(Transform):
    """Transforms data to the unit (0 to 1) interval.
    """
    def __init__(self, x):
        """Create UnitTransform of array
        """
        self._max = np.nanmax(x, axis=0)

    def transform(self, x):
        """Transform to unit interval
        """
        return x/self._max

    def untransform(self, x):
        """Transform from unit interval back to original units.
        """
        return x*self._max


class LogZTransform(ZTransform):
    """Log transform then takes z-score.
    """

    def __init__(self, x):
        """Create a LogZTransform for x.
        """
        log_x = np.log(x)
        super().__init__(log_x)

    def transform(self, x):
        """Transform to log z-score

        Logs the data tehn standardizes to zero mean and unit variance.
        """
        log_x = np.log(x)
        return super().transform(log_x)

    def untransform(self, z=None):
        """Reverse log z-score transformation.
        """
        log_x = super().untransform(z)
        return np.exp(log_x)


class Dmatrix(Transform):
    """Transform for spline design matrix (Unused)
    """
    def __init__(self, knots, degree, form):
        """Create a Dmatrix object (Unused)
        """
        self.form = f"{form}(stage, knots=knots, degree={degree}, include_intercept=True) - 1"
        self.knots = knots

    def transform(self, stage):
        """Transform (Unused)
        """
        return patsy.dmatrix(self.form, {"stage": stage, "knots": self.knots[1:-1]})
