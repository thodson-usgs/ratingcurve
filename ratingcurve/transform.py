"""Data transformations to improve optimization"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from patsy import dmatrix, build_design_matrices

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class Transform:
    """
    Transformation class.

    All children of Transform must have transfom and untransform methods.
    """
    
    def __init__(self, x):
        """
        Create empty Transform object.
        """
        self.mean_ = None
        self.std_ = None

    def transform(self, x: ArrayLike) -> ArrayLike:
        """
        Transform data.

        Parameters
        ----------
        x : array_like
            Data to be transformed.

        Returns
        -------
        transformed_x : array_like
            Transformed data.
        """
        return x

    def untransform(self, transformed_x: ArrayLike) -> ArrayLike:
        """
        Undo data transform (untransform).

        Parameters
        ----------
        transformed_x : array_like
            Transformed data.

        Returns
        -------
        x: array_like
            Untransformed data.
        """
        return x


class ZTransform(Transform):
    """
    Z-transforms data to z-score scale (i.e., zero mean and unit variance).
    """

    def __init__(self, x: ArrayLike):
        """
        Create a ZTransform object.

        Parameters
        ----------
        x : array_like
            Data that defines the transform.
        """
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)

    def transform(self, x: ArrayLike) -> ArrayLike:
        """
        Transform to z-score scale (standardize x).

        Parameters
        ----------
        x : array_like
            Data to be transformed.

        Returns
        -------
        transformed_x : array_like
            Data standardized to zero mean and unit variance (z-score).
        """
        return (x - self.mean_) / self.std_

    def untransform(self, transformed_x: ArrayLike) -> ArrayLike:
        """
        Transform from z-score scale back to original units.

        Parameters
        ----------
        transformed_x : array_like
            Z-Transformed data.

        Returns
        -------
        x : array_like
            Untransformed data back in original units.
        """
        return transformed_x * self.std_ + self.mean_


class LogZTransform(ZTransform):
    """
    Z-transforms logarithm of data to z-score scale (i.e., log of data has 
    zero mean and unit variance).
    """

    def __init__(self, x: ArrayLike):
        """
        Create a LogZTransform object.

        Parameters
        ----------
        x : array_like
            Data that defines the transform.
        """
        log_x = np.log(x)
        super().__init__(log_x)

    def transform(self, x: ArrayLike) -> ArrayLike:
        """
        Transform to log z-score scale (standardize log(x)).

        Parameters
        ----------
        x : array_like
            Data to be transformed.

        Returns
        -------
        transformed_x : array_like
            Log transformed data standardized to zero mean and unit variance (log z-score).
        """
        log_x = np.log(x)
        return super().transform(log_x)

    def untransform(self, transformed_x: ArrayLike) -> ArrayLike:
        """
        Transform from log z-score scale back to original units.

        Parameters
        ----------
        transformed_x : array_like
            LogZTransformed data.

        Returns
        -------
        x : array_like
            Untransformed data back in original units.
        """
        log_x = super().untransform(transformed_x)
        return np.exp(log_x)


class UnitTransform(Transform):
    """
    Transforms data to the unit (0 to 1) interval.
    """
    def __init__(self, x: ArrayLike):
        """
        Create a UnitTransform object.

        Parameters
        ----------
        x : array_like
            Data that defines the transform.
        """
        self.max_ = np.nanmax(x, axis=0)

    def transform(self, x: ArrayLike) -> ArrayLike:
        """
        Transform to unit interval.

        Parameters
        ----------
        x : array_like
            Data to be transformed.
        
        Returns
        -------
        transformed_x : array_like
            Transformed data to unit interval (0 to 1).
        """
        return x / self.max_

    def untransform(self, transformed_x: ArrayLike) -> ArrayLike:
        """
        Transform from unit interval back to original units.

        Parameters
        ----------
        transformed_x : array_like
            UnitTransformed data.

        Returns
        -------
        x: array_like
            Untransformed data back in original units.
        """
        return transformed_x * self.max_


class Dmatrix():
    """
    Transform for spline design matrix.
    """
    def __init__(self, x: ArrayLike, df: int, form: str = 'cr') -> None:
        """
        Create a design matrix (Dmatrix) object.

        Create a design matrix for a natural cubic spline, which is a cubic
        spline that is additionally constrained to be linear at the boundaries.
        Due to this constraint, the total degrees of freedom equals the number
        of knots minus 1.
        
        Parameters
        ----------
        x : array_like
            Data that defines the transform.
        df : int
            Degrees of freedom.
        form : str, optional
            Spline form.
        """
        n_knots = df - 1
        self.knots = compute_knots(x.min(), x.max(), n=n_knots)
        temp = dmatrix(f"{form}(x, df={df}) - 1", {"x": x})
        self.design_info = temp.design_info

    def transform(self, x: ArrayLike) -> ArrayLike:
        """
        Transform data using spline design matrix.
 
        Parameters
        ----------
        x : array-like
            Data to be transformed.

        Returns
        -------
        transformed_x : array_like
            Transformed data into spline design matrix.
        """
        return np.asarray(build_design_matrices([self.design_info], {"x": x})).squeeze()


def compute_knots(minimum: float, maximum: float, n: int) -> ArrayLike:
    """
    Return list of spline knots.

    Parameters
    ----------
    minimum: float
        Minimum data value to consider.
    maximum : float
        Maximum data value to consider.
    n : int
        Number of knots.

    Returns
    -------
    knots : array_like
        List of spline knots.
    """
    return np.linspace(minimum, maximum, n)
