"""Example datasets for rating curve analysis."""
from __future__ import annotations
from typing import TYPE_CHECKING

import pkg_resources

from intake import open_catalog

if TYPE_CHECKING:
    from pandas import DataFrame

cat = open_catalog('ratingcurve/data/catalog.yaml')

def list() -> tuple:
    """
    Returns a tuple of names for the tutorial datasets.

    Returns
    -------
    datasets : tuple of str
        Tuple of names of tutorial datasets.
    """
    datasets = tuple(cat)
    
    return datasets


def load(name: str) -> DataFrame:
    """
    Opens a tutorial dataset.

    Parameters
    ----------
    name : str
        Name of the dataset (e.g., 'green channel').

    Returns
    -------
    dataset : DataFrame
        Dataframe with the tutorial data. Columns include `h` (stage) and `q` (discharge), and
        potentially `q_sigma` (discharge uncertainty).
    """
    if name not in tuple(cat):
        raise ValueError(f'Dataset "{name}" does not exist. Valid values are: {tuple(cat)}')
    
    return cat[name].read()


def describe(name: str):
    """
    Prints description of a tutorial dataset.

    Parameters
    ----------
    name : str
        Name of the dataset (e.g., 'green channel').
    """
    if name not in tuple(cat):
        raise ValueError(f'Dataset "{name}" does not exist. Valid values are: {tuple(cat)}')

    print(cat[name].description)
    