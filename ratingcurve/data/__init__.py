"""Example datasets for rating curve analysis."""
from __future__ import annotations
from typing import TYPE_CHECKING

import pkg_resources

from intake import open_catalog

if TYPE_CHECKING:
    from pandas import DataFrame

cat = open_catalog('catalog.yaml')

def list() -> list:
    """
    Returns a list of names for the tutorial datasets.

    Returns
    -------
    datasets : list of str
        List of names of tutorial datasets.
    """
    return list(cat)


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
    if name not in list(cat):
        raise ValueError(f'Dataset "{name}" does not exist. Valid values are: {list(cat)}')
    
    return cat[name].read()


def describe(name: str):
    """
    Prints description of a tutorial dataset.

    Parameters
    ----------
    name : str
        Name of the dataset (e.g., 'green channel').
    """
    if name not in list(cat):
        raise ValueError(f'Dataset "{name}" does not exist. Valid values are: {list(cat)}')

    print(cat[name].description)
    