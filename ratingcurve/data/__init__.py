"""Example datasets for rating curve analysis.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import pkg_resources

from pandas import read_csv


if TYPE_CHECKING:
    from pandas import DataFrame

DATASETS = {
    'chalk artificial': 'chalk_artificial',
    'co channel': 'co_channel',
    'green channel': 'green_channel',
    'provo natural': 'provo_natural',
    '3-segment simulated': 'simulated_rating',
    'mahurangi artificial': 'mahurangi_artificial',
    'nordura': 'nordura',
    'skajalfandafljot': 'skajalfandafljot',
    'isere': 'isere',
}


def list() -> list:
    """Returns a list of tutorial datasets

    Returns
    -------
    list
        List of tutorial datasets
    """
    return [key for key in DATASETS.keys()]


def load(name: str) -> DataFrame:
    """Opens a tutorial dataset

    Parameters
    ----------
    name : str
        Name of the dataset.
        e.g., 'green channel'

    Returns
    """
    if name not in DATASETS.keys():
        raise ValueError(f'Dataset "{name}" does not exist. Valid values are: {list()}')

    filename = DATASETS.get(name) + '.csv'
    stream = pkg_resources.resource_stream(__name__, filename)
    return read_csv(stream)


def describe(name) -> str:
    """Describes a tutorial dataset

    Parameters
    ----------
    name : str
        Name of the dataset.
        e.g., 'green channel'

    Returns
    -------
    str
        Description of the dataset
    """
    if name not in DATASETS.keys():
        raise ValueError(f'Dataset "{name}" does not exist. Valid values are: {list()}')

    filename = DATASETS.get(name) + '.md'
    stream = pkg_resources.resource_stream(__name__, filename)
    print(stream.read().decode('utf-8'))
