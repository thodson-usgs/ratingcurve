"""Example datasets for rating curve analysis.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from importlib import resources

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
    source = resources.files(__package__).joinpath(filename)
    with resources.as_file(source) as path:
        return read_csv(path)


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
    source = resources.files(__package__).joinpath(filename)
    return source.read_text(encoding='utf-8')
