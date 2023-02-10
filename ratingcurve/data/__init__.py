import pkg_resources

from pandas import read_csv

DATASETS = {
    'chalk artificial': 'chalk_artificial.csv',
    'co channel': 'co_channel.csv',
    'green channel': 'green_channel.csv',
    'provo natural': 'provo_natural.csv',
    '3-segment simulated': 'simulated_rating.csv'
}


def list():
    """Returns a list of tutorial datasets
    """
    return [key for key in DATASETS.keys()]


def load(name):
    """Opens a tutorial dataset

    Parameters
    ----------
    name : str
        Name of the dataset.
        e.g., 'green channel'
    """
    filename = DATASETS.get(name)
    stream = pkg_resources.resource_stream(__name__, filename)
    return read_csv(stream)
