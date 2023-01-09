"""Tutorial datasets"""

import intake


CATALOG_URL = 'https://raw.githubusercontent.com/thodson-usgs/ratingcurve/main/data/rating_data_catalog.yml'


def list_datasets():
    """Returns a list of tutorial datasets
    """
    cat = intake.open_catalog(CATALOG_URL)
    return list(cat)


def open_dataset(name):
    """Opens a tutorial dataset

    Parameters
    ----------
    name : str
        Name of the dataset.
        e.g., 'green_channel'
    """
    cat = intake.open_catalog(CATALOG_URL)
    df = cat[name].read()
    return df
