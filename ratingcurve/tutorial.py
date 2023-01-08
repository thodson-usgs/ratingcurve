import intake

catalog_url = 'https://raw.githubusercontent.com/thodson-usgs/ratingcurve/main/data/rating_data_catalog.yml'


def list_datasets():
    """Returns a list of tutorial datasets
    """
    cat = intake.open_catalog(catalog_url)
    return list(cat)


def open_dataset(name):
    """Opens a tutorial dataset

    Parameters
    ----------
    name : str
        Name of the dataset.
        e.g., 'green_channel'
    """
    cat = intake.open_catalog(catalog_url)
    df = cat[name].read()
    return df