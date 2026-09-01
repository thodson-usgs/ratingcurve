from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ratingcurve")
except PackageNotFoundError:
    # package is not installed
    __version__ = "dev"
