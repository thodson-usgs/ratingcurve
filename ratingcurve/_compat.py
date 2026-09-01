"""Compatibility helpers for ArviZ 0.x and 1.x.

PyMC 6 requires ArviZ >= 1.1, which replaced ``arviz.InferenceData`` with
xarray's ``DataTree``. The two objects expose the same inference data under
different APIs, so these helpers keep `ratingcurve` working on both PyMC 5
(ArviZ 0.x) and PyMC 6 (ArviZ 1.x).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from xarray import Dataset


def merge_idata(idata: Any, other: Any) -> None:
    """Merge the groups of one inference data object into another.

    ArviZ 0.x merges groups with ``InferenceData.extend``, which ArviZ 1.x
    replaced with ``DataTree.update``. The two disagree over groups that are
    already present, so groups already in `idata` are kept in both, matching
    ``extend``'s default of ``join='left'``.

    Parameters
    ----------
    idata : arviz.InferenceData or xarray.DataTree
        The object to merge into, modified in place.
    other : arviz.InferenceData or xarray.DataTree
        The object supplying the groups to merge.

    """
    if hasattr(idata, 'extend'):
        idata.extend(other)  # ArviZ 0.x, joins 'left'
    else:
        # ArviZ 1.x. `DataTree.update` lets the incoming group win, whereas
        # `InferenceData.extend` defaults to join='left' and keeps the group
        # already present, so skip those to match.
        for group, node in other.children.items():
            if group not in idata.children:
                idata[group] = node


def to_dataset(group: Any) -> Dataset:
    """Return an inference data group as an `xarray.Dataset`.

    ArviZ 0.x stores each group as an `xarray.Dataset`, whereas ArviZ 1.x
    stores it as a ``DataTree`` node that must be converted first.

    Parameters
    ----------
    group : xarray.Dataset or xarray.DataTree
        A single group of an inference data object, such as ``idata.fit_data``.

    Returns
    -------
    xarray.Dataset
        The group as a dataset.

    """
    # ArviZ 1.x DataTree nodes convert to a Dataset; 0.x groups already are one
    return group.to_dataset() if hasattr(group, 'to_dataset') else group
