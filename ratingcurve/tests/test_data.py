import pytest

from .. import data

@pytest.mark.parametrize('idx', list(range(len(data.list()))))
def test_data_load(idx):
    data_list = data.list()
    df = data.load(data_list[idx])

    assert len(df) > 0
    assert all(df.stage >= 0)
    assert all(df.q >= 0)
