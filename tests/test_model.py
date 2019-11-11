import emc2
import numpy as np


def test_model():
    model = emc2.core.model.TestModel()
    assert model.hydrometeor_classes == ["cl", "ci", "pl", "pi"]
    assert model.num_hydrometeor_classes == 4
    model.num_subcolumns = 2
    assert model.ds['subcolumn'].values[0] == 0
    assert model.ds['subcolumn'].values[1] == 1
    assert 'subcolumn' in model.ds.dims
