import emc2


def test_model():
    model = emc2.core.model.TestModel()
    assert model.hydrometeor_classes == ["cl", "ci", "pl", "pi"]
    assert model.num_hydrometeor_classes == 4
