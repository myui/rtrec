import pytest
from rtrec.utils.regularization import Regularization, PassThrough, L1, L2, ElasticNet, get_regularization

def test_pass_through():
    reg = PassThrough(lambda_=0.01)
    weight = 1.0
    gradient = 0.5
    assert reg.regularize(weight, gradient) == gradient
    assert reg.regularization_term(weight) == 0.0

def test_l1_regularization():
    reg = L1(lambda_=0.1)
    weight_positive = 1.0
    weight_negative = -1.0
    gradient = 0.5

    assert reg.regularization_term(weight_positive) == 1.0
    assert reg.regularization_term(weight_negative) == -1.0
    assert reg.regularize(weight_positive, gradient) == gradient - 0.1 * 1.0
    assert reg.regularize(weight_negative, gradient) == gradient + 0.1 * 1.0

def test_l2_regularization():
    reg = L2(lambda_=0.1)
    weight = 1.0
    gradient = 0.5

    assert reg.regularization_term(weight) == weight
    assert reg.regularize(weight, gradient) == gradient - 0.1 * weight

def test_elastic_net_regularization():
    reg = ElasticNet(lambda_=0.1, l1_ratio=0.7)
    weight_positive = 1.0
    weight_negative = -1.0
    gradient = 0.5

    l1_reg_term = 0.7 * (1.0 if weight_positive > 0 else -1.0)
    l2_reg_term = (1.0 - 0.7) * weight_positive
    expected_reg_term = l1_reg_term + l2_reg_term

    assert reg.regularization_term(weight_positive) == expected_reg_term
    assert reg.regularize(weight_positive, gradient) == gradient - 0.1 * expected_reg_term

def test_elastic_net_invalid_l1_ratio():
    import re
    with pytest.raises(ValueError, match=re.escape("L1 ratio should be in [0.0, 1.0], but got 1.5")):
        ElasticNet(lambda_=0.01, l1_ratio=1.5)

def test_get_regularization():
    # Test PassThrough
    reg = get_regularization('pass_through')
    assert isinstance(reg, PassThrough)

    # Test L1
    reg = get_regularization('l1', lambda_=0.2)
    assert isinstance(reg, L1)
    assert reg.lambda_ == 0.2

    # Test L2
    reg = get_regularization('l2', lambda_=0.3)
    assert isinstance(reg, L2)
    assert reg.lambda_ == 0.3

    # Test ElasticNet
    reg = get_regularization('elastic_net', lambda_=0.4, l1_ratio=0.6)
    assert isinstance(reg, ElasticNet)
    assert reg.lambda_ == 0.4
    assert reg.l1_ratio == 0.6

    # Test invalid regularization type
    with pytest.raises(ValueError):
        get_regularization('unknown_type')

    # Test ElasticNet with invalid l1_ratio
    with pytest.raises(ValueError):
        get_regularization('elastic_net', l1_ratio=1.5)

if __name__ == "__main__":
    pytest.main()
