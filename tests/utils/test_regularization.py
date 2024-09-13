import pytest
from rtrec.utils.regularization import Regularization, PassThrough, L1, L2, ElasticNet

def test_pass_through():
    reg = PassThrough(lambda_=0.01)
    weight = 1.0
    gradient = 0.5
    assert reg.regularize(weight, gradient) == gradient
    assert reg.regularization_term(weight) == 0.0

def test_l1():
    reg = L1(lambda_=0.01)
    weight_positive = 1.0
    weight_negative = -1.0
    gradient = 0.5
    assert reg.regularize(weight_positive, gradient) == gradient + 0.01
    assert reg.regularization_term(weight_positive) == 1.0
    assert reg.regularize(weight_negative, gradient) == gradient - 0.01
    assert reg.regularization_term(weight_negative) == -1.0

def test_l2():
    reg = L2(lambda_=0.01)
    weight = 1.0
    gradient = 0.5
    assert reg.regularize(weight, gradient) == gradient + 0.01
    assert reg.regularization_term(weight) == weight

def test_elastic_net():
    reg = ElasticNet(lambda_=0.01, l1_ratio=0.7)
    weight = 1.0
    gradient = 0.5
    l1_reg = L1(lambda_=0.01).regularization_term(weight)
    l2_reg = L2(lambda_=0.01).regularization_term(weight)
    expected_reg = 0.7 * l1_reg + 0.3 * l2_reg
    assert reg.regularize(weight, gradient) == gradient + 0.01 * expected_reg
    assert reg.regularization_term(weight) == expected_reg

def test_elastic_net_invalid_l1_ratio():
    import re
    with pytest.raises(ValueError, match=re.escape("L1 ratio should be in [0.0, 1.0], but got 1.5")):
        ElasticNet(lambda_=0.01, l1_ratio=1.5)

if __name__ == "__main__":
    pytest.main()
