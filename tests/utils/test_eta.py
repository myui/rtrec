import pytest
from math import isclose
from rtrec.utils.eta import FixedEtaEstimator, InvscalingEtaEstimator

# Example tolerance for floating point comparisons
EPSILON = 1e-6

def test_fixed_eta_estimator():
    initial_eta = 0.01
    fixed_eta_estimator = FixedEtaEstimator(initial_eta=initial_eta)
    
    # Test that the learning rate is always fixed
    assert isclose(fixed_eta_estimator.get_eta(), initial_eta, abs_tol=EPSILON)
    
    # Test that the update method does not change the learning rate
    fixed_eta_estimator.update()
    assert isclose(fixed_eta_estimator.get_eta(), initial_eta, abs_tol=EPSILON)
    
def test_invscaling_eta_estimator():
    initial_eta = 0.01
    power_t = 0.5
    invscaling_eta_estimator = InvscalingEtaEstimator(initial_eta=initial_eta, power_t=power_t)
    
    # Test learning rate at the beginning (t=0)
    assert isclose(invscaling_eta_estimator.get_eta(), initial_eta, abs_tol=EPSILON)
    
    # Update and test learning rate after several iterations
    invscaling_eta_estimator.update()  # t = 1
    assert isclose(invscaling_eta_estimator.get_eta(), initial_eta / 1**power_t, abs_tol=EPSILON)
    
    invscaling_eta_estimator.update()  # t = 2
    assert isclose(invscaling_eta_estimator.get_eta(), initial_eta / 2**power_t, abs_tol=EPSILON)
    
    invscaling_eta_estimator.update()  # t = 3
    assert isclose(invscaling_eta_estimator.get_eta(), initial_eta / 3**power_t, abs_tol=EPSILON)
    
    # Test learning rate after more updates
    for t in range(4, 10):
        invscaling_eta_estimator.update()  # increment t
        expected_eta = initial_eta / t**power_t
        assert isclose(invscaling_eta_estimator.get_eta(), expected_eta, abs_tol=EPSILON)
