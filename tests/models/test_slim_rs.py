import pytest
import time

from rtrec.models import Fast_SLIM_MSE, SLIM_MSE  # Ensure this matches the module name in your Rust crate.

@pytest.fixture
def slim():
    # Create a SlimMse instance with example hyperparameters
    return SLIM_MSE(alpha=0.1, beta=1.0, lambda1=0.0002, lambda2=0.0001)

def test_fit(slim):
    # Sample user interactions for fitting with timestamps
    current_time = time.time()
    interactions = [
        (0, 1, current_time, 5.0),
        (0, 2, current_time, 3.0),
        (1, 1, current_time, 4.0),
        (1, 3, current_time, 2.0)
    ]
    # Fit the model with sample interactions
    slim.fit(interactions)

    # Check empirical loss is greater than 0 after fitting
    assert slim.get_empirical_loss() > 0.0

def test_predict_rating(slim):
    # Add interactions with timestamps
    current_time = time.time()
    interactions = [
        (0, 1, current_time, 5.0),
        (0, 2, current_time, 3.0),
    ]
    slim.fit(interactions)

    # Predict rating for a known interaction
    predicted_rating = slim._predict_rating(0, 1)
    assert isinstance(predicted_rating, float)

def test_recommend(slim):
    # Add interactions with timestamps
    current_time = time.time()
    interactions = [
        (0, 1, current_time, 5.0),
        (0, 2, current_time, 3.0),
        (0, 3, current_time, 4.0),
        (1, 4, current_time, 3.0),
        (1, 2, current_time, 4.0),  # both user 0 and 1 interacted with item 2
    ]
    slim.fit(interactions)

    # Get top-k recommendations for a user
    recommendations = slim.recommend(0, top_k=2)

    # Check that we have the correct number of recommendations
    assert len(recommendations) == 1
    assert recommendations[0] == 4
    # Ensure the recommendations are a list of integers
    assert all(isinstance(item_id, int) for item_id in recommendations)

def test_get_empirical_loss(slim):
    # Before any interaction, empirical loss should be zero
    assert slim.get_empirical_loss() == 0.0

    # Add interactions with timestamps
    current_time = time.time()
    interactions = [
        (0, 1, current_time, 5.0),
        (0, 2, current_time, 3.0),
    ]
    slim.fit(interactions)

    # Check empirical loss is greater than 0
    assert slim.get_empirical_loss() > 0.0

if __name__ == "__main__":
    pytest.main()
