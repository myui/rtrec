import pytest
import time
import numpy as np

from rtrec.models import SLIM_MSE as SlimMSE

@pytest.fixture
def slim():
    # Create a SlimMse instance with example hyperparameters
    return SlimMSE(alpha=0.1, beta=1.0, lambda1=0.0002, lambda2=0.0001, min_value=-np.inf, max_value=np.inf)

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

def test_predict_mean_close_to_zero(slim):
    np.random.seed(42)  # For reproducibility
    current_time = time.time()

    # Generate random interactions with normal distribution (mean=0, stddev=5)
    num_interactions = 10000
    user_ids = [f"user{i}" for i in range(num_interactions)] # unique user IDs
    item_ids = np.random.randint(0, 50, num_interactions)  # 50 items
    ratings = np.random.normal(0, 5, num_interactions)     # Normal distribution (mean=0, stddev=5)

    interactions = [
        (user, item, current_time, rating)
        for user, item, rating in zip(user_ids, item_ids, ratings)
    ]

    # Fit the model with the generated interactions
    slim.fit(interactions)

    # Predict for all user-item pairs in the interactions
    sum_predicted = 0.0
    max_predicted = -np.inf
    min_predicted = np.inf
    for user, item, _, _ in interactions:
        predicted = slim.predict_rating(user, item)
        sum_predicted += predicted
        max_predicted = max(max_predicted, predicted)
        min_predicted = min(min_predicted, predicted)
    mean_prediction = sum_predicted / len(interactions)

    max_rating = max(ratings)
    assert abs(max_predicted - max_rating) < 0.001, f"Max prediction {max_predicted} is not close to max rating {max_rating}"
    min_rating = min(ratings)
    assert abs(min_predicted - min_rating) < 0.001, f"Min prediction {min_predicted} is not close to min rating {min_rating}"

    # Assert the mean prediction is close to 0
    assert abs(mean_prediction) < 0.001, f"Mean prediction {mean_prediction} is not close to 0"

if __name__ == "__main__":
    pytest.main()
