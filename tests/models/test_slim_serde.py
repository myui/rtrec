import os
import pytest
import tempfile

from rtrec.models import Fast_SLIM_MSE as SlimMSE

@pytest.fixture
def sample_model():
    """Create a sample SlimMSE model instance with test data."""
    # Create a new SlimMSE model with default parameters
    model = SlimMSE(
        alpha=0.5,
        beta=1.0,
        lambda1=0.0002,
        lambda2=0.0001,
        min_value=-5.0,
        max_value=10.0,
        decay_in_days=None,
    )

    # Add sample interactions
    user_interactions = [
        (1, 2, 1627776000.0, 5.0),
        (1, 3, 1627776000.0, 3.0),
        (2, 3, 1627776000.0, 4.0),
        (2, 4, 1627776000.0, 2.0),
    ]
    model.fit(user_interactions)
    return model

def test_save_and_load(sample_model):
    """Test saving and loading the model using the save() and load() methods."""
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".msgpack") as temp_file:
        file_path = temp_file.name

    try:
        # Save the model to the temporary file
        sample_model.save(file_path)

        # Load the model back from the temporary file
        loaded_model = SlimMSE.load(file_path)

        # Verify that the loaded model is equal to the original model
        # Note: This assumes that the __eq__ method is implemented correctly in SlimMSE
        assert sample_model.get_empirical_loss() == loaded_model.get_empirical_loss(), \
            "Empirical loss of the loaded model does not match the original model."
        assert sample_model.recommend(1, 2) == loaded_model.recommend(1, 2), \
            "Recommendations of the loaded model do not match the original model."
    finally:
        # Clean up the temporary file after the test
        if os.path.exists(file_path):
            os.remove(file_path)

def test_save_file_exists(sample_model):
    """Test whether the file is created when the model is saved."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".msgpack") as temp_file:
        file_path = temp_file.name

    try:
        # Save the model to the temporary file
        sample_model.save(file_path)

        # Check if the file exists after saving
        assert os.path.exists(file_path), "The file does not exist after saving the model."
    finally:
        # Clean up the temporary file after the test
        if os.path.exists(file_path):
            os.remove(file_path)

def test_load_nonexistent_file():
    """Test loading a model from a nonexistent file."""
    with pytest.raises(OSError):
        # Try to load from a nonexistent file path
        SlimMSE.load("nonexistent_file.msgpack")

def test_empty_model_load():
    """Test loading a newly created empty model without any interactions."""
    # Create a new model instance
    empty_model = SlimMSE(
        alpha=0.5,
        beta=1.0,
        lambda1=0.0002,
        lambda2=0.0001,
        min_value=-5.0,
        max_value=10.0,
        decay_in_days=None,
    )

    # Create a temporary file for saving and loading
    with tempfile.NamedTemporaryFile(delete=False, suffix=".msgpack") as temp_file:
        file_path = temp_file.name

    try:
        # Save the empty model
        empty_model.save(file_path)

        # Load the model back
        loaded_model = SlimMSE.load(file_path)

        # Verify that the loaded empty model is equal to the original empty model
        assert empty_model.get_empirical_loss() == loaded_model.get_empirical_loss(), \
            "Empirical loss of the empty model does not match the original empty model."
    finally:
        # Clean up the temporary file after the test
        if os.path.exists(file_path):
            os.remove(file_path)

def test_save_and_load_local(sample_model):
    """Test saving and loading the model using the local file system."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".msgpack") as temp_file:
        file_path = temp_file.name

    try:
        sample_model.save(f"file://{file_path}")
        loaded_model = SlimMSE.load(f"file://{file_path}")

        assert sample_model.get_empirical_loss() == loaded_model.get_empirical_loss()
        assert sample_model.recommend(1, 2) == loaded_model.recommend(1, 2)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
