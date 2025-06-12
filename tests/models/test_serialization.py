#!/usr/bin/env python3
"""
Test script for model serialization functionality.
"""

import tempfile
from io import BytesIO

from rtrec.models.hybrid import HybridSlimFM
from rtrec.models.lightfm import LightFM
from rtrec.models.slim import SLIM


def test_slim_serialization():
    """Test SLIM model serialization and deserialization."""
    print("Testing SLIM serialization...")

    # Create and train a simple SLIM model
    model = SLIM()

    # Add some test interactions
    interactions = [
        ("user1", "item1", 1.0, 5.0),
        ("user1", "item2", 2.0, 4.0),
        ("user2", "item1", 3.0, 3.0),
        ("user2", "item3", 4.0, 5.0),
    ]

    model.fit(interactions, progress_bar=False)

    # Test serialization to BytesIO
    with BytesIO() as buffer:
        bytes_written = model.save(buffer)
        print(f"SLIM model serialized: {bytes_written} bytes written")

        # Test loading from BytesIO
        buffer.seek(0)
        loaded_model = SLIM.load(buffer)
        print("SLIM model loaded successfully from BytesIO")

    # Test serialization to file
    with tempfile.NamedTemporaryFile() as tmp_file:
        bytes_written = model.save(tmp_file)
        print(f"SLIM model saved to file: {bytes_written} bytes written")

        # Test loading from file
        tmp_file.seek(0)
        loaded_model = SLIM.load(tmp_file)
        print("SLIM model loaded successfully from file")

    # Test loads method
    with BytesIO() as buffer:
        model.save(buffer)
        data = buffer.getvalue()
        loaded_model = SLIM.loads(data)
        print("SLIM model loaded successfully with loads() method")

    print("SLIM serialization tests passed!")

def test_lightfm_serialization():
    """Test LightFM model serialization and deserialization."""
    print("Testing LightFM serialization...")

    # Create and train a simple LightFM model
    model = LightFM(no_components=10, epochs=1)

    # Add some test interactions
    interactions = [
        ("user1", "item1", 1.0, 5.0),
        ("user1", "item2", 2.0, 4.0),
        ("user2", "item1", 3.0, 3.0),
        ("user2", "item3", 4.0, 5.0),
    ]

    model.fit(interactions, progress_bar=False)

    # Test serialization to BytesIO
    with BytesIO() as buffer:
        bytes_written = model.save(buffer)
        print(f"LightFM model serialized: {bytes_written} bytes written")

        # Test loading from BytesIO
        buffer.seek(0)
        loaded_model = LightFM.load(buffer)
        print("LightFM model loaded successfully from BytesIO")

    # Test serialization to file
    with tempfile.NamedTemporaryFile() as tmp_file:
        bytes_written = model.save(tmp_file)
        print(f"LightFM model saved to file: {bytes_written} bytes written")

        # Test loading from file
        tmp_file.seek(0)
        loaded_model = LightFM.load(tmp_file)
        print("LightFM model loaded successfully from file")

    # Test loads method
    with BytesIO() as buffer:
        model.save(buffer)
        data = buffer.getvalue()
        loaded_model = LightFM.loads(data)
        print("LightFM model loaded successfully with loads() method")

    print("LightFM serialization tests passed!")

def test_hybrid_serialization():
    """Test HybridSlimFM model serialization and deserialization."""
    print("Testing HybridSlimFM serialization...")

    # Create and train a simple HybridSlimFM model
    model = HybridSlimFM(no_components=10, epochs=1, similarity_weight_factor=2.0)

    # Add some test interactions
    interactions = [
        ("user1", "item1", 1.0, 5.0),
        ("user1", "item2", 2.0, 4.0),
        ("user2", "item1", 3.0, 3.0),
        ("user2", "item3", 4.0, 5.0),
    ]

    model.fit(interactions, progress_bar=False)

    # Test serialization to BytesIO
    with BytesIO() as buffer:
        bytes_written = model.save(buffer)
        print(f"HybridSlimFM model serialized: {bytes_written} bytes written")

        # Test loading from BytesIO
        buffer.seek(0)
        loaded_model = HybridSlimFM.load(buffer)
        print("HybridSlimFM model loaded successfully from BytesIO")

    # Test serialization to file
    with tempfile.NamedTemporaryFile() as tmp_file:
        bytes_written = model.save(tmp_file)
        print(f"HybridSlimFM model saved to file: {bytes_written} bytes written")

        # Test loading from file
        tmp_file.seek(0)
        loaded_model = HybridSlimFM.load(tmp_file)
        print("HybridSlimFM model loaded successfully from file")

    # Test loads method
    with BytesIO() as buffer:
        model.save(buffer)
        data = buffer.getvalue()
        loaded_model = HybridSlimFM.loads(data)
        print("HybridSlimFM model loaded successfully with loads() method")

    print("HybridSlimFM serialization tests passed!")

def test_recommendations_after_load():
    """Test that loaded models can still make recommendations."""
    print("Testing recommendations after model loading...")

    # Test SLIM
    model = SLIM()
    interactions = [
        ("user1", "item1", 1.0, 5.0),
        ("user1", "item2", 2.0, 4.0),
        ("user2", "item1", 3.0, 3.0),
        ("user2", "item3", 4.0, 5.0),
    ]
    model.fit(interactions, progress_bar=False)

    # Get recommendations before serialization
    recs_before = model.recommend("user1", top_k=2)

    # Save and load model
    with BytesIO() as buffer:
        model.save(buffer)
        buffer.seek(0)
        loaded_model = SLIM.load(buffer)

    # Get recommendations after loading
    recs_after = loaded_model.recommend("user1", top_k=2)

    print(f"SLIM recommendations before: {recs_before}")
    print(f"SLIM recommendations after:  {recs_after}")
    assert recs_before == recs_after, "SLIM recommendations differ after loading"

    # Test LightFM
    model = LightFM(no_components=10, epochs=1)
    model.fit(interactions, progress_bar=False)

    # Get recommendations before serialization
    recs_before = model.recommend("user1", top_k=2)

    # Save and load model
    with BytesIO() as buffer:
        model.save(buffer)
        buffer.seek(0)
        loaded_model = LightFM.load(buffer)

    # Get recommendations after loading
    recs_after = loaded_model.recommend("user1", top_k=2)

    print(f"LightFM recommendations before: {recs_before}")
    print(f"LightFM recommendations after:  {recs_after}")
    assert recs_before == recs_after, "LightFM recommendations differ after loading"

    # Test HybridSlimFM
    model = HybridSlimFM(no_components=10, epochs=1, similarity_weight_factor=2.0)
    model.fit(interactions, progress_bar=False)

    # Get recommendations before serialization
    recs_before = model.recommend("user1", top_k=2)

    # Save and load model
    with BytesIO() as buffer:
        model.save(buffer)
        buffer.seek(0)
        loaded_model = HybridSlimFM.load(buffer)

    # Get recommendations after loading
    recs_after = loaded_model.recommend("user1", top_k=2)

    print(f"HybridSlimFM recommendations before: {recs_before}")
    print(f"HybridSlimFM recommendations after:  {recs_after}")
    assert recs_before == recs_after, "HybridSlimFM recommendations differ after loading"

    print("Recommendation consistency tests passed!")

if __name__ == "__main__":
    try:
        test_slim_serialization()
        test_lightfm_serialization()
        test_hybrid_serialization()
        test_recommendations_after_load()
        print("All serialization tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()