from typing import Dict
import argparse

from .datasets import load_dataset
from .split import leave_one_last_item, random_split, temporal_split, temporal_user_split
from ..recommender import Recommender
from ..models import Fast_SLIM_MSE, SLIM_MSE

def run_experiment(
    dataset_name: str,model_name: str, split_method: str = "temporal", epochs: int = 10, batch_size: int = 1000
) -> Dict[str, float]:
    """
    Run an experiment to evaluate the recommender on the specified dataset.

    Args:
        dataset_name (str): Name of the dataset to evaluate the recommender on.
        model_name (str): Name of the model to use for recommendation.
        split_method (str): Method for splitting the data into train and test sets.
        epochs (int): Number of epochs (iterations) over the dataset.
        batch_size (int): Batch size for training the model.

    Returns:
        Dict[str, float]: A dictionary of evaluation metrics on the test set.
    """

    print(f"Evaluating recommender on {dataset_name}...")
    # Load and preprocess the dataset
    df = load_dataset(dataset_name)

    # Split into train and test sets
    if split_method == "temporal":
        train_data, test_data = temporal_split(df, test_frac=0.2)
    elif split_method == "temporal_user":
        train_data, test_data = temporal_user_split(df, test_frac=0.2)
    elif split_method == "leave_one_last":
        train_data, test_data = leave_one_last_item(df)
    elif split_method == "random":
        train_data, test_data = random_split(df, test_frac=0.2)
    else:
        raise ValueError(f"Unsupported split method: {split_method}")

    # Initialize and train the recommender
    if model_name == "fast_slim_mse":
        model = Fast_SLIM_MSE()
    elif model_name == "slim_mse":
        model = SLIM_MSE()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    recommender = Recommender(model)
    recommender.fit(train_data, epochs=epochs, batch_size=batch_size, random_seed=43)

    # Evaluate the recommender
    return recommender.evaluate(test_data)

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="fast_slim_mse", help="Name of the model to use for recommendation. Options: fast_slim_mse (default), slim_mse.")
    parser.add_argument("--dataset", type=str, default="movielens_1m", help="Name of the dataset to evaluate the recommender on. Options: movielens_[1m|20m|25m], yelp, amazon_[music|electronics].")
    parser.add_argument("--split", type=str, default="temporal", help="Method for splitting the data into train and test sets. Options: temporal (default), temporal_user, leave_one_last, random.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (iterations) over the dataset. Default: 10.")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for training the model. Default: 1000.")
    args = parser.parse_args()

    # run the experiment
    results = run_experiment(args.dataset, args.model, args.split, args.epochs, args.batch_size)

    print(f"Results: {results}")
