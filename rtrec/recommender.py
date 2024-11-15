import pandas as pd
import time

from tqdm import tqdm
from typing import Dict, Generator, Iterator, Tuple, Iterable, Optional, Any, List

from rtrec.utils.metrics import compute_scores
from rtrec.models import Fast_SLIM_MSE

class Recommender:

    def __init__(self, model):
        self.model = model
        # Rust module do not support Python generators
        self.use_generator = not isinstance(model, Fast_SLIM_MSE)

    def get_model(self):
        return self.model

    def partial_fit(self, user_interactions: Iterable[Tuple[int, int, int, float]]) -> None:
        """
        Incrementally fit the recommender model on new interactions.
        """
        self.model.fit(user_interactions)

    def fit(
        self,
        train_data: pd.DataFrame,
        epochs: int = 1,
        batch_size: int = 1_000,
        random_seed: Optional[int] = None
    ) -> None:
        """
        Fit the recommender model on the given DataFrame of interactions.

        Parameters:
            train_data (pd.DataFrame): The DataFrame containing interactions with columns (user, item, tstamp, rating).
            epochs (int): Number of epochs (iterations) over the dataset. Defaults to 1.
            batch_size (int): The number of interactions per mini-batch. Defaults to 1000.
        """
        train_data = train_data[["user", "item", "tstamp", "rating"]]

        user_item_pairs = train_data[['user', 'item']].apply(tuple, axis=1).tolist()
        identified_pairs = self.model.bulk_identify(user_item_pairs)
        train_data[['user', 'item']] = pd.DataFrame(identified_pairs, index=train_data.index)
        del user_item_pairs, identified_pairs

        # Iterate over epochs
        for epoch in tqdm(range(epochs)):
            # Shuffle the training data at the beginning of each epoch
            train_data = train_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

            print(f"Starting epoch {epoch + 1}/{epochs}")
            start_time = time.time()
            for batch in generate_batches(train_data, batch_size, as_generator=self.use_generator):
                self.model.fit_identified(batch, update_interaction=epoch > 0)
            end_time = time.time()
            print(f"Epoch {epoch + 1} completed in {end_time - start_time:.2f} seconds")
            print(f"Throughput: {len(train_data) / (end_time - start_time):.2f} samples/sec")
            print(f"Empirical loss after epoch {epoch + 1}: {self.model.get_empirical_error(reset=True)}")

    def predict_rating(self, user: Any, item: Any) -> float:
        """
        Predict the rating for a given user-item pair.
        """
        return self.model.predict_rating(user, item)

    def recommend(self, user: Any, top_k: int = 10, filter_interacted: bool = True) -> List[Any]:
        """
        Recommend top-K items for a given user.
        :param user: User index
        :param top_k: Number of top items to recommend
        :param filter_interacted: Whether to filter out items the user has already interacted with
        :return: List of top-K item indices recommended for the user
        """
        return self.model.recommend(user, top_k, filter_interacted)

    def similar_items(self, query_items: List[Any], top_k: int = 10, filter_query_items: bool = True) -> List[List[Any]]:
        """
        Find similar items for a list of query items.
        :param query_items: List of query items
        :param top_k: Number of top similar items to return
        :param filter_interacted: Whether to filter out items in the query_items list
        :return: List of top-K similar items for each query item
        """
        return self.model.similar_items(query_items, top_k, filter_query_items)

    def evaluate(self, test_data: pd.DataFrame, recommend_size: int = 10) -> Dict[str, float]:
        """
        Evaluates the model using batch evaluation metrics on the test data.

        Parameters:
            test_data (pd.DataFrame): DataFrame with columns ['user', 'item'] containing ground truth interactions.
            recommend_size (int): Number of items to recommend per user for evaluation.

        Returns:
            Dict[str, float]: Dictionary with averaged evaluation metrics across all users.
        """
        # Group the test data by user to get ground truth items per user
        grouped_data = test_data.groupby('user')['item'].apply(list).to_dict()

        # Use a generator to yield recommendations and ground truth for each user
        def generate_evaluation_pairs() -> Iterable[Tuple[List[Any], List[Any]]]:
            for user, ground_truth_items in tqdm(grouped_data.items()):
                recommended_items = self.recommend(user, recommend_size)
                yield recommended_items, ground_truth_items

        # Compute and return the evaluation metrics using the generator
        return compute_scores(generate_evaluation_pairs(), recommend_size)

@staticmethod
def generate_batches(df: pd.DataFrame, batch_size: int = 1_000, as_generator: bool = False) -> Iterator[Iterable[Tuple[int, int, int, float]]]:
    """
    Converts a DataFrame to an iterable of mini-batches.

    Parameters:
        df (pd.DataFrame): The DataFrame to convert to mini-batches.
        batch_size (int): The number of rows per mini-batch.
        as_generator (bool): Whether to return a generator or a list of mini-batches.

    Returns:
        Iterator[Iterable[Tuple[int, int, int, float]]]: An iterator of mini-batches.
    """
    num_rows = len(df)
    if as_generator:
        for start in range(0, num_rows, batch_size):
            batch = df.iloc[start:start + batch_size]
            yield batch.itertuples(index=False, name=None)
    else:
        for start in range(0, num_rows, batch_size):
            batch = df.iloc[start:start + batch_size]
            yield list(batch.itertuples(index=False, name=None))
