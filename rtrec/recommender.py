import pandas as pd
import time

from tqdm import tqdm
from typing import Dict, Iterator, Tuple, Iterable, Optional, Any, List

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
        random_seed: Optional[int] = None,
        no_shuffle: bool = False,
    ) -> None:
        """
        Fit the recommender model on the given DataFrame of interactions.

        Parameters:
            train_data (pd.DataFrame): The DataFrame containing interactions with columns (user, item, tstamp, rating).
            epochs (int): Number of epochs (iterations) over the dataset. Defaults to 1.
            batch_size (int): The number of interactions per mini-batch. Defaults to 1000.
            random_seed (Optional[int]): Random seed for reproducibility. Defaults to None.
            no_shuffle (bool): Whether to disable shuffling of the training data between epochs. Defaults to False.
        """
        train_data = train_data[["user", "item", "tstamp", "rating"]]

        # Iterate over epochs
        for epoch in tqdm(range(epochs)):
            if not no_shuffle:
                # Shuffle the training data at the beginning of each epoch
                train_data = train_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

            print(f"Starting epoch {epoch + 1}/{epochs}")
            start_time = time.time()
            for batch in generate_batches(train_data, batch_size, as_generator=self.use_generator):
                self.model.fit(batch, update_interaction=epoch >= 1)
            end_time = time.time()
            print(f"Epoch {epoch + 1} completed in {end_time - start_time:.2f} seconds")
            print(f"Throughput: {len(train_data) / (end_time - start_time):.2f} samples/sec")

    def fit_single_batch(self, train_data: pd.DataFrame, batch_size: int = 10_000) -> None:
        """
        Fit the recommender model on the given DataFrame of interactions in a single batch.

        Parameters:
            train_data (pd.DataFrame): The DataFrame containing interactions with columns (user, item, tstamp, rating).
        """
        train_data = train_data[["user", "item", "tstamp", "rating"]]

        start_time = time.time()
        for batch in generate_batches(train_data, batch_size, as_generator=self.use_generator):
            self.model.add_interactions(batch)
        self.model.bulk_fit()
        end_time = time.time()
        print(f"Fit completed in {end_time - start_time:.2f} seconds")
        print(f"Throughput: {len(train_data) / (end_time - start_time):.2f} samples/sec")

    def recommend(self, user: Any, top_k: int = 10, filter_interacted: bool = True) -> List[Any]:
        """
        Recommend top-K items for a given user.
        :param user: User index
        :param top_k: Number of top items to recommend
        :param filter_interacted: Whether to filter out items the user has already interacted with
        :return: List of top-K item indices recommended for the user
        """
        return self.model.recommend(user, top_k, filter_interacted)

    def recommend_batch(self, users: List[Any], top_k: int = 10, filter_interacted: bool = True) -> List[List[Any]]:
        """
        Recommend top-K items for a list of users.
        :param users: List of user indices
        :param top_k: Number of top items to recommend
        :param filter_interacted: Whether to filter out items the user has already interacted with
        :return: List of top-K item indices recommended for each user
        """
        return [self.model.recommend(user, top_k, filter_interacted) for user in tqdm(users)]
        #return self.model.recommend_batch(users, top_k, filter_interacted)

    def similar_items(self, query_items: List[Any], top_k: int = 10) -> List[List[Any]]:
        """
        Find similar items for a list of query items.
        :param query_items: List of query items
        :param top_k: Number of top similar items to return
        :return: List of top-K similar items for each query item
        """
        return [self.model.similar_items(item, top_k) for item in query_items]

    def evaluate(self, test_data: pd.DataFrame, recommend_size: int = 10, batch_size=100, filter_interacted: bool = True) -> Dict[str, float]:
        """
        Evaluates the model using batch evaluation metrics on the test data.

        Parameters:
            test_data (pd.DataFrame): DataFrame with columns ['user', 'item'] containing ground truth interactions.
            recommend_size (int): Number of items to recommend per user for evaluation.
            batch_size (int): Number of users to evaluate in each batch.
            filter_interacted (bool): Whether to filter out items the user has already interacted with during evaluation.

        Returns:
            Dict[str, float]: Dictionary with averaged evaluation metrics across all users.
        """
        # Group the test data by user to get ground truth items per user
        grouped_data = test_data.groupby('user')['item'].apply(list).to_dict()

        # Use a generator to yield recommendations and ground truth for each user
        def generate_evaluation_pairs() -> Iterable[Tuple[List[Any], List[Any]]]:
            # Split the grouped data into batches of users
            users = list(grouped_data.keys())

            for i in tqdm(range(0, len(users), batch_size)):
                batch_users = users[i:i + batch_size]
                # Get recommended items for the batch of users
                batch_results = self.recommend_batch(batch_users, recommend_size, filter_interacted)

                # Yield recommendations and ground truth for each user in the batch
                for user, recommended_items in zip(batch_users, batch_results):
                    ground_truth_items = grouped_data[user]
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
