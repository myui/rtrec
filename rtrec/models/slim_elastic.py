from typing import List
import numpy as np
from numpy.typing import ArrayLike
import scipy.sparse as sp
from sklearn.linear_model import ElasticNet, SGDRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
from scipy.sparse.linalg import cg

class ColumnarView:
    def __init__(self, csr_matrix: sp.csr_matrix):
        self.csr_matrix = csr_matrix
        # create the columnar view of the matrix
        ind = self.csr_matrix.copy()
        ind.data = np.arange(len(ind.data)) # store the original data indices
        self.col_view = ind.tocsc()

    @property
    def shape(self) -> tuple:
        return self.csr_matrix.shape
    
    def get_col(self, j: int) -> sp.csc_matrix:
        """
        Return the j-th column of the matrix.

        Parameters
        ----------
        j : int
            The column index.

        Returns
        -------
        scipy.sparse.csc_matrix
            The j-th column of the matrix.
        """

        col = self.col_view[:, j].copy()
        col.data = self.csr_matrix.data[col.data]
        return col
    
    def set_col(self, j: int, values: ArrayLike) -> None:
        """
        Set the j-th column of the matrix to the given values.

        Parameters
        ----------
        j : int
            The column index.
        values : ArrayLike
            The new values for the column.
        """
        self.csr_matrix.data[self.col_view[:, j].data] = values

class FeatureSelectionWrapper:

    def __init__(self, model: ElasticNet, n_neighbors: int = 30):
        assert n_neighbors > 0, f"n_neighbors must be a positive integer: {n_neighbors}"
        self.model = model
        self.n_neighbors = n_neighbors
        self.coef_ = None

    def fit(self, X: sp.csr_matrix, y: np.ndarray):
         # Compute dot products between items and the target
        feature_scores = X.T.dot(y).flatten()
        # Select the top-k similar items to the target item by sorting the dot products
        selected_features = np.argsort(feature_scores)[-1:-1-self.n_neighbors:-1]

        # Only fit the model with the selected features
        self.model.fit(X[:, selected_features], y)
        
        # Store the coefficients of the fitted model
        coef = np.zeros(X.shape[1])
        coef[selected_features] = self.model.coef_
        self.coef_ = coef
        return self

class SLIMElastic:
    """
    SLIMElastic is a sparse linear method for top-K recommendation, which learns
    a sparse aggregation coefficient matrix by solving an L1-norm and L2-norm
    regularized optimization problem.
    """

    def __init__(self, config: dict={}):
        """
        Initialize the SLIMElastic model.
        
        Args:
            alpha (float): Regularization strength.
            l1_ratio (float): The ratio between L1 and L2 regularization.
            positive_only (bool): Whether to enforce positive coefficients.
        """
        self.model_name = config.get("model_name", "cd")
        self.eta0 = config.get("eta0", 0.001) # Learning rate used only for SGD
        self.alpha = config.get("alpha", 0.1) # Regularization strength
        self.l1_ratio = config.get("l1_ratio", 0.1) # mostly for L2 regularization for SLIM
        self.positive_only = config.get("positive_only", True)
        self.max_iter = config.get("max_iter", 30)
        self.tol = config.get("tol", 1e-4)
        self.random_state = config.get("random_state", 43)
        self.nn_feature_selection = config.get("nn_feature_selection", None)

        # Initialize an empty item similarity matrix (will be computed during fit)
        self.item_similarity = None

    def get_model(self):
        if self.model_name == "cd" or self.model_name == "coordinate_descent":
            return ElasticNet(
                    alpha=self.alpha, # Regularization strength
                    l1_ratio=self.l1_ratio,
                    positive=self.positive_only, # Enforce positive coefficients
                    fit_intercept=False,
                    copy_X=False, # Avoid copying the input matrix
                    precompute=True, # Precompute Gram matrix for faster computation
                    max_iter=self.max_iter,
                    tol=self.tol,
                    selection='random', # Randomize the order of features
                    random_state=self.random_state,
                )
        elif self.model_name == "sgd":
            return SGDRegressor(
                    penalty='elasticnet',
                    eta0 = self.eta0,
                    alpha=self.alpha,
                    l1_ratio=self.l1_ratio,
                    fit_intercept=False,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state,
                )
        else:
            raise ValueError(f"Invalid model name: {self.model_name}")

    def fit(self, interaction_matrix: sp.csr_matrix):
        """
        Fit the SLIMElastic model to the interaction matrix.

        Args:
            interaction_matrix (csr_matrix): User-item interaction matrix (sparse).
        """
        if not isinstance(interaction_matrix, sp.csr_matrix):
            raise ValueError("Interaction matrix must be a scipy.sparse.csr_matrix of user-item interactions.")

        X = ColumnarView(interaction_matrix)
        num_items = X.shape[1]

        self.item_similarity = np.zeros((num_items, num_items))  # Initialize similarity matrix
        
        model = self.get_model()

        if self.nn_feature_selection is not None:
            model = FeatureSelectionWrapper(model, n_neighbors=int(self.nn_feature_selection))

        # Ignore convergence warnings for ElasticNet
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)

            # Iterate through each item (column) and fit the model
            for j in tqdm(range(num_items), desc="Fitting SLIMElastic"):
                # Target column (current item)
                y = X.get_col(j)

                # Set the target item column to 0
                X.set_col(j, np.zeros_like(y.data))

                # Fit the model
                model.fit(X.csr_matrix, y.toarray().ravel())

                # Update the item similarity matrix with new coefficients (weights for each user-item interaction)
                self.item_similarity[:, j] = model.coef_

                # Reattach the item column after training
                X.set_col(j, y.data)

        return self

    def partial_fit(self, interaction_matrix: sp.csr_matrix, user_ids: List[int]):
        """
        Incrementally fit the SLIMElastic model with new or updated users.

        Args:
            interaction_matrix (coo_matrix): user-item interaction matrix (sparse).
            user_ids (list): List of user indices that were updated.
        """        
        user_items = set()
        for user_id in user_ids:
            user_items.update(interaction_matrix[user_id].indices.tolist())
        return self.partial_fit_items(interaction_matrix, list(user_items))

    def partial_fit_items(self, interaction_matrix: sp.csr_matrix, updated_items: List[int]):
        """
        Incrementally fit the SLIMElastic model with new or updated items.

        Args:
            interaction_matrix (coo_matrix): user-item interaction matrix (sparse).
            updated_items (list): List of item indices that were updated.
        """
        if not isinstance(interaction_matrix, sp.csr_matrix):
            raise ValueError("Interaction matrix must be a scipy.sparse.csr_matrix of user-item interactions.")

        X = ColumnarView(interaction_matrix)

        model = self.get_model()

        if self.nn_feature_selection is not None:
            model = FeatureSelectionWrapper(model, n_neighbors=int(self.nn_feature_selection))

        old_size = self.item_similarity.shape[1]
        new_size = max(updated_items) + 1
        if new_size > old_size:
            # Expand both rows and columns symmetrically
            expanded_similarity = np.zeros((new_size, new_size))
            expanded_similarity[:old_size, :old_size] = self.item_similarity
            self.item_similarity = expanded_similarity

        # Iterate through the updated items and fit the model incrementally
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)

            for j in tqdm(updated_items):
                # Target column (current item)
                y = X.get_col(j)

                # Set the target item column to 0
                X.set_col(j, np.zeros_like(y.data))

                # Fit the model for the updated item
                model.fit(X.csr_matrix, y.toarray().ravel())

                # Update the item similarity matrix with new coefficients (weights for each user-item interaction)
                self.item_similarity[:, j] = model.coef_

                # Reattach the item column after training
                X.set_col(j, y.data)

        return self

    def predict(self, user_id: int, interaction_matrix: sp.csr_matrix):
        """
        Compute the predicted scores for a specific user across all items.

        Args:
            user_id (int): The user ID (row index in interaction_matrix).
            interaction_matrix (csr_matrix): User-item interaction matrix.

        Returns:
            numpy.ndarray: Predicted scores for the user across all items.
        """
        if self.item_similarity is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        # Compute the predicted scores by performing dot product between the user interaction vector
        # and the item similarity matrix
        return interaction_matrix[user_id].dot(self.item_similarity)

    def predict_all(self, interaction_matrix: sp.csr_matrix):
        """
        Compute the predicted scores for all users and items.

        Args:
            interaction_matrix (csr_matrix): User-item interaction matrix.

        Returns:
            numpy.ndarray: Predicted scores for all users and items.
        """
        if self.item_similarity is None:
            raise RuntimeError("Model must be fitted before calling predict_all.")

        # Compute the predicted scores for all users by performing dot product between the interaction matrix
        # and the item similarity matrix
        return interaction_matrix.dot(self.item_similarity)

    def similar_items(self, item_id: int, top_k: int=10):
        """
        Get the top-K most similar items to a given item.

        Args:
            item_id (int): The item ID (column index in the interaction matrix).
            top_k (int): Number of similar items to retrieve.

        Returns:
            List of similar item indices.
        """
        if self.item_similarity is None:
            raise RuntimeError("Model must be fitted before calling similar_items.")

        # Get the item similarity vector for the given item
        item_similarity = self.item_similarity[item_id]

        # Get the top-K similar items by sorting the similarity scores in descending order
        similar_items = np.argsort(item_similarity)[-1:-1-top_k:-1]
        return similar_items

    def recommend(self, user_id: int, interaction_matrix: sp.csr_matrix, top_k: int=10, exclude_seen: bool=True) -> List[int]:
        """
        Recommend top-K items for a given user.

        Args:
            user_id (int): ID of the user (row index in interaction_matrix).
            interaction_matrix (csr_matrix): User-item interaction matrix (sparse).
            top_k (int): Number of recommendations to return.
            exclude_seen (bool): Whether to exclude items the user has already interacted with.

        Returns:
            List of recommended item indices.
        """
        # Get predicted scores for all items for the given user
        user_scores = self.predict(user_id, interaction_matrix).ravel()

        # Exclude items that the user has already interacted with
        if exclude_seen:
            seen_items = interaction_matrix[user_id].indices
            user_scores[seen_items] = -np.inf  # Exclude seen items by setting scores to -inf

        # Get the top-K items by sorting the predicted scores in descending order
        # [::-1] reverses the order to get the items with the highest scores first
        top_items = np.argsort(user_scores)[-top_k:][::-1]

        return top_items