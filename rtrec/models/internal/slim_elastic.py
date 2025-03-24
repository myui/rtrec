import logging
import os
from typing import Any, Dict, List, Optional, Self, Tuple
import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike
import scipy.sparse as sp
from sklearn.linear_model import SGDRegressor, ElasticNet
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
from sklearn.utils.extmath import safe_sparse_dot
from multiprocessing import Pool, shared_memory
from functools import partial

from rtrec.utils.multiprocessing import create_shared_array

class CSRMatrixWrapper:
    """
    CSRMatrixWrapper is a wrapper class for a CSR matrix that provides efficient access to columns.
    """

    def __init__(self, csr_matrix: sp.csr_matrix):
        if not isinstance(csr_matrix, sp.csr_matrix):
            raise ValueError("Input matrix must be a scipy.sparse.csr_matrix.")
        self.csr_matrix = csr_matrix
        # create the columnar view of the matrix
        ind = self.csr_matrix.copy()
        ind.data = np.arange(len(ind.data)) # store the original data indices
        self.col_view = ind.tocsc()

    @property
    def matrix(self) -> sp.csr_matrix:
        return self.csr_matrix

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

class CSCMatrixWrapper:

    def __init__(self, csc_matrix: sp.csc_matrix):
        if not isinstance(csc_matrix, sp.csc_matrix):
            raise ValueError("Input matrix must be a scipy.sparse.csc_matrix.")
        self.csc_matrix = csc_matrix

    @property
    def matrix(self) -> sp.csc_matrix:
        return self.csc_matrix

    @property
    def shape(self) -> tuple:
        return self.csc_matrix.shape
    
    def get_col(self, j: int, copy: bool = True) -> sp.spmatrix:
        """
        Return the j-th column of the matrix.

        Parameters
        ----------
        j : int
            The column index.

        Returns
        -------
        scipy.sparse.spmatrix
            The j-th column of the matrix.
        """
        col = self.csc_matrix.getcol(j)
        if copy:
            return col.copy()
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
        start, end = self.csc_matrix.indptr[j], self.csc_matrix.indptr[j+1]
        assert len(values) == end - start, f"Values must have the same length as the column: {len(values)} != {end - start}"
        self.csc_matrix.data[start:end] = values

class FeatureSelectionWrapper:

    def __init__(self, model: ElasticNet, n_neighbors: int = 30):
        assert n_neighbors > 0, f"n_neighbors must be a positive integer: {n_neighbors}"
        self.model = model
        self.n_neighbors = n_neighbors
        self.sparse_coef_ = None

    def fit(self, X: sp.spmatrix, y: np.ndarray):
         # Compute dot products between items and the target
        feature_scores = X.T.dot(y).flatten()
        # Select the top-k similar items to the target item by sorting the dot products
        selected_features = np.argsort(feature_scores)[-1:-1-self.n_neighbors:-1]

        # Only fit the model with the selected features
        # TODO: Implement a more efficient way to select the features for csr_matrix
        self.model.fit(X[:, selected_features], y)
        
        # Store the coefficients of the fitted model
        coeff = self.model.coef_ # of shape (n_neighbors,)

        # Create a sparse representation (1, X.shape[1]) of the coefficients with only the selected features
        self.sparse_coef_ = sp.csr_matrix((coeff, (np.zeros_like(coeff), selected_features)), shape=(1, X.shape[1]))
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
            config (dict): Configuration parameters for the model

        Configuration:
            optimizer (str): Optimization method (cd or cg)
            eta0 (float): Learning rate used only for SGD
            alpha (float): Regularization strength
            l1_ratio (float): ElasticNet mixing parameter
            positive_only (bool): Whether to enforce positive coefficients
            max_iter (int): Maximum number of iterations
            tol (float): Tolerance for stopping criteria
            random_state (int): Random seed
            nn_feature_selection (int): Number of nearest neighbors for feature
                selection. If None, all features are used.
        """
        self.optim_name = config.get("optim", "cd") # optimization method (cd or cg)
        self.eta0 = config.get("eta0", 0.001) # Learning rate used only for SGD
        self.alpha = config.get("alpha", 0.1) # Regularization strength
        self.l1_ratio = config.get("l1_ratio", 0.1) # mostly for L2 regularization for SLIM
        self.positive_only = config.get("positive_only", True)
        self.max_iter = config.get("max_iter", 100)
        self.tol = config.get("tol", 1e-4)
        self.random_state = config.get("random_state", 43)
        self.nn_feature_selection = config.get("nn_feature_selection", None)

        # Initialize an empty item similarity matrix (will be computed during fit) of type scipy.sparse.csc_matrix
        self.item_similarity = None

    def get_model(self) -> ElasticNet | SGDRegressor | FeatureSelectionWrapper:
        if self.optim_name == "cd":
            model = ElasticNet(
                alpha=self.alpha, # Regularization strength
                l1_ratio=self.l1_ratio,
                fit_intercept=False,
                precompute=True, # Precompute Gram matrix for faster computation
                max_iter=self.max_iter,
                copy_X=False, # Avoid copying the input matrix
                tol=self.tol,
                positive=self.positive_only, # Enforce positive coefficients
                random_state=self.random_state,
                selection='random', # Randomize the order of features
            )
        elif self.optim_name == "sgd":
            model = SGDRegressor(
                loss="squared_error",
                penalty="elasticnet",
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=False,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                learning_rate="invscaling",
                eta0=self.eta0,
                average=False,
            )
        else:
            raise ValueError(f"Invalid Optimizer name: {self.optim_name}")
        if self.nn_feature_selection is not None:
            model = FeatureSelectionWrapper(model, n_neighbors=int(self.nn_feature_selection))
        return model

    def fit(self, interaction_matrix: sp.csc_matrix | sp.csr_matrix, parallel: bool=False, progress_bar: bool=False) -> Self:
        """
        Fit the SLIMElastic model to the interaction matrix.

        Args:
            interaction_matrix (csc_matrix | csr_matrix): User-item interaction matrix (sparse).
            parallel (bool): Whether to use parallel processing for fitting.
            progress_bar (bool): Whether to show a progress bar during training.
        """
        if isinstance(interaction_matrix, sp.csc_matrix):
            if parallel:
                return self.fit_in_parallel(interaction_matrix, progress_bar=progress_bar)
            X = CSCMatrixWrapper(interaction_matrix)
        elif isinstance(interaction_matrix, sp.csr_matrix):
            if parallel:
                logging.warning("Multiprocessing is only supported for CSC format. Fitting in single process.")
            X = CSRMatrixWrapper(interaction_matrix)
        else:
            raise ValueError("Interaction matrix must be a scipy.sparse.csr_matrix or scipy.sparse.csc_matrix.")

        num_items = X.shape[1]

        item_similarity = sp.lil_matrix((num_items, num_items))  # Start with LIL matrix

        model = self.get_model()

        # Ignore convergence warnings for ElasticNet
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)

            # Iterate through each item (column) and fit the model
            for j in tqdm(range(num_items), desc="Fitting SLIMElastic") if progress_bar else range(num_items):
                # Target column (current item)
                y = X.get_col(j)

                # Set the target item column to 0
                X.set_col(j, np.zeros_like(y.data))

                # Fit the model
                model.fit(X.matrix, y.toarray().ravel())

                # Update the item similarity matrix with new coefficients (weights for each user-item interaction)
                # item_similarity[:, j] = model.coef_
                for i, value in zip(model.sparse_coef_.indices, model.sparse_coef_.data):
                    item_similarity[i, j] = value

                # Reattach the item column after training
                X.set_col(j, y.data)

        # Convert item_similarity to CSC format for efficient access
        self.item_similarity = item_similarity.tocsc(copy=False)
        return self

    def fit_in_parallel(
        self, interaction_matrix: sp.csc_matrix, item_ids: Optional[ndarray] = None, progress_bar: bool = False, chunk_size: int = 100, num_workers: Optional[int] = None
    ) -> Self:
        """
        Fit the SLIM ElasticNet model in parallel.

        Args:
            interaction_matrix (sp.csc_matrix): Sparse interaction matrix.
            item_ids (ndarray): List of item indices to fit. If None, fit all items.
            progress_bar (bool): Whether to display a progress bar.
            chunk_size (int): Number of items per chunk for parallel processing.
            num_workers (int): Number of worker processes to use. Defaults to 70% of available CPU cores.

        Returns:
            Self: The fitted SLIM ElasticNet model.
        """
        if num_workers is None:
            num_workers = int(os.cpu_count() * 0.7)  # Default to 70% of available CPU cores

        if not isinstance(interaction_matrix, sp.csc_matrix):
            raise ValueError("Interaction matrix must be in CSC format for parallel processing.")

        # Create shared memory for interaction matrix
        shared_data = create_shared_array(interaction_matrix.data)
        shared_indices = create_shared_array(interaction_matrix.indices)
        shared_indptr = create_shared_array(interaction_matrix.indptr)

        # Create shapes for shared arrays
        shm_shapes = (
            interaction_matrix.data.shape,  # shape of the entire matrix
            interaction_matrix.indices.shape,  # shape of the indices array
            interaction_matrix.indptr.shape,  # shape of the indptr array
        )

        shm_dtypes = (interaction_matrix.data.dtype, interaction_matrix.indices.dtype, interaction_matrix.indptr.dtype)

        matrix_shape = interaction_matrix.shape
        num_items = matrix_shape[1]
        if self.item_similarity is None:
            item_similarity = sp.lil_matrix((num_items, num_items), dtype=np.float32)
        else:
            # ensure the item similarity matrix is large enough to accommodate the new items
            item_similarity = self.item_similarity.tolil()
            item_similarity.resize((num_items, num_items))

        if item_ids is None:
            item_ids = np.arange(num_items)

        # Ignore convergence warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)

            config = {
                "alpha": self.alpha,
                "l1_ratio": self.l1_ratio,
                "positive_only": self.positive_only,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "random_state": self.random_state,
                "nn_feature_selection": self.nn_feature_selection,
                "optim": self.optim_name,
                "eta0": self.eta0,
            }

            # Prepare partial function for multiprocessing
            worker_fn = partial(
                self._fit_items,
                shared_data_name = shared_data.name,
                shared_indices_name = shared_indices.name,
                shared_indptr_name = shared_indptr.name,
                shm_shapes = shm_shapes,
                shm_dtypes = shm_dtypes,
                matrix_shape = matrix_shape,
                config = config,
            )

            # Prepare batches of item indices using np.array_split
            item_chunks = np.array_split(item_ids, int(num_items / chunk_size))

            # Use multiprocessing with imap_unordered
            with Pool(processes=num_workers) as pool:
                results = pool.imap_unordered(worker_fn, item_chunks)

                if progress_bar:
                    results = tqdm(results, total=len(item_chunks), desc="Fitting SLIMElastic in parallel")

                for result in results:
                    for j, (indices, values) in result.items():
                        for i, value in zip(indices, values):
                            item_similarity[i, j] = value

        # Cleanup shared memory
        shared_data.close()
        shared_indices.close()
        shared_indptr.close()
        shared_data.unlink()
        shared_indices.unlink()
        shared_indptr.unlink()

        # Convert item similarity to CSC format
        self.item_similarity = item_similarity.tocsc(copy=False)
        return self

    @staticmethod
    def _fit_items(
        item_ids: ndarray,
        shared_data_name: str,
        shared_indices_name: str,
        shared_indptr_name: str,
        shm_shapes: Tuple[Tuple[int, int], Tuple[int], Tuple[int]],
        shm_dtypes: Tuple[np.dtype, np.dtype, np.dtype],
        matrix_shape: Tuple[int, int],
        config: dict[str, Any],
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Worker function for fitting multiple items using shared memory.

        Args:
            shared_data_name (str): Name of shared memory for `data`.
            shared_indices_name (str): Name of shared memory for `indices`.
            shared_indptr_name (str): Name of shared memory for `indptr`.
            shm_shapes (Tuple[Tuple[int, int], Tuple[int], Tuple[int]]): Shapes of the shared arrays (`data`, `indices`, `indptr`).
            matrix_shape (Tuple[int, int]): Shape of the full interaction matrix.
            items (List[int]): List of target column indices.
            alpha (float): Regularization strength.
            l1_ratio (float): ElasticNet mixing parameter.

        Returns:
            Dict[int, Tuple[np.ndarray, np.ndarray]]: Mapping of item index to (non-zero indices, coefficients).
        """
        # Access shared memory
        shared_data = shared_memory.SharedMemory(name=shared_data_name)
        shared_indices = shared_memory.SharedMemory(name=shared_indices_name)
        shared_indptr = shared_memory.SharedMemory(name=shared_indptr_name)

        # Reconstruct CSC matrix with provided shapes
        X = sp.csc_matrix(
            (
                np.ndarray(shm_shapes[0], dtype=shm_dtypes[0], buffer=shared_data.buf),
                np.ndarray(shm_shapes[1], dtype=shm_dtypes[1], buffer=shared_indices.buf),
                np.ndarray(shm_shapes[2], dtype=shm_dtypes[2], buffer=shared_indptr.buf),
            ),
            shape=matrix_shape,
        )

        results = {}

        model = SLIMElastic._get_model(config)

        for j in item_ids:
            y = X.getcol(j).copy()

            # Temporarily zero-out the target column
            X.data[X.indptr[j]:X.indptr[j + 1]] = 0

            # Fit the model
            model.fit(X, y.toarray().ravel())

            # Restore the column
            X.data[X.indptr[j]:X.indptr[j + 1]] = y.data

            # Collect results
            results[j] = (model.sparse_coef_.indices, model.sparse_coef_.data)

        # Close shared memory
        shared_data.close()
        shared_indices.close()
        shared_indptr.close()

        return results

    @staticmethod
    def _get_model(config: dict[str, Any]) -> ElasticNet | SGDRegressor | FeatureSelectionWrapper:        
        optim_name = config.get("optim", "cd")
        if optim_name == "cd":
            model = ElasticNet(
                alpha=config["alpha"],
                l1_ratio=config["l1_ratio"],
                fit_intercept=False,
                precompute=True,
                max_iter=config["max_iter"],
                copy_X=False,
                tol=config["tol"],
                positive=config["positive_only"],
                random_state=config["random_state"],
                selection="random",
            )
        elif optim_name == "sgd":
            model = SGDRegressor(
                loss="squared_error",
                penalty="elasticnet",
                alpha=config["alpha"],
                l1_ratio=config["l1_ratio"],
                fit_intercept=False,
                max_iter=config["max_iter"],
                tol=config["tol"],
                random_state=config["random_state"],
                learning_rate="invscaling",
                eta0=config["eta0"],
                average=False,
            )
        else:
            raise ValueError(f"Invalid Optimizer name: {optim_name}")

        nn_feature_selection = config.get("nn_feature_selection", None)
        if nn_feature_selection is not None:
            model = FeatureSelectionWrapper(model, n_neighbors=int(nn_feature_selection))

        return model

    def partial_fit(self, interaction_matrix: sp.csr_matrix, user_ids: List[int], parallel: bool=False, progress_bar: bool=False) -> Self:
        """
        Incrementally fit the SLIMElastic model with new or updated users.

        Args:
            interaction_matrix (csr_matrix): user-item interaction matrix (sparse).
            user_ids (list): List of user indices that were updated.
            parallel (bool): Whether to use parallel processing for fitting.
            progress_bar (bool): Whether to show a progress bar during training.
        """        
        user_items = set()
        for user_id in user_ids:
            user_items.update(interaction_matrix[user_id, :].indices.tolist())
        return self.partial_fit_items(interaction_matrix, list(user_items), progress_bar)

    def partial_fit_items(self, interaction_matrix: sp.csc_matrix | sp.csr_matrix, updated_items: List[int], parallel: bool=False, progress_bar: bool=False) -> Self:
        """
        Incrementally fit the SLIMElastic model with new or updated items.

        Args:
            interaction_matrix (csc_matrix | csr_matrix): user-item interaction matrix (sparse).
            updated_items (list): List of item indices that were updated.
            parallel (bool): Whether to use parallel processing for fitting.
            progress_bar (bool): Whether to show a progress bar during training.
        """
        if isinstance(interaction_matrix, sp.csc_matrix):
            if parallel:
                return self.fit_in_parallel(interaction_matrix, item_ids=np.array(updated_items), progress_bar=progress_bar)
            X = CSCMatrixWrapper(interaction_matrix)
        elif isinstance(interaction_matrix, sp.csr_matrix):
            X = CSRMatrixWrapper(interaction_matrix)
        else:
            raise ValueError("Interaction matrix must be a scipy.sparse.csr_matrix or scipy.sparse.csc_matrix.")

        model = self.get_model()

        num_items = X.shape[1]
        if self.item_similarity is None:
            item_similarity = sp.lil_matrix((num_items, num_items), dtype=np.float32)
        else:
            # ensure the item similarity matrix is large enough to accommodate the new items
            item_similarity = self.item_similarity.tolil()
            item_similarity.resize((num_items, num_items))

        # Iterate through the updated items and fit the model incrementally
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)

            for j in tqdm(updated_items, desc="Fitting SLIMElastic") if progress_bar else updated_items:
                # Target column (current item)
                y = X.get_col(j)

                # Set the target item column to 0
                X.set_col(j, np.zeros_like(y.data))

                # Fit the model for the updated item
                model.fit(X.matrix, y.toarray().ravel())

                # Update the item similarity matrix with new coefficients (weights for each user-item interaction)
                # self.item_similarity[:, j] = model.coef_
                for i, value in zip(model.sparse_coef_.indices, model.sparse_coef_.data):
                    item_similarity[i, j] = value

                # Reattach the item column after training
                X.set_col(j, y.data)

        # Convert item_similarity to CSC format for efficient access
        self.item_similarity = item_similarity.tocsc(copy=False)
        return self

    def predict(self, user_id: int, interaction_matrix: sp.csr_matrix, dense_output: bool=True) -> ndarray:
        """
        Compute the predicted scores for a specific user across all items.

        Args:
            user_id (int): The user ID (row index in interaction_matrix).
            interaction_matrix (csr_matrix): User-item interaction matrix.
            dense_output (bool): Whether to return a dense output.

        Returns:
            numpy.ndarray: Predicted scores for the user across all items of shape (n_items,)
        """
        if self.item_similarity is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        # Compute the predicted scores by performing dot product between the user interaction vector
        # and the item similarity matrix
        return safe_sparse_dot(interaction_matrix[user_id, :], self.item_similarity, dense_output=dense_output)

    def predict_selected(self, user_id: int, item_ids: List[int], interaction_matrix: sp.csr_matrix, dense_output: bool=True) -> ndarray:
        """
        Compute the predicted scores for a specific user and a subset of items.

        Args:
            user_id (int): The user ID (row index in interaction_matrix).
            item_ids (List[int]): List of item indices to compute the predicted scores for.
            interaction_matrix (csr_matrix): User-item interaction matrix.
            dense_output (bool): Whether to return a dense output.

        Returns:
            numpy.ndarray: Predicted scores for the user and selected items of shape (len(item_ids),)
        """
        if self.item_similarity is None:
            raise RuntimeError("Model must be fitted before calling predict_selected.")

        # Compute the predicted scores for the selected items by performing dot product between the user interaction vector
        # and the item similarity matrix
        # return interaction_matrix[user_id, :].dot(self.item_similarity[:, item_ids])
        return safe_sparse_dot(interaction_matrix[user_id, :], self.item_similarity[:, item_ids], dense_output=dense_output)

    def predict_all(self, interaction_matrix: sp.csr_matrix, dense_output: bool=True) -> ndarray | sp.csr_matrix:
        """
        Compute the predicted scores for all users and items.

        Args:
            interaction_matrix (csr_matrix): User-item interaction matrix.
            dense_output (bool): Whether to return a dense output.

        Returns:
            numpy.ndarray | scipy.sparse.csr_matrix: Predicted scores for all users and items.
        """
        if self.item_similarity is None:
            raise RuntimeError("Model must be fitted before calling predict_all.")

        # Compute the predicted scores for all users by performing dot product between the interaction matrix
        # and the item similarity matrix
        return safe_sparse_dot(interaction_matrix, self.item_similarity, dense_output=dense_output)

    def recommend(self,
                  user_id: int,
                  interaction_matrix: sp.csr_matrix,
                  candidate_item_ids: Optional[List[int]]=None,
                  top_k: int=10,
                  filter_interacted: bool=True,
                  dense_output: bool=True
    ) -> List[int]:
        """
        Recommend top-K items for a given user.

        Args:
            user_id (int): ID of the user (row index in interaction_matrix).
            interaction_matrix (csr_matrix): User-item interaction matrix (sparse).
            candidate_item_ids (List[int]): List of candidate item indices to recommend from. If None, recommend from all items.
            top_k (int): Number of recommendations to return.
            filter_interacted (bool): Whether to exclude items the user has already interacted with. Ignored if candidate_item_ids is provided.
            dense_output (bool): Whether to return dense item IDs at prediction time.

        Returns:
            List[int]: List of top-K item indices recommended for the user.
        """
        # Get predicted scores for all items for the given user
        if candidate_item_ids is None:
            scores = self.predict(user_id, interaction_matrix, dense_output=dense_output)
            if dense_output:
                return self._dense_topk_indicies(scores, top_k, user_id, interaction_matrix, filter_interacted)
            else:
                return self._sparse_topk_indicies(scores, top_k, user_id, interaction_matrix, filter_interacted)
        else:
            scores = self.predict_selected(user_id, candidate_item_ids, interaction_matrix, dense_output=True)
            scores = scores.ravel()
            assert len(scores) == len(candidate_item_ids), f"Predicted scores must have the same length as candidate_item_ids: {len(scores)} != {len(candidate_item_ids)}"
            # sort the candidate_item_ids by user_scores and take top-k
            top_items = [candidate_item_ids[i] for i in np.argsort(scores)[-top_k:][::-1]]
            return top_items

    @staticmethod
    def _dense_topk_indicies(scores: ndarray, top_k: int, user_id: int, interaction_matrix: sp.csr_matrix, filter_interacted: bool=True) -> List[int]:
        """
        Get the top-K indices for a given dense matrix.

        Args:
            scores (ndarray): Dense matrix of scores.
            top_k (int): Number of top indices to retrieve.
            user_id (int): User index.
            interaction_matrix (csr_matrix): User-item interaction matrix.
            filter_interacted (bool): Whether to filter out items the user has already interacted with.

        Returns:
            List[int]: List of top-K indices.
        """
        scores = scores.ravel()
        # Exclude items that the user has already interacted with
        if filter_interacted:
            interacted_items = interaction_matrix[user_id, :].indices
            scores[interacted_items] = -np.inf  # Exclude interacted items by setting scores to -inf

        # Get the top-K items by sorting the predicted scores in descending order
        # [::-1] reverses the order to get the items with the highest scores first
        top_items = np.argsort(scores)[-top_k:][::-1]

        # Filter out items with -np.inf scores
        if len(top_items) > 0:
            valid_indices = scores[top_items] != -np.inf
            top_items = top_items[valid_indices]

        return top_items.tolist() # Convert numpy array to list

    @staticmethod
    def _sparse_topk_indicies(scores: sp.csr_matrix, top_k: int, user_id: int, interaction_matrix: sp.csr_matrix, filter_interacted: bool=True) -> List[int]:
        """
        Get the top-K indices for a given sparse matrix.

        Args:
            scores (csr_matrix): Sparse matrix of scores.
            top_k (int): Number of top indices to retrieve.
            filter_interacted (bool): Whether to filter out items the user has already interacted with.

        Returns:
            List[int]: List of top-K indices.
            """
        # Extract non-zero scores and their indices from the sparse matrix
        score_data = scores.data.tolist()
        score_indices = scores.indices.tolist()

        if filter_interacted:
            # Filter out scores for interacted items
            interacted_items = set(interaction_matrix[user_id].indices.tolist())
            filtered_scores = [
                (idx, score) for idx, score in zip(score_indices, score_data) if idx not in interacted_items
            ]
        else:
            filtered_scores = [(idx, score) for idx, score in zip(score_indices, score_data)]

        # Sort by score in descending order
        top_items = sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:top_k]

        # Extract and return the top-k indices
        return [idx for idx, _ in top_items]

    def similar_items(self, item_id: int, top_k: int=10) -> List[Tuple[int, float]]:
        """
        Get the top-K most similar items to a given item.

        Args:
            item_id (int): The item ID (column index in the interaction matrix).
            top_k (int): Number of similar items to retrieve.

        Returns:
            List[int]: List of top-K similar item indices
        """
        if self.item_similarity is None:
            raise RuntimeError("Model must be fitted before calling similar_items.")

        # Get the item similarity vector for the given item
        item_similarity: sp.csr_matrix = self.item_similarity[:,item_id]

        # Get non-zero indices and their corresponding similarity scores
        indices = item_similarity.indices
        scores = item_similarity.data
        # Exclude the query item itself
        valid_mask = indices != item_id
        valid_indices = indices[valid_mask]
        valid_scores = scores[valid_mask]

        # Sort the indices by similarity scores in descending order
        # return sorted(zip(valid_indices, valid_scores), key=lambda x: x[1], reverse=True)[:top_k]
        top_k_indices = np.argsort(-valid_scores)[:top_k]
        ids = valid_indices[top_k_indices].tolist()
        scores = valid_scores[top_k_indices].tolist()
        return list(zip(ids, scores))
