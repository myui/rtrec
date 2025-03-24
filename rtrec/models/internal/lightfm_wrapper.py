from lightfm import LightFM
from sklearn.base import clone
from typing import Optional
# require typing-extensions >= 4.5
# from typing import override
from scipy.sparse import csr_matrix, coo_matrix

class LightFMWrapper(LightFM):
    """
    A LightFM that resizes the model to accomodate new users, items, and features.

    See https://github.com/lyst/lightfm/issues/347#issuecomment-707829342
    """

    def __init__(
        self,
        no_components=10,
        k=5,             # for k-OS training, the k-th positive example will be selected from the n positive examples sampled for every user.
        n=10,            # for k-OS training, maximum number of positives sampled for each update.
        learning_schedule="adagrad",
        loss="warp",
        learning_rate=0.05,
        rho=0.95,        # moving average coefficient for the adadelta learning schedule.
        epsilon=1e-6,    # conditioning parameter for the adadelta learning schedule.
        item_alpha=1e-7, # L2 penalty on item features
        user_alpha=1e-7, # L2 penalty on user features
        max_sampled=10,  # maximum number of negative samples used during WARP fitting
        random_state=None,
        **kwargs,
    ):
        super().__init__(
            no_components,
            k,
            n,
            learning_schedule,
            loss,
            learning_rate,
            rho,
            epsilon,
            item_alpha,
            user_alpha,
            max_sampled,
            random_state,
        )

    #@override
    def fit_partial(
        self,
        interactions: coo_matrix,
        user_features: Optional[csr_matrix] = None,
        item_features: Optional[csr_matrix] = None,
        sample_weight: Optional[coo_matrix] = None,
        epochs: int = 1,
        num_threads : int = 1,
        verbose: bool = False,
    ):
        """
        Fit the model on a new batch of interactions, user and item features.
        This method resizes the model to accomodate new users, items, and features.

        Parameters
        ----------
        interactions : coo_matrix
            The user-item interactions matrix of shape [n_users, n_items].
        user_features : csr_matrix, optional
            The user features matrix of shape [n_users, n_user_features].
        item_features : csr_matrix, optional
            The item features matrix of shape [n_items, n_item_features].
        sample_weight : coo_matrix, optional
            The sample weights matrix of shape [n_users, n_items].
        epochs : int, optional
            The number of epochs to run.
        num_threads : int, optional
            The number of threads to use.
        verbose : bool, optional
            Whether to print progress information.
        """
        try:
            self._check_initialized()
            self._resize(interactions, user_features, item_features)
        except ValueError:
            # This is the first call so just fit without resizing
            pass

        super().fit_partial(
            interactions,
            user_features,
            item_features,
            sample_weight,
            epochs,
            num_threads,
            verbose,
        )

        return self

    def _resize(self, interactions, user_features=None, item_features=None):
        """Resizes the model to accommodate new users/items/features"""

        no_components = self.no_components
        no_user_features, no_item_features = interactions.shape  # default

        if hasattr(user_features, "shape"):
            no_user_features = user_features.shape[-1]
        if hasattr(item_features, "shape"):
            no_item_features = item_features.shape[-1]

        if (
            no_user_features == self.user_embeddings.shape[0]
            and no_item_features == self.item_embeddings.shape[0]
        ):
            return self

        new_model = clone(self)
        new_model._initialize(no_components, no_item_features, no_user_features)

        # update all attributes from self._check_initialized
        for attr in (
            "item_embeddings",
            "item_embedding_gradients",
            "item_embedding_momentum",
            "item_biases",
            "item_bias_gradients",
            "item_bias_momentum",
            "user_embeddings",
            "user_embedding_gradients",
            "user_embedding_momentum",
            "user_biases",
            "user_bias_gradients",
            "user_bias_momentum",
        ):
            # extend attribute matrices with new rows/cols from
            # freshly initialized model with right shape
            old_array = getattr(self, attr)
            old_slice = [slice(None, i) for i in old_array.shape]
            new_array = getattr(new_model, attr)
            new_array[tuple(old_slice)] = old_array
            setattr(self, attr, new_array)

        return self
