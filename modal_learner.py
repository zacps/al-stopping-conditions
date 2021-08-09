from typing import Optional, Callable

import scipy
import numpy as np
from sklearn.base import BaseEstimator

from modAL import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from modAL.utils.data import modALinput
from modAL.utils.data import data_vstack


class IndexLearner(ActiveLearner):
    """
    Active learner which utilizes index sets instead of array modifications.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        X_training: Optional[modALinput],
        y_training: Optional[modALinput],
        X_unlabelled: Optional[modALinput],
        y_unlabelled: Optional[modALinput],
        query_strategy: Callable = uncertainty_sampling,
        bootstrap_init: bool = False,
        on_transformed: bool = False,
        **fit_kwargs,
    ) -> None:
        self.X_unlabelled = X_unlabelled
        self.y_unlabelled = y_unlabelled

        # See https://github.com/modAL-python/modAL/issues/103
        self.bootstrap_init = bootstrap_init

        self.taught_idx = np.array([], dtype=int)

        super().__init__(
            estimator,
            query_strategy,
            X_training,
            y_training,
            bootstrap_init,
            on_transformed,
            **fit_kwargs,
        )

    @property
    def X_training(self):
        return data_vstack((self._X_training, self.X_unlabelled[self.taught_idx]))

    @property
    def y_training(self):
        return np.concatenate((self._y_training, self.y_unlabelled[self.taught_idx]))

    @X_training.setter
    def X_training(self, X):
        self._X_training = X

    @y_training.setter
    def y_training(self, y):
        self._y_training = y

    def teach(
        self, query_idx: modALinput, bootstrap: bool = False, **fit_kwargs
    ) -> None:
        # assert one dimensional array
        assert len(query_idx.shape) == 1
        # Assert non-overlapping index sets
        overlap = np.in1d(self.taught_idx, query_idx)
        if np.count_nonzero(overlap) != 0:
            raise Exception(
                "Attempt to add an example to the training pool which has already been learnt."
                f"\nThe examples at indexes {self.taught_idx[overlap]} exist at indexes {np.where(overlap)[0]}."
                f"\nThere are currently {len(self.taught_idx)} learnt examples"
            )
        assert np.count_nonzero(overlap) == 0, str(
            np.count_nonzero(np.in1d(self.taught_idx, query_idx))
        )

        self.taught_idx = np.concatenate((self.taught_idx, query_idx))

        X_training = data_vstack((self.X_training, self.X_unlabelled[self.taught_idx]))
        y_training = data_vstack((self.y_training, self.y_unlabelled[self.taught_idx]))

        self.estimator.fit(X_training, y_training, **fit_kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        clear_attrs = ["_X_training", "_y_training", "X_unlabelled", "y_unlabelled"]
        for attr in clear_attrs:
            if hasattr(state, attr):
                del state[attr]
        for k, v in state.items():
            if isinstance(v, scipy.sparse.csr_matrix):
                raise UserWarning(f"Serialized learner has a sparse matrix field {k}")
        return state

    def __setstate__(self, state):
        # Pools are restored by compressedstore on load.
        self.__dict__.update(state)
