import itertools
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import normalize as _normalize, LabelEncoder
import numpy as np
from scipy.linalg import block_diag

from utils import subspace_bases
from utils import mean_square_singular_values


class SMBase(BaseEstimator, ClassifierMixin):
    """
    Base class of Subspace Method
    """
    param_names = {'normalize', 'n_subdims'}

    def __init__(self, n_subdims, normalize=False, faster_mode=False):
        """
        Parameters
        ----------
        n_subdims : int
            The dimension of subspace. it must be smaller than the dimension of original space.
        normalize : boolean, optional (default=True)
            If this is True, all vectors are normalized as |v| = 1
        """
        self.n_subdims = n_subdims
        self.normalize = normalize
        self.faster_mode = faster_mode
        self.le = LabelEncoder()
        self.dic = None
        self.labels = None
        self.n_classes = None
        self._test_n_subdims = None
        self.params = ()

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in self.param_names}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _prepare_X(self, X):
        """
        preprocessing data matrices X.
        normalize and transpose
        Parameters
        ----------
        X: list of 2d-arrays, (n_classes, n_samples, n_dims)
        """
        # normalize each vectors
        if self.normalize:
            X = [_normalize(_X) for _X in X]

        # transpose to make feature vectors as column vectors
        # this makes it easy to implement referring to formula
        X = [_X.T for _X in X]

        return X

    def _prepare_y(self, y):
        # converted labels
        y = self.le.fit_transform(y)
        self.labels = y

        # number of classes
        self.n_classes = self.le.classes_.size

        # number of data
        self.n_data = len(y)

        return y

    def fit(self, X, y):
        """
        Fit the model using the given data and parameters
        Parameters
        ----------
        X: list of 2d-arrays, (n_classes, n_samples, n_dims)
            Training vectors. n_classes is count of classes.
            n_samples is number of vectors of samples, this is variable across each classes.
            n_dims is number of dimensions of vectors.
        y: integer array, (n_classes)
            Class labels of training vectors.
        """

        # preprocessing data matrices
        # ! X[i] will transposed for conventional
        X = self._prepare_X(X)
        y = self._prepare_y(y)

        self._fit(X, y)

    def _fit(self, X, y):
        """
        Parameters
        ----------
        X: list of 2d-arrays, (n_classes, n_dims, n_samples)
        y: array, (n_classes)
        """

        dic = [subspace_bases(_X, self.n_subdims) for _X in X]
        # dic,  (n_classes, n_dims, n_subdims)
        dic = np.array(dic)
        self.dic = dic

    def predict(self, X):
        """
        Predict classes
        Parameters:
        -----------
        X: list of 2d-arrays, (n_vector_sets, n_samples, n_dims)
            List of input vector sets.
        Returns:
        --------
        pred: array, (n_vector_sets)
            Prediction array
        """

        if self.faster_mode and hasattr(self, 'fast_predict_proba'):
            proba = self.fast_predict_proba(X)
        else:
            proba = self.predict_proba(X)

        salt = 1e-3
        assert proba.min() > 0.0 - salt, 'some probabilities are smaller than 0! min value is {}'.format(proba.min())
        assert proba.max() < 1.0 + salt, 'some probabilities are bigger than 1! max value is {}'.format(proba.max())
        proba = np.clip(proba, 0, 1)
        return self.proba2class(proba)

    def proba2class(self, proba):
        pred = self.labels[np.argmax(proba, axis=1)]
        return self.le.inverse_transform(pred)

    def predict_proba(self, X):
        """
        Predict class probabilities
        Parameters:
        -----------
        X: 2d-array, (n_samples, n_dims)
            Matrix of input vectors.
        Returns:
        --------
        pred: array-like, shape: (n_samples)
            Prediction array
        """

        # preprocessing data matrices
        X = self._prepare_X([X])[0]
        pred = self._predict_proba(X)
        return pred

    def _predict_proba(self, X):
        """
        Parameters
        ----------
        X: arrays, (n_dims, n_samples)
        """
        raise NotImplementedError('_predict is not implemented')


class CVMSMInterface(object):
    """
    Prediction interface of Mutual Subspace Method
    """

    def predict_proba(self, X):
        """
        Predict class probabilities
        Parameters:
        -----------
        X: list of 2d-arrays, (n_vector_sets, n_samples, n_dims)
            List of input vector sets.
        Returns:
        --------
        pred: array, (n_vector_sets)
            Prediction array
        """

        # preprocessing data matrices
        X = self._prepare_X(X)

        pred = []
        for _X in X:
            # gramians, (n_classes, n_subdims, n_subdims)
            gramians = self._get_gramians(_X)

            # i_th singular value of gramian of subspace bases is
            # square root of cosine of i_th canonical angles
            # average of square of them is canonical angle between subspaces
            c = [mean_square_singular_values(g) for g in gramians]
            pred.append(c)
        return np.array(pred)

    def _get_gramians(self, X):
        """
        Parameters
        ----------
        X: array, (n_dims, n_samples)
        Returns
        -------
        G: array, (n_class, n_subdims, n_subdims)
            gramian matrices of references of each class
        """
        raise NotImplementedError('_get_gramians is not implemented')

    @property
    def test_n_subdims(self):
        if self._test_n_subdims is None:
            return self.n_subdims
        return self._test_n_subdims

    @test_n_subdims.setter
    def test_n_subdims(self, v):
        assert isinstance(v, int)
        self._test_n_subdims = v

    @test_n_subdims.deleter
    def test_n_subdims(self):
        self._test_n_subdims = None