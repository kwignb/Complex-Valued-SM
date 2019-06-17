import numpy as np
from scipy.linalg import eigh


def mean_square_singular_values(X):
    """
    calculate mean square of singular values of X
    Parameters:
    -----------
    X : array-like, shape: (n, m)
    Returns:
    --------
    c: mean square of singular values
    """

    # _, s, _ = np.linalg.svd(X)
    # mssv = (s ** 2).mean()

    # Frobenius norm means square root of
    # sum of square singular values
    mssv = (X * np.conjugate(X)).sum() / min(X.shape)
    return mssv


def _eigh(X, eigvals=None):
    """
    A wrapper function of numpy.linalg.eigh and scipy.linalg.eigh
    Parameters
    ----------
    X: array-like, shape (a, a)
        target symmetric matrix
    eigvals: tuple, (lo, hi)
        Indexes of the smallest and largest (in ascending order) eigenvalues and corresponding eigenvectors
        to be returned: 0 <= lo <= hi <= M-1. If omitted, all eigenvalues and eigenvectors are returned.
    Returns
    -------
    e: array-like, shape (a) or (n_dims)
        eigenvalues with descending order
    V: array-like, shape (a, a) or (a, n_dims)
        eigenvectors
    """

    if eigvals != None:
        e, V = eigh(X, eigvals=eigvals)
    else:
        # numpy's eigh is faster than scipy's when all calculating eigenvalues and eigenvectors
        e, V = np.linalg.eigh(X)

    e, V = e[::-1], V[:, ::-1]

    return e, V


def _eigen_basis(X, eigvals=None):
    """
    Return subspace basis using PCA
    Parameters
    ----------
    X: array-like, shape (a, a)
        target matrix
    n_dims: integer
        number of basis
    Returns
    -------
    e: array-like, shape (a) or (n_dims)
        eigenvalues with descending order
    V: array-like, shape (a, a) or (a, n_dims)
        eigenvectors
    """

    try:
        e, V = _eigh(X, eigvals=eigvals)
    except np.linalg.LinAlgError:
        # if it not converges, try with tiny salt
        salt = 1e-8 * np.eye(X.shape[0])
        e, V = eigh(X + salt, eigvals=eigvals)

    return e, V


def _get_eigvals(n, n_subdims, higher):
    """
    Culculate eigvals for eigh

    Parameters
    ----------
    n: int
    n_subdims: int, dimension of subspace
    higher: boolean, if True, use higher `n_subdim` basis
    Returns
    -------
    eigvals: tuple of 2 integers
    """

    if n_subdims is None:
        return None

    if higher:
        low = max(0, n - n_subdims)
        high = n - 1
    else:
        low = 0
        high = min(n - 1, n_subdims - 1)

    return low, high


def subspace_bases(X, n_subdims=None, higher=True, return_eigvals=False):
    """
    Return subspace basis using PCA
    Parameters
    ----------
    X: array-like, shape (n_dimensions, n_vectors)
        data matrix
    n_subdims: integer
        number of subspace dimension
    higher: bool
        if True, this function returns eigenvectors collesponding
        top-`n_subdims` eigenvalues. default is True.
    return_eigvals: bool
        if True, this function also returns eigenvalues.
    Returns
    -------
    V: array-like, shape (n_dimensions, n_subdims)
        bases matrix
    w: array-like shape (n_subdims)
        eigenvalues
    """

    if X.shape[0] <= X.shape[1]:
        eigvals = _get_eigvals(X.shape[0], n_subdims, higher)
        # get eigenvectors of autocorrelation matrix X @ X.T
        w, V = _eigen_basis(X @ np.conjugate(X.T), eigvals=eigvals)
    else:
        # use linear kernel to get eigenvectors
        A, w = dual_vectors(np.conjugate(X.T) @ X, n_subdims=n_subdims, higher=higher)
        V = X @ A

    if return_eigvals:
        return V, w
    else:
        return V