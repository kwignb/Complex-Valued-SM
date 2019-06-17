import numpy as np

from base_method import SMBase, CVMSMInterface
from utils import subspace_bases


class ComplexValuedMSM(CVMSMInterface, SMBase):
    """
    Complex Valued Mutual Subspace Method
    """

    def _get_gramians(self, X):
        """
        Parameters
        ----------
        X: array, (n_dims, n_samples)
        Returns
        -------
        G: array, (n_class, n_subdims, n_subdims)
            gramian matricies of references of each class
        """

        # bases, (n_dims, n_subdims)
        bases = subspace_bases(X, self.test_n_subdims)

        # grammians, (n_classes, n_subdims, n_subdims or greater)
        dic = self.dic[:, :, :self.n_subdims]
        gramians = np.dot(np.conjugate(dic.transpose(0, 2, 1)), bases)

        return gramians