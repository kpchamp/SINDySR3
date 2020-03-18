import warnings

import numpy as np
from scipy.integrate import odeint
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from pysindy.optimizers import BaseOptimizer
from pysindy.utils import get_prox
from utils import get_reg


class SR3Forcing(BaseOptimizer):
    """
    Sparse relaxed regularized regression with parameterized forcing.

    Parameters
    ----------
    n_forcing_params : int
        The number of parameters in the parameterized forcing. 

    forcing_functions : list of functions
        List of functions that make up the forcing terms. Each function
        should take two parameters: (1) an array of the forcing
        parameters and (2) a (time-dependent) forcing input.

    threshold : float, optional (default 0.1)
        Determines the strength of the regularization. When the
        regularization function R is the l0 norm, the regularization
        is equivalent to performing hard thresholding, and lambda
        is chosen to threshold at the value given by this parameter.
        This is equivalent to choosing lambda = threshold^2 / (2 * nu).

    nu : float, optional (default 1)
        Determines the level of relaxation. Decreasing nu encourages
        w and v to be close, whereas increasing nu allows the
        regularized coefficients v to be farther from w.

    tol : float, optional (default 1e-5)
        Tolerance used for determining convergence of the optimization
        algorithm.

    thresholder : string, optional (default 'l0')
        Regularization function to use. Currently implemented options
        are 'l0' (l0 norm), 'l1' (l1 norm), and 'cad' (clipped
        absolute deviation).

    max_iter : int, optional (default 30)
        Maximum iterations of the optimization algorithm.

    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    normalize : boolean, optional (default False)
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Regularized weight vector(s). This is the v in the objective
        function.

    coef_full_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s) that are not subjected to the regularization.
        This is the w in the objective function.

    forcing_params_ : array, shape (n_forcing_params,)
        Vector containing the values of the forcing parameters found during
        the optimization.
    """

    def __init__(
        self,
        n_forcing_params,
        forcing_functions,
        threshold=0.1,
        nu=1.0,
        tol=1e-5,
        thresholder="l0",
        max_iter=30,
        normalize=False,
        fit_intercept=False,
        copy_X=True,
    ):
        super(SR3Forcing, self).__init__(
            max_iter=max_iter,
            normalize=normalize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
        )

        if threshold < 0:
            raise ValueError("threshold cannot be negative")
        if nu <= 0:
            raise ValueError("nu must be positive")
        if tol <= 0:
            raise ValueError("tol must be positive")

        self.threshold = threshold
        self.nu = nu
        self.tol = tol
        self.thresholder = thresholder
        self.prox = get_prox(thresholder)
        self.reg = get_reg(thresholder)

        self.n_forcing_params = n_forcing_params
        self.forcing_params_ = None
        self.forcing_functions = forcing_functions

    def _update_sparse_coef(self, coef_full):
        """Update the regularized weight vector
        """
        coef_sparse = self.prox(coef_full, self.threshold)
        self.history_.append(coef_sparse.T)
        return coef_sparse

    def _compute_x_forcing(self, x, forcing_input):
        # assume time-dependent forcing for now
        forcing_cols = np.array(
            [f(self.forcing_params_, forcing_input) for f in self.forcing_functions]
        ).T
        return np.concatenate((x, forcing_cols), axis=1)

    def _objective(self, x, y, coef_full, coef_sparse):
        """objective function"""
        R2 = (y - np.dot(x, coef_full)) ** 2
        D2 = (coef_full - coef_sparse) ** 2

        return (
            0.5 * np.sum(R2)
            + self.reg(coef_full, 0.5 * self.threshold ** 2 / self.nu)
            + 0.5 * np.sum(D2) / self.nu
        )

    def _convergence_criterion(self):
        """Calculate the convergence criterion for the optimization
        """
        this_coef = self.history_[-1]
        if len(self.history_) > 1:
            last_coef = self.history_[-2]
        else:
            last_coef = np.zeros_like(this_coef)
        return np.sum((this_coef - last_coef) ** 2)

    def _reduce(self, x, y, forcing_input=None, initial_forcing_params=None):
        """
        Performs joint fitting of the SINDy  
        """
        if initial_forcing_params is None:
            initial_forcing_params = np.zeros(self.n_forcing_params)
        if forcing_input is None:
            raise ArgumentError("no forcing variable provided")

        forcing_params = initial_forcing_params.copy()

        vp = VariableProjection(
            x, y, forcing_input, forcing_params, self.forcing_functions, self.nu
        )
        self.history_ = [vp.coef_sparse.copy().T]

        vp.project_params()

        obj_his = []
        for _ in range(self.max_iter):
            coef_sparse = self._update_sparse_coef(vp.coef_full)
            vp.coef_sparse = coef_sparse
            vp.project_params()

            obj_his.append(self._objective(vp.x_forcing, y, vp.coef_full, coef_sparse))
            if self._convergence_criterion() < self.tol:
                # Could not (further) select important features
                break
        else:
            warnings.warn(
                "SR3._reduce did not converge after {} iterations.".format(
                    self.max_iter
                ),
                ConvergenceWarning,
            )

        self.coef_ = coef_sparse.T
        self.coef_full_ = vp.coef_full.T
        self.forcing_params_ = vp.forcing_params
        self.obj_his = obj_his

    def predict(self, x, forcing_input=None):
        check_is_fitted(self)

        x = check_array(x)
        x_forcing = self._compute_x_forcing(x, forcing_input)
        return np.dot(x_forcing, self.coef_.T) + self.intercept_


class VariableProjection:
    def __init__(self, x, y, forcing_input, forcing_params, forcing_functions, nu):
        self.x = x
        self.y = y
        self.forcing_input = forcing_input
        self.forcing_params = forcing_params
        self.forcing_functions = forcing_functions
        self.x_forcing = self.recompute_x_forcing(x, forcing_input, forcing_params)
        self.nu = nu
        self.coef_full = np.linalg.lstsq(self.x_forcing, y, rcond=None)[0]
        self.coef_sparse = self.coef_full.copy()

    def recompute_x_forcing(self, x, forcing_input, forcing_params):
        forcing_cols = np.array(
            [f(forcing_params, forcing_input) for f in self.forcing_functions]
        ).T
        return np.concatenate((x, forcing_cols), axis=1)

    def update_coef_full(self):
        self.x_forcing = self.recompute_x_forcing(
            self.x, self.forcing_input, self.forcing_params
        )
        A = np.dot(self.x_forcing.T, self.x_forcing) + 1.0 / self.nu * np.eye(
            self.x_forcing.shape[1]
        )
        b = np.dot(self.x_forcing.T, self.y) + 1.0 / self.nu * self.coef_sparse
        self.coef_full = np.linalg.solve(A, b)

    def params_function(self, forcing_params, project_coef=True):
        if project_coef:
            self.forcing_params = forcing_params
            self.update_coef_full()
            x_forcing = self.x_forcing
        else:
            x_forcing = self.recompute_x_forcing(
                self.x, self.forcing_input, forcing_params
            )
        return (
            0.5 * np.sum((self.y - np.dot(x_forcing, self.coef_full)) ** 2)
            + 0.5 * np.sum((self.coef_full - self.coef_sparse) ** 2) / self.nu
        )

    def params_grad(self, forcing_params):
        self.forcing_params = forcing_params
        self.update_coef_full()
        h = 1e-10
        params_g = np.zeros(forcing_params.shape)
        params_c = forcing_params + 0j * np.zeros(forcing_params.shape)
        params_i = np.copy(params_c)
        for i in range(len(forcing_params)):
            # restore alpha_c
            np.copyto(params_i, params_c)
            # pertube the i-th element
            params_i[i] += 1j * h
            # obtain the objective value
            f_i = self.params_function(params_i, project_coef=False)
            # extract the imaginary part as the derivative
            params_g[i] = f_i.imag / h
        #
        return params_g

    def project_params(self):
        res = minimize(
            self.params_function,
            self.forcing_params,
            jac=self.params_grad,
            method="BFGS",
        )
        self.forcing_params = res.x
        self.update_coef_full()
