import warnings
from numpy import isscalar
from numpy import newaxis
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from pysindy import SINDy
from pysindy.differentiation import FiniteDifference
from pysindy.optimizers import SINDyOptimizer
from pysindy.utils.base import drop_nan_rows
from pysindy.utils.base import validate_input
from sr3_forcing import SR3Forcing


class SINDyForcing(SINDy):
    """
    Model object for SINDy with parameterized forcing.

    Parameters
    ----------
    n_forcing_params : int
        The number of parameters in the parameterized forcing. 

    forcing_functions : list of functions
        List of functions that make up the forcing terms. Each function
        should take two parameters: (1) an array of the forcing
        parameters and (2) a (time-dependent) forcing input.

    feature_library : feature library object, optional
        Default is polynomial features of degree 2.

    differentiation_method : differentiation object, optional
        Method for differentiating the data. This must be an object that
        extends the sindy.differentiation_methods.BaseDifferentiation class.
        Default is centered difference.

    feature_names : list of string, length n_input_features, optional
        Names for the input features. If None, will use ['x0','x1',...].

    discrete_time : boolean, optional (default False)
        If True, dynamical system is treated as a map. Rather than predicting
        derivatives, the right hand side functions step the system forward by
        one time step. If False, dynamical system is assumed to be a flow
        (right hand side functions predict continuous time derivatives).

    n_jobs : int, optional (default 1)
        The number of parallel jobs to use when fitting, predicting with, and
        scoring the model.

    Attributes
    ----------
    model : sklearn.pipeline.Pipeline object
        The fitted SINDy model.
    """
    def __init__(
        self,
        n_forcing_params,
        forcing_functions,
        feature_library=PolynomialFeatures(),
        differentiation_method=FiniteDifference(),
        feature_names=None,
        discrete_time=False,
        n_jobs=1,
        **optimizer_kws,
    ):
        optimizer = SR3Forcing(n_forcing_params, forcing_functions, **optimizer_kws)
        super(SINDyForcing, self).__init__(
            optimizer=optimizer,
            feature_library=feature_library,
            differentiation_method=differentiation_method,
            feature_names=feature_names,
            discrete_time=discrete_time,
            n_jobs=n_jobs,
        )

    def fit(
        self,
        x,
        t=1,
        x_dot=None,
        forcing_input=None,
        initial_forcing_params=None,
        multiple_trajectories=False,
        unbias=True,
        quiet=False,
    ):
        if multiple_trajectories:
            x, x_dot = self.process_multiple_trajectories(x, t, x_dot)
        else:
            x = validate_input(x, t)

            if self.discrete_time:
                if x_dot is None:
                    x_dot = x[1:]
                    x = x[:-1]
                else:
                    x_dot = validate_input(x)
            else:
                if x_dot is None:
                    x_dot = self.differentiation_method(x, t)
                else:
                    x_dot = validate_input(x_dot, t)

        # Drop rows where derivative isn't known
        x, x_dot = drop_nan_rows(x, x_dot)

        steps = [("features", self.feature_library), ("model", self.optimizer)]
        self.model = Pipeline(steps)

        action = "ignore" if quiet else "default"
        with warnings.catch_warnings():
            warnings.filterwarnings(action, category=ConvergenceWarning)
            warnings.filterwarnings(action, category=LinAlgWarning)
            warnings.filterwarnings(action, category=UserWarning)

            self.model.fit(
                x,
                x_dot,
                model__forcing_input=forcing_input,
                model__initial_forcing_params=initial_forcing_params,
            )

        self.n_input_features_ = self.model.steps[0][1].n_input_features_
        self.n_output_features_ = self.model.steps[0][1].n_output_features_

        if self.feature_names is None:
            feature_names = []
            for i in range(self.n_input_features_):
                feature_names.append("x" + str(i))
            self.feature_names = feature_names

        return self

    def predict(self, x, forcing_input, multiple_trajectories=False):
        """
        Predict the time derivatives using the SINDy model.

        Parameters
        ----------
        x: array-like or list of array-like, shape (n_samples, n_input_features)
            Samples.

        multiple_trajectories: boolean, optional (default False)
            If True, x contains multiple trajectories and must be a list of
            data from each trajectory. If False, x is a single trajectory.

        Returns
        -------
        x_dot: array-like or list of array-like, shape (n_samples, n_input_features)
            Predicted time derivatives
        """
        if hasattr(self, "model"):
            if multiple_trajectories:
                x = [validate_input(xi) for xi in x]
                return [self.model.predict(xi) for xi in x]
            else:
                x = validate_input(x)
                if hasattr(self, "model"):
                    return self.model.predict(x, forcing_input=forcing_input)
        else:
            raise NotFittedError("SINDy model must be fit before predict can be called")

    def simulate(
        self,
        x0,
        t,
        forcing_input,
        integrator=odeint,
        stop_condition=None,
        **integrator_kws,
    ):
        if self.discrete_time:
            if not isinstance(t, int):
                raise ValueError(
                    "For discrete time model, t must be an integer (indicating"
                    "the number of steps to predict)"
                )

            x = zeros((t, self.n_input_features_))
            x[0] = x0
            for i in range(1, t):
                x[i] = self.predict(x[i - 1 : i], forcing_input[i - 1 : i])
                if stop_condition is not None and stop_condition(x[i]):
                    return x[: i + 1]
            return x
        else:
            if isscalar(t):
                raise ValueError(
                    "For continuous time model, t must be an array of time"
                    " points at which to simulate"
                )
            if t.shape[0] != forcing_input.shape[0]:
                raise ArgumentError("Time and forcing input must be same length")

            if forcing_input.ndim == 1:
                forcing_func = interp1d(
                    t,
                    forcing_input[:, newaxis],
                    axis=0,
                    bounds_error=False,
                    fill_value="extrapolate",
                )
            else:
                forcing_func = interp1d(
                    t, forcing_input, axis=0, bounds_error=False, fill_value="extrapolate"
                )

            def rhs(x, t):
                return self.predict(x[newaxis, :], forcing_func(t))[0]

            return integrator(rhs, x0, t, **integrator_kws)
