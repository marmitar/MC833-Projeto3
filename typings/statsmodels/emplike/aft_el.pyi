"""
This type stub file was generated by pyright.
"""

from .descriptive import _OptFuncts

"""

Accelerated Failure Time (AFT) Model with empirical likelihood inference.

AFT regression analysis is applicable when the researcher has access
to a randomly right censored dependent variable, a matrix of exogenous
variables and an indicatior variable (delta) that takes a value of 0 if the
observation is censored and 1 otherwise.

AFT References
--------------

Stute, W. (1993). "Consistent Estimation Under Random Censorship when
Covariables are Present." Journal of Multivariate Analysis.
Vol. 45. Iss. 1. 89-103

EL and AFT References
---------------------

Zhou, Kim And Bathke. "Empirical Likelihood Analysis for the Heteroskedastic
Accelerated Failure Time Model." Manuscript:
URL: www.ms.uky.edu/~mai/research/CasewiseEL20080724.pdf

Zhou, M. (2005). Empirical Likelihood Ratio with Arbitrarily Censored/
Truncated Data by EM Algorithm.  Journal of Computational and Graphical
Statistics. 14:3, 643-656.


"""
class OptAFT(_OptFuncts):
    """
    Provides optimization functions used in estimating and conducting
    inference in an AFT model.

    Methods
    ------

    _opt_wtd_nuis_regress:
        Function optimized over nuisance parameters to compute
        the profile likelihood

    _EM_test:
        Uses the modified Em algorithm of Zhou 2005 to maximize the
        likelihood of a parameter vector.
    """
    def __init__(self) -> None:
        ...



class emplikeAFT:
    """

    Class for estimating and conducting inference in an AFT model.

    Parameters
    ----------

    endog: nx1 array
        Response variables that are subject to random censoring

    exog: nxk array
        Matrix of covariates

    censors: nx1 array
        array with entries 0 or 1.  0 indicates a response was
        censored.

    Attributes
    ----------
    nobs : float
        Number of observations
    endog : ndarray
        Endog attay
    exog : ndarray
        Exogenous variable matrix
    censors
        Censors array but sets the max(endog) to uncensored
    nvar : float
        Number of exogenous variables
    uncens_nobs : float
        Number of uncensored observations
    uncens_endog : ndarray
        Uncensored response variables
    uncens_exog : ndarray
        Exogenous variables of the uncensored observations

    Methods
    -------

    params:
        Fits model parameters

    test_beta:
        Tests if beta = b0 for any vector b0.

    Notes
    -----

    The data is immediately sorted in order of increasing endogenous
    variables

    The last observation is assumed to be uncensored which makes
    estimation and inference possible.
    """
    def __init__(self, endog, exog, censors) -> None:
        ...

    def fit(self): # -> AFTResults:
        """

        Fits an AFT model and returns results instance

        Parameters
        ----------
        None


        Returns
        -------
        Results instance.

        Notes
        -----
        To avoid dividing by zero, max(endog) is assumed to be uncensored.
        """
        ...

    def predict(self, params, endog=...): # -> Any:
        ...



class AFTResults(OptAFT):
    def __init__(self, model) -> None:
        ...

    def params(self): # -> Any:
        """

        Fits an AFT model and returns parameters.

        Parameters
        ----------
        None


        Returns
        -------
        Fitted params

        Notes
        -----
        To avoid dividing by zero, max(endog) is assumed to be uncensored.
        """
        ...

    def test_beta(self, b0_vals, param_nums, ftol=..., maxiter=..., print_weights=...): # -> tuple[Any, ndarray[_AnyShape, dtype[Any]] | ndarray[tuple[()], dtype[Any]]] | tuple[float, Literal[0]]:
        """
        Returns the profile log likelihood for regression parameters
        'param_num' at 'b0_vals.'

        Parameters
        ----------
        b0_vals : list
            The value of parameters to be tested
        param_num : list
            Which parameters to be tested
        maxiter : int, optional
            How many iterations to use in the EM algorithm.  Default is 30
        ftol : float, optional
            The function tolerance for the EM optimization.
            Default is 10''**''-5
        print_weights : bool
            If true, returns the weights tate maximize the profile
            log likelihood. Default is False

        Returns
        -------

        test_results : tuple
            The log-likelihood and p-pvalue of the test.

        Notes
        -----

        The function will warn if the EM reaches the maxiter.  However, when
        optimizing over nuisance parameters, it is possible to reach a
        maximum number of inner iterations for a specific value for the
        nuisance parameters while the resultsof the function are still valid.
        This usually occurs when the optimization over the nuisance parameters
        selects parameter values that yield a log-likihood ratio close to
        infinity.

        Examples
        --------

        >>> import statsmodels.api as sm
        >>> import numpy as np

        # Test parameter is .05 in one regressor no intercept model
        >>> data=sm.datasets.heart.load()
        >>> y = np.log10(data.endog)
        >>> x = data.exog
        >>> cens = data.censors
        >>> model = sm.emplike.emplikeAFT(y, x, cens)
        >>> res=model.test_beta([0], [0])
        >>> res
        (1.4657739632606308, 0.22601365256959183)

        #Test slope is 0 in  model with intercept

        >>> data=sm.datasets.heart.load()
        >>> y = np.log10(data.endog)
        >>> x = data.exog
        >>> cens = data.censors
        >>> model = sm.emplike.emplikeAFT(y, sm.add_constant(x), cens)
        >>> res = model.test_beta([0], [1])
        >>> res
        (4.623487775078047, 0.031537049752572731)
        """
        ...

    def ci_beta(self, param_num, beta_high, beta_low, sig=...): # -> tuple[tuple[Any, RootResults] | Any, tuple[Any, RootResults] | Any]:
        """
        Returns the confidence interval for a regression
        parameter in the AFT model.

        Parameters
        ----------
        param_num : int
            Parameter number of interest
        beta_high : float
            Upper bound for the confidence interval
        beta_low : float
            Lower bound for the confidence interval
        sig : float, optional
            Significance level.  Default is .05

        Notes
        -----
        If the function returns f(a) and f(b) must have different signs,
        consider widening the search area by adjusting beta_low and
        beta_high.

        Also note that this process is computational intensive.  There
        are 4 levels of optimization/solving.  From outer to inner:

        1) Solving so that llr-critical value = 0
        2) maximizing over nuisance parameters
        3) Using  EM at each value of nuisamce parameters
        4) Using the _modified_Newton optimizer at each iteration
           of the EM algorithm.

        Also, for very unlikely nuisance parameters, it is possible for
        the EM algorithm to not converge.  This is not an indicator
        that the solver did not find the correct solution.  It just means
        for a specific iteration of the nuisance parameters, the optimizer
        was unable to converge.

        If the user desires to verify the success of the optimization,
        it is recommended to test the limits using test_beta.
        """
        ...
