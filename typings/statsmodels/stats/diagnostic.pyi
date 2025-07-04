"""
This type stub file was generated by pyright.
"""

from statsmodels.compat.pandas import deprecate_kwarg

"""
Various Statistical Tests

Author: josef-pktd
License: BSD-3

Notes
-----
Almost fully verified against R or Gretl, not all options are the same.
In many cases of Lagrange multiplier tests both the LM test and the F test is
returned. In some but not all cases, R has the option to choose the test
statistic. Some alternative test statistic results have not been verified.

TODO
* refactor to store intermediate results

missing:

* pvalues for breaks_hansen
* additional options, compare with R, check where ddof is appropriate
* new tests:
  - breaks_ap, more recent breaks tests
  - specification tests against nonparametric alternatives
"""
__all__ = ["kstest_fit", "lilliefors", "kstest_normal", "kstest_exponential", "normal_ad", "compare_cox", "compare_j", "acorr_breusch_godfrey", "acorr_ljungbox", "acorr_lm", "het_arch", "het_breuschpagan", "het_goldfeldquandt", "het_white", "spec_white", "linear_lm", "linear_rainbow", "linear_harvey_collier", "anderson_statistic"]
NESTED_ERROR = ...
class ResultsStore:
    def __str__(self) -> str:
        ...



def compare_cox(results_x, results_z, store=...): # -> tuple[Any, ndarray[_AnyShape, dtype[Any]] | ndarray[tuple[()], dtype[Any]], ResultsStore] | tuple[Any, ndarray[_AnyShape, dtype[Any]] | ndarray[tuple[()], dtype[Any]]]:
    """
    Compute the Cox test for non-nested models

    Parameters
    ----------
    results_x : Result instance
        result instance of first model
    results_z : Result instance
        result instance of second model
    store : bool, default False
        If true, then the intermediate results are returned.

    Returns
    -------
    tstat : float
        t statistic for the test that including the fitted values of the
        first model in the second model has no effect.
    pvalue : float
        two-sided pvalue for the t statistic
    res_store : ResultsStore, optional
        Intermediate results. Returned if store is True.

    Notes
    -----
    Tests of non-nested hypothesis might not provide unambiguous answers.
    The test should be performed in both directions and it is possible
    that both or neither test rejects. see [1]_ for more information.

    Formulas from [1]_, section 8.3.4 translated to code

    Matches results for Example 8.3 in Greene

    References
    ----------
    .. [1] Greene, W. H. Econometric Analysis. New Jersey. Prentice Hall;
       5th edition. (2002).
    """
    ...

def compare_j(results_x, results_z, store=...): # -> tuple[Any, Any, ResultsStore] | tuple[Any, Any]:
    """
    Compute the J-test for non-nested models

    Parameters
    ----------
    results_x : RegressionResults
        The result instance of first model.
    results_z : RegressionResults
        The result instance of second model.
    store : bool, default False
        If true, then the intermediate results are returned.

    Returns
    -------
    tstat : float
        t statistic for the test that including the fitted values of the
        first model in the second model has no effect.
    pvalue : float
        two-sided pvalue for the t statistic
    res_store : ResultsStore, optional
        Intermediate results. Returned if store is True.

    Notes
    -----
    From description in Greene, section 8.3.3. Matches results for Example
    8.3, Greene.

    Tests of non-nested hypothesis might not provide unambiguous answers.
    The test should be performed in both directions and it is possible
    that both or neither test rejects. see Greene for more information.

    References
    ----------
    .. [1] Greene, W. H. Econometric Analysis. New Jersey. Prentice Hall;
       5th edition. (2002).
    """
    ...

def compare_encompassing(results_x, results_z, cov_type=..., cov_kwargs=...): # -> DataFrame:
    r"""
    Davidson-MacKinnon encompassing test for comparing non-nested models

    Parameters
    ----------
    results_x : Result instance
        result instance of first model
    results_z : Result instance
        result instance of second model
    cov_type : str, default "nonrobust
        Covariance type. The default is "nonrobust` which uses the classic
        OLS covariance estimator. Specify one of "HC0", "HC1", "HC2", "HC3"
        to use White's covariance estimator. All covariance types supported
        by ``OLS.fit`` are accepted.
    cov_kwargs : dict, default None
        Dictionary of covariance options passed to ``OLS.fit``. See OLS.fit
        for more details.

    Returns
    -------
    DataFrame
        A DataFrame with two rows and four columns. The row labeled x
        contains results for the null that the model contained in
        results_x is equivalent to the encompassing model. The results in
        the row labeled z correspond to the test that the model contained
        in results_z are equivalent to the encompassing model. The columns
        are the test statistic, its p-value, and the numerator and
        denominator degrees of freedom. The test statistic has an F
        distribution. The numerator degree of freedom is the number of
        variables in the encompassing model that are not in the x or z model.
        The denominator degree of freedom is the number of observations minus
        the number of variables in the nesting model.

    Notes
    -----
    The null is that the fit produced using x is the same as the fit
    produced using both x and z. When testing whether x is encompassed,
    the model estimated is

    .. math::

        Y = X\beta + Z_1\gamma + \epsilon

    where :math:`Z_1` are the columns of :math:`Z` that are not spanned by
    :math:`X`. The null is :math:`H_0:\gamma=0`. When testing whether z is
    encompassed, the roles of :math:`X` and :math:`Z` are reversed.

    Implementation of  Davidson and MacKinnon (1993)'s encompassing test.
    Performs two Wald tests where models x and z are compared to a model
    that nests the two. The Wald tests are performed by using an OLS
    regression.
    """
    ...

def acorr_ljungbox(x, lags=..., boxpierce=..., model_df=..., period=..., return_df=..., auto_lag=...): # -> DataFrame:
    """
    Ljung-Box test of autocorrelation in residuals.

    Parameters
    ----------
    x : array_like
        The data series. The data is demeaned before the test statistic is
        computed.
    lags : {int, array_like}, default None
        If lags is an integer then this is taken to be the largest lag
        that is included, the test result is reported for all smaller lag
        length. If lags is a list or array, then all lags are included up to
        the largest lag in the list, however only the tests for the lags in
        the list are reported. If lags is None, then the default maxlag is
        min(10, nobs // 5). The default number of lags changes if period
        is set.
    boxpierce : bool, default False
        If true, then additional to the results of the Ljung-Box test also the
        Box-Pierce test results are returned.
    model_df : int, default 0
        Number of degrees of freedom consumed by the model. In an ARMA model,
        this value is usually p+q where p is the AR order and q is the MA
        order. This value is subtracted from the degrees-of-freedom used in
        the test so that the adjusted dof for the statistics are
        lags - model_df. If lags - model_df <= 0, then NaN is returned.
    period : int, default None
        The period of a Seasonal time series.  Used to compute the max lag
        for seasonal data which uses min(2*period, nobs // 5) if set. If None,
        then the default rule is used to set the number of lags. When set, must
        be >= 2.
    auto_lag : bool, default False
        Flag indicating whether to automatically determine the optimal lag
        length based on threshold of maximum correlation value.

    Returns
    -------
    DataFrame
        Frame with columns:

        * lb_stat - The Ljung-Box test statistic.
        * lb_pvalue - The p-value based on chi-square distribution. The
          p-value is computed as 1 - chi2.cdf(lb_stat, dof) where dof is
          lag - model_df. If lag - model_df <= 0, then NaN is returned for
          the pvalue.
        * bp_stat - The Box-Pierce test statistic.
        * bp_pvalue - The p-value based for Box-Pierce test on chi-square
          distribution. The p-value is computed as 1 - chi2.cdf(bp_stat, dof)
          where dof is lag - model_df. If lag - model_df <= 0, then NaN is
          returned for the pvalue.

    See Also
    --------
    statsmodels.regression.linear_model.OLS.fit
        Regression model fitting.
    statsmodels.regression.linear_model.RegressionResults
        Results from linear regression models.
    statsmodels.stats.stattools.q_stat
        Ljung-Box test statistic computed from estimated
        autocorrelations.

    Notes
    -----
    Ljung-Box and Box-Pierce statistic differ in their scaling of the
    autocorrelation function. Ljung-Box test is has better finite-sample
    properties.

    References
    ----------
    .. [*] Green, W. "Econometric Analysis," 5th ed., Pearson, 2003.
    .. [*] J. Carlos Escanciano, Ignacio N. Lobato
          "An automatic Portmanteau test for serial correlation".,
          Volume 151, 2009.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.sunspots.load_pandas().data
    >>> res = sm.tsa.ARMA(data["SUNACTIVITY"], (1,1)).fit(disp=-1)
    >>> sm.stats.acorr_ljungbox(res.resid, lags=[10], return_df=True)
           lb_stat     lb_pvalue
    10  214.106992  1.827374e-40
    """
    ...

@deprecate_kwarg("maxlag", "nlags")
def acorr_lm(resid, nlags=..., store=..., *, period=..., ddof=..., cov_type=..., cov_kwargs=...): # -> tuple[Any | float, ndarray[_AnyShape, dtype[Any]] | ndarray[tuple[()], dtype[Any]] | float, float, float, ResultsStore] | tuple[Any | float, ndarray[_AnyShape, dtype[Any]] | ndarray[tuple[()], dtype[Any]] | float, float, float]:
    """
    Lagrange Multiplier tests for autocorrelation.

    This is a generic Lagrange Multiplier test for autocorrelation. Returns
    Engle's ARCH test if resid is the squared residual array. Breusch-Godfrey
    is a variation on this test with additional exogenous variables.

    Parameters
    ----------
    resid : array_like
        Time series to test.
    nlags : int, default None
        Highest lag to use.
    store : bool, default False
        If true then the intermediate results are also returned.
    period : int, default none
        The period of a Seasonal time series.  Used to compute the max lag
        for seasonal data which uses min(2*period, nobs // 5) if set. If None,
        then the default rule is used to set the number of lags. When set, must
        be >= 2.
    ddof : int, default 0
        The number of degrees of freedom consumed by the model used to
        produce resid. The default value is 0.
    cov_type : str, default "nonrobust"
        Covariance type. The default is "nonrobust` which uses the classic
        OLS covariance estimator. Specify one of "HC0", "HC1", "HC2", "HC3"
        to use White's covariance estimator. All covariance types supported
        by ``OLS.fit`` are accepted.
    cov_kwargs : dict, default None
        Dictionary of covariance options passed to ``OLS.fit``. See OLS.fit for
        more details.

    Returns
    -------
    lm : float
        Lagrange multiplier test statistic.
    lmpval : float
        The p-value for Lagrange multiplier test.
    fval : float
        The f statistic of the F test, alternative version of the same
        test based on F test for the parameter restriction.
    fpval : float
        The pvalue of the F test.
    res_store : ResultsStore, optional
        Intermediate results. Only returned if store=True.

    See Also
    --------
    het_arch
        Conditional heteroskedasticity testing.
    acorr_breusch_godfrey
        Breusch-Godfrey test for serial correlation.
    acorr_ljung_box
        Ljung-Box test for serial correlation.

    Notes
    -----
    The test statistic is computed as (nobs - ddof) * r2 where r2 is the
    R-squared from a regression on the residual on nlags lags of the
    residual.
    """
    ...

@deprecate_kwarg("maxlag", "nlags")
def het_arch(resid, nlags=..., store=..., ddof=...): # -> tuple[Any | float, ndarray[_AnyShape, dtype[Any]] | ndarray[tuple[()], dtype[Any]] | float, float, float, ResultsStore] | tuple[Any | float, ndarray[_AnyShape, dtype[Any]] | ndarray[tuple[()], dtype[Any]] | float, float, float]:
    """
    Engle's Test for Autoregressive Conditional Heteroscedasticity (ARCH).

    Parameters
    ----------
    resid : ndarray
        residuals from an estimation, or time series
    nlags : int, default None
        Highest lag to use.
    store : bool, default False
        If true then the intermediate results are also returned
    ddof : int, default 0
        If the residuals are from a regression, or ARMA estimation, then there
        are recommendations to correct the degrees of freedom by the number
        of parameters that have been estimated, for example ddof=p+q for an
        ARMA(p,q).

    Returns
    -------
    lm : float
        Lagrange multiplier test statistic
    lmpval : float
        p-value for Lagrange multiplier test
    fval : float
        fstatistic for F test, alternative version of the same test based on
        F test for the parameter restriction
    fpval : float
        pvalue for F test
    res_store : ResultsStore, optional
        Intermediate results. Returned if store is True.

    Notes
    -----
    verified against R:FinTS::ArchTest
    """
    ...

@deprecate_kwarg("results", "res")
def acorr_breusch_godfrey(res, nlags=..., store=...): # -> tuple[Any, ndarray[_AnyShape, dtype[Any]] | ndarray[tuple[()], dtype[Any]], float, float, ResultsStore] | tuple[Any, ndarray[_AnyShape, dtype[Any]] | ndarray[tuple[()], dtype[Any]], float, float]:
    """
    Breusch-Godfrey Lagrange Multiplier tests for residual autocorrelation.

    Parameters
    ----------
    res : RegressionResults
        Estimation results for which the residuals are tested for serial
        correlation.
    nlags : int, optional
        Number of lags to include in the auxiliary regression. (nlags is
        highest lag).
    store : bool, default False
        If store is true, then an additional class instance that contains
        intermediate results is returned.

    Returns
    -------
    lm : float
        Lagrange multiplier test statistic.
    lmpval : float
        The p-value for Lagrange multiplier test.
    fval : float
        The value of the f statistic for F test, alternative version of the
        same test based on F test for the parameter restriction.
    fpval : float
        The pvalue for F test.
    res_store : ResultsStore
        A class instance that holds intermediate results. Only returned if
        store=True.

    Notes
    -----
    BG adds lags of residual to exog in the design matrix for the auxiliary
    regression with residuals as endog. See [1]_, section 12.7.1.

    References
    ----------
    .. [1] Greene, W. H. Econometric Analysis. New Jersey. Prentice Hall;
      5th edition. (2002).
    """
    ...

def het_breuschpagan(resid, exog_het, robust=...): # -> tuple[Any, ndarray[_AnyShape, dtype[Any]] | ndarray[tuple[()], dtype[Any]], Any, Any]:
    r"""
    Breusch-Pagan Lagrange Multiplier test for heteroscedasticity

    The tests the hypothesis that the residual variance does not depend on
    the variables in x in the form

    .. :math: \sigma_i = \sigma * f(\alpha_0 + \alpha z_i)

    Homoscedasticity implies that :math:`\alpha=0`.

    Parameters
    ----------
    resid : array_like
        For the Breusch-Pagan test, this should be the residual of a
        regression. If an array is given in exog, then the residuals are
        calculated by the an OLS regression or resid on exog. In this case
        resid should contain the dependent variable. Exog can be the same as x.
    exog_het : array_like
        This contains variables suspected of being related to
        heteroscedasticity in resid.
    robust : bool, default True
        Flag indicating whether to use the Koenker version of the
        test (default) which assumes independent and identically distributed
        error terms, or the original Breusch-Pagan version which assumes
        residuals are normally distributed.

    Returns
    -------
    lm : float
        lagrange multiplier statistic
    lm_pvalue : float
        p-value of lagrange multiplier test
    fvalue : float
        f-statistic of the hypothesis that the error variance does not depend
        on x
    f_pvalue : float
        p-value for the f-statistic

    Notes
    -----
    Assumes x contains constant (for counting dof and calculation of R^2).
    In the general description of LM test, Greene mentions that this test
    exaggerates the significance of results in small or moderately large
    samples. In this case the F-statistic is preferable.

    **Verification**

    Chisquare test statistic is exactly (<1e-13) the same result as bptest
    in R-stats with defaults (studentize=True).

    **Implementation**

    This is calculated using the generic formula for LM test using $R^2$
    (Greene, section 17.6) and not with the explicit formula
    (Greene, section 11.4.3), unless `robust` is set to False.
    The degrees of freedom for the p-value assume x is full rank.

    References
    ----------
    .. [1] Greene, W. H. Econometric Analysis. New Jersey. Prentice Hall;
       5th edition. (2002).
    .. [2]  Breusch, T. S.; Pagan, A. R. (1979). "A Simple Test for
       Heteroskedasticity and Random Coefficient Variation". Econometrica.
       47 (5): 1287–1294.
    .. [3] Koenker, R. (1981). "A note on studentizing a test for
       heteroskedasticity". Journal of Econometrics 17 (1): 107–112.
    """
    ...

def het_white(resid, exog): # -> tuple[Any, ndarray[_AnyShape, dtype[Any]] | ndarray[tuple[()], dtype[Any]], Any, Any]:
    """
    White's Lagrange Multiplier Test for Heteroscedasticity.

    Parameters
    ----------
    resid : array_like
        The residuals. The squared residuals are used as the endogenous
        variable.
    exog : array_like
        The explanatory variables for the variance. Squares and interaction
        terms are automatically included in the auxiliary regression.

    Returns
    -------
    lm : float
        The lagrange multiplier statistic.
    lm_pvalue :float
        The p-value of lagrange multiplier test.
    fvalue : float
        The f-statistic of the hypothesis that the error variance does not
        depend on x. This is an alternative test variant not the original
        LM test.
    f_pvalue : float
        The p-value for the f-statistic.

    Notes
    -----
    Assumes x contains constant (for counting dof).

    question: does f-statistic make sense? constant ?

    References
    ----------
    Greene section 11.4.1 5th edition p. 222. Test statistic reproduces
    Greene 5th, example 11.3.
    """
    ...

def het_goldfeldquandt(y, x, idx=..., split=..., drop=..., alternative=..., store=...): # -> tuple[Any, ndarray[_AnyShape, dtype[Any]] | ndarray[tuple[()], dtype[Any]] | Any, Literal['increasing', 'decreasing', 'two-sided'], ResultsStore] | tuple[Any, ndarray[_AnyShape, dtype[Any]] | ndarray[tuple[()], dtype[Any]] | Any, Literal['increasing', 'decreasing', 'two-sided']]:
    """
    Goldfeld-Quandt homoskedasticity test.

    This test examines whether the residual variance is the same in 2
    subsamples.

    Parameters
    ----------
    y : array_like
        endogenous variable
    x : array_like
        exogenous variable, regressors
    idx : int, default None
        column index of variable according to which observations are
        sorted for the split
    split : {int, float}, default None
        If an integer, this is the index at which sample is split.
        If a float in 0<split<1 then split is interpreted as fraction
        of the observations in the first sample. If None, uses nobs//2.
    drop : {int, float}, default None
        If this is not None, then observation are dropped from the middle
        part of the sorted series. If 0<split<1 then split is interpreted
        as fraction of the number of observations to be dropped.
        Note: Currently, observations are dropped between split and
        split+drop, where split and drop are the indices (given by rounding
        if specified as fraction). The first sample is [0:split], the
        second sample is [split+drop:]
    alternative : {"increasing", "decreasing", "two-sided"}
        The default is increasing. This specifies the alternative for the
        p-value calculation.
    store : bool, default False
        Flag indicating to return the regression results

    Returns
    -------
    fval : float
        value of the F-statistic
    pval : float
        p-value of the hypothesis that the variance in one subsample is
        larger than in the other subsample
    ordering : str
        The ordering used in the alternative.
    res_store : ResultsStore, optional
        Storage for the intermediate and final results that are calculated

    Notes
    -----
    The Null hypothesis is that the variance in the two sub-samples are the
    same. The alternative hypothesis, can be increasing, i.e. the variance
    in the second sample is larger than in the first, or decreasing or
    two-sided.

    Results are identical R, but the drop option is defined differently.
    (sorting by idx not tested yet)
    """
    ...

@deprecate_kwarg("result", "res")
def linear_reset(res, power=..., test_type=..., use_f=..., cov_type=..., cov_kwargs=...): # -> Any:
    r"""
    Ramsey's RESET test for neglected nonlinearity

    Parameters
    ----------
    res : RegressionResults
        A results instance from a linear regression.
    power : {int, List[int]}, default 3
        The maximum power to include in the model, if an integer. Includes
        powers 2, 3, ..., power. If an list of integers, includes all powers
        in the list.
    test_type : str, default "fitted"
        The type of augmentation to use:

        * "fitted" : (default) Augment regressors with powers of fitted values.
        * "exog" : Augment exog with powers of exog. Excludes binary
          regressors.
        * "princomp": Augment exog with powers of first principal component of
          exog.
    use_f : bool, default False
        Flag indicating whether an F-test should be used (True) or a
        chi-square test (False).
    cov_type : str, default "nonrobust
        Covariance type. The default is "nonrobust` which uses the classic
        OLS covariance estimator. Specify one of "HC0", "HC1", "HC2", "HC3"
        to use White's covariance estimator. All covariance types supported
        by ``OLS.fit`` are accepted.
    cov_kwargs : dict, default None
        Dictionary of covariance options passed to ``OLS.fit``. See OLS.fit
        for more details.

    Returns
    -------
    ContrastResults
        Test results for Ramsey's Reset test. See notes for implementation
        details.

    Notes
    -----
    The RESET test uses an augmented regression of the form

    .. math::

       Y = X\beta + Z\gamma + \epsilon

    where :math:`Z` are a set of regressors that are one of:

    * Powers of :math:`X\hat{\beta}` from the original regression.
    * Powers of :math:`X`, excluding the constant and binary regressors.
    * Powers of the first principal component of :math:`X`. If the
      model includes a constant, this column is dropped before computing
      the principal component. In either case, the principal component
      is extracted from the correlation matrix of remaining columns.

    The test is a Wald test of the null :math:`H_0:\gamma=0`. If use_f
    is True, then the quadratic-form test statistic is divided by the
    number of restrictions and the F distribution is used to compute
    the critical value.
    """
    ...

def linear_harvey_collier(res, order_by=..., skip=...): # -> TtestResult:
    """
    Harvey Collier test for linearity

    The Null hypothesis is that the regression is correctly modeled as linear.

    Parameters
    ----------
    res : RegressionResults
        A results instance from a linear regression.
    order_by : array_like, default None
        Integer array specifying the order of the residuals. If not provided,
        the order of the residuals is not changed. If provided, must have
        the same number of observations as the endogenous variable.
    skip : int, default None
        The number of observations to use for initial OLS, if None then skip is
        set equal to the number of regressors (columns in exog).

    Returns
    -------
    tvalue : float
        The test statistic, based on ttest_1sample.
    pvalue : float
        The pvalue of the test.

    See Also
    --------
    statsmodels.stats.diadnostic.recursive_olsresiduals
        Recursive OLS residual calculation used in the test.

    Notes
    -----
    This test is a t-test that the mean of the recursive ols residuals is zero.
    Calculating the recursive residuals might take some time for large samples.
    """
    ...

def linear_rainbow(res, frac=..., order_by=..., use_distance=..., center=...): # -> tuple[Any, ndarray[_AnyShape, dtype[Any]] | ndarray[tuple[()], dtype[Any]]]:
    """
    Rainbow test for linearity

    The null hypothesis is the fit of the model using full sample is the same
    as using a central subset. The alternative is that the fits are difference.
    The rainbow test has power against many different forms of nonlinearity.

    Parameters
    ----------
    res : RegressionResults
        A results instance from a linear regression.
    frac : float, default 0.5
        The fraction of the data to include in the center model.
    order_by : {ndarray, str, List[str]}, default None
        If an ndarray, the values in the array are used to sort the
        observations. If a string or a list of strings, these are interpreted
        as column name(s) which are then used to lexicographically sort the
        data.
    use_distance : bool, default False
        Flag indicating whether data should be ordered by the Mahalanobis
        distance to the center.
    center : {float, int}, default None
        If a float, the value must be in [0, 1] and the center is center *
        nobs of the ordered data.  If an integer, must be in [0, nobs) and
        is interpreted as the observation of the ordered data to use.

    Returns
    -------
    fstat : float
        The test statistic based on the F test.
    pvalue : float
        The pvalue of the test.

    Notes
    -----
    This test assumes residuals are homoskedastic and may reject a correct
    linear specification if the residuals are heteroskedastic.
    """
    ...

def linear_lm(resid, exog, func=...): # -> tuple[Any, ndarray[_AnyShape, dtype[Any]] | ndarray[tuple[()], dtype[Any]], Any]:
    """
    Lagrange multiplier test for linearity against functional alternative

    # TODO: Remove the restriction
    limitations: Assumes currently that the first column is integer.
    Currently it does not check whether the transformed variables contain NaNs,
    for example log of negative number.

    Parameters
    ----------
    resid : ndarray
        residuals of a regression
    exog : ndarray
        exogenous variables for which linearity is tested
    func : callable, default None
        If func is None, then squares are used. func needs to take an array
        of exog and return an array of transformed variables.

    Returns
    -------
    lm : float
       Lagrange multiplier test statistic
    lm_pval : float
       p-value of Lagrange multiplier tes
    ftest : ContrastResult instance
       the results from the F test variant of this test

    Notes
    -----
    Written to match Gretl's linearity test. The test runs an auxiliary
    regression of the residuals on the combined original and transformed
    regressors. The Null hypothesis is that the linear specification is
    correct.
    """
    ...

def spec_white(resid, exog): # -> tuple[Any, ndarray[_AnyShape, dtype[Any]] | ndarray[tuple[()], dtype[Any]], Any]:
    """
    White's Two-Moment Specification Test

    Parameters
    ----------
    resid : array_like
        OLS residuals.
    exog : array_like
        OLS design matrix.

    Returns
    -------
    stat : float
        The test statistic.
    pval : float
        A chi-square p-value for test statistic.
    dof : int
        The degrees of freedom.

    See Also
    --------
    het_white
        White's test for heteroskedasticity.

    Notes
    -----
    Implements the two-moment specification test described by White's
    Theorem 2 (1980, p. 823) which compares the standard OLS covariance
    estimator with White's heteroscedasticity-consistent estimator. The
    test statistic is shown to be chi-square distributed.

    Null hypothesis is homoscedastic and correctly specified.

    Assumes the OLS design matrix contains an intercept term and at least
    one variable. The intercept is removed to calculate the test statistic.

    Interaction terms (squares and crosses of OLS regressors) are added to
    the design matrix to calculate the test statistic.

    Degrees-of-freedom (full rank) = nvar + nvar * (nvar + 1) / 2

    Linearly dependent columns are removed to avoid singular matrix error.

    References
    ----------
    .. [*] White, H. (1980). A heteroskedasticity-consistent covariance matrix
       estimator and a direct test for heteroscedasticity. Econometrica, 48:
       817-838.
    """
    ...

@deprecate_kwarg("olsresults", "res")
def recursive_olsresiduals(res, skip=..., lamda=..., alpha=..., order_by=...): # -> tuple[NDArray[float64], NDArray[float64], NDArray[float64], Any, NDArray[float64], Any, Any]:
    """
    Calculate recursive ols with residuals and Cusum test statistic

    Parameters
    ----------
    res : RegressionResults
        Results from estimation of a regression model.
    skip : int, default None
        The number of observations to use for initial OLS, if None then skip is
        set equal to the number of regressors (columns in exog).
    lamda : float, default 0.0
        The weight for Ridge correction to initial (X'X)^{-1}.
    alpha : {0.90, 0.95, 0.99}, default 0.95
        Confidence level of test, currently only two values supported,
        used for confidence interval in cusum graph.
    order_by : array_like, default None
        Integer array specifying the order of the residuals. If not provided,
        the order of the residuals is not changed. If provided, must have
        the same number of observations as the endogenous variable.

    Returns
    -------
    rresid : ndarray
        The recursive ols residuals.
    rparams : ndarray
        The recursive ols parameter estimates.
    rypred : ndarray
        The recursive prediction of endogenous variable.
    rresid_standardized : ndarray
        The recursive residuals standardized so that N(0,sigma2) distributed,
        where sigma2 is the error variance.
    rresid_scaled : ndarray
        The recursive residuals normalize so that N(0,1) distributed.
    rcusum : ndarray
        The cumulative residuals for cusum test.
    rcusumci : ndarray
        The confidence interval for cusum test using a size of alpha.

    Notes
    -----
    It produces same recursive residuals as other version. This version updates
    the inverse of the X'X matrix and does not require matrix inversion during
    updating. looks efficient but no timing

    Confidence interval in Greene and Brown, Durbin and Evans is the same as
    in Ploberger after a little bit of algebra.

    References
    ----------
    jplv to check formulas, follows Harvey
    BigJudge 5.5.2b for formula for inverse(X'X) updating
    Greene section 7.5.2

    Brown, R. L., J. Durbin, and J. M. Evans. “Techniques for Testing the
    Constancy of Regression Relationships over Time.”
    Journal of the Royal Statistical Society. Series B (Methodological) 37,
    no. 2 (1975): 149-192.
    """
    ...

def breaks_hansen(olsresults): # -> tuple[Any, NDArray[Any]]:
    """
    Test for model stability, breaks in parameters for ols, Hansen 1992

    Parameters
    ----------
    olsresults : RegressionResults
        Results from estimation of a regression model.

    Returns
    -------
    teststat : float
        Hansen's test statistic.
    crit : ndarray
        The critical values at alpha=0.95 for different nvars.

    Notes
    -----
    looks good in example, maybe not very powerful for small changes in
    parameters

    According to Greene, distribution of test statistics depends on nvar but
    not on nobs.

    Test statistic is verified against R:strucchange

    References
    ----------
    Greene section 7.5.1, notation follows Greene
    """
    ...

def breaks_cusumolsresid(resid, ddof=...): # -> tuple[Any, ndarray[_AnyShape, dtype[Any]] | ndarray[tuple[()], dtype[Any]], list[tuple[int, float]]]:
    """
    Cusum test for parameter stability based on ols residuals.

    Parameters
    ----------
    resid : ndarray
        An array of residuals from an OLS estimation.
    ddof : int
        The number of parameters in the OLS estimation, used as degrees
        of freedom correction for error variance.

    Returns
    -------
    sup_b : float
        The test statistic, maximum of absolute value of scaled cumulative OLS
        residuals.
    pval : float
        Probability of observing the data under the null hypothesis of no
        structural change, based on asymptotic distribution which is a Brownian
        Bridge
    crit: list
        The tabulated critical values, for alpha = 1%, 5% and 10%.

    Notes
    -----
    Tested against R:structchange.

    Not clear: Assumption 2 in Ploberger, Kramer assumes that exog x have
    asymptotically zero mean, x.mean(0) = [1, 0, 0, ..., 0]
    Is this really necessary? I do not see how it can affect the test statistic
    under the null. It does make a difference under the alternative.
    Also, the asymptotic distribution of test statistic depends on this.

    From examples it looks like there is little power for standard cusum if
    exog (other than constant) have mean zero.

    References
    ----------
    Ploberger, Werner, and Walter Kramer. “The Cusum Test with OLS Residuals.”
    Econometrica 60, no. 2 (March 1992): 271-285.
    """
    ...
