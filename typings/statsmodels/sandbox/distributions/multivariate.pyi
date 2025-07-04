"""
This type stub file was generated by pyright.
"""

import numpy as np

'''Multivariate Distribution

Probability of a multivariate t distribution

Now also mvstnormcdf has tests against R mvtnorm

Still need non-central t, extra options, and convenience function for
location, scale version.

Author: Josef Perktold
License: BSD (3-clause)

Reference:
Genz and Bretz for formula

'''
def chi2_pdf(self, x, df):
    '''pdf of chi-square distribution'''
    ...

def chi_pdf(x, df): # -> Any:
    ...

def chi_logpdf(x, df):
    ...

def funbgh(s, a, b, R, df): # -> Any:
    ...

def funbgh2(s, a, b, R, df): # -> Any:
    ...

def bghfactor(df): # -> Any:
    ...

def mvstdtprob(a, b, R, df, ieps=..., quadkwds=..., mvstkwds=...):
    """
    Probability of rectangular area of standard t distribution

    assumes mean is zero and R is correlation matrix

    Notes
    -----
    This function does not calculate the estimate of the combined error
    between the underlying multivariate normal probability calculations
    and the integration.
    """
    ...

def multivariate_t_rvs(m, S, df=..., n=...): # -> NDArray[float64]:
    '''generate random variables of multivariate t distribution

    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))

    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable


    '''
    ...

if __name__ == '__main__':
    corr = np.asarray([[1, 0, 0.5], [0, 1, 0], [0.5, 0, 1]])
    corr_indep = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    corr_equal = np.asarray([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
    R = corr_equal
    a = np.array([-np.inf, -np.inf, -100])
    a = np.array([-0.96, -0.96, -0.96])
    b = np.array([0, 0, 0])
    b = np.array([0.96, 0.96, 0.96])
    df = ...
    sqrt_df = ...
    s = ...
    df = ...
    S = np.array([[1, 0.5], [0.5, 1]])
    nobs = ...
    rvst = multivariate_t_rvs([10, 20], S, 2, nobs)
