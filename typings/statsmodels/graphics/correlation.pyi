"""
This type stub file was generated by pyright.
"""

'''correlation plots

Author: Josef Perktold
License: BSD-3

example for usage with different options in
statsmodels/sandbox/examples/thirdparty/ex_ratereturn.py

'''
def plot_corr(dcorr, xnames=..., ynames=..., title=..., normcolor=..., ax=..., cmap=...): # -> Figure:
    """Plot correlation of many variables in a tight color grid.

    Parameters
    ----------
    dcorr : ndarray
        Correlation matrix, square 2-D array.
    xnames : list[str], optional
        Labels for the horizontal axis.  If not given (None), then the
        matplotlib defaults (integers) are used.  If it is an empty list, [],
        then no ticks and labels are added.
    ynames : list[str], optional
        Labels for the vertical axis.  Works the same way as `xnames`.
        If not given, the same names as for `xnames` are re-used.
    title : str, optional
        The figure title. If None, the default ('Correlation Matrix') is used.
        If ``title=''``, then no title is added.
    normcolor : bool or tuple of scalars, optional
        If False (default), then the color coding range corresponds to the
        range of `dcorr`.  If True, then the color range is normalized to
        (-1, 1).  If this is a tuple of two numbers, then they define the range
        for the color bar.
    ax : AxesSubplot, optional
        If `ax` is None, then a figure is created. If an axis instance is
        given, then only the main plot but not the colorbar is created.
    cmap : str or Matplotlib Colormap instance, optional
        The colormap for the plot.  Can be any valid Matplotlib Colormap
        instance or name.

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.graphics.api as smg

    >>> hie_data = sm.datasets.randhie.load_pandas()
    >>> corr_matrix = np.corrcoef(hie_data.data.T)
    >>> smg.plot_corr(corr_matrix, xnames=hie_data.names)
    >>> plt.show()

    .. plot:: plots/graphics_correlation_plot_corr.py
    """
    ...

def plot_corr_grid(dcorrs, titles=..., ncols=..., normcolor=..., xnames=..., ynames=..., fig=..., cmap=...): # -> Figure:
    """
    Create a grid of correlation plots.

    The individual correlation plots are assumed to all have the same
    variables, axis labels can be specified only once.

    Parameters
    ----------
    dcorrs : list or iterable of ndarrays
        List of correlation matrices.
    titles : list[str], optional
        List of titles for the subplots.  By default no title are shown.
    ncols : int, optional
        Number of columns in the subplot grid.  If not given, the number of
        columns is determined automatically.
    normcolor : bool or tuple, optional
        If False (default), then the color coding range corresponds to the
        range of `dcorr`.  If True, then the color range is normalized to
        (-1, 1).  If this is a tuple of two numbers, then they define the range
        for the color bar.
    xnames : list[str], optional
        Labels for the horizontal axis.  If not given (None), then the
        matplotlib defaults (integers) are used.  If it is an empty list, [],
        then no ticks and labels are added.
    ynames : list[str], optional
        Labels for the vertical axis.  Works the same way as `xnames`.
        If not given, the same names as for `xnames` are re-used.
    fig : Figure, optional
        If given, this figure is simply returned.  Otherwise a new figure is
        created.
    cmap : str or Matplotlib Colormap instance, optional
        The colormap for the plot.  Can be any valid Matplotlib Colormap
        instance or name.

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.api as sm

    In this example we just reuse the same correlation matrix several times.
    Of course in reality one would show a different correlation (measuring a
    another type of correlation, for example Pearson (linear) and Spearman,
    Kendall (nonlinear) correlations) for the same variables.

    >>> hie_data = sm.datasets.randhie.load_pandas()
    >>> corr_matrix = np.corrcoef(hie_data.data.T)
    >>> sm.graphics.plot_corr_grid([corr_matrix] * 8, xnames=hie_data.names)
    >>> plt.show()

    .. plot:: plots/graphics_correlation_plot_corr_grid.py
    """
    ...
