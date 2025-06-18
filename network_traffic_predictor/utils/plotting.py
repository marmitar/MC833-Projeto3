"""
Utilities for plotting with matplotlib and seaborn.
"""

import itertools
from collections.abc import Iterator
from typing import Any, Literal

import seaborn as sns
from matplotlib import colormaps
from matplotlib.colors import to_hex


def colormap(name: str, *, cycle: bool = True) -> Iterator[str]:
    """
    Get the colors of a colormap, possibly as a cycle.
    """
    cmap = colormaps.get_cmap(name)

    colors = (to_hex(cmap(i)) for i in range(cmap.N))
    if not cycle:
        return colors

    return itertools.cycle(colors)


_EMPTY_RC: dict[str, Any] = {}


def set_theme(
    context: Literal['paper', 'notebook', 'talk', 'poster'] = 'notebook',
    style: Literal['white', 'dark', 'whitegrid', 'darkgrid', 'ticks'] = 'darkgrid',
    palette: Literal['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind'] = 'deep',
    color_codes: bool = True,
    font: str = 'serif',
    font_scale: float = 1,
    rc: dict[str, Any] = _EMPTY_RC,
) -> None:
    """
    Set a modified seaborn theme as default.
    """
    sns.set_theme(
        style=style,
        context=context,
        palette=palette,
        color_codes=color_codes,
        font=font,
        font_scale=font_scale,
        rc={
            'axes.labelpad': 10,
            'savefig.bbox': 'tight',
            'figure.autolayout': True,
            'figure.figsize': (3.2, 3.2),
            'font.family': ['serif'],
            'font.size': 11,
            'axes.titlesize': 'large',
            'axes.labelsize': 'medium',
            'xtick.labelsize': 'small',
            'ytick.labelsize': 'small',
            'axes.formatter.useoffset': False,
            'axes.formatter.limits': (-10, 10),
            'xtick.bottom': True,
            'ytick.left': True,
            'text.usetex': True,
            'pgf.rcfonts': False,
            'text.latex.preamble': r"""
                \usepackage[brazilian]{babel}
                \usepackage[T1]{fontenc}
                \usepackage[utf8]{inputenc}
                \usepackage{siunitx}
            """,
            'pgf.preamble': r"""
                \usepackage[brazilian]{babel}
                \usepackage[T1]{fontenc}
                \usepackage[utf8]{inputenc}
                \usepackage{siunitx}
            """,
            **rc,
        },
    )
