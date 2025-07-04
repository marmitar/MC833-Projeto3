% aga-homework.cls
%
% A LaTeX2e document class for preparing homework solutions.

% --- METADATA -----------------------------------------------------------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{homework}

% --- OPTION PROCESSING -------------------------------------------------------
\RequirePackage{kvoptions}
\SetupKeyvalOptions{
    family = homework,
    prefix = homework@
}

% pass unrecognized options to `article` class
\DeclareDefaultOption{%
    \PassOptionsToClass{\CurrentOption}{article}%
}

% process all options
\ProcessKeyvalOptions{homework}

% load parent class with some default options
\LoadClass[a4paper, 11pt]{article}

\RequirePackage[brazilian]{babel}
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
% --- PACKAGES AND SETUP -------------------------------------------------------
\RequirePackage[
    a4paper,
    headheight = 20pt,
    margin = 1cm,
    top = \dimexpr 2.5cm - 10pt \relax,
    bottom = \dimexpr 2cm - 5pt \relax,
    footskip = 0pt,
]{geometry} % set up nicer margins
\addtolength{\footskip}{20pt}
\RequirePackage{fancyhdr} % for headers and footers
\RequirePackage{graphicx} % for including figures
\RequirePackage{mathpazo} % use Palation font
\RequirePackage{amsmath} % use AMS math package
\RequirePackage{amssymb} % use AMS symbols
\RequirePackage{amsthm} % for writing proofs
\RequirePackage{array} % for setting up arguments to columns
\RequirePackage{booktabs} % for professional tables
\RequirePackage{xcolor} % for colored text (comments in algorithms)
\RequirePackage{trimspaces, xstring} % for multiple author parsing
\RequirePackage[bottom]{footmisc}

% look for figures in the figures directory
% \graphicspath{{figures/}}

% set up headers and footers; same for all pages
\setlength{\headheight}{14pt}
\fancyhf{}
\fancyhead[L]{\sffamily Universidade Estadual de Campinas}
\fancyhead[R]{\sffamily MC833 Programação em Redes de Computadores, 2025 S1}
\fancyfoot[L]{\sffamily IC Instituto de Computação}
\fancyfoot[R]{\sffamily\bfseries\thepage}
\renewcommand{\headrulewidth}{0.5pt}
\renewcommand{\footrulewidth}{0.5pt}
\pagestyle{fancy}

\fancypagestyle{footnotes}{%
    \setlength{\footskip}{20pt}
}
\setlength{\skip\footins}{24pt plus 3pt minus 3pt}
\fancypagestyle{main}{%
    \setlength{\footskip}{44pt plus 3pt minus 3pt}
}
\renewcommand{\footnotelayout}{%
    \thispagestyle{footnotes}
}
