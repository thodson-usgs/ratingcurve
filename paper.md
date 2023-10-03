---
title: 'ratingcurve: A Simple Parameterization for Segmented Rating Curves'
tags:
  - Python
  - Rating Curve
  - Stage-Discharge Relation
  - Bayesian Inference
  - Uncertainty
  - Manning's Formula
authors:
  - name: Timothy O. Hodson
    orcid: 0000-0003-0962-5130
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
  - name: Keith Doore
    orcid: 0000-0001-5035-4016
    equal-contrib: true
    affiliation: 1
  - name: Terry A. Kenney
    affiliation: 2
  - name: Thomas M. Over
    orcid: 0000-0001-8280-4368
    affiliation: 1
  - name: Muluken B. Yeheyis
    affiliation: 3
affiliations:
 - name: U.S. Geological Survey Central Midwest Water Science Center, Urbana, Illinois, USA
   index: 1
 - name: U.S. Geological Survey Water Resources Mission Area, West Valley City, Utah, USA
   index: 2
 - name: Environment and Climate Change Canada, Gatineau, Quebec, Canada
   index: 3
date: 3 October 2023
bibliography: paper.bib
---

# Summary

Streamflow is one of the most important variables in hydrology, but it is difficult to measure continuously.
As a result, nearly all streamflow time series are estimated from rating curves
that define a mathematical relationship between streamflow and some easy-to-measure proxy like water-surface elevation (stage).
Despite the existence of automated methods,
most ratings are still fit manually, which can be time consuming and subjective.
To improve that process, the U.S. Geological Survey (USGS), among others, is evaluating algorithms to automate that fitting.
<!-- Although several such methods already exist,
each parameterizes the rating curve slightly differently.
Because of the nonconvex nature of the problem,
those differences can greatly affect performance. -->
In this work, we develop a parameterization of the classic segment segmented power law
that works reliably with minimial data.
To improve fitting performance, minimize data requirements,
and account for uncertainty in the modeling process,
we implemented the model using Bayesian algorithms within a modern probabilistic machine-learning framework.
The implementation is simple, open source, and easily modified
so that others can contribute to improving the quality of USGS streamflow data.

# Statement of need




\section{Introduction}

Streamflow time series are widely used in hydrologic research, water resource management, engineering design, and flood forecasting,
but they are difficult to measure directly.
In nearly all time-series applications, streamflow is estimated from rating curves or ``ratings'' that describe the relation between streamflow and an easy-to-measure proxy like stage.
The shape of the rating is specific to each streamgage and is governed by channel conditions at or downstream from the gage, referred to as controls.
Section controls, like natural riffles or artificial weirs, occur downstream from the gage, 
whereas channel controls, like the geometry of the banks, represent conditions along the stream reach
(the upstream and downstream vicinity of the gage).
Regardless of the type, the behavior of each control is often well-approximated with standard hydraulic equations that take the general form of a power law with an offset parameter
\begin{equation}
    q = C(h - h_0)^{b}
\end{equation}
where $q$ is the discharge (streamflow);
$h$ is the height of the water above some datum (stage);
$h_0$ is the stage of zero flow (the offset parameter);
$(h-h_0)$ is the hydraulic head;
$b$ is the slope of the rating curve when plotted in log-log space; and
$C$ is a scale factor equal to the discharge when the head is equal to one \citep{ISO18320_2020}.
When multiple controls are present, the rating curve is divided into segments
with one power law corresponding to each control resulting in a multi-segment or compound rating.

Although automated methods exist, most ratings are still fit manually using a graphical method of plotting stage and discharge in log-log space.
With the appropriate location parameter, each control can be fit to a straight-line segment in log space \citep{Kennedy_1984, ISO18320_2020}.
Variants of this method have been used for decades,
first with pencil and log paper
and now with computer-aided software.
However, the fitting processes is still done manually by adjusting parameters until an acceptable fit is achieved.

Single-segment ratings are relatively easy to fit by automated methods \citep{Venetis_1970},
but compound ratings are more challenging because their solution is nonconvex or multimodal \citep{Reitan_2006}.
As a result, optimization algorithms can become stuck in local optima and fail to converge to the global optimum.
General function approximators, such as natural splines \citep{Fenton_2018} or neural networks,
are sometimes able to avoid these optimization issues, making them easier to fit.
However, their generality comes at the cost of requiring more data to constrain their greater flexibility and prevent overfitting.
In contrast, power-law rating models are physically based on the hydraulic equations governing
uniform open-channel flow, 
like the Manning equation \citep{Manning_1891}.
Due to that physical basis, power laws are potentially more robust than other generic curve-fitting functions:
requiring less data to achieve the same fit and being less prone to overfitting.

Several algorithms for fitting segmented power laws already exist.
Some are more physical like power laws, meaning their structure corresponds to the governing hydraulic equations \citep{Reitan_2008, Le_Coz_2014};
some are more data-driven with more flexible structures like splines \citep{Fenton_2018} or local regression \citep{Coxon_2015};
and some are a hybrid of the two \citep{Hrafnkelsson_2021}.
Each approach has different tradeoffs.
More physical approaches require less data but may be nonconvex,
which makes them challenging to fit,
whereas data-driven approaches are easier to fit but require more data.

While increasing a dataset size may not alway be possible,
minimizing fitting issues due to nonconvex settings can be achieved by an appropriate fitting algorithm. 
For example, the physics-based parameterizations tend to use Bayesian sampling algorithms \citep[as opposed to optimization;][]{Ma_2019},
which incorporate priors to help mitigate nonconvex problems that occur.
Examples of priors include constraining the exponent $b$ to be around 5/3,
constraining the number of rating segments,
or constraining the transitions between segments around a particular stage.
Being Bayesian, these algorithms inherently estimate uncertainty in the fitted parameters and discharge,
which is important for many applications.
However, since these parameterizations differ in their exact formulation,
and because of their nonconvex nature,
slight differences can greatly affect their performance.

In this paper, we develop a parameterization approximating the classic power-law form used in the manual fitting method.
Our algorithm distinguishes itself in two ways:
(1) its simple implementation, which uses a community-developed open-source probabilistic programming library,
and (2) its robust parameterization.
These two aspects are in a way interrelated.
With a community-developed library,
the underlying numerical code is maintained by a broader community of developers,
so instead of developing that code,
we could focus on testing different parameterizations for the rating curve.

\section{Implementation}
We implemented our power-law model, along with some additional plotting and evaluation tools,
test datasets, and tutorials, as a Python package called \texttt{ratingcurve}.
The \emph{Data Availability Statement} provides links to the source code repository.

\subsection{Parameterization}
Our rating curve algorithm is similar to the segmented power law used in the manual method \citep{Kennedy_1984, ISO18320_2020},
as well as in automated approaches \citep{Reitan_2008, Le_Coz_2014},
but differs in its parameterization.
Conceptually, the \citep{Reitan_2008} parameterization slices the channel cross section horizontally to form each segment:
segments stack one on top of the other.
Once the stage rises beyond the range of a particular control, that control is "drowned out" and flow through that segment ceases to increase with stage.
Our parameterization slices the channel cross-section vertically, so controls never drown out.
The \citep{Le_Coz_2014} parameterization can slice in either direction but differs in that the segments are summed after transforming them
back to their original scale; whereas, \citep{Reitan_2008} sum the segments in log.

After testing several parameterizations, one seemed especially reliable and simple:
slicing the channel cross section vertically into control segments and summing them in log,
which is somewhat like a ReLU (rectified linear unit) neural network with hydraulic controls as neurons

\begin{align}
    \label{eq:parameterization}
    X &= \ln(\max(h - h_s, 0) + h_o) \\
    \ln(q) &= a + b^T X + \epsilon + \epsilon_o
\end{align}

where
$h_s$ are the unknown segment breakpoints;
$h_o$ is a vector of offsets, the first is 0 and the rest are 1;
$\max$ is the element-wise maximum, which returns a vector of size $h_s$;
$a$ is a bias parameter equal to $\log(C)$, the scale factor;
$b$ are the slopes of each log-transformed segment;
$\epsilon$ is the residual error;
and $\epsilon_o$ is the uncertainty in the discharge observations (optional).
The offset vector $h_o$ ensures that $X \ge 0$, so additional segments
never subtract discharge.

The default priors and settings are documented in the package;
in general, they do not need to be modified.
Besides selecting the number of segments, the user can specify a prior distribution on the breakpoints.
The default assumes the breakpoints are monotonically ordered and uniformly distributed across the range of the data,
$h_{s1} < \min(h) < h_{s2} < \cdots < h_{sn} < \max(h)$.
Alternatively, the user can specify approximate locations for each breakpoint and their uncertainty as normal distributions.

Uncertainty in the discharge observations is typically reported as relative standard error (RSE).
For convenience, we convert that relative error to a geometric error as $\epsilon_o \sim N(0, \ln(1 + \text{RSE}/q)^2)$.
For small uncertainties, the difference is negligible,
and for large uncertainties, it is not known which is better.
Like \citet{Reitan_2008}, we assume $\epsilon$ is normally distributed with mean zero and variance $\sigma^2$, $\epsilon \sim N(0, \sigma^2)$.
That simplification can create unaccounted heteroscedasticity \citep{Petersen_Overleir_2004}
but generally yields a reasonable estimate for the rating and its uncertainty.

\subsection{Fitting}
The algorithm uses \texttt{PyMC} \citep{Salvatier_2016},
an open-source Python library for Bayesian statistical modeling and probabilistic machine learning.
Using \texttt{PyMC}, the core model can be expressed in several lines of code,
making it easier to extend or modify,
like changing the priors,
the parameterization of the rating curve,
or the inference algorithm to achieve different tradeoffs of speed and accuracy.
This paper demonstrates two such algorithms:
Automatic Differentiation Variational Inference (ADVI) \citep{Kucukelbir_2017}
and Hamiltonian Monte Carlo with the No-U-Turn Sampler (NUTS) \citep{Hoffman_2014}.
ADVI is an Bayesian optimization algorithm, whereas NUTS is a Markov chain Monte Carlo (MCMC) sampling algorithm.
In general, MCMC sampling is slower than optimization but better for nonconvex problems \citep{Ma_2019}.

\subsection{Usage}
Refer to the \emph{Data Availability Statement} for links to the source code repository and packaged versions of \texttt{ratingcurve}.
Given observations of discharge (q), stage (h), and, optionally, the standard error of the discharge observations (e),
a two-segment rating is fit with
\begin{verbatim}
rating = PowerLawRating(q, h, e, 2)
trace = rating.fit()
rating.plot(trace)
\end{verbatim}
and produces a plot like Figure \ref{fig:rating-plot}.

A rating curve can also be exported as a table for use by other applications, shown in Table \ref{tab:rating-table}.
In addition to the mean discharge for each stage, the table gives the median and geometric standard error (GSE),
which is multiplied and divided by the median to estimate prediction intervals \citep{Limpert_2001}.


\section{Results}
We compared the performance of the segment power law against a log-transformed natural spline on a simulated 3-segment rating curve.
Both models use log transformations, which helps with heteroscedasticity.
The power law is strictly increasing;
otherwise, both approaches use log transformations,
and both are flexible enough to approximate a wide variety of functions.
Unlike the spline, the parameters in the power law have physical meaning in that they correspond to parameters in standard hydraulic equations for approximating open-channel flow.
The segmented power law is notoriously difficult to calibrate \citep{Reitan_2008}, however,
and its performance depends, in large part, on its parameterization
---we tested several, some mathematically equivalent, some slicing the cross section vertically or horizontally---
as well as its priors.
If the calibration challenges are overcome, 
the power law should yield better fits with fewer observations \citep{Reitan_2008}.

Figure \ref{fig:rating-comparison} shows a side-by-side comparison of a spline and power law fit with 6, 12, 24, and 48 stage-discharge observations.
For best accuracy, the curves were fit using NUTS.
We also specified that the power law had 3 segments and that the spline had 8 degrees of freedom,
the same as the power law (1 bias, 3 offsets, 3 slopes, and 1 uncertainty).
Otherwise, default settings were used for both.

Although the natural spline was 5-20x faster, it yielded poorer fits,
particularly when $n=6$.
Reducing the degrees of freedom might improve performance when $n=6$
but also sacrifices flexibility when $n=48$.
By comparison, the power law yielded a good fit with six observations---
two fewer than the number of model parameters.
Our intent is not to disparage all splines---
both parameterizations are technically splines.
Rather, we wanted to demonstrate a classic tradeoff between being easy to fit or being accurate,
which is a characteristic of data-driven and physical approaches.

In general, the accuracy of data-driven approaches is highly dependent on the availability of data.
For example, \citet{Coxon_2015} recommends a minimum of twenty stage-discharge measurements for their data-driven approach.
Taken over the lifetime of a streamgage, twenty may be manageable.
However, ratings shift through time from erosion, deposition, vegetation growth, debris/ice jams, etc. \citep{Herschy_2014, Mansanarez_2019},
and it may be impracticable to collect twenty measurements between each shift.
In that case, a more physical approach like the power law may be a better choice,
because they require fewer observations.

This paper focuses on one parameterization of the classic multi-segment power law,
but undoubtedly more will emerge, which might achieve better tradeoffs of ease and accuracy.
For example, our comparison uses NUTS, which is accurate but slow.
With 6 observations, NUTS fit the 3-segment power law in around 10 minutes.
With 48 observations, NUTS completed in 1 minute; a 10x speedup.
In general, stronger priors, more observations, or fewer segments would reduce that time.
By comparison, ADVI generally achieved a NUTS-like fit in several seconds, 
but it occasionally failed to converge on the optimum solution.
A better parameterization might yield better convergence with a faster inference algorithm.

\section{Conclusions}
Despite the existence of automated methods, most stage-discharge rating curves are still fit manually.
Although the governing hydraulic equations are relatively simple and well-understood,
they are notoriously difficult to solve for multiple controls.
Among the automated methods, 
no parameterization has emerged as the standard,
and functionally equivalent parameterizations may vary greatly in performance.

Here, we implement a simple parameterization that works well with minimal data and prior information.
Notably, it does not address shifts in the rating curve through time or hysteresis, and the curve is continuous but
not smooth (twice differentiable).
Such limitations could be addressed,
and any such effort will depend, in part, on building from a good starting parameterization.
Therefore, our simple-yet-reliable parameterization,
use of a community-developed probabilistic programming library,
and packaging provide a benchmark for operationalizing automated methods
that could promote more widespread use, testing, and refinement by the hydrologic community.


%\subsection{Figures}
%Frontiers requires figures to be submitted individually, in the same order as they are referred to in the manuscript. Figures will then be automatically embedded at the bottom of the submitted manuscript. Kindly ensure that each table and figure is mentioned in the text and in numerical order. Figures must be of sufficient resolution for publication \href{https://www.frontiersin.org/about/author-guidelines#ImageSizeRequirements}{see here for examples and minimum requirements}. Figures which are not according to the guidelines will cause substantial delay during the production process. Please see \href{https://www.frontiersin.org/about/author-guidelines#FigureRequirementsStyleGuidelines}{here} for full figure guidelines. Cite figures with subfigures as figure \ref{fig:Subfigure 1} and \ref{fig:Subfigure 2}.


\subsubsection{Permission to Reuse and Copyright}
Figures, tables, and images will be published under a Creative Commons CC-BY licence and permission must be obtained for use of copyrighted material from other sources (including re-published/adapted/modified/partial figures and images from the internet). It is the responsibility of the authors to acquire the licenses, to follow any citation instructions requested by third-party rights holders, and cover any supplementary charges.
%%Figures, tables, and images will be published under a Creative Commons CC-BY licence and permission must be obtained for use of copyrighted material from other sources (including re-published/adapted/modified/partial figures and images from the internet). It is the responsibility of the authors to acquire the licenses, to follow any citation instructions requested by third-party rights holders, and cover any supplementary charges.

\subsection{Tables}

\begin{table}[h!]
    \centering
    \caption{Rating table generated by \texttt{rating.table(trace)}.
             Units are feet (ft) and cubic feet per second (ft$^3$ s$^{-1}$); geometric standard error (GSE)
             is a unitless factor.}
    \label{tab:rating-table}
    \begin{tabular}{ cccc }
        \hline
               & Mean      & Median    &      \\
        Stage  & Discharge & Discharge & GSE  \\
        \hline
        ft     & ft$^3$ s$^{-1}$ & ft$^3$ s$^{-1}$  &  -   \\
        \hline
        2.20 &    1376.14 & 1376.16 & 1.0107 \\
        2.21 &    1388.27 & 1388.27 & 1.0107 \\
        2.22 &    1400.41 & 1400.40 & 1.0107 \\
        2.23 &    1412.57 & 1412.55 & 1.0106 \\
        2.24 &    1424.74 & 1424.73 & 1.0106 \\
        \dots&      \dots &   \dots &  \dots \\
        \hline
    \end{tabular}
\end{table}

%Tables should be inserted at the end of the manuscript. Please build your table directly in LaTeX. Tables provided as jpeg/tiff files will not be accepted. Please note that very large tables (covering several pages) cannot be included in the final PDF for reasons of space. These tables will be published as \href{http://home.frontiersin.org/about/author-guidelines#SupplementaryMaterial}{Supplementary Material} on the online article page at the time of acceptance. The author will be notified during the typesetting of the final article if this is the case. 

%\subsection{International Phonetic Alphabet}
%To include international phonetic alphabet (IPA) symbols, please include the following functions:
%Under useful packages, include:\begin{verbatim}\usepackage{tipa}\end{verbatim} 
%In the main text, when inputting symbols, use the following format:\begin{verbatim}\text[symbolname]\end{verbatim}e.g.\begin{verbatim}\textgamma\end{verbatim}

%\section{Nomenclature}

%\subsection{Resource Identification Initiative}
%To take part in the Resource Identification Initiative, please use the corresponding catalog number and RRID in your current manuscript. For more information about the project and for steps on how to search for an RRID, please click \href{http://www.frontiersin.org/files/pdf/letter_to_author.pdf}{here}.

%\subsection{Life Science Identifiers}
%Life Science Identifiers (LSIDs) for ZOOBANK registered names or nomenclatural acts should be listed in the manuscript before the keywords. For more information on LSIDs please see \href{https://www.frontiersin.org/about/author-guidelines#Nomenclature}{Inclusion of Zoological Nomenclature} section of the guidelines.


%\section{Additional Requirements}
%
%For additional requirements for specific article types and further information please refer to \href{http://www.frontiersin.org/about/AuthorGuidelines#AdditionalRequirements}{Author Guidelines}.

\section*{Conflict of Interest Statement}
%All financial, commercial or other relationships that might be perceived by the academic community as representing a potential conflict of interest must be disclosed. If no such relationship exists, authors will be asked to confirm the following statement: 

The authors declare that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest.

\section*{Author Contributions}
TOH: Conceptualization, Methodology, Software, Writing -- Original Draft;
KJD: Methodology, Software, Writing -- Review and Editing;
TAK: Conceptualization, Project administration, Writing -- Review and Editing;
TMO: Conceptualization, Writing -- Review and Editing; and
MBY: Conceptualization, Writing -- Review and Editing.

\section*{Funding}
This research was funded by the U.S. Geological Survey National Hydrologic Monitoring Program.
Any use of trade, firm, or product names is for descriptive purposes only and does not imply endorsement by the U.S. Government.

\section*{Acknowledgments}
The reviewers.

%\section*{Supplemental Data}
% \href{http://home.frontiersin.org/about/author-guidelines#SupplementaryMaterial}{Supplementary Material} should be uploaded separately on submission, if there are Supplementary Figures, please include the caption in the same file as the figure. LaTeX Supplementary Material templates can be found in the Frontiers LaTeX folder.

\section*{Data Availability Statement}
The latest version of \texttt{ratingcurve} is available at
\url{https://github.com/thodson-usgs/ratingcurve},
as well as \url{https://code.usgs.gov/wma/uncertainty/ratingcurve}.
Packaged versions are available via PyPI and conda-forge.
A link to the official release of the version used in this paper will appear here in due course.

%The datasets [GENERATED/ANALYZED] for this study can be found in the [NAME OF REPOSITORY] [LINK].
% Please see the availability of data guidelines for more information, at https://www.frontiersin.org/about/author-guidelines#AvailabilityofData

\bibliographystyle{Frontiers-Harvard} %  Many Frontiers journals use the Harvard referencing system (Author-date), to find the style and resources for the journal you are submitting to: https://zendesk.frontiersin.org/hc/en-us/articles/360017860337-Frontiers-Reference-Styles-by-Journal. For Humanities and Social Sciences articles please include page numbers in the in-text citations 
%\bibliographystyle{Frontiers-Vancouver} % Many Frontiers journals use the numbered referencing system, to find the style and resources for the journal you are submitting to: https://zendesk.frontiersin.org/hc/en-us/articles/360017860337-Frontiers-Reference-Styles-by-Journal
\bibliography{references}

%%% Make sure to upload the bib file along with the tex file and PDF
%%% Please see the test.bib file for some examples of references

\section*{Figure captions}

\begin{figure}[h!]
    \begin{center}
    \includegraphics[width=8.3cm]{figure1.pdf}
    \end{center}
    \caption{Two-segment rating curve, with 95-percent prediction interval, for the Green River near Jensen, Utah
    (U.S. Geological Survey streamgage 09261000);
    generated by \texttt{rating.plot(trace)} and fit using ADVI.
    The circles with error bars show the observations and their uncertainty.
    Horizontal dotted lines show the segment breakpoints and their prediction intervals.}
    \label{fig:rating-plot}
\end{figure}


\begin{figure*}[h!]
    \begin{center}
    \includegraphics[width=12cm]{figure2.pdf}
    \end{center}
    \caption{Segmented-power law (top) and natural spline (bottom) fit with different numbers of observations (n).
    The dashed red line is the true rating and the circles are the simulated observations.
    Horizontal dotted lines show segment breakpoints and knot locations for the power law and spline, respectively.
    The shaded regions depict 95 percent prediction intervals for the rating and breakpoints.}
    \label{fig:rating-comparison}
\end{figure*}



%%% Please be aware that for original research articles we only permit a combined number of 15 figures and tables, one figure with multiple subfigures will count as only one figure.
%%% Use this if adding the figures directly in the mansucript, if so, please remember to also upload the files when submitting your article
%%% There is no need for adding the file termination, as long as you indicate where the file is saved. In the examples below the files (logo1.eps and logos.eps) are in the Frontiers LaTeX folder
%%% If using *.tif files convert them to .jpg or .png
%%%  NB logo1.eps is required in the path in order to correctly compile front page header %%%

%\begin{figure}[h!]
%\begin{center}
%\includegraphics[width=10cm]{logo1}% This is a *.eps file
%\end{center}
%\caption{ Enter the caption for your figure here.  Repeat as  necessary for each of your figures}\label{fig:1}
%\end{figure}
%
%\setcounter{figure}{2}
%\setcounter{subfigure}{0}
%\begin{subfigure}
%\setcounter{figure}{2}
%\setcounter{subfigure}{0}
%    \centering
%    \begin{minipage}[b]{0.5\textwidth}
%        \includegraphics[width=\linewidth]{logo1.eps}
%        \caption{This is Subfigure 1.}
%        \label{fig:Subfigure 1}
%    \end{minipage}  
%   
%\setcounter{figure}{2}
%\setcounter{subfigure}{1}
%    \begin{minipage}[b]{0.5\textwidth}
%        \includegraphics[width=\linewidth]{logo2.eps}
%        \caption{This is Subfigure 2.}
%        \label{fig:Subfigure 2}
%    \end{minipage}
%
%\setcounter{figure}{2}
%\setcounter{subfigure}{-1}
%    \caption{Enter the caption for your subfigure here. \textbf{(A)} This is the caption for Subfigure 1. \textbf{(B)} This is the caption for Subfigure 2.}
%    \label{fig: subfigures}
%\end{subfigure}

%%% If you don't add the figures in the LaTeX files, please upload them when submitting the article.
%%% Frontiers will add the figures at the end of the provisional pdf automatically
%%% The use of LaTeX coding to draw Diagrams/Figures/Structures should be avoided. They should be external callouts including graphics.

\end{document}
