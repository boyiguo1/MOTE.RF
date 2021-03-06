% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/predict.R
\name{predict.MOTE.forest}
\alias{predict.MOTE.forest}
\title{Ranger prediction}
\usage{
\method{predict}{MOTE.forest}(
  object,
  data,
  predict.all = FALSE,
  num.trees = object$num.trees,
  seed = NULL,
  num.threads = NULL,
  verbose = TRUE,
  inbag.counts = NULL,
  q,
  ...
)
}
\arguments{
\item{object}{Ranger \code{ranger.forest} object.}

\item{data}{New test data of class \code{data.frame}.}

\item{num.trees}{Number of trees used for prediction. The first \code{num.trees} in the forest are used.}

\item{seed}{Random seed. Default is \code{NULL}, which generates the seed from \code{R}. Set to \code{0} to ignore the \code{R} seed. The seed is used in case of ties in classification mode.}

\item{num.threads}{Number of threads. Default is number of CPUs available.}

\item{verbose}{Verbose output on or off.}

\item{inbag.counts}{Number of times the observations are in-bag in the trees.}

\item{...}{further arguments passed to or from other methods.}
}
\value{
Object of class \code{ranger.prediction} with elements
\tabular{ll}{
\code{predictions}    \tab Predicted classes/values (only for classification and regression)  \cr
\code{unique.death.times} \tab Unique death times (only for survival). \cr
\code{chf} \tab Estimated cumulative hazard function for each sample (only for survival). \cr
\code{survival} \tab Estimated survival function for each sample (only for survival). \cr
\code{num.trees}   \tab Number of trees. \cr
\code{num.independent.variables} \tab Number of independent variables. \cr
\code{treetype}    \tab Type of forest/tree. Classification, regression or survival. \cr
\code{num.samples}     \tab Number of samples.
}
}
\description{
Prediction with new data and a saved forest from Ranger.
}
\details{
For \code{type = 'response'} (the default), the predicted classes (classification), predicted numeric values (regression), predicted probabilities (probability estimation) or survival probabilities (survival) are returned.

For classification and \code{predict.all = TRUE}, a factor levels are returned as numerics.
To retrieve the corresponding factor levels, use \code{rf$forest$levels}, if \code{rf} is the ranger object.
}
\references{
\itemize{
\item Wright, M. N. & Ziegler, A. (2017). ranger: A Fast Implementation of Random Forests for High Dimensional Data in C++ and R. J Stat Softw 77:1-17. \url{https://doi.org/10.18637/jss.v077.i01}.
\item Wager, S., Hastie T., & Efron, B. (2014). Confidence Intervals for Random Forests: The Jackknife and the Infinitesimal Jackknife. J Mach Learn Res 15:1625-1651. \url{http://jmlr.org/papers/v15/wager14a.html}.
}
}
\seealso{
\code{\link[=ranger]{ranger()}}
}
\author{
Marvin N. Wright
}
