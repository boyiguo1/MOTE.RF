# -------------------------------------------------------------------------------
#   This file is part of MOTE.
#
# Written by:
#
# Boyi Guo
# Department of Biostatistics
# University of Alabama at Birmingham
# -------------------------------------------------------------------------------

##' MOTE is a fast implementation of MOTE RF (Guo et al. 2021), a random forest model that simultaneously estimates individualized treatment effects of multivariate outcomes.
##'
##' Forests of extremely randomized trees (Geurts et al. 2006) can be grown.
##'
##' `r lifecycle::badge('experimental')`
##'
##' @details 
##'
##' Note that: 
##' * nodes with size smaller than `min.node.size` can occur, as in original Random Forests.
##' * The mtry variable that randomly selected variables doesn't exist in our framework, as we are using oblique splitting rules.
##' * For factor covriates, a reference coding design matrix will be created accordingly. TODO: it is possible to extend this later by extending to ordered factors.
##' * The importance of variables/features are calculated by accumulating the coefficients in the oblique splitting rules weighted by node sample size.
##' * The model yet to provide handling of missing value. Data containing missing value can promote computation error and warning.
##' 
##' For a large number of variables and data frames as input data the formula interface can be slow or impossible to use.
##' TODO: need improve this part a little bit
##' Alternatively `dependent.variable.name` and `status.variable.name` for treatment category or `x` and `y` can be used.
##' Use `x` and `y` with a matrix for `x` to avoid conversions and save memory.
##' 
##' TODO: 
##' Since our algorithm is oriented to analyze microbiome data, we can directly use phyloseq object.
##' Case weights is not supported temporarily
##' 
##' See <https://github.com/boyiguo1/MOTE.RF> for the development version.
##' 
##' With recent R versions, multithreading on Windows platforms should just work. 
##' If you compile yourself, the new RTools toolchain is required.
##' 
##' @title MOTE
##' @param formula Object of class `formula` or `character` describing the model to fit. Interaction terms supported only for numerical variables.
##' @param data Training data of class `data.frame`, `matrix`, `dgCMatrix` (Matrix).
##' @param num.trees Number of trees.
##' @param write.forest Save `MOTE.forest` object, required for prediction. Set to `FALSE` to reduce memory usage if no prediction intended.
##' @param min.node.size Minimal node size. Default 1 for classification, 5 for regression, 3 for survival, and 10 for probability.
##' @param max.depth Maximal tree depth. A value of NULL or 0 (the default) corresponds to unlimited depth, 1 to tree stumps (1 split per tree).
##' @param replace Sample with replacement. 
##' @param sample.fraction Fraction of observations to sample. Default is 1 for sampling with replacement and 0.632 for sampling without replacement. For classification, this can be a vector of class-specific values. 
##' @param class.weights Weights for the outcome classes (in order of the factor levels) in the splitting rule (cost sensitive learning). Classification and probability prediction only. For classification the weights are also applied in the majority vote in terminal nodes.
# TODO: Update the definition for minprop and num.random.splits
##' @param num.random.splits For "extratrees" splitrule.: Number of random splits to consider for each candidate splitting variable.
##' @param minprop For "maxstat" splitrule: Lower quantile of covariate distribution to be considered for splitting.
##' @param keep.inbag Save how often observations are in-bag in each tree. 
##' @param inbag Manually set observations per tree. List of size num.trees, containing inbag counts for each observation. Can be used for stratified sampling.
##' @param holdout Hold-out mode. Hold-out all samples with case weight 0 and use these for variable importance and prediction error.
##' @param oob.error Compute OOB prediction error. Set to `FALSE` to save computation time, e.g. for large survival forests.
##' @param num.threads Number of threads. Default is number of CPUs available.
##' @param save.memory Use memory saving (but slower) splitting mode. No effect for survival and GWAS data. Warning: This option slows down the tree growing, use only if you encounter memory problems.
##' @param verbose Show computation status and estimated runtime.
##' @param seed Random seed. Default is `NULL`, which generates the seed from `R`. Set to `0` to ignore the `R` seed. 
##' @param dependent.variable.name Name of dependent variable, needed if no formula given. For survival forests this is the time variable.
##' @param trt.variable.name Name of treatment variable. Treatment variable should be a factor variable
##' @param x Predictor data (independent variables), alternative interface to data with formula or dependent.variable.name.
##' @param y Response vector (dependent variable), alternative interface to data with formula or dependent.variable.name. For survival use a `Surv()` object or a matrix with time and status.
##' @param ... Further arguments passed to or from other methods (currently ignored).
##' @return Object of class `ranger` with elements
##'   \item{`forest`}{Saved forest (If write.forest set to TRUE). Note that the variable IDs in the `split.varIDs` object do not necessarily represent the column number in R.}
##'   \item{`predictions`}{Predicted classes/values, based on out of bag samples (classification and regression only).}
##'   \item{`variable.importance`}{Variable importance for each independent variable.}
##'   \item{`prediction.error`}{Overall out of bag prediction error. For classification this is the fraction of missclassified samples, for probability estimation the Brier score, for regression the mean squared error and for survival one minus Harrell's C-index.}
##'   \item{`r.squared`}{R squared. Also called explained variance or coefficient of determination (regression only). Computed on out of bag data.}
##'   \item{`confusion.matrix`}{Contingency table for classes and predictions based on out of bag samples (classification only).}
##'   \item{`call`}{Function call.}
##'   \item{`num.trees`}{Number of trees.}
##'   \item{`num.independent.variables`}{Number of independent variables.}
##'   \item{`mtry`}{Value of mtry used.}
##'   \item{`min.node.size`}{Value of minimal node size used.}
##'   \item{`num.samples`}{Number of samples.}
##'   \item{`inbag.counts`}{Number of times the observations are in-bag in the trees.}
##' @examples
##' set.seed(1)
##' 
##' p = 10
##' q = 3
##'
##' B = create.B(p)
##' Z = create.Z(p,q)
##'
##' sim.dat <- sim_MOTE_data(B = B, Z = Z)
##' 
##'
##' 
##'
##'
##' @author Boyi Guo
##' @references
##' \itemize{
##'   \item Guo et al. (Submitted). Estimating Heterogeneous Treatment Effect on Multivariate Responses using Random Forests. <https://doi.org/xxxx>.
##'   }
##' @seealso [predict.MOTE()]
##' @useDynLib MOTE.RF, .registration = TRUE
##' @importFrom Rcpp evalCpp
##' @import stats 
##' @import utils
##' @importFrom Matrix Matrix
##' @export
MOTE <- function(#formula = NULL, data = NULL, 
                  x.b, x.e,
                  treat,
                  y.b, y.e,
                  Z,     # Covariates
                  num.trees = 300, min.node.size = NULL, max.depth = NULL,
                  num.random.splits = 1,  minprop = 0.1,
                  write.forest = TRUE, 
                  replace = TRUE, 
                  sample.fraction = ifelse(replace, 1, 0.632), 
                  # case.weights = NULL, 
                  class.weights = NULL,
                  keep.inbag = FALSE, inbag = NULL, holdout = FALSE,
                  oob.error = TRUE,
                  num.threads = 1,
                  verbose = TRUE, seed = NULL, 
                  dependent.variable.name = NULL, 
                  trt.variable.name = NULL, 
                  ...) {
  
  ## Handle ... arguments
  if (length(list(...)) > 0) {
    warning(paste("Unused arguments:", paste(names(list(...)), collapse = ", ")))
  }
  
  
  # TODO: reorganize data
  # TODO: Create diff
  # TODO: check treatment variable
  # TODO: create design matrix for categorical variables
  # creating X.b, X.e, Covariates
  # if (is.null(data)) {
  #   ## x/y interface
  #   if (is.null(x) | is.null(y)) {
  #     stop("Error: Either data or x and y is required.")
  #   }
  # }  else {
  #   
  #   ## Formula interface. Use whole data frame if no formula provided and depvarname given
  #   if (is.null(formula)) {
  #     if (is.null(dependent.variable.name)) {
  #       if (is.null(y) | is.null(x)) {
  #         stop("Error: Please give formula, dependent variable name or x/y.")
  #       } 
  #     } else {
  #       if (is.null(status.variable.name)) {
  #         y <- data[, dependent.variable.name, drop = TRUE]
  #         x <- data[, !(colnames(data) %in% dependent.variable.name), drop = FALSE]
  #       } else {
  #         y <- survival::Surv(data[, dependent.variable.name], data[, status.variable.name]) 
  #         x <- data[, !(colnames(data) %in% c(dependent.variable.name, status.variable.name)), drop = FALSE]
  #       }
  #     }
  #   } else {
  #     formula <- formula(formula)
  #     if (!inherits(formula, "formula")) {
  #       stop("Error: Invalid formula.")
  #     }
  #     data.selected <- parse.formula(formula, data, env = parent.frame())
  #     y <- data.selected[, 1]
  #     x <- data.selected[, -1, drop = FALSE]
  #   }
  # }
  
  ## Sparse matrix data
  # if (inherits(x, "Matrix")) {
  #   if (!inherits(x, "dgCMatrix")) {
  #     stop("Error: Currently only sparse data of class 'dgCMatrix' supported.")
  #   } 
  #   if (!is.null(formula)) {
  #     stop("Error: Sparse matrices only supported with alternative interface. Use dependent.variable.name or x/y instead of formula.")
  #   }
  # }
  
  ## Check missing values
  # if (any(is.na(x))) {
  #   offending_columns <- colnames(x)[colSums(is.na(x)) > 0]
  #   stop("Missing data in columns: ",
  #        paste0(offending_columns, collapse = ", "), ".", call. = FALSE)
  # }
  # if (any(is.na(y))) {
  #   stop("Missing data in dependent variable.", call. = FALSE)
  # }
  
  
  if(!all(is.matrix(x.b), is.matrix(x.e), is.matrix(y.b), is.matrix(y.e)))
    stop("Error Message: x.b, x.e, y.b, y.e must be numeric matrices")
  
  if(missing(Z)) {
    Z <- NULL
    # Z <- matrix(0, 2, 2)
  }
  else {
    if(!is.matrix(Z))
      stop("Error Message: x.b, x.e, y.b, y.e must be numeric matrices")
  }
  
  if(length(unique(
    nrow(x.b),nrow(x.e),length(treat),nrow(y.b), nrow(y.b), nrow(Z)
  ))!=1)
    stop("Error Message: incosistent observation numbers in x.b, x.e, treat, y.b, y.e, Z")
  
  treat <- factor(treat)
  if(length(levels(treat)) != 2)
    stop("Error Message: Incorrect number of treatment groups. Must be 2 groups")
  trt.lvl <- levels(treat)
  

  
  
  ## Number of trees
  if (!is.numeric(num.trees) || num.trees < 1) {
    stop("Error: Invalid value for num.trees.")
  }
  
  
  ## Seed
  if (is.null(seed)) {
    seed <- runif(1 , 0, .Machine$integer.max)
  }
  
  ## Keep inbag
  if (!is.logical(keep.inbag)) {
    stop("Error: Invalid value for keep.inbag")
  }
  
  ## Num threads
  ## Default 0 -> detect from system in C++.
  if (is.null(num.threads)) {
    num.threads = 0
  } else if (!is.numeric(num.threads) || num.threads < 0) {
    stop("Error: Invalid value for num.threads")
  }
  
  ## Minumum node size
  if (is.null(min.node.size)) {
    min.node.size <- 0
  } else if (!is.numeric(min.node.size) || min.node.size < 0) {
    stop("Error: Invalid value for min.node.size")
  }
  
  ## Tree depth
  if (is.null(max.depth)) {
    max.depth <- 0
  } else if (!is.numeric(max.depth) || max.depth < 0) {
    stop("Error: Invalid value for max.depth. Please give a positive integer.")
  }
  
  ## Sample fraction
  # TODO: need to figure out how sample.fraction works
  if (!is.numeric(sample.fraction)) {
    stop("Error: Invalid value for sample.fraction. Please give a value in (0,1] or a vector of values in [0,1].")
  }
  if (length(sample.fraction) > 1) {
    if (any(sample.fraction < 0) || any(sample.fraction > 1)) {
      stop("Error: Invalid value for sample.fraction. Please give a value in (0,1] or a vector of values in [0,1].")
    }
    if (sum(sample.fraction) <= 0) {
      stop("Error: Invalid value for sample.fraction. Sum of values must be >0.")
    }
    if (length(sample.fraction) != nlevels(treat)) {
      stop("Error: Invalid value for sample.fraction. Expecting ", nlevels(treat), " values, provided ", length(sample.fraction), ".")
    }
    if (!replace & any(sample.fraction * length(treat) > table(treat))) {
      idx <- which(sample.fraction * length(treat) > table(treat))[1]
      stop("Error: Not enough samples in class ", names(idx), 
           "; available: ", table(treat)[idx], 
           ", requested: ", (sample.fraction * length(treat))[idx], ".")
    }
    # if (!is.null(case.weights)) {
    #   stop("Error: Combination of case.weights and class-wise sampling not supported.")
    # }
    # Fix order (C++ needs sample.fraction in order as classes appear in data)
    sample.fraction <- sample.fraction[as.numeric(unique(treat))]
  } else {
    if (sample.fraction <= 0 || sample.fraction > 1) {
      stop("Error: Invalid value for sample.fraction. Please give a value in (0,1] or a vector of values in [0,1].")
    }
    sample.fraction <- rep(sample.fraction, nlevels(treat))
  }
  
  
  ## Case weights: NULL for no weights or all weights equal
  # TODO: check what this does
  # if (is.null(case.weights) || length(unique(case.weights)) == 1) {
  #   case.weights <- c(0,0)
  #   use.case.weights <- FALSE
  #   if (holdout) {
  #     stop("Error: Case weights required to use holdout mode.")
  #   }
  # } else {
  #   use.case.weights <- TRUE
  #   
  #   ## Sample from non-zero weights in holdout mode
  #   if (holdout) {
  #     sample.fraction <- sample.fraction * mean(case.weights > 0)
  #   }
  #   
  #   if (!replace && sum(case.weights > 0) < sample.fraction * nrow(x)) {
  #     stop("Error: Fewer non-zero case weights than observations to sample.")
  #   }
  # }
  
  ## Manual inbag selection
  # TODO: check what this does
  if (is.null(inbag)) {
    inbag <- list(c(0,0))
    use.inbag <- FALSE
  } else if (is.list(inbag)) {
    use.inbag <- TRUE
    # if (use.case.weights) {
    #   stop("Error: Combination of case.weights and inbag not supported.")
    # }
    if (length(sample.fraction) > 1) {
      stop("Error: Combination of class-wise sampling and inbag not supported.")
    }
    if (length(inbag) != num.trees) {
      stop("Error: Size of inbag list not equal to number of trees.")
    }
  } else {
    stop("Error: Invalid inbag, expects list of vectors of size num.trees.")
  }
  
  ## Class weights: NULL for no weights (all 1)
  # TODO: check what this does
  if (is.null(class.weights)) {
    class.weights <- rep(1, nlevels(treat))
  } else {
    if (!is.numeric(class.weights) || any(class.weights < 0)) {
      stop("Error: Invalid value for class.weights. Please give a vector of non-negative values.")
    }
    if (length(class.weights) != nlevels(treat)) {
      stop("Error: Number of class weights not equal to number of classes.")
    }
    
    ## Reorder (C++ expects order as appearing in the data)
    class.weights <- class.weights[unique(as.numeric(treat))]
  }
  
  if (minprop < 0 || minprop > 0.5) {
    stop("Error: Invalid value for minprop, please give a value between 0 and 0.5.")
  }
  
  if (!is.numeric(num.random.splits) || num.random.splits < 1) {
    stop("Error: Invalid value for num.random.splits, please give a positive integer.")
  }
  
  
  ## Prediction mode always false. Use predict.MOTE() method.
  prediction.mode <- FALSE
  predict.all <- FALSE
  prediction.type <- 1
  
  ## No loaded forest object
  loaded.forest <- list()
  
  x.b.new <- cbind(Z, x.b)
  x.diff <- x.e - x.b
  independent.variable.names <- colnames(x.b.new)
  if(is.null(independent.variable.names))
    independent.variable.names <- paste0("Var", seq(1, ncol(x.b.new)))
  
  all.independent.variable.names <- independent.variable.names
  
  treat_num <- ifelse(treat==trt.lvl[1], 1, -1)
  
  ## Call MOTE
  result <- MOTECpp(  x.b.new, x.diff,
                      y.e-y.b,
                      treat_num,
                    independent.variable.names, 
                    num.trees, verbose, seed, num.threads, write.forest, min.node.size,
                    prediction.mode,  loaded.forest, 
                    replace, # case.weights, use.case.weights,
                    class.weights, 
                    predict.all, keep.inbag, sample.fraction,
                    minprop, holdout, #prediction.type, 
                    num.random.splits, #sparse.x, use.sparse.data, 
                    oob.error, max.depth, 
                    inbag, use.inbag 
  )
  
  if (length(result) == 0) {
    stop("User interrupt or internal error.")
  }
  
  ## Prepare results
  # Varable importance
  colnames(result$variable.importance) <- all.independent.variable.names
  
  # if (oob.error) {
    # # TODO:organize the structure for prediction results
    # if (is.list(result$predictions)) {
    #   result$predictions <- do.call(rbind, result$predictions)
    # }
    # if (is.vector(result$predictions)) {
    #   result$predictions <- matrix(result$predictions, nrow = 1)
    # }
  # }
  
  # TODO: is it possible to calculate r.squared for us
  # result$r.squared <- 1 - result$prediction.error / var(y)
  result$call <- sys.call()
  # if (use.sparse.data) {
  #   result$num.samples <- nrow(sparse.x)
  # } else {
    result$num.samples <- nrow(x.b.new)
  # }
  result$replace <- replace
  result$q <- ncol(y.b)
  
  ## Write forest object
  if (write.forest) {
    # if (is.factor(y)) {
    #   result$forest$levels <- levels(y)
    # }
    result$forest$independent.variable.names <- independent.variable.names
    class(result$forest) <- "MOTE.forest"
    
    ## In 'ordered' mode, save covariate levels
    ## TODO: How to save the reference leveling for factor analysis
    # if (respect.unordered.factors == "order" && ncol(x) > 0) {
    #   result$forest$covariate.levels <- covariate.levels
    # }
  }
  
  class(result) <- "MOTE"
  
  return(result)
}
