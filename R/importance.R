# -------------------------------------------------------------------------------
#   This file is part of Ranger.
#
# Ranger is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Ranger is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Ranger. If not, see <http://www.gnu.org/licenses/>.
#
# Written by:
#
#   Marvin N. Wright
# Institut fuer Medizinische Biometrie und Statistik
# Universitaet zu Luebeck
# Ratzeburger Allee 160
# 23562 Luebeck
# Germany
#
# http://www.imbs-luebeck.de
# -------------------------------------------------------------------------------

##' @export
importance <- function(x, ...)  UseMethod("importance")

##' Extract variable importance of MOTE object.
##'
##'
##' `r lifecycle::badge('experimental')`
##'
##' @title MOTE variable importance
##' @param x MOTE object.
##' @param ... Further arguments passed to or from other methods.
##' @return Variable importance measures.
##' @seealso [MOTE()]
##' @author Marvin N. Wright
##' @aliases importance
##' @export 
importance.MOTE <- function(x, ...) {
  if (!inherits(x, "MOTE")) {
    stop("Object ist no MOTE object.")
  }
  if (is.null(x$variable.importance) || ncol(x$variable.importance) < 1) {
    stop("No variable importance found. Please use 'importance' option when growing the forest.")
  }
  
  FI_eigen <- eigen(x$variable.importance)
  

  eval <- FI_eigen$values
  evec <- FI_eigen$vectors
  
  res <- vector()
  for(i in 1:length(eval))
    res <- cbind(res,eval[i]*evec[,i])
  
  FI_res <-rowSums(abs(res))
  FI_res <- FI_res/max(FI_res)
  names(FI_res) <- colnames(x$variable.importance)
  
  return(FI_res)
}
