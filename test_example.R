 library(tidyverse)

## -----------------------------------------------------------------------------
set.seed(39)

 sim.dat <- sim_MOTE_data(B = create.B(10), Z = create.Z(10,3))
#
#
#  # Organize data by standardize
 train.dat <- sim.dat$train
#
 x.b <- train.dat$x.b
 x.e <- train.dat$x.e
 y.b <- train.dat$y.b
 y.e <- train.dat$y.e
#
 treat <- train.dat$trt
#
#
#  # scaling test.dat
 test.dat <- sim.dat$test
 test.x.b <- test.dat$x.b
 # colnames(test.x.b) <- paste0("Var", 1:10)
#  # true.trt.diff <- test.dat$y.e.1 - test.dat$y.e.2
#

## -----------------------------------------------------------------------------
#  # extreme random forest
 RF.mdl <- MOTE(x.b = x.b, x.e = x.e,
                treat = treat,
                y.b = y.b, y.e = y.e,
                # replace=F,
                num.trees = 1,
                num.random.splits = 100,
                max.depth=2,
                sample.fraction = rep(0.8,2),
                num.threads = 1,
                verbose=T)
 
 
 predict(RF.mdl, test.x.b,
         num.threads = 1)
 