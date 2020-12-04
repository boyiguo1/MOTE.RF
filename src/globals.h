/*-------------------------------------------------------------------------------
 This file is part of ranger.
 
 Copyright (c) [2014-2018] [Marvin N. Wright]
 
 This software may be modified and distributed under the terms of the MIT license.
 
 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
#-------------------------------------------------------------------------------*/
 
#ifndef GLOBALS_H_
#define GLOBALS_H_
 
 namespace MOTE {
 
 // Old/new Win build
#ifdef WIN_R_BUILD
#if __cplusplus < 201103L
#define OLD_WIN_R_BUILD
#else
#define NEW_WIN_R_BUILD
#endif
#endif
 
 typedef unsigned int uint;
 
 // Prediction type
 enum PredictionType {
         RESPONSE = 1,
         TERMINALNODES = 2
 };
 
 
 const uint DEFAULT_NUM_RANDOM_SPLITS = 1;
 const uint DEFAULT_MAXDEPTH = 0;
 const double DEFAULT_MINPROP = 0.1;
 const uint DEFAULT_NUM_TREE = 500;
 const uint DEFAULT_NUM_THREADS = 0;
 const PredictionType DEFAULT_PREDICTIONTYPE = RESPONSE;
 const uint DEFAULT_MIN_NODE_SIZE_REGRESSION = 5;
 
 
 
 } // namespace MOTE
 
#endif /* GLOBALS_H_ */