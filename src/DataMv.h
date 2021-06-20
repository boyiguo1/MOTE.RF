/*-------------------------------------------------------------------------------
 This file is part of ranger.
 
 Copyright (c) [2014-2018] [Marvin N. Wright]
 
 This software may be modified and distributed under the terms of the MIT license.
 
 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
#-------------------------------------------------------------------------------*/
 
#ifndef DATAMV_H_
#define DATAMV_H_
 
#include <vector>
#include <iostream>
#include <exception>
 
// #include <numeric>
// #include <random>
// #include <algorithm>
 
// #include "globals.h"
// #include <Rcpp.h>
// using namespace Rcpp;

#include <RcppArmadillo.h>

using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

 
 namespace MOTE {
 
 class DataMv {
 public:
   DataMv():
    num_rows(0), num_cols(0)//, num_rows_rounded(0), 
     // externalData(true), 
    //index_data(0), max_num_unique_values(0)
    {
   }
   
   DataMv(mat& x, mat& y,
            std::vector<std::string> variable_names, size_t num_rows, size_t num_cols) {
     this->x = x;
     this->y = y;
     
     this->variable_names = variable_names;
     this->num_rows = num_rows;
     this->num_cols = num_cols;
   }
   
   
   
   DataMv(const DataMv&) = delete;
   DataMv& operator=(const DataMv&) = delete;
   
   size_t getNumRows() const {
      return num_rows;
   }
   
   size_t getNumCols() const {
      return num_cols;
   }
   
   size_t get_y_cols() const {
      return y.n_cols;
   }
   
   
   mat get_y_rows(uvec pos) const {
      return y.rows(pos);
   }
   
   mat get_x_rows(uvec pos) const {
      return x.rows(pos);
   }
   
   rowvec get_x_rows(size_t pos) const {
      return x.row(pos);
   }
   
   
   //    return y_diff.n_rows;
   // }
   

   
 protected:
   std::vector<std::string> variable_names;
   size_t num_rows;
   // size_t num_rows_rounded;
   size_t num_cols;
   
   
   mat x;
   mat y;
   
   
   // bool externalData;
   
   // std::vector<size_t> index_data;
   // std::vector<std::vector<double>> unique_data_values;
   // size_t max_num_unique_values;
   

   
   // For each varID true if ordered
   // std::vector<bool> is_ordered_variable;
   
   // Permuted samples for corrected impurity importance
   // std::vector<size_t> permuted_sampleIDs;
   
 };
 
 } // namespace ranger
 
#endif /* DATAMV_H_ */
 