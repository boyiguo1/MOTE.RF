/*-------------------------------------------------------------------------------
 This file is part of ranger.
 
 Copyright (c) [2014-2018] [Marvin N. Wright]
 
 This software may be modified and distributed under the terms of the MIT license.
 
 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
#-------------------------------------------------------------------------------*/
 
#ifndef DATA_H_
#define DATA_H_
 
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
 
 class Data {
 public:
   Data():
    num_rows(0), num_cols(0)//, num_rows_rounded(0), 
     // externalData(true), 
    //index_data(0), max_num_unique_values(0)
    {
   }
   
   Data(mat& x_b, mat& x_diff,
        // Rcpp::mat& y_b, Rcpp::mat& y_e, 
        mat& y_diff,
            vec trt,
         // mat& Z,
            std::vector<std::string> variable_names, size_t num_rows, size_t num_cols) {
     this->x_b = x_b;
     // this->y_b = y_b;
     // this->x_e = x_e;
     this->x_diff = x_diff;
     // this->y_e = y_e;
     this->y_diff = y_diff;
     // this ->Z = Z;
     this->trt = trt;
     
     // Debug line: if trt could only contains two values
     if(!all((trt==1) || (trt==-1)))
       throw std::runtime_error("Trt must be either 1 or -1");
     
     this->variable_names = variable_names;
     this->num_rows = num_rows;
     this->num_cols = num_cols;
   }
   
   
   
   Data(const Data&) = delete;
   Data& operator=(const Data&) = delete;
   
   size_t getNumRows() const {
      return num_rows;
   }
   
   size_t getNumCols() const {
      return num_cols;
   }
   
   vec get_trt() const {
      return trt;
   }
   
   vec get_trt(uvec pos) const {
      return trt.elem(pos);
   }
   
   mat get_y_diff_rows(uvec pos) const {
      return y_diff.rows(pos);
   }
   
   mat get_x_b_rows(uvec pos) const {
      return x_b.rows(pos);
   }
   
   rowvec get_x_b_rows(size_t pos) const {
      return x_b.row(pos);
   }
   
   mat get_x_diff_rows(uvec pos) const{
      return x_diff.rows(pos);
   }
   
   size_t n_y_diff_rows() const{
      
      if(!((y_diff.n_rows==x_diff.n_rows) & (x_b.n_rows==y_diff.n_rows) & (y_diff.n_rows== trt.n_elem)))
         throw std::runtime_error("Inconsistent sample size for Data structure");
      
      return y_diff.n_rows;
   }
   

   
 protected:
   std::vector<std::string> variable_names;
   size_t num_rows;
   // size_t num_rows_rounded;
   size_t num_cols;
   
   // NOTE:
   // trt[i]==1 when trt is in the first level
   // trt[i]==-1 when trt is in the second level
   vec trt;
   
   mat x_b;
   mat x_diff;
   // mat x_e;
   // mat y_b;
   // mat y_e;
   mat y_diff;
   // mat Z;
   
   
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
 
#endif /* DATA_H_ */
 