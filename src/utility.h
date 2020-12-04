/*-------------------------------------------------------------------------------
 This file is part of ranger.
 
 Copyright (c) [2014-2018] [Marvin N. Wright]
 
 This software may be modified and distributed under the terms of the MIT license.
 
 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
#-------------------------------------------------------------------------------*/
 
#ifndef UTILITY_H_
#define UTILITY_H_
 
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <cstddef> 
#include <memory> 
#include <type_traits> 
#include <utility> 
 
#ifdef R_BUILD
#include <Rinternals.h>
#endif
 
#include "globals.h"
#include "Data.h"
 
// #include <Rcpp.h>
 using namespace Rcpp;

#include <RcppArmadillo.h>
using namespace arma;
 
 namespace MOTE {
 
 
 // Provide make_unique (not available in C++11)
 namespace detail {
 
 template<class T> struct _Unique_if {
   typedef std::unique_ptr<T> _Single_object;
 };
 
 template<class T> struct _Unique_if<T[]> {
   typedef std::unique_ptr<T[]> _Unknown_bound;
 };
 
 template<class T, size_t N> struct _Unique_if<T[N]> {
   typedef void _Known_bound;
 };
 
 } // namespace detail
 
 template<class T, class ... Args>
 typename detail::_Unique_if<T>::_Single_object make_unique(Args&&... args) {
   return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
 }
 
 template<class T>
 typename detail::_Unique_if<T>::_Unknown_bound make_unique(size_t n) {
   typedef typename std::remove_extent<T>::type U;
   return std::unique_ptr<T>(new U[n]());
 }
 
 template<class T, class ... Args>
 typename detail::_Unique_if<T>::_Known_bound make_unique(Args&&...) = delete;
 
 void shuffleAndSplitAppend(std::vector<size_t>& first_part, std::vector<size_t>& second_part, size_t n_all,
                            size_t n_first, const std::vector<size_t>& mapping, std::mt19937_64 random_number_generator);
 
 
 // Matrix operations
 rowvec colSums(const mat& X);
 mat center(const mat& X); // Centering a matrix
 mat times(const mat& X, const vec& col); // Each row of matrix times the corrsponding scalar in col
 
 // CCA wrapper functions
 mat cancor(mat& matr_1, mat& matr_2);
 
 /**
  * Split sequence start..end in num_parts parts with sizes as equal as possible.
  * @param result Result vector of size num_parts+1. Ranges for the parts are then result[0]..result[1]-1, result[1]..result[2]-1, ..
  * @param start minimum value
  * @param end maximum value
  * @param num_parts number of parts
  */
 void equalSplit(std::vector<uint>& result, uint start, uint end, uint num_parts);
 
 /**
  * Convert a unsigned integer to string
  * @param number Number to convert
  * @return Converted number as string
  */
 std::string uintToString(uint number);
 
 /**
  * Beautify output of time.
  * @param seconds Time in seconds
  * @return Time in days, hours, minutes and seconds as string
  */
 std::string beautifyTime(uint seconds);
 
 } // namespace ranger
 
#endif /* UTILITY_H_ */