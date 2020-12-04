#include <math.h>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "utility.h"

namespace MOTE {

void shuffleAndSplitAppend(std::vector<size_t>& first_part, std::vector<size_t>& second_part, size_t n_all,
                           size_t n_first, const std::vector<size_t>& mapping, std::mt19937_64 random_number_generator) {
  // Old end is start position for new data
  size_t first_old_size = first_part.size();
  size_t second_old_size = second_part.size();
  
  // Reserve space
  first_part.resize(first_old_size + n_all);
  std::vector<size_t>::iterator first_start_pos = first_part.begin() + first_old_size;
  
  // Fill with 0..n_all-1 and shuffle
  std::iota(first_start_pos, first_part.end(), 0);
  std::shuffle(first_start_pos, first_part.end(), random_number_generator);
  
  // Mapping
  for (std::vector<size_t>::iterator j = first_start_pos; j != first_part.end(); ++j) {
    *j = mapping[*j];
  }
  
  // Copy to second part
  second_part.resize(second_part.size() + n_all - n_first);
  std::vector<size_t>::iterator second_start_pos = second_part.begin() + second_old_size;
  std::copy(first_start_pos + n_first, first_part.end(), second_start_pos);
  
  // Resize first part
  first_part.resize(first_old_size + n_first);
}


rowvec colSums(const mat& X){
  
  int nCols = X.n_cols;
  rowvec out(nCols);
  for(int i = 0; i < nCols; i++){
    out(i) = sum(X.col(i));
  }
  return(out);
}

// mat center(const mat& X, const rowvec& center){
//   mat out(X);
//   // int nRows = X.n_rows;
//   for(int i = 0; i < X.n_rows; i++){
//     out.row(i) = X.row(i)-center;
//   }
//   return(out);
// }

mat center(const mat& X){
  mat out(X);
  int nRows = X.n_rows;
  rowvec center = mean(X,0);

  for(int i = 0; i < nRows; i++){
    out.row(i) = X.row(i)-center;
  }
  return(out);
}

mat times(const mat& X, const vec& y){
  mat out(X);
  
  for(size_t i = 0; i < X.n_rows; i++){
        out.row(i) = X.row(i) * y(i);
      }
      return(out);
}

mat cancor(mat& matr_1, mat& matr_2){

  Function f("cancor");   
  List cca_res = f(matr_1, matr_2, false, false);
  
  return(cca_res["xcoef"]);
}

// NumericVector colMeans(NumericMatrix mat){
//   int nCols = mat.ncol();
//   NumericVector out (nCols);
// }

} // namespace ranger
