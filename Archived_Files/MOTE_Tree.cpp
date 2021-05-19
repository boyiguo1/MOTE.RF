#include <vector>
#include <sstream>
#include <exception>
#include <memory>
#include <utility>

#include "globals.h"
#include "Data.h"
#include "Tree.h"
#include "utility.h"


// #include <Rcpp.h>
// using namespace Rcpp;

#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace MOTE;


void MOTE_Tree(arma::mat& input_x_b, arma::mat& input_x_e, // mat& input_y_b, mat& input_y_e,
               arma::mat& y_diff,
               arma::vec trt,
               arma::mat& Z,
               std::vector<std::string> variable_names, // uint mtry,
               uint num_trees, uint min_node_size,
               uint seed,
               bool sample_with_replacement,
               bool memory_saving_splitting,
               bool keep_inbag,
               uint num_random_splits,
               uint max_depth,
               std::vector<double>& case_weights,
               std::vector<double>& sample_fraction,
               double minprop,
               bool verbose) {
  try {
    std::unique_ptr<Tree> tree { };
    std::unique_ptr<Data> data { };
    
    std::ostream* verbose_out;
    if (verbose) {
      verbose_out = &Rcpp::Rcout;
    } else {
      verbose_out = new std::stringstream;
    }
    
    size_t num_rows = input_x_b.n_rows;
    size_t num_cols= input_x_b.n_cols;
    // size_t num_samples;
    
    mat Z;
    
    data = make_unique<Data>(input_x_b, input_x_e, 
                             // input_y_b, input_y_e,
                             y_diff,
                             trt, Z, variable_names, num_rows, num_cols);
    
    tree = make_unique<Tree>();
    
   size_t num_samples = data->getNumRows();
    
    // ?? need to be removed after fixing manual_inbag
    std::vector<size_t>* manual_inbag = 0;
    // Saving all the tree parameter to tree object.
    // setting up for the root node
    tree->init( data.get(), // mtry,
                num_samples, seed, min_node_size,
                sample_with_replacement, memory_saving_splitting,
                // ?? Manual_inbag needs work
               // &case_weights,
               &manual_inbag[0], keep_inbag,
               &sample_fraction,
               minprop, //bool holdout,
              num_random_splits,
              max_depth);

    // Initiate Variable Importance
    vec variable_importance(num_cols, fill::zeros);
    // variable_importance.resize(num_cols, 0);
    tree->grow(&variable_importance);
    
  } catch (std::exception& e) {
    if (strcmp(e.what(), "User interrupt.") != 0) {
      Rcpp::Rcerr << "Error: " << e.what() << " MOTE.RF will EXIT now." << std::endl;
    }
  }
}