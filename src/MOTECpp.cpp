/*-------------------------------------------------------------------------------
 This file is part of Ranger.
 
 Ranger is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 Ranger is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with Ranger. If not, see <http://www.gnu.org/licenses/>.
 
 Written by:
 
 Marvin N. Wright
 Institut für Medizinische Biometrie und Statistik
 Universität zu Lübeck
 Ratzeburger Allee 160
 23562 Lübeck
 
 http://www.imbs-luebeck.de
#-------------------------------------------------------------------------------*/
 
 // #include <RcppEigen.h>
#include <vector>
#include <sstream>
#include <memory>
#include <utility>
 
#include "globals.h"
#include "Forest.h"
#include "Data.h"
 // #include "DataSparse.h"
#include "utility.h"
 
 using namespace MOTE;
 
 // [[Rcpp::export]]
 Rcpp::List MOTECpp(arma::mat& x_b, arma::mat& x_diff, // mat& input_y_b, mat& input_y_e,
                    arma::mat& y_diff,
                    arma::vec trt, // arma::mat& Z,
                    std::vector<std::string> variable_names,
                    uint num_trees, bool verbose, uint seed, uint num_threads,
                    bool write_forest, uint min_node_size,
                    bool prediction_mode, Rcpp::List loaded_forest, 
                    bool sample_with_replacement,  // std::vector<double>& case_weights, bool use_case_weights,
                    std::vector<double>& class_weights, bool predict_all, bool keep_inbag,
                    std::vector<double>& sample_fraction, double minprop, bool holdout, 
                    uint num_random_splits, 
                    bool oob_error, uint max_depth, 
                    std::vector<std::vector<size_t>>& inbag, bool use_inbag
 ) {
    
    Rcpp::List result;
    
    try {
       std::unique_ptr<Forest> forest { };
       std::unique_ptr<Data> data { };
       
       
       // if (!use_case_weights) {
       //    case_weights.clear();
       // }
       if (!use_inbag) {
          inbag.clear();
       }
       
       std::ostream* verbose_out;
       if (verbose) {
          verbose_out = &Rcpp::Rcout;
       } else {
          verbose_out = new std::stringstream;
       }
       
       size_t num_rows;
       size_t num_cols;
       
       num_rows = x_b.n_rows;
       num_cols = x_b.n_cols;
       
       
       // Initialize data 
       // TODO: why this is DataRcpp
       // data = make_unique<DataRcpp>(input_x, input_y, variable_names, num_rows, num_cols);
       data = make_unique<Data>(x_b, x_diff, 
                                y_diff,
                                trt, //Z, 
                                variable_names, num_rows, num_cols);
       
       forest = make_unique<Forest>();
       
       // Init Ranger
       forest->initR(std::move(data),
                     num_trees, verbose_out, seed,
                     num_threads,
                     min_node_size,
                     prediction_mode, sample_with_replacement,
                     // case_weights, 
                     inbag, predict_all,
                     keep_inbag, sample_fraction,
                     minprop, holdout,
                     num_random_splits,
                     max_depth
       );
       
       // Load forest object if in prediction mode
       // TODO: need to implement this part
       // if (prediction_mode) {
       //   std::vector<std::vector<std::vector<size_t>> > child_nodeIDs = loaded_forest["child.nodeIDs"];
       //   std::vector<std::vector<size_t>> split_varIDs = loaded_forest["split.varIDs"];
       //   std::vector<std::vector<double>> split_values = loaded_forest["split.values"];
       //   std::vector<bool> is_ordered = loaded_forest["is.ordered"];
       // 
       //   if (treetype == TREE_CLASSIFICATION) {
       //     std::vector<double> class_values = loaded_forest["class.values"];
       //     auto& temp = dynamic_cast<ForestClassification&>(*forest);
       //     temp.loadForest(num_trees, child_nodeIDs, split_varIDs, split_values, class_values,
       //                     is_ordered);
       //   } else if (treetype == TREE_REGRESSION) {
       //     auto& temp = dynamic_cast<ForestRegression&>(*forest);
       //     temp.loadForest(num_trees, child_nodeIDs, split_varIDs, split_values, is_ordered);
       //   } else if (treetype == TREE_SURVIVAL) {
       //     std::vector<std::vector<std::vector<double>> > chf = loaded_forest["chf"];
       //     std::vector<double> unique_timepoints = loaded_forest["unique.death.times"];
       //     auto& temp = dynamic_cast<ForestSurvival&>(*forest);
       //     temp.loadForest(num_trees, child_nodeIDs, split_varIDs, split_values, chf,
       //                     unique_timepoints, is_ordered);
       //   } else if (treetype == TREE_PROBABILITY) {
       //     std::vector<double> class_values = loaded_forest["class.values"];
       //     std::vector<std::vector<std::vector<double>>> terminal_class_counts = loaded_forest["terminal.class.counts"];
       //     auto& temp = dynamic_cast<ForestProbability&>(*forest);
       //     temp.loadForest(num_trees, child_nodeIDs, split_varIDs, split_values, class_values,
       //                     terminal_class_counts, is_ordered);
       //   }
       // } else {
       //   // Set class weights
       //   if (treetype == TREE_CLASSIFICATION && !class_weights.empty()) {
       //     auto& temp = dynamic_cast<ForestClassification&>(*forest);
       //     temp.setClassWeights(class_weights);
       //   } else if (treetype == TREE_PROBABILITY && !class_weights.empty()) {
       //     auto& temp = dynamic_cast<ForestProbability&>(*forest);
       //     temp.setClassWeights(class_weights);
       //   }
       // }

       // Run Ranger
       forest->run(false, oob_error);        // original line
       // forest->run(true, oob_error);        // Debug Line


       // Use first non-empty dimension of predictions
       // TODO: need to change
       // const std::vector<std::vector<std::vector<double>>>& predictions = forest->getPredictions();
       // if (predictions.size() == 1) {
       //   if (predictions[0].size() == 1) {
       //     result.push_back(forest->getPredictions()[0][0], "predictions");
       //   } else {
       //     result.push_back(forest->getPredictions()[0], "predictions");
       //   }
       // } else {
       //   result.push_back(forest->getPredictions(), "predictions");
       // }
       //
       // Return output
       // result.push_back(forest->getNumTrees(), "num.trees");
       // result.push_back(forest->getNumIndependentVariables(), "num.independent.variables");
       // 
       // if (!prediction_mode) {
       //    result.push_back(forest->getMinNodeSize(), "min.node.size");
       //    result.push_back(forest->getVariableImportance(), "variable.importance");
       // 
       //    // TODO: expand this function
       //    //   result.push_back(forest->getOverallPredictionError(), "prediction.error");
       // }
       // 
       // if (keep_inbag) {
       //    result.push_back(forest->getInbagCounts(), "inbag.counts");
       // }
       // 
       // // Save forest if needed
       // if (write_forest) {
       //    Rcpp::List forest_object;
       //    forest_object.push_back(forest->getNumTrees(), "num.trees");
       //    // TODO: customize to accomadate to our forest structure
       //    //   forest_object.push_back(forest->getChildNodeIDs(), "child.nodeIDs");
       //    //   forest_object.push_back(forest->getSplitVarIDs(), "split.varIDs");
       //    //   forest_object.push_back(forest->getSplitValues(), "split.values");
       // 
       //    result.push_back(forest_object, "forest");
       // }

       if (!verbose) {
          delete verbose_out;
       }
    } catch (std::exception& e) {
       if (strcmp(e.what(), "User interrupt.") != 0) {
          Rcpp::Rcerr << "Error: " << e.what() << " Ranger will EXIT now." << std::endl;
       }
       return result;
    }
    
    return result;
 }
 
 