/*-------------------------------------------------------------------------------
 This file is part of ranger.
 
 Copyright (c) [2014-2018] [Marvin N. Wright]
 
 This software may be modified and distributed under the terms of the MIT license.
 
 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
#-------------------------------------------------------------------------------*/
 
 // #include <iterator>
 
#include  <RcppArmadillo.h>
 
#include "Forest.h"
#include "utility.h"
 
 using namespace arma;
 
 
 namespace MOTE {
 
 Forest::Forest() : verbose_out(0), num_trees(DEFAULT_NUM_TREE), min_node_size(0), num_independent_variables(0), seed(0), num_samples(
                 0), prediction_mode(false), //memory_mode(MEM_DOUBLE),
                 sample_with_replacement(true), // memory_saving_splitting(false), splitrule(DEFAULT_SPLITRULE), 
                 predict_all(false), keep_inbag(false), sample_fraction( { 1 }), holdout(false), // prediction_type(DEFAULT_PREDICTIONTYPE), 
                 num_random_splits(DEFAULT_NUM_RANDOM_SPLITS), max_depth(DEFAULT_MAXDEPTH),
                 minprop(DEFAULT_MINPROP), num_threads(DEFAULT_NUM_THREADS), data { }, overall_prediction_error(NAN), progress(0) {
                 }
 
 void Forest::initR(std::unique_ptr<Data> input_data, //uint mtry, 
                    uint num_trees, std::ostream* verbose_out, uint seed,
                    uint num_threads, //ImportanceMode importance_mode, 
                    uint min_node_size,
                    // std::vector<std::vector<double>>& split_select_weights,
                    // const std::vector<std::string>& always_split_variable_names, 
                    bool prediction_mode, bool sample_with_replacement,
                    // const std::vector<std::string>& unordered_variable_names, bool memory_saving_splitting, SplitRule splitrule,
                    std::vector<double>& case_weights, std::vector<std::vector<size_t>>& manual_inbag, bool predict_all,
                    bool keep_inbag, std::vector<double>& sample_fraction, //double alpha, 
                    double minprop, bool holdout,
                    // PredictionType prediction_type, 
                    uint num_random_splits, //bool order_snps, 
                    uint max_depth) {
         
         this->verbose_out = verbose_out;
         
         // Call other init function
         init(// MEM_DOUBLE, 
              std::move(input_data),  "", num_trees, seed, num_threads, min_node_size,
              prediction_mode, sample_with_replacement, 
              predict_all, sample_fraction,  minprop, holdout,  num_random_splits, max_depth);
         
         
         // Set case weights
         if (!case_weights.empty()) {
                 if (case_weights.size() != num_samples) {
                         throw std::runtime_error("Number of case weights not equal to number of samples.");
                 }
                 this->case_weights = case_weights;
         }
         
         // Set manual inbag
         if (!manual_inbag.empty()) {
                 this->manual_inbag = manual_inbag;
         }
         
         // Keep inbag counts
         this->keep_inbag = keep_inbag;
                        }
 
 void Forest::init(
                   std::unique_ptr<Data> input_data, std::string output_prefix,
                   uint num_trees, uint seed, uint num_threads, uint min_node_size,
                   bool prediction_mode, bool sample_with_replacement, 
                    bool predict_all, std::vector<double>& sample_fraction,
                   double minprop, bool holdout,  uint num_random_splits,
                   uint max_depth) {
         
         // Initialize data with memmode
         this->data = std::move(input_data);
         
         // Initialize random number generator and set seed
         if (seed == 0) {
                 std::random_device random_device;
                 random_number_generator.seed(random_device());
         } else {
                 random_number_generator.seed(seed);
         }
         
         // Set number of threads
         if (num_threads == DEFAULT_NUM_THREADS) {
#ifdef OLD_WIN_R_BUILD
                 this->num_threads = 1;
#else
                 this->num_threads = std::thread::hardware_concurrency();
#endif
         } else {
                 this->num_threads = num_threads;
         }
         
         // Set member variables
         this->num_trees = num_trees;
         // this->mtry = mtry;
         this->seed = seed;
         this->output_prefix = output_prefix;
         // this->importance_mode = importance_mode;
         this->min_node_size = min_node_size;
         // this->memory_mode = memory_mode;
         this->prediction_mode = prediction_mode;
         this->sample_with_replacement = sample_with_replacement;
         // this->memory_saving_splitting = memory_saving_splitting;
         // this->splitrule = splitrule;
         this->predict_all = predict_all;
         this->sample_fraction = sample_fraction;
         this->holdout = holdout;
         // this->alpha = alpha;
         this->minprop = minprop;
         // this->prediction_type = prediction_type;
         this->num_random_splits = num_random_splits;
         this->max_depth = max_depth;
         // this->regularization_factor = regularization_factor;
         // this->regularization_usedepth = regularization_usedepth;
         
         // Set number of samples and variables
         num_samples = data->getNumRows();
         num_independent_variables = data->getNumCols();
         
         // Set unordered factor variables
         // if (!prediction_mode) {
                 // data->setIsOrderedVariable(unordered_variable_names);
         // }
         
         initInternal();
         
         // // Init split select weights
         // split_select_weights.push_back(std::vector<double>());
         
         // Init manual inbag
         manual_inbag.push_back(std::vector<size_t>());
         
         // Check if mtry is in valid range
         // if (this->mtry > num_independent_variables) {
         //         throw std::runtime_error("mtry can not be larger than number of variables in data.");
         // }
         
         // Check if any observations samples
         if ((size_t) num_samples * sample_fraction[0] < 1) {
                 throw std::runtime_error("sample_fraction too small, no observations sampled.");
         }
         
         // Permute samples for corrected Gini importance
         // if (importance_mode == IMP_GINI_CORRECTED) {
         //         data->permuteSampleIDs(random_number_generator);
         // }
         
         // Order SNP levels if in "order" splitting
         // if (!prediction_mode && order_snps) {
         //         data->orderSnpLevels((importance_mode == IMP_GINI_CORRECTED));
         // }
         
         // Regularization
         // if (regularization_factor.size() > 0) {
         //         if (regularization_factor.size() == 1 && num_independent_variables > 1) {
         //                 double single_regularization_factor = regularization_factor[0];
         //                 this->regularization_factor.resize(num_independent_variables, single_regularization_factor);
         //         } else if (regularization_factor.size() != num_independent_variables) {
         //                 throw std::runtime_error("Use 1 or p (the number of predictor variables) regularization factors.");
         //         }
         //         
         //         // Set all variables to not used
         //         split_varIDs_used.resize(num_independent_variables, false);
         // }
 }
 
 void Forest::initInternal() {
         
         // If mtry not set, use floored square root of number of independent variables
         // if (mtry == 0) {
         //         unsigned long temp = sqrt((double) num_independent_variables);
         //         mtry = std::max((unsigned long) 1, temp);
         // }
         // 
         // Set minimal node size
         if (min_node_size == 0) {
                 min_node_size = DEFAULT_MIN_NODE_SIZE_REGRESSION;
         }
         
         // Error if beta splitrule used with data outside of [0,1]
         // if (splitrule == BETA && !prediction_mode) {
         //         for (size_t i = 0; i < num_samples; ++i) {
         //                 double y = data->get_y(i, 0);
         //                 if (y < 0 || y > 1) {
         //                         throw std::runtime_error("Beta splitrule applicable to regression data with outcome between 0 and 1 only.");
         //                 }
         //         }
         // }
         
         // Sort data if memory saving mode
         // if (!memory_saving_splitting) {
         //         data->sort();
         // }
 }
 
 void Forest::run(bool verbose, bool compute_oob_error) {
         
         if (prediction_mode) {
                 if (verbose && verbose_out) {
                         *verbose_out << "Predicting .." << std::endl;
                 }
                 // TODO: implement predict function
                 // predict();
         } else {
                 if (verbose && verbose_out) {
                         *verbose_out << "Growing trees .." << std::endl;
                 }
                 
                 grow();
                 
                 if (verbose && verbose_out) {
                         *verbose_out << "Computing prediction error .." << std::endl;
                 }
                 
                 // TODO: implement this part
                 // if (compute_oob_error) {
                 //         computePredictionError();
                 // }
                 
                 // unneccssary files
                 // if (importance_mode == IMP_PERM_BREIMAN || importance_mode == IMP_PERM_LIAW || importance_mode == IMP_PERM_RAW
                 //             || importance_mode == IMP_PERM_CASEWISE) {
                 //         if (verbose && verbose_out) {
                 //                 *verbose_out << "Computing permutation variable importance .." << std::endl;
                 //         }
                 //         computePermutationImportance();
                 // }
         }
 }
 
 
 } // namespace MOTE
 