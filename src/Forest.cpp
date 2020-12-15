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
                 minprop(DEFAULT_MINPROP), num_threads(DEFAULT_NUM_THREADS), data { }, //overall_prediction_error(NAN),
                 progress(0) {
                 }
 
 void Forest::initR(std::unique_ptr<Data> input_data, //uint mtry, 
                    uint num_trees, std::ostream* verbose_out, uint seed,
                    uint num_threads, //ImportanceMode importance_mode, 
                    uint min_node_size,
                    // std::vector<std::vector<double>>& split_select_weights,
                    // const std::vector<std::string>& always_split_variable_names, 
                    bool prediction_mode, bool sample_with_replacement,
                    // const std::vector<std::string>& unordered_variable_names, bool memory_saving_splitting, SplitRule splitrule,
                    // std::vector<double>& case_weights, 
                    std::vector<std::vector<size_t>>& manual_inbag, bool predict_all,
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
         // if (!case_weights.empty()) {
         //         if (case_weights.size() != num_samples) {
         //                 throw std::runtime_error("Number of case weights not equal to number of samples.");
         //         }
         //         this->case_weights = case_weights;
         // }
         // 
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
         
         // Set number of samples and variables
         num_samples = data->getNumRows();
         num_independent_variables = data->getNumCols();
         
         
         initInternal();
         
         
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
         
         
         // create class value and class ID
         // TODO: TO DELTE
         // if (!prediction_mode) {
         //         for (size_t i = 0; i < num_samples; ++i) {
         //                 double value = data->get_trt(i, 0);
         //                 
         //                 // If classID is already in class_values, use ID. Else create a new one.
         //                 uint classID = find(class_values.begin(), class_values.end(), value) - class_values.begin();
         //                 if (classID == class_values.size()) {
         //                         class_values.push_back(value);
         //                 }
         //                 response_classIDs.push_back(classID);
         //         }
         // }
         
         
         // Hard code for creation of sampleIDs_per_class
         vec tmp_trt = data->get_trt();
         uvec idx_1 = find(tmp_trt==1); // Indices for trtment lvl 1 (ref level)
         uvec idx_2 = find(tmp_trt==-1); // Indices for trtment lvl 2
         
         sampleIDs_per_class.resize(sample_fraction.size());   // How to be 2
         
         sampleIDs_per_class[0] = conv_to<std::vector<size_t>>::from(idx_1);
         sampleIDs_per_class[1] =  conv_to<std::vector<size_t>>::from(idx_2);       
         
         // Create sampleIDs_per_class i.e. trtment
         // if (sample_fraction.size() > 1) {
         //         sampleIDs_per_class.resize(sample_fraction.size());
         //         for (auto& v : sampleIDs_per_class) {
         //                 v.reserve(num_samples);
         //         }
         //         for (size_t i = 0; i < num_samples; ++i) {
         //                 // Somehow, change this to trt levels_num of 0, 1s
         //                 size_t classID = response_classIDs[i];
         //                 sampleIDs_per_class[classID].push_back(i);
         //         }
         // }
         
         // TODO: Delete this section
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
                 if (compute_oob_error) {
                         computePredictionError();
                 }
                 
                 // unneccssary files
                 // TODO: Delete this part
                 // if (importance_mode == IMP_PERM_BREIMAN || importance_mode == IMP_PERM_LIAW || importance_mode == IMP_PERM_RAW
                 //             || importance_mode == IMP_PERM_CASEWISE) {
                 //         if (verbose && verbose_out) {
                 //                 *verbose_out << "Computing permutation variable importance .." << std::endl;
                 //         }
                 //         computePermutationImportance();
                 // }
         }
 }
 
 void Forest::grow() {
         
         // Create thread ranges
         equalSplit(thread_ranges, 0, num_trees - 1, num_threads);
         
         // Init trees, create a seed for each tree, based on main seed
         // // Allocate space for tree vector
         trees.reserve(num_trees);
         for (size_t i = 0; i < num_trees; ++i) {
                 trees.push_back(make_unique<Tree>());
         }
         
         
         std::uniform_int_distribution<uint> udist;
         for (size_t i = 0; i < num_trees; ++i) {
                 uint tree_seed;
                 if (seed == 0) {
                         tree_seed = udist(random_number_generator);
                 } else {
                         tree_seed = (i + 1) * seed;
                 }
                 
                 // Get inbag counts for tree
                 std::vector<size_t>* tree_manual_inbag;
                 if (manual_inbag.size() > 1) {
                         tree_manual_inbag = &manual_inbag[i];
                 } else {
                         tree_manual_inbag = &manual_inbag[0];
                 }
                 // TODO: adding sampleIDs_per_class to initiator of tree
                 trees[i]->init(data.get(), num_samples, tree_seed, min_node_size,
                                sample_with_replacement, //memory_saving_splitting, 
                                // TODO: do we need case_weights?
                                // &case_weights,
                                &sampleIDs_per_class,
                                tree_manual_inbag, keep_inbag,
                                &sample_fraction, minprop, holdout, num_random_splits, max_depth);
         }
         
         *verbose_out << "Finish Initialize "<< num_trees<<" Trees in Grow" << std::endl;               //Debug line
         
         // Init variable importance
         // variable_importance.resize(num_independent_variables, 0);
         variable_importance = vec(num_independent_variables, fill::zeros);
         
         // Debug case when there is only tree
         // trees[1]->grow(&variable_importance); //Debug line
         // *verbose_out << variable_importance << std::endl;               //Debug line
         // throw std::runtime_error("Finish Building the example tree");   //Debug line
         
         // *verbose_out << "Initialize Variable_imporantace" << std::endl;               //Debug line
         
         // Grow trees in multiple threads
#ifdef OLD_WIN_R_BUILD
         // #nocov start
         progress = 0;
         clock_t start_time = clock();
         clock_t lap_time = clock();
         for (size_t i = 0; i < num_trees; ++i) {
                 trees[i]->grow(&variable_importance);
                 progress++;
                 showProgress("Growing trees..", start_time, lap_time);
         }
         // *verbose_out << "End Building Tree Single Thred" << std::endl;               //Debug line
         // #nocov end
#else
         progress = 0;
#ifdef R_BUILD
         aborted = false;
         aborted_threads = 0;
#endif
         
         std::vector<std::thread> threads;
         threads.reserve(num_threads);
         
         // Initialize importance per thread
         std::vector<vec> variable_importance_threads(num_threads);
         *verbose_out << "Start to Fit Tree Parallell" << std::endl;               //Debug line
         for (uint i = 0; i < num_threads; ++i) {
                 // TODO: Do I need to create for each one
                 // Yes I need
                 // if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED || importance_mode == IMP_GINI_OOB) {
                 variable_importance_threads[i] = vec(num_independent_variables, fill::zeros);
                 // }
                 threads.emplace_back(&Forest::growTreesInThread, this, i, &(variable_importance_threads[i]));
         }
         showProgress("Growing trees..", num_trees);
         for (auto &thread : threads) {
                 thread.join();
         }
         
#ifdef R_BUILD
         if (aborted_threads > 0) {
                 throw std::runtime_error("User interrupt.");
         }
#endif
         
         // Sum thread importances
         //          if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED || importance_mode == IMP_GINI_OOB) {
         variable_importance = vec(num_independent_variables, fill::zeros);
         for (size_t i = 0; i < num_independent_variables; ++i) {
                 for (uint j = 0; j < num_threads; ++j) {
                         variable_importance[i] += variable_importance_threads[j][i];
                 }
         }
         // TODO: remove useless vec
         //                  variable_importance_threads.clear();
         // }
         
#endif
         //          
         //          // Divide importance by number of trees
         //          if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED || importance_mode == IMP_GINI_OOB) {
         //                  for (auto& v : variable_importance) {
         //                          v /= num_trees;
         //                  }
         //          }
         // *verbose_out << "End Growing Trees" << std::endl;               //Debug line
 }
 
 
 
#ifndef OLD_WIN_R_BUILD
 // TODO: change variable_important to Vec * if keeping
 void Forest::growTreesInThread(uint thread_idx, vec* variable_importance) {
         if (thread_ranges.size() > thread_idx + 1) {
                 for (size_t i = thread_ranges[thread_idx]; i < thread_ranges[thread_idx + 1]; ++i) {
                         trees[i]->grow(variable_importance);
                         
                         // Check for user interrupt
#ifdef R_BUILD
                         if (aborted) {
                                 std::unique_lock<std::mutex> lock(mutex);
                                 ++aborted_threads;
                                 condition_variable.notify_one();
                                 return;
                         }
#endif
                         
                         // Increase progress by 1 tree
                         std::unique_lock<std::mutex> lock(mutex);
                         *verbose_out << "Finish Growing Tree " << i << std::endl;         // Debug Line
                         ++progress;
                         condition_variable.notify_one();
                 }
         }
 }
#endif
 
#ifdef OLD_WIN_R_BUILD
 // #nocov start
 void Forest::showProgress(std::string operation, clock_t start_time, clock_t& lap_time) {
         
         // Check for user interrupt
         if (checkInterrupt()) {
                 throw std::runtime_error("User interrupt.");
         }
         
         double elapsed_time = (clock() - lap_time) / CLOCKS_PER_SEC;
         if (elapsed_time > STATUS_INTERVAL) {
                 double relative_progress = (double) progress / (double) num_trees;
                 double time_from_start = (clock() - start_time) / CLOCKS_PER_SEC;
                 uint remaining_time = (1 / relative_progress - 1) * time_from_start;
                 // if (verbose_out) {
                 if (true) { // debugline
                         *verbose_out << operation << " Progress: " << round(100 * relative_progress)
                                      << "%. Estimated remaining time: " << beautifyTime(remaining_time) << "." << std::endl;
                 }
                 lap_time = clock();
         }
 }
 // #nocov end
#else
 void Forest::showProgress(std::string operation, size_t max_progress) {
         using std::chrono::steady_clock;
         using std::chrono::duration_cast;
         using std::chrono::seconds;
         
         steady_clock::time_point start_time = steady_clock::now();
         steady_clock::time_point last_time = steady_clock::now();
         std::unique_lock<std::mutex> lock(mutex);
         
         // Wait for message from threads and show output if enough time elapsed
         while (progress < max_progress) {
                 condition_variable.wait(lock);
                 seconds elapsed_time = duration_cast<seconds>(steady_clock::now() - last_time);
                 
                 // Check for user interrupt
#ifdef R_BUILD
                 if (!aborted && checkInterrupt()) {
                         aborted = true;
                 }
                 if (aborted && aborted_threads >= num_threads) {
                         return;
                 }
#endif
                 
                 if (progress > 0 && elapsed_time.count() > STATUS_INTERVAL) {
                         double relative_progress = (double) progress / (double) max_progress;
                         seconds time_from_start = duration_cast<seconds>(steady_clock::now() - start_time);
                         uint remaining_time = (1 / relative_progress - 1) * time_from_start.count();
                         if (verbose_out) {
                                 *verbose_out << operation << " Progress: " << round(100 * relative_progress) << "%. Estimated remaining time: "
                                              << beautifyTime(remaining_time) << "." << std::endl;
                         }
                         last_time = steady_clock::now();
                 }
         }
 }
#endif
 
 
 void Forest::computePredictionError() {
         
         // Predict trees in multiple threads
#ifdef OLD_WIN_R_BUILD
         // #nocov start
         progress = 0;
         clock_t start_time = clock();
         clock_t lap_time = clock();
         for (size_t i = 0; i < num_trees; ++i) {
                 trees[i]->predict(data.get(), true);
                 progress++;
                 showProgress("Predicting..", start_time, lap_time);
         }
         // #nocov end
#else
         std::vector<std::thread> threads;
         threads.reserve(num_threads);
         progress = 0;
         for (uint i = 0; i < num_threads; ++i) {
                 threads.emplace_back(&Forest::predictTreesInThread, this, i, data.get(), true);
         }
         showProgress("Computing prediction error..", num_trees);
         for (auto &thread : threads) {
                 thread.join();
         }
         
#ifdef R_BUILD
         if (aborted_threads > 0) {
                 throw std::runtime_error("User interrupt.");
         }
#endif
#endif
         Rcpp::Rcout << "Finished Tree Predicting" << std::endl;        // Debug Line
         
         
         // Call special function for subclasses
         computePredictionErrorInternal();
 }
 
 void Forest::predictTreesInThread(uint thread_idx, const Data* prediction_data, bool oob_prediction) {
         if (thread_ranges.size() > thread_idx + 1) {
                 for (size_t i = thread_ranges[thread_idx]; i < thread_ranges[thread_idx + 1]; ++i) {
                         trees[i]->predict(prediction_data, oob_prediction);
                         
                         // Check for user interrupt
#ifdef R_BUILD
                         if (aborted) {
                                 std::unique_lock<std::mutex> lock(mutex);
                                 ++aborted_threads;
                                 condition_variable.notify_one();
                                 return;
                         }
#endif
                         
                         // Increase progress by 1 tree
                         std::unique_lock<std::mutex> lock(mutex);
                         ++progress;
                         condition_variable.notify_one();
                 }
         }
 }
 
 
 /* OOB is calculated by averaging the predicted treat effect  by the number of trees*/
 void Forest::computePredictionErrorInternal() {
         
         // For each sample sum over trees where sample is OOB
         // TODO: need to revise this part
         std::vector<size_t> samples_oob_count;

         // predictions = std::vector<std::vector<std::vector<double>>>(1,
         //                                                             std::vector<std::vector<double>>(1, std::vector<double>(num_samples, 0)));
        //
        
        size_t q = data->get_y_cols();
        
        mat outcome_1 = mat(num_samples, q, fill::zeros);
        mat outcome_2 = mat(num_samples, q, fill::zeros);
        uvec size_1 = uvec(num_samples, fill::zeros);
        uvec size_2 = uvec(num_samples, fill::zeros);
         samples_oob_count.resize(num_samples, 0);
         
         Rcpp::Rcout << "After init space" << std::endl;        // Debug Line
         for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
                 for (size_t sample_idx = 0; sample_idx < trees[tree_idx]->getNumSamplesOob(); ++sample_idx) {
                         size_t sampleID = trees[tree_idx]->getOobSampleIDs()[sample_idx];
                         
                         Node* tmp_node = getTreePrediction(tree_idx, sample_idx);
                         
                         //update weighted ave info
                         
                         // predictions[0][0][sampleID] += value;
                         outcome_1.row(sampleID) += tmp_node -> get_outcome1();
                         outcome_2.row(sampleID) += tmp_node -> get_outcome2();
                         size_1(sampleID) += tmp_node->get_n1();
                         size_2(sampleID) += tmp_node->get_n2();
                         
                         ++samples_oob_count[sampleID];
                 }
         }
         
         Rcpp::Rcout << "After retrieving data" << std::endl;        // Debug Line
         // TODO: update Prediction
         // TODO: Allocate memory for prediction & Update the definition in Forest.h
         predictions = mat(num_samples, q, fill::zeros);
         for (size_t i = 0; i < predictions.n_rows; ++i) {
                 if (samples_oob_count[i] > 0) {
                         // Update Prediction with diff of the weighted average
                         predictions.row(i) = (outcome_2.row(i)/size_1(i)) - (outcome_1.row(i)/size_1(i));
                         // TODO: remove followings;
                         // predictions[0][0][i] /= (double) samples_oob_count[i];
                         // double predicted_value = predictions[0][0][i];
                         // double real_value = data->get_y(i, 0);
                 } else {
                         // fill with NAN's
                         // predictions.row(i) = NAN;           // REmove
                         // Rcpp::Rcout << "Setting Nan for prediction" << std::endl;        // Debug Line
                         // predictions.row(i) = datum::nan;
                         predictions.row(i).fill(datum::nan);
                         // Rcpp::Rcout << "Success" << std::endl;        // Debug Line
                 }
         }
         

         
 }
 
 // TODO update return type to a list
 Node* Forest::getTreePrediction(size_t tree_idx, size_t sample_idx) const {
         // TODO: test if the non-casting version works
         const auto& tree = dynamic_cast<const Tree&>(*trees[tree_idx]);
         return tree.getPrediction(sample_idx);
 }
 
 
 } // namespace MOTE
 