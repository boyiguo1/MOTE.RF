/*-------------------------------------------------------------------------------
 This file is part of ranger.
 
 Copyright (c) [2014-2018] [Marvin N. Wright]
 
 This software may be modified and distributed under the terms of the MIT license.
 
 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
#-------------------------------------------------------------------------------*/
 
#ifndef TREE_H_
#define TREE_H_
 //  
#include <vector>
#include <random>
 // #include <iostream>
 // #include <stdexcept>
 //  
#include "globals.h"
#include "Data.h"
#include "Node.h"
 
 namespace MOTE {
 
 class Tree {
 public:
    Tree();
    
    // Create from loaded forest
    // Tree(std::vector<std::vector<size_t>>& child_nodeIDs, std::vector<size_t>& split_varIDs,
    //      std::vector<double>& split_values);
    // 
    
    Tree(const Tree&) = delete;
    Tree& operator=(const Tree&) = delete;
    
    // Constructer for a tree, without any computation
    void init(const Data* data, // uint mtry,
              size_t num_samples, uint seed, uint min_node_size,
              bool sample_with_replacement, //bool memory_saving_splitting,
              //std::vector<double>* case_weights, 
              std::vector<size_t>* manual_inbag, bool keep_inbag,
              std::vector<double>* sample_fraction,  double minprop, bool holdout, //TODO: do we need holdout
              uint num_random_splits,
              uint max_depth);
    
    void grow(vec* variable_importance);
    
    void predict(const Data* prediction_data, bool oob_prediction);
    
    
 protected:
    
    void createEmptyNode();
    
    // Functions creating sample to training sample for each tree
    // Bootstrap / Manual
    void bootstrapClassWise();
    void bootstrapWithoutReplacementClassWise();
    void setManualInbag();
    
    // Clean up functions
    void cleanUpInternal();
    
    // Function to make splits
    bool splitNode(size_t nodeID);
    bool splitNodeInternal(size_t nodeID);
    bool findBestSplit(size_t nodeID);
    // Calculate the variance reduction
    void findBestSplitValue(size_t nodeID, // size_t varID, size_t num_classes,
                            // const std::vector<size_t>& class_counts, 
                            // size_t num_samples_node, 
                            const vec& proj_x, const vec& proj_y, const vec& split_can_final,
                            double& best_value, //rowvec& best_coefs,
                                double& best_decrease);
    
    // Update Variance Importance 
    void addImportance(size_t nodeID, const vec& coefs);
    
    
    // uint mtry;
    
    // Number of samples (all samples, not only inbag for this tree)
    size_t num_samples;
    
    // Number of OOB samples
    size_t num_samples_oob;
    
    // Minimum node size to split, like in original RF nodes of smaller size can be produced
    uint min_node_size;
    
    // Bootstrap weights
    // const std::vector<double>* case_weights;
    
    // Pre-selected bootstrap samples
    const std::vector<size_t>* manual_inbag;
    
    
    // Value to split at for each node, for now only binary split
    // For terminal nodes the prediction value is saved here
    std::vector<std::unique_ptr<Node>> child_nodes;
    
    // Vector of left and right child node IDs, 0 for no child
    std::vector<std::vector<size_t>> child_nodeIDs;
    
    // All sampleIDs in the tree, will be re-ordered while splitting
    std::vector<size_t> sampleIDs;
    
    // TODO: need to write the constructor for it.
    const std::vector<std::vector<size_t>>* sampleIDs_per_class;
    
    // For each node a vector with start and end positions
    std::vector<size_t> start_pos;
    std::vector<size_t> end_pos;
    
    // IDs of OOB individuals, sorted
    std::vector<size_t> oob_sampleIDs;
    
    // Holdout mode
     bool holdout;
    
    // Inbag counts
    bool keep_inbag;
    std::vector<size_t> inbag_counts;
    
    // Random number generator
    std::mt19937_64 random_number_generator;
    
    // Pointer to original data
    const Data* data;
    
    // Regularization
    // bool regularization;
    // std::vector<double>* regularization_factor;
    // bool regularization_usedepth;
    // std::vector<bool>* split_varIDs_used;
    
    // Variable importance for all variables
    vec* variable_importance;
    // ImportanceMode importance_mode;
    
    // When growing here the OOB set is used
    // Terminal nodeIDs for prediction samples
    std::vector<size_t> prediction_terminal_nodeIDs;
    
    bool sample_with_replacement;
    const std::vector<double>* sample_fraction;
    
    // TODO: remove the memory_save_splitting option
    bool memory_saving_splitting;
    // SplitRule splitrule;
    // double alpha;
    // NOTE: this is the same as leftout in previous implementation
    double minprop;
    uint num_random_splits;
    uint max_depth;
    uint depth;
    size_t last_left_nodeID;
 };
 
 } // namespace MOTE
 
#endif /* TREE_H_ */
 