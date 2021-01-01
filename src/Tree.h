/*-------------------------------------------------------------------------------
 This file is part of MOTE. Some of the infrastructure is based on ranger.
 
 Copyright (c) [2020-2021] [Boyi Guo]
#-------------------------------------------------------------------------------*/
 
#ifndef TREE_H_
#define TREE_H_
 
#include <vector>
#include <random>
 // #include <iostream>
 // #include <stdexcept>
 
#include "globals.h"
#include "Data.h"
#include "Node.h"
 
 namespace MOTE {
 
 class Tree {
 public:
     Tree();
     
     // Create from loaded forest
     Tree(std::vector<std::vector<size_t>>& child_nodeIDs,
          std::vector<Rcpp::List>& child_nodes
              // std::vector<size_t>& split_varIDs,std::vector<double>& split_values
     );
     
     Tree(const Tree&) = delete;
     Tree& operator=(const Tree&) = delete;
     
     // Update Parameters of tree after initialization (No computation)
     void init(const Data* data, 
               size_t num_samples, uint seed, uint min_node_size,
               bool sample_with_replacement, 
               const std::vector<std::vector<size_t>>* sampleIDs_per_class,
               std::vector<size_t>* manual_inbag, bool keep_inbag,
               std::vector<double>* sample_fraction,  double minprop, //bool holdout, //TODO: do we need holdout
               uint num_random_splits,
               uint max_depth);
     
     void grow(vec* variable_importance);
     
     void predict(const Data* prediction_data, bool oob_prediction);
     
     /*-----------------------------------------------------------------------
      Getter Functions
    #-----------------------------------------------------------------------*/
     const std::vector<size_t>& getOobSampleIDs() const {
         return oob_sampleIDs;
     }
     
     size_t getNumSamplesOob() const {
         return num_samples_oob;
     }
     const std::vector<size_t>& getInbagCounts() const {
         return inbag_counts;
     }
     
     //TODO: return type may need to be a pointer, since child_nodes contains unique pointers
     Node* getPrediction(size_t sampleID) const {
         // Node* getPrediction(size_t sampleID) {
         size_t terminal_nodeID = prediction_terminal_nodeIDs[sampleID];
         return child_nodes[terminal_nodeID].get();
     }
     
     std::vector<Rcpp::List> getNodes() const{
         std::vector<Rcpp::List> result;
         for(auto& node : child_nodes){
             Rcpp::List tmp;
             tmp.push_back(node->get_value(), "split_value");
             tmp.push_back(node->get_coefs(), "coefs");
             tmp.push_back(node->get_n1(), "n1");
             tmp.push_back(node->get_n2(), "n2");
             tmp.push_back(node->get_outcome1(), "Outcome_1");
             tmp.push_back(node->get_outcome2(), "Outcome_2");
             result.push_back(tmp);
             
         }
         return result;
         
     }
     
     // std::vector<std::unique_ptr<Node>>& getNodes(){
     //     return child_nodes;
     // }
     
     
     const std::vector<std::vector<size_t>>& getChildNodeIDs() const {
         return child_nodeIDs;
     }
     
     
 protected:
     
     void createEmptyNode();
     
     // Functions creating sample to training sample for each tree
     // Bootstrap / Manual
     void bootstrapClassWise();
     void bootstrapWithoutReplacementClassWise();
     void setManualInbag();
     
     // Clean up functions
     // TODO: seems useless function and be removed
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
     // void addImportance(size_t nodeID, const vec& coefs);
     
     /*------------------ Functions Ends --------------------*/
     
     /*------------------ Variables Starts --------------------*/
     
     /*-----------------------------------------------------------------------
     Tree Parameters
    #-----------------------------------------------------------------------*/
     const Data* data;               // Pointer to original data
     size_t num_samples;             // Number of samples (all samples, not only inbag for this tree)
     uint min_node_size;
     double minprop;                 // proportion of minimum samples in each class
     uint num_random_splits;
     uint max_depth;
     
     // TODO: need to write the constructor for it.
     const std::vector<std::vector<size_t>>* sampleIDs_per_class;
     
     // Bootstrapping & Inbag
     bool keep_inbag;
     std::vector<size_t> inbag_counts;
     // Pre-selected bootstrap samples
     const std::vector<size_t>* manual_inbag;
     bool sample_with_replacement;
     const std::vector<double>* sample_fraction;

     // Out-of-bag samples
     size_t num_samples_oob;         // Number of OOB samples
     std::vector<size_t> oob_sampleIDs;
     
     
     std::mt19937_64 random_number_generator;        // Random number generator
     
     
     /*------------------ Tree Structure --------------------*/
     /*-----------------------------------------------------------------------
      Each tree is stored as of vectors:
      * integer: depth of the tree
      * Vector of node ID
      * Vector of node information (split value, split coefficients, etc.)
      * Vector of ID of samples used in this tree
      * Vector of start_pos & end_pos (samples in the corresponding node)
     #-----------------------------------------------------------------------*/

      // All sampleIDs that are used to construct tree, are re-ordered while splitting
      std::vector<size_t> sampleIDs;
      
      uint depth;                                         // Depth of the tree
      
      std::vector<std::vector<size_t>> child_nodeIDs;     // Vector of left and right child node IDs, 0 for no child
      std::vector<std::unique_ptr<Node>> child_nodes;     // Information related to each node

      std::vector<size_t> start_pos;                      // For each node a vector with start and end positions
      std::vector<size_t> end_pos;                        // containing the sample in this node
      
      size_t last_left_nodeID;
      
      
      /*------------------ Variable Importance --------------------*/
      vec* variable_importance;   // Variable importance for all variables
      
      
      /*------------------ Prediction --------------------*/
      std::vector<size_t> prediction_terminal_nodeIDs;      // Terminal node IDs for each predicting sample
      
 };
 
 } // namespace MOTE
 
#endif /* TREE_H_ */
 