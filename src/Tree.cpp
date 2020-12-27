/*-------------------------------------------------------------------------------
 This file is part of ranger.
 
 Copyright (c) [2014-2018] [Marvin N. Wright]
 
 This software may be modified and distributed under the terms of the MIT license.
 
 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
#-------------------------------------------------------------------------------*/
 
 // #include <iterator>
 
#include  <RcppArmadillo.h>
 
#include "Tree.h"
#include "utility.h"
 
 using namespace arma;
 
 
 namespace MOTE {
 
 Tree::Tree() :
 // mtry(0), 
 num_samples(0), num_samples_oob(0), min_node_size(0),
 // case_weights(0), 
 manual_inbag(0), oob_sampleIDs(0), //holdout(false),
 keep_inbag(false), data(0), 
 variable_importance(0), 
 sample_with_replacement(true), sample_fraction(0), memory_saving_splitting(false), minprop(DEFAULT_MINPROP),
 num_random_splits(DEFAULT_NUM_RANDOM_SPLITS), max_depth(DEFAULT_MAXDEPTH), depth(0), last_left_nodeID(0) {
 }
 
 Tree::Tree(std::vector<std::vector<size_t>>& child_nodeIDs, 
            std::vector<Rcpp::List>& child_nodes
            //std::vector<size_t>& split_varIDs,std::vector<double>& split_values
            ) :
   num_samples(0), num_samples_oob(0), min_node_size(0), manual_inbag(0), //split_varIDs(split_varIDs), split_values(split_values), 
   child_nodeIDs(child_nodeIDs), oob_sampleIDs(
         0), holdout(false), keep_inbag(false), data(0), variable_importance(0),  sample_with_replacement(true), sample_fraction(
                 0), memory_saving_splitting(false), minprop(
                     DEFAULT_MINPROP), num_random_splits(DEFAULT_NUM_RANDOM_SPLITS), max_depth(DEFAULT_MAXDEPTH), depth(0), last_left_nodeID(
                         0) {
   // make List child nodes to Nodes
   size_t tmp_n = child_nodes.size();
   this->child_nodes.reserve(tmp_n);
   for(size_t i = 0; i < tmp_n; ++i){
     
     arma::rowvec Outcome_1(as<NumericVector>((child_nodes[i])["Outcome_1"]));
     arma::rowvec Outcome_2(as<NumericVector>((child_nodes[i])["Outcome_2"]));
     arma::vec coefs = (as<NumericVector>((child_nodes[i])["coefs"]));
     
     this->child_nodes.push_back(
       make_unique<Node>(Outcome_1, Outcome_2,
                         coefs,
                         as<double>((child_nodes[i])["split_value"]),
                         as<size_t>((child_nodes[i])["n1"]),
                         as<size_t>((child_nodes[i])["n2"]))
     );
   }
   
 }
 
 void Tree::init(const Data* data, // uint mtry,
                 size_t num_samples, uint seed, uint min_node_size,
                 bool sample_with_replacement, //bool memory_saving_splitting,
                 // std::vector<double>* case_weights,
                 const std::vector<std::vector<size_t>>* sampleIDs_per_class,
                 std::vector<size_t>* manual_inbag, bool keep_inbag,
                 std::vector<double>* sample_fraction,  double minprop, bool holdout,
                 uint num_random_splits,
                 uint max_depth) {
         
         this->data = data;
         // this->mtry = mtry;
         this->num_samples = num_samples;
         this->sampleIDs_per_class = sampleIDs_per_class;;
         // this->memory_saving_splitting = memory_saving_splitting;
         
         // Create root node, assign bootstrap sample and oob samples
         child_nodeIDs.push_back(std::vector<size_t>());
         child_nodeIDs.push_back(std::vector<size_t>());
         createEmptyNode();
         
         // Initialize random number generator and set seed
         random_number_generator.seed(seed);
         
         this->min_node_size = min_node_size;
         this->sample_with_replacement = sample_with_replacement;
         // this->case_weights = case_weights;
         this->manual_inbag = manual_inbag;
         this->keep_inbag = keep_inbag;
         this->sample_fraction = sample_fraction;
         this->holdout = holdout;
         this->minprop = minprop;
         this->num_random_splits = num_random_splits;
         this->max_depth = max_depth;
 }
 
 void Tree::grow(vec* variable_importance) {
         Rcpp::Rcout << "Start tree::grow" << std::endl;        // Debug Line

         // Allocate memory for tree growing
         // TODO: figure out what to do with this
         // allocateMemory();           // I don't think this is necessary

         this->variable_importance = variable_importance;

         // Bootstrap, dependent if weighted or not and with or without replacement
         // NOTE: Could implement in the future
         // if (!case_weights->empty()) {
         //         if (sample_with_replacement) {
         //                 bootstrapWeighted();
         //         } else {
         //                 bootstrapWithoutReplacementWeighted();
         //         }
         // } 
         
         if(sample_fraction->size() !=2) {
                 throw std::runtime_error("sample_fraction must contains only 2 elements");
         }


         if (!manual_inbag->empty()) {
                 // TODO: Need to debug
                 setManualInbag();
         } else {
                 if (sample_with_replacement) {
                         // Rcpp::Rcout << "Start tree::bootstrapClassWise" << std::endl;        // Debug Line    
                         bootstrapClassWise();
                 } else {
                         // TODO: Need to debug
                         // Rcpp::Rcout << "Start tree::bootstrapWithoutReplacementClassWise" << std::endl;        // Debug Line         
                         bootstrapWithoutReplacementClassWise();
                 }
         }
         
         // Init start and end positions
         start_pos[0] = 0;
         end_pos[0] = sampleIDs.size();
         
         // While not all nodes terminal, split next node
         size_t num_open_nodes = 1;
         size_t i = 0;
         depth = 0;
         
         // Rcpp::Rcout << "Starting creating Split" << std::endl;       //Debug Line
         
         while (num_open_nodes > 0) {
                 // Split node
                 // TODO: debug the node splitting
                 bool is_terminal_node = splitNode(i);
                 if (is_terminal_node) {
                         --num_open_nodes;
                 } else {
                         ++num_open_nodes;
                         if (i >= last_left_nodeID) {
                                 // If new level, increase depth
                                 // (left_node saves left-most node in current level, new level reached if that node is splitted)
                                 
                                 // last_left_nodeID = split_varIDs.size() - 2;
                                 // NOTE:: I changed from split_varIDs.size() to child_nodes.size() They should both work
                                 last_left_nodeID = child_nodes.size() - 2;
                                 ++depth;
                         }
                 }
                 // Rcpp::Rcout << "Resolve a Node" << std::endl;        // Debug Line
                 ++i;
         }
         // 
         // Delete sampleID vector to save memory
         // TODO: figure out if more space can be freed
         // sampleIDs.clear();
         // sampleIDs.shrink_to_fit();
         // cleanUpInternal();
         
         Rcpp::Rcout << "End growing tree" << std::endl;        // Debug Line
 }
 

 bool Tree::splitNode(size_t nodeID) {
         // 
         // // Select random subset of variables to possibly split at
         // Boyi:: Doesn't apply to our methods. We don't needs mtry
         // std::vector<size_t> possible_split_varIDs;
         // createPossibleSplitVarSubset(possible_split_varIDs);

         // Rcpp::Rcout << "Split Node Started" << std::endl;        // Debug Line
         
         // // Call subclass method, sets split_varIDs and split_values
         bool stop = splitNodeInternal(nodeID);
         if (stop) {
                 // Rcpp::Rcout << "Terminal node L231" << std::endl;        // Debug Line
                 return true;
         }
         // 
         // size_t split_varID = split_varIDs[nodeID];
         
         // Rcpp::Rcout << "Node ID" << nodeID << std::endl;        // Debug Line
         // Rcpp::Rcout << "child nodes size" << child_nodes.size() << std::endl;        // Debug Line
         
         
         double split_value = child_nodes[nodeID]->get_value();
         vec tmp_coef = child_nodes[nodeID]-> get_coefs();
         
         // Rcpp::Rcout << "Got Split  Value" << std::endl;        // Debug Line
         
         // Create child nodes
         size_t left_child_nodeID = child_nodes.size();
         child_nodeIDs[0][nodeID] = left_child_nodeID;
         createEmptyNode();
         start_pos[left_child_nodeID] = start_pos[nodeID];

         size_t right_child_nodeID = child_nodes.size();
         child_nodeIDs[1][nodeID] = right_child_nodeID;
         createEmptyNode();
         start_pos[right_child_nodeID] = end_pos[nodeID];
         
         // Rcpp::Rcout << "Finished Creating Child Noe" << std::endl;        // Debug Line
         
         // For each sample in node, assign to left or right child
         // Ordered: left is <= splitval and right is > splitval
         size_t pos = start_pos[nodeID];
         while (pos < start_pos[right_child_nodeID]) {
                 size_t sampleID = sampleIDs[pos];
                 double tmp_proj_x = dot(data->get_x_b_rows(sampleID), tmp_coef);
                 // If proj value smaller than split_value
                 if ( tmp_proj_x <= split_value) {
                         // If going to left, do nothing
                         ++pos;
                 } else {
                         // If going to right, move to right end
                         --start_pos[right_child_nodeID];
                         std::swap(sampleIDs[pos], sampleIDs[start_pos[right_child_nodeID]]);
                 }
         }
         
         // End position of left child is start position of right child
         end_pos[left_child_nodeID] = start_pos[right_child_nodeID];
         end_pos[right_child_nodeID] = end_pos[nodeID];

         // Rcpp::Rcout << "After Sording start pos and end pos" << std::endl;        // Debug Line
         
         // No terminal node
         return false;
 }
 
 bool Tree::splitNodeInternal(size_t nodeID) {
         
         size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
         // Rcpp::Rcout << "Internal Splitting in Node ID "<< nodeID << std::endl;        // Debug Line

         if(start_pos.size() != end_pos.size()){
                 Rcpp::Rcout << "end_pos.size  "<< end_pos.size() << std::endl;        // Debug Line
                 Rcpp::Rcout << "start_pos.size  "<< start_pos.size() << std::endl;        // Debug Line 
                 throw std::runtime_error("Inconsistent end_pos & start_pos size");
         }

         // finding all the sampleIDs in the current node
         IntegerVector indices;
         for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
                 // Rcpp::Rcout << "SampleID  "<< sampleIDs[pos] << std::endl;        // Debug Line
                 indices.push_back(sampleIDs[pos]);
         }
         // Rcpp::Rcout << "# of Samples: " << indices.size() << std::endl;        // Debug Line
         // Rcpp::Rcout << "After getting y diff rows for datasets " << data->n_y_diff_rows() << std::endl;       // Debug Line
         
         vec tmp_trt = data->get_trt(as<uvec>(indices));
         // Rcpp::Rcout << "After Finding Treatmnents" << std::endl;        // Debug Line
         
         // The following index variables are the index fo the indices
         uvec idx_1 = find(tmp_trt==1);
         uvec idx_2 = find(tmp_trt==-1);
         
         size_t n_outcome1 = idx_1.n_elem;
         size_t n_outcome2 = idx_2.n_elem;
         // Rcpp::Rcout << "After Calculate outcome SIze" << std::endl;        // Debug Line
         rowvec sum_outcome1 = colSums(data->get_y_diff_rows(as<uvec>(indices[as<NumericVector>(wrap(idx_1))])));
         // Rcpp::Rcout << "After Calculate Average SIze for outcome 1" << std::endl;        // Debug Line
         // Rcpp::Rcout << idx_2 << std::endl;        // Debug Line
         rowvec sum_outcome2 = colSums(data->get_y_diff_rows(as<uvec>(indices[as<NumericVector>(wrap(idx_2))])));

         
         // Rcpp::Rcout << "Summairze Outcomes" << std::endl;        // Debug Line
         // Stop if maximum node size or depth reached
         if (num_samples_node <= min_node_size || (nodeID >= last_left_nodeID && max_depth > 0 && depth >= max_depth)) {
                 
                 // child_nodes[nodeID]->set_leaf(true);
                 child_nodes[nodeID]->set_n(n_outcome1, n_outcome2);
                 child_nodes[nodeID]->set_sum(sum_outcome1, sum_outcome2);
                 return true;
         }
         
         // Rcpp::Rcout << "Before Find Best Split" << std::endl;        // Debug Line
         // TODO: debug findBestSplit(nodeID) function
         bool stop = findBestSplit(nodeID);
         
         if (stop) {
                 // child_nodes[nodeID]->set_leaf(true);
                 child_nodes[nodeID]->set_n(n_outcome1, n_outcome2);
                 child_nodes[nodeID]->set_sum(sum_outcome1, sum_outcome2);
                 return true;
         }
         
         // Rcpp::Rcout << "End SplitNodeInternal" << std::endl;        // Debug Line
         
         return false;
 }
 
 
 
 void Tree::createEmptyNode() {
         child_nodeIDs[0].push_back(0);
         child_nodeIDs[1].push_back(0);
         start_pos.push_back(0);
         end_pos.push_back(0);
         
         child_nodes.push_back(make_unique<Node>());
 }
 
 void Tree::bootstrapClassWise() {
         // Number of samples is sum of sample fraction * number of samples
         size_t num_samples_inbag = 0;
         double sum_sample_fraction = 0;
         for (auto& s : *sample_fraction) {
                 num_samples_inbag += (size_t) num_samples * s;
                 sum_sample_fraction += s;
         }
         
         // Reserve space, reserve a little more to be save)
         sampleIDs.reserve(num_samples_inbag);
         oob_sampleIDs.reserve(num_samples * (exp(-sum_sample_fraction) + 0.1));
         
         // Start with all samples OOB
         inbag_counts.resize(num_samples, 0);
         
         // Draw samples for each class
         for (size_t i = 0; i < sample_fraction->size(); ++i) {
                 // Draw samples of class with replacement as inbag and mark as not OOB
                 size_t num_samples_class = (*sampleIDs_per_class)[i].size();
                 size_t num_samples_inbag_class = round(num_samples * (*sample_fraction)[i]);
                 std::uniform_int_distribution<size_t> unif_dist(0, num_samples_class - 1);
                 for (size_t s = 0; s < num_samples_inbag_class; ++s) {
                         size_t draw = (*sampleIDs_per_class)[i][unif_dist(random_number_generator)];
                         sampleIDs.push_back(draw);
                         ++inbag_counts[draw];
                 }
         }
         
         // Save OOB samples
         for (size_t s = 0; s < inbag_counts.size(); ++s) {
                 if (inbag_counts[s] == 0) {
                         oob_sampleIDs.push_back(s);
                 }
         }
         num_samples_oob = oob_sampleIDs.size();
         
         if (!keep_inbag) {
                 inbag_counts.clear();
                 inbag_counts.shrink_to_fit();
         }
 }
 
 void Tree::bootstrapWithoutReplacementClassWise() {
         // Draw samples for each class
         for (size_t i = 0; i < sample_fraction->size(); ++i) {
                 size_t num_samples_class = (*sampleIDs_per_class)[i].size();
                 size_t num_samples_inbag_class = round(num_samples * (*sample_fraction)[i]);
                 
                 shuffleAndSplitAppend(sampleIDs, oob_sampleIDs, num_samples_class, num_samples_inbag_class,
                                       (*sampleIDs_per_class)[i], random_number_generator);
         }
         num_samples_oob = oob_sampleIDs.size();
         
         if (keep_inbag) {
                 // All observation are 0 or 1 times inbag
                 inbag_counts.resize(num_samples, 1);
                 for (size_t i = 0; i < oob_sampleIDs.size(); i++) {
                         inbag_counts[oob_sampleIDs[i]] = 0;
                 }
         }
 }
 
 void Tree::setManualInbag() {
         // Select observation as specified in manual_inbag vector
         sampleIDs.reserve(manual_inbag->size());
         inbag_counts.resize(num_samples, 0);
         for (size_t i = 0; i < manual_inbag->size(); ++i) {
                 size_t inbag_count = (*manual_inbag)[i];
                 if ((*manual_inbag)[i] > 0) {
                         for (size_t j = 0; j < inbag_count; ++j) {
                                 sampleIDs.push_back(i);
                         }
                         inbag_counts[i] = inbag_count;
                 } else {
                         oob_sampleIDs.push_back(i);
                 }
         }
         num_samples_oob = oob_sampleIDs.size();
         
         // Shuffle samples
         std::shuffle(sampleIDs.begin(), sampleIDs.end(), random_number_generator);
         
         if (!keep_inbag) {
                 inbag_counts.clear();
                 inbag_counts.shrink_to_fit();
         }
 }
 
 // TODO: need some customization here
 void Tree::cleanUpInternal() {
         // counter.clear();
         // counter.shrink_to_fit();
         // counter_per_class.clear();
         // counter_per_class.shrink_to_fit();
         // sums.clear();
         // sums.shrink_to_fit();
 }
 
 bool Tree::findBestSplit(size_t nodeID) {
         
         // size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
         
         // Retrieve Data
         IntegerVector indices;
         for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
                 indices.push_back(sampleIDs[pos]);
         }
         
         // Rcpp::Rcout << "Finding Best Split for Node IDs" << std::endl;        // Debug Line
         
         // Retrieve data
         vec tmp_trt = data->get_trt(as<uvec>(indices));
         uvec idx_1 = find(tmp_trt==1);
         uvec idx_2 = find(tmp_trt==-1);
         
         
         // Rcpp::Rcout << "# of idx_1 " << idx_1.n_elem << std::endl;        // Debug Line
         // Rcpp::Rcout << "# of idx_2 " << idx_2.n_elem << std::endl;        // Debug Line     
         // Rcpp::Rcout << "idx_1" << idx_1 << std::endl;        // Debug Line  
         // Rcpp::Rcout << "idx_2" << idx_2  << std::endl;        // Debug Line          
         
         // size_t n_outcome1 = idx_1.n_elem;
         // size_t n_outcome2 = idx_2.n_elem;
         
         
         // Retrieve X_b
         mat x_b = data->get_x_b_rows(as<uvec>(indices));
         // Retrieve x_diff;
         // mat x_e = (data->get_x_e_rows(as<uvec>(indices)));
         mat x_diff = data->get_x_diff_rows(as<uvec>(indices));
         // Retrieve y_diff
         mat y_diff = data->get_y_diff_rows(as<uvec>(indices));
         
         // size_t ncol_x_b = x_b.n_cols;
         
         
         size_t p = x_b.n_cols;
         size_t q = y_diff.n_cols;
         size_t n = x_b.n_rows;
         
         // Centering
         mat x_b_centered = center(x_b); // center X_b
         mat T_Delta_X = center(times(x_diff,tmp_trt)); // center T_Delta_X
         mat T_Delta_Y = center(times(y_diff,tmp_trt)); // center T_Delta_Y
         
         // Rcpp::Rcout << "Before Matrix Augmentation" << std::endl;        // Debug Line
         
         // Create concatenated matrix
         mat left_mat = join_vert(
                 // join_vert(
                         join_horiz(x_b_centered, mat(n, p+q, fill::zeros)), // cbind(.x.b,matrix(0,nrow=n,ncol=p+q))
                         join_horiz(x_b_centered, mat(n, p+q, fill::zeros)) // cbind(.x.b,matrix(0,nrow=n,ncol=p+q))
                 // )
                 ,
                 join_horiz( // cbind(matrix(0,nrow=n,ncol=p),T.delta.x, matrix(0,nrow=n,ncol=q))
                         join_horiz(mat(n, p, fill::zeros),
                                    T_Delta_X),
                                    mat(n, q, fill::zeros)
                 )
         );
         mat right_mat = join_vert(
                 join_vert(
                         join_horiz(mat(n, 2*p, fill::zeros), T_Delta_Y), //  cbind(matrix(0,nrow=n,ncol=2*p),T.delta.y)
                         join_horiz( // cbind(matrix(0,nrow=n,ncol=p),T.delta.x, matrix(0,nrow=n,ncol=q))
                                 join_horiz(mat(n, p, fill::zeros),
                                            T_Delta_X),
                                            mat(n, q, fill::zeros)
                         ) 
                 ),
                 join_horiz(mat(n, 2*p, fill::zeros), T_Delta_Y)//  cbind(matrix(0,nrow=n,ncol=2*p),T.delta.y)
         );
         // Rcpp::Rcout << "After Matrix Augmentation" << std::endl;        // Debug Line
         
         if((left_mat.n_cols!=right_mat.n_cols)|(left_mat.n_rows!=right_mat.n_rows)|(left_mat.n_rows!=3*n)){
                 Rcpp::Rcout << "left_mat columns" << left_mat.n_cols << std::endl;        // Debug Line              
                 Rcpp::Rcout << "left_mat rows" << left_mat.n_rows << std::endl;        // Debug Line 
                 
                 Rcpp::Rcout << "right_mat columns" << right_mat.n_cols << std::endl;        // Debug Line              
                 Rcpp::Rcout << "right_mat rows" << right_mat.n_rows << std::endl;        // Debug Line 
                 
                 throw std::runtime_error("Inconsistent Dimension in Matrix Augmentation");
         }

         
         mat left = join_vert(left_mat , right_mat);
         mat right = join_vert(right_mat, left_mat);
        
         // Run CCA
         // mat cca_res = cancor(join_vert(left_mat , right_mat),
         //                      join_vert(right_mat, left_mat));
         mat cca_res = cancor(left, right);         // Debug Line  
         // Rcpp::Rcout << "After CCA" << std::endl;        // Debug Line         
         // if degenerate solution, stop
         if(std::min(cca_res.n_cols, cca_res.n_rows) < (2*p+q)){
                 // Rcpp::Rcout << "CCA_res columns" << cca_res.n_cols << std::endl;        // Debug Line              
                 // Rcpp::Rcout << "CCA_res rows" << cca_res.n_rows << std::endl;        // Debug Line                 
                 // Rcpp::Rcout << "Terminal Node: Degernated CCA" << std::endl;        // Debug Line    
                 return true;
         }
         

         
         // Extract CCA coef for X_b and y_diff
         vec coef_x = (cca_res.col(0)).subvec(0, p-1);
         vec coef_y = (cca_res.col(0)).subvec(2*p, 2*p+q-1);
         // Calculate projections
         vec proj_x = x_b_centered * coef_x;
         vec proj_y = T_Delta_Y * coef_y;
         
         vec tmp_proj = (unique(proj_x));       // Debug Line
         // Rcpp::Rcout << "# of Unique Proj Values: " << tmp_proj.n_elem << std::endl;        // Debug Line
         
         // TODO: extract this part to a function
         // Find Possible Splits
         // vec split_can_1 = unique(proj_x.elem(as<uvec>(indices[as<NumericVector>(wrap(idx_1))])));
         vec split_can_1 = unique(proj_x.elem(idx_1));                                                  // Debug Line
         // Rcpp::Rcout << "# of Unique Proj Values for trt 1: " << split_can_1.n_elem << std::endl;        // Debug Line
         // vec split_can_2 = unique(proj_x.elem(as<uvec>(indices[as<NumericVector>(wrap(idx_2))])));
         vec split_can_2 = unique(proj_x.elem(idx_2));                                                  // Debug Line    
         // Rcpp::Rcout << "# of Unique Proj Values for trt 2: " << split_can_2.n_elem << std::endl;        // Debug Line
         vec bnd_quantile = {minprop, 1-minprop};
         // Rcpp::Rcout << "bound Quantile: "<< bnd_quantile << std::endl;        // Debug Line     
         
         vec treat1_bnd = quantile(split_can_1, bnd_quantile);
         // Rcpp::Rcout << "trt1 bound: "<< treat1_bnd << std::endl;        // Debug Line 
         vec treat2_bnd = quantile(split_can_2, bnd_quantile);
         // Rcpp::Rcout << "trt2 bound: "<< treat2_bnd << std::endl;        // Debug Line 
         // Rcpp::Rcout << "trt1 bound[1]: "<< treat1_bnd[0] << std::endl;        // Debug Line 
         // Rcpp::Rcout << "trt1 bound[2]: "<< treat1_bnd[1] << std::endl;        // Debug Line 
         // Rcpp::Rcout << "trt2 bound[1]: "<< treat2_bnd[0] << std::endl;        // Debug Line 
         // Rcpp::Rcout << "trt2 bound[2]: "<< treat2_bnd[1] << std::endl;        // Debug Line     
         
         if(std::max(treat1_bnd(0), treat2_bnd(0)) > std::min(treat1_bnd(1), treat2_bnd(1))){
            Rcpp::Rcout << "Terminal Node: LB > UB" << std::endl;        // Debug Line    
            return true;
         }

         vec split_can = unique(
                 clamp(
                         proj_x, std::max(treat1_bnd(0), treat2_bnd(0)),
                         std::min(treat1_bnd(1), treat2_bnd(1))
                 )
         );
         
         if(split_can.n_elem < 2){
                 Rcpp::Rcout << "Terminal Node: Not Enough Split_candidate" << std::endl;        // Debug Line    
                 return true;
         }
         
         // Rcpp::Rcout << "# of Qualified Split_can: "<< split_can.n_elem << std::endl;        // Debug Line    
         // Rcpp::Rcout << "Qualified Split_can: "<< split_can << std::endl;        // Debug Line 
         // Limit number of splits
         vec split_can_final = split_can.elem(randperm(split_can.n_elem, std::min(num_random_splits, split_can.n_elem)));
  
          // Rcpp::Rcout << "# of Randpom Split_can: "<< split_can_final.n_elem << std::endl;        // Debug Line          
         
         // Initiate variables for the final split results
         double best_decrease = -1;
         // vec best_coefs(ncol_x_b);
         double best_value = 0;
         rowvec x_b_center = mean(x_b,0);
         
         // TODO: calculate the variance reduction & Choose the best split
         findBestSplitValue(nodeID, //varID, sum_node, num_samples_node,
                            proj_x, proj_y, split_can_final,
                            best_value, best_decrease);
         
         
         
         // Stop if no good split found
         if (best_decrease < 0) {
                 Rcpp::Rcout << "Terminal Node: Not decrease in variance reduction" << std::endl;        // Debug Line    
                 return true;
         }
         
         // Rcpp::Rcout << "Best Value" << std::endl; 
         
         // 
         // Save best values
         // split_varIDs[nodeID] = best_varID;
         // split_values[nodeID] = best_value;
         // child_nodes[nodeID]->set_coef(best_coefs);
         child_nodes[nodeID]->set_coef(coef_x);
         // child_nodes[nodeID]->set_center(x_b_center);
         child_nodes[nodeID]->set_value(best_value + dot(x_b_center, coef_x));
         // child_nodes[nodeID]->set_leaf(false);
         
         // Rcpp::Rcout << "Best Coef is " << coef_x << std::endl;  // debug line
         // Rcpp::Rcout << "x_b_center is " << x_b_center << std::endl;  // debug line  
         // Rcpp::Rcout << "dot prod is " << dot(x_b_center, coef_x) << std::endl;  // debug line  
         
         // Rcpp::Rcout << "Best Split is " << best_value << std::endl;  // debug line
         // Rcpp::Rcout << "Best Split after is " << best_value + dot(x_b_center, coef_x) << std::endl; // debug line
         
         // uvec L_indices_proj = find(proj_x < best_value); // debug line
         // uvec R_indices_proj = find(proj_x >= best_value);// debug line
         // Rcpp::Rcout << "[Use Proj] # of Left Child is " << L_indices_proj.n_elem << std::endl;  // debug line
         // Rcpp::Rcout << "[Use Proj] # of Right Child is " << R_indices_proj.n_elem << std::endl; // debug line
         
         // uvec L_indices = find(x_b*coef_x < best_value + dot(x_b_center, coef_x)); // debug line
         // uvec R_indices = find(x_b*coef_x >= best_value + dot(x_b_center, coef_x));// debug line
         // Rcpp::Rcout << "[add center] # of Left Child is " << L_indices.n_elem << std::endl;  // debug line
         // Rcpp::Rcout << "[add center] # of Right Child is " << R_indices.n_elem << std::endl; // debug line
         
         
         // throw std::runtime_error("Finished calculating Best Splits");          // debug line
         
         // 
         // Compute Varaible Importance
         // addImportance(nodeID, coef_x);  // TO Delete
         // Rcpp::Rcout <<  "Sample Size" << child_nodes[nodeID]->get_samplesize() << std::endl; // Debug Line
         // Rcpp::Rcout << coef_x << std::endl; // Debug Line
         vec incrmt  = n * coef_x;
         // Rcpp::Rcout << "incrmt:" << incrmt << std::endl; // Debug Line
         (*variable_importance) += incrmt;
         // Rcpp::Rcout << "variable_importance:" << *variable_importance << std::endl; // Debug Line
         
         return false;
 }
 
 // TODO: Debug this function
 void Tree::findBestSplitValue(size_t nodeID, // size_t varID, size_t num_classes,
                               // const std::vector<size_t>& class_counts, 
                               // size_t num_samples_node, 
                               const vec& proj_x, const vec& proj_y, const vec& split_can_final,
                               double& best_value, //rowvec& best_coefs,
                               double& best_decrease) {
         
         
         size_t num_unique = split_can_final.n_elem;
         
         // For loop to iterate over each unique value
         for (size_t i = 0; i < num_unique - 1; ++i) {
                 
                 // Make the cut
                 uvec L_indices = find(proj_x <= split_can_final[i]);
                 uvec R_indices = find(proj_x > split_can_final[i]);
                 size_t L_length = L_indices.n_elem;
                 size_t R_length = R_indices.n_elem;
                 
                 // calculate impurity
                 double decrease = (L_length-1)*var(proj_y.elem(L_indices)) + (R_length-1)*var(proj_y.elem(R_indices));
                 
                 // Check with best
                 // If better than before, use this
                 if (decrease > best_decrease) {
                         
                         best_value = split_can_final[i];
                         best_decrease = decrease;
                 }
         }
         
 }
 
 
 /* returns the terminal node ID*/
 void Tree::predict(const Data* prediction_data, bool oob_prediction) {
         
         size_t num_samples_predict;
         if (oob_prediction) {
                 num_samples_predict = num_samples_oob;
         } else {
                 num_samples_predict = prediction_data->getNumRows();
         }
         
         prediction_terminal_nodeIDs.resize(num_samples_predict, 0);
         
         // For each sample start in root, drop down the tree and return final value
         for (size_t i = 0; i < num_samples_predict; ++i) {
                 size_t sample_idx;
                 if (oob_prediction) {
                         sample_idx = oob_sampleIDs[i];
                 } else {
                         sample_idx = i;
                 }
                 size_t nodeID = 0;
                 while (1) {
                         
                         // Break if terminal node
                         if (child_nodeIDs[0][nodeID] == 0 && child_nodeIDs[1][nodeID] == 0) {
                                 break;
                         }
                         
                         // Move to child
                         // TODO: delete this line
                         // size_t split_varID = split_varIDs[nodeID];
                         
                         double value = dot(prediction_data->get_x_b_rows(sample_idx), child_nodes[nodeID]->get_coefs());
                         // if (prediction_data->isOrderedVariable(split_varID)) {
                         if (value <= child_nodes[nodeID]->get_value()) {
                                 // Move to left child
                                 nodeID = child_nodeIDs[0][nodeID];
                         } else {
                                 // Move to right child
                                 nodeID = child_nodeIDs[1][nodeID];
                         }
                         // } else {
                         //         size_t factorID = floor(value) - 1;
                         //         size_t splitID = floor(split_values[nodeID]);
                         //         
                         //         // Left if 0 found at position factorID
                         //         if (!(splitID & (1ULL << factorID))) {
                         //                 // Move to left child
                         //                 nodeID = child_nodeIDs[0][nodeID];
                         //         } else {
                         //                 // Move to right child
                         //                 nodeID = child_nodeIDs[1][nodeID];
                         //         }
                         // }
                 }
                 
                 prediction_terminal_nodeIDs[i] = nodeID;
         }
 }
 
 // TODO: could be removed
 // void Tree::addImportance(size_t nodeID, const vec& coefs){
 //   
 //        // if((child_nodes[nodeID]->get_samplesize())==0)    // Debug Line
 //        //   throw std::runtime_error("sample size is 0");   // Debug Line
 //        Rcpp::Rcout <<  "Sample Size" << child_nodes[nodeID]->get_samplesize() << std::endl; // Debug Line
 //   
 //   Rcpp::Rcout << coefs << std::endl; // Debug Line
 //   
 //         vec incrmt  = (child_nodes[nodeID]->get_samplesize()) * coefs;
 //        Rcpp::Rcout << "incrmt:" << incrmt << std::endl; // Debug Line
 //        
 //         
 //         (*variable_importance) += incrmt;
 // }
 // 
 

 } // namespace MOTE
 