/*-------------------------------------------------------------------------------
 This file is part of MOTE. Some of the infrastructure is based on ranger.
 
 Copyright (c) [2020-2021] [Boyi Guo]
#-------------------------------------------------------------------------------*/
 
 
#include  <RcppArmadillo.h>
 
#include "Tree.h"
#include "utility.h"
 
 using namespace arma;
 
 
 namespace MOTE {
 
 Tree::Tree() :
 data(0), num_samples(0),  min_node_size(0),
 minprop(DEFAULT_MINPROP), num_random_splits(DEFAULT_NUM_RANDOM_SPLITS), max_depth(DEFAULT_MAXDEPTH),
 keep_inbag(false), manual_inbag(0), sample_with_replacement(true), sample_fraction(0),
 num_samples_oob(0), oob_sampleIDs(0), 
 depth(0), last_left_nodeID(0),
 variable_importance(0) {
 }
 
 Tree::Tree(std::vector<std::vector<size_t>>& child_nodeIDs, 
            std::vector<Rcpp::List>& child_nodes
 ) :
 data(0), num_samples(0), min_node_size(0),
 minprop(DEFAULT_MINPROP), num_random_splits(DEFAULT_NUM_RANDOM_SPLITS), max_depth(DEFAULT_MAXDEPTH),
 keep_inbag(false), manual_inbag(0), sample_with_replacement(true), sample_fraction(0),
 num_samples_oob(0),oob_sampleIDs(0), 
 depth(0), child_nodeIDs(child_nodeIDs), last_left_nodeID(0),
 variable_importance(0) {
     
     // Convert type List child nodes to Nodes
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
 
 void Tree::init(const Data* data, 
                 size_t num_samples, uint seed, uint min_node_size,
                 bool sample_with_replacement, 
                 const std::vector<std::vector<size_t>>* sampleIDs_per_class,
                 std::vector<size_t>* manual_inbag, bool keep_inbag,
                 std::vector<double>* sample_fraction,  double minprop, 
                 uint num_random_splits,
                 uint max_depth) {
     
     this->data = data;
     this->num_samples = num_samples;
     this->sampleIDs_per_class = sampleIDs_per_class;
     this->min_node_size = min_node_size;
     this->sample_with_replacement = sample_with_replacement;
     this->manual_inbag = manual_inbag;
     this->keep_inbag = keep_inbag;
     this->sample_fraction = sample_fraction;
     this->minprop = minprop;
     this->num_random_splits = num_random_splits;
     this->max_depth = max_depth;
     

     child_nodeIDs.push_back(std::vector<size_t>());    // Vector for left child nodes
     child_nodeIDs.push_back(std::vector<size_t>());    // Vector for right child nodes
     // Create root node
     createEmptyNode();
     
     // Initialize random number generator and set seed
     random_number_generator.seed(seed);
 }
 
 void Tree::grow(vec* variable_importance) {
     
     // Rcpp::Rcout << "Start tree::grow" << std::endl;        // Debug Line
     
     this->variable_importance = variable_importance;
     
     if( sample_fraction->size() != 2) {
         throw std::runtime_error("sample_fraction could opnly contain 2 elements, one for each treatment arm.");
     }
     
     /*-----------------------------------------------------------------------
        Create In-bag samples
     #-----------------------------------------------------------------------*/
     // TODO: need to test setManualInbag(), bootstrapClassWise(), bootstrapWithoutReplacementClassWise(). These functions is working now.
     if (!manual_inbag->empty()) {
         setManualInbag();
     } else {
         if (sample_with_replacement) {
             // Rcpp::Rcout << "Start tree::bootstrapClassWise" << std::endl;        // Debug Line    
             bootstrapClassWise();
         } else {
             // Rcpp::Rcout << "Start tree::bootstrapWithoutReplacementClassWise" << std::endl;        // Debug Line         
             bootstrapWithoutReplacementClassWise();
         }
     }
     
     // Init start and end positions
     start_pos[0] = 0;
     end_pos[0] = sampleIDs.size();
     
     
     /*-----------------------------------------------------------------------
        Create Splits
     #-----------------------------------------------------------------------*/
     
     // While not all nodes terminal, split next node
     size_t num_open_nodes = 1;
     size_t i = 0;
     depth = 0;

     while (num_open_nodes > 0) {
         // Split node
         bool is_terminal_node = splitNode(i);
         if (is_terminal_node) {
             --num_open_nodes;
         } else {
             ++num_open_nodes;
             if (i >= last_left_nodeID) {
                 // NOTE: If new level, increase depth
                 // NOTE: (left_node saves left-most node in current level, new level reached if that node is splitted)
                 last_left_nodeID = child_nodes.size() - 2;
                 ++depth;
             }
         }
         ++i;
     }
     
     /*-----------------------------------------------------------------------
        Free Spaces
     #-----------------------------------------------------------------------*/
     // NOTE: It is possible Delete sampleID vector to save memory
     // sampleIDs.clear();
     // sampleIDs.shrink_to_fit();
     
     // Rcpp::Rcout << "End growing tree" << std::endl;        // Debug Line
 }
 
 
 bool Tree::splitNode(size_t nodeID) {
     
     // Rcpp::Rcout << "Start Split Node " << nodeID << std::endl;        // Debug Line
     
     bool stop = splitNodeInternal(nodeID);
     if (stop) {
         // Rcpp::Rcout << "Terminal node L231" << std::endl;        // Debug Line
         return true;
     }

     /*-----------------------------------------------------------------------
        Create Child Nodes
     #-----------------------------------------------------------------------*/
     size_t left_child_nodeID = child_nodes.size();
     child_nodeIDs[0][nodeID] = left_child_nodeID;
     createEmptyNode();
     start_pos[left_child_nodeID] = start_pos[nodeID];
     
     size_t right_child_nodeID = child_nodes.size();
     child_nodeIDs[1][nodeID] = right_child_nodeID;
     createEmptyNode();
     start_pos[right_child_nodeID] = end_pos[nodeID];
     // NOTE: Because of the swapping in the next section, start pos of the right child is the end pos
     
     
     /*-----------------------------------------------------------------------
        Sort Sample ID
     #-----------------------------------------------------------------------*/
     double split_value = child_nodes[nodeID]->get_value();
     vec tmp_coef = child_nodes[nodeID]-> get_coefs();
     
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
     
     // Not a terminal node
     return false;
 }
 
 
 bool Tree::splitNodeInternal(size_t nodeID) {
     
     size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
     
     //TODO: this part could be removed
     //NOTE: was set up trying to debug
     // if(start_pos.size() != end_pos.size()){                                   // Debug Line
     //     Rcpp::Rcout << "end_pos.size  "<< end_pos.size() << std::endl;        // Debug Line
     //     Rcpp::Rcout << "start_pos.size  "<< start_pos.size() << std::endl;        // Debug Line 
     //     throw std::runtime_error("Inconsistent end_pos & start_pos size");     // Debug Line
     // }                                                                          // Debug Line
     
     // finding all the sampleIDs in the current node
     // IntegerVector indices;
     // for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
     //     indices.push_back(sampleIDs[pos]);
     // }
     
     uvec indices(num_samples_node);
     size_t offset = start_pos[nodeID];
     for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
         indices(pos-offset) = sampleIDs[pos];
     }
     
     if(indices.n_elem == 0 || indices.has_nan()) std::runtime_error("Empty Indices");                // Debug Line
     
     // NOTE: idx_1, and idx_2 are the index of the IntegerVector indices,
     //       instead of the index in the original dataset.
     //       See example below when calculating sum
     // vec tmp_trt = data->get_trt(as<uvec>(indices));
     vec tmp_trt = data->get_trt(indices);
     uvec idx_1 = find(tmp_trt==1);
     uvec idx_2 = find(tmp_trt==-1);
     
     size_t n_outcome1 = idx_1.n_elem;
     size_t n_outcome2 = idx_2.n_elem;
     size_t n_1_unique = 0;      
     size_t n_2_unique = 0;
     
     size_t q = data->get_y_cols();
     size_t p = data->getNumCols();
     
     rowvec sum_outcome1 = rowvec( q, fill::zeros);
     rowvec sum_outcome2 = rowvec( q, fill::zeros);
     
     
     // Rcpp::Rcout << "Index Conversion seg faul starts" << std::endl;        // Debug Line
     //TODO: to remove later
     if(n_outcome1!=0){
         // uvec data_idx_1 = as<uvec>(indices[as<NumericVector>(wrap(idx_1))]);
         uvec data_idx_1 = indices.elem(idx_1);
         uvec tmp1 = unique(data_idx_1);
         n_1_unique = tmp1.n_elem;
         sum_outcome1 = colSums(data->get_y_diff_rows(data_idx_1));
     }
         
     if(n_outcome2!=0){                                       
         // uvec data_idx_2 = as<uvec>(indices[as<NumericVector>(wrap(idx_2))]);
         uvec data_idx_2 = indices.elem(idx_2);
         uvec tmp2 = unique(data_idx_2); 
         n_2_unique = tmp2.n_elem;
         sum_outcome2 = colSums(data->get_y_diff_rows(data_idx_2));
     }
     // Rcpp::Rcout << "Index Conversion seg faul starts" << std::endl;        // Debug Line
    
    /*-----------------------------------------------------------------------
        Base Cases
    #-----------------------------------------------------------------------*/
     if ( 
         std::min(n_1_unique, n_2_unique) <= std::max(p, q) || // TODO: added debug line
         std::min(n_outcome1, n_outcome2) <= std::max(p, q) || // Stop if sample size too small for CCA
         n_outcome2 == 0 || n_outcome1 == 0 ||       // Stop if no Samples in any Treatment Arm
         num_samples_node <= min_node_size ||       // Stop if maximum node size or depth reached
         (nodeID >= last_left_nodeID && max_depth > 0 && depth >= max_depth) // Stop if  depth reached
             ) {
         
         child_nodes[nodeID]->set_n(n_outcome1, n_outcome2);
         child_nodes[nodeID]->set_sum(sum_outcome1, sum_outcome2);
         return true;
     }
     
     
     /*-----------------------------------------------------------------------
          Find Best Split
     #-----------------------------------------------------------------------*/
     bool stop = findBestSplit(nodeID);
     if (stop) {
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
 
 
 bool Tree::findBestSplit(size_t nodeID) {
     
     /*-----------------------------------------------------------------------
        Fetching Data
     #-----------------------------------------------------------------------*/
     // NOTE: This is repetitive of in SplitNodeInternal
     IntegerVector indices;
     for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
         indices.push_back(sampleIDs[pos]);
     }
     vec tmp_trt = data->get_trt(as<uvec>(indices));
     uvec idx_1 = find(tmp_trt==1);
     uvec idx_2 = find(tmp_trt==-1);
     
     // Retrieve Matrices
     mat x_b = data->get_x_b_rows(as<uvec>(indices));
     mat x_diff = data->get_x_diff_rows(as<uvec>(indices));
     mat y_diff = data->get_y_diff_rows(as<uvec>(indices));
     
     size_t p = x_b.n_cols;
     size_t q = y_diff.n_cols;
     size_t n = x_b.n_rows;
     
     /*-----------------------------------------------------------------------
        Prepare Data:
            * Calculate Modified Outcomes
            * Centering Matrix
     #-----------------------------------------------------------------------*/
     mat x_b_centered = center(x_b); // center X_b
     mat T_Delta_X = center(times(x_diff,tmp_trt)); // center T_Delta_X
     mat T_Delta_Y = center(times(y_diff,tmp_trt)); // center T_Delta_Y
     
     
     /*-----------------------------------------------------------------------
        Matrix Augmentation
     #-----------------------------------------------------------------------*/
     mat left_mat = join_vert(
         join_horiz(x_b_centered, mat(n, p+q, fill::zeros)), // cbind(.x.b,matrix(0,nrow=n,ncol=p+q))
         join_horiz(x_b_centered, mat(n, p+q, fill::zeros)), // cbind(.x.b,matrix(0,nrow=n,ncol=p+q))
         join_horiz(                                         // cbind(matrix(0,nrow=n,ncol=p),T.delta.x, matrix(0,nrow=n,ncol=q))
             join_horiz(mat(n, p, fill::zeros),
                        T_Delta_X),
                        mat(n, q, fill::zeros)
         )
     );
     
     mat right_mat = join_vert(
         join_vert(
             join_horiz(mat(n, 2*p, fill::zeros), T_Delta_Y), // cbind(matrix(0,nrow=n,ncol=2*p),T.delta.y)
             join_horiz(                                      // cbind(matrix(0,nrow=n,ncol=p),T.delta.x, matrix(0,nrow=n,ncol=q))
                 join_horiz(
                     mat(n, p, fill::zeros),
                     T_Delta_X
                 ),
                 mat(n, q, fill::zeros)
             )
         ),
         join_horiz(mat(n, 2*p, fill::zeros), T_Delta_Y)     // cbind(matrix(0,nrow=n,ncol=2*p),T.delta.y)
     );
     
     // Error Prevention: Consistent Matrices Dimensions
     if((left_mat.n_cols!=right_mat.n_cols)|(left_mat.n_rows!=right_mat.n_rows)|(left_mat.n_rows!=3*n)){
         Rcpp::Rcout << "left_mat columns" << left_mat.n_cols << std::endl;         // Debug Line              
         Rcpp::Rcout << "left_mat rows" << left_mat.n_rows << std::endl;            // Debug Line 
         Rcpp::Rcout << "right_mat columns" << right_mat.n_cols << std::endl;       // Debug Line              
         Rcpp::Rcout << "right_mat rows" << right_mat.n_rows << std::endl;          // Debug Line 
         
         throw std::runtime_error("Inconsistent Dimension in Matrix Augmentation");
     }
     
     
     mat left = join_vert(left_mat , right_mat);
     mat right = join_vert(right_mat, left_mat);
     
     /*-----------------------------------------------------------------------
        CCA Step
     #-----------------------------------------------------------------------*/
     // TODO: find a C++ implementation of CCA, such that parallel computing is possible
     mat cca_res = cancor(left, right); 
     
     // Rcpp::Rcout << "CCA seg faul starts" << std::endl;        // Debug Line
     // Base case: degenerated CCA solution
     if(std::min(cca_res.n_cols, cca_res.n_rows) < (2*p+q)){    
         // Rcpp::Rcout << "CCA_res columns" << cca_res.n_cols << std::endl;        // Debug Line              
         // Rcpp::Rcout << "CCA_res rows" << cca_res.n_rows << std::endl;        // Debug Line                 
         // Rcpp::Rcout << "Terminal Node: Degernated CCA" << std::endl;        // Debug Line
         return true;
     }

     // Extract CCA coef for X_b and y_diff
     vec coef_x = (cca_res.col(0)).subvec(0, p-1);
     vec coef_y = (cca_res.col(0)).subvec(2*p, 2*p+q-1);
     // Rcpp::Rcout << "CCA seg faul ends" << std::endl;        // Debug Line

     // Calculate projections
     vec proj_x = x_b_centered * coef_x;
     vec proj_y = T_Delta_Y * coef_y;
     
     // Find Possible Splits
     vec split_can_1 = proj_x.elem(idx_1);
     vec split_can_2 = proj_x.elem(idx_2);    
     vec bnd_quantile = {minprop, 1-minprop};
     
     vec treat1_bnd = quantile(split_can_1, bnd_quantile);
     vec treat2_bnd = quantile(split_can_2, bnd_quantile);
     // Rcpp::Rcout << "trt1 bound: "<< treat1_bnd << std::endl;        // Debug Line
     // Rcpp::Rcout << "trt2 bound: "<< treat2_bnd << std::endl;        // Debug Line
     // Rcpp::Rcout << "trt1 bound[1]: "<< treat1_bnd[0] << std::endl;        // Debug Line 
     // Rcpp::Rcout << "trt1 bound[2]: "<< treat1_bnd[1] << std::endl;        // Debug Line 
     // Rcpp::Rcout << "trt2 bound[1]: "<< treat2_bnd[0] << std::endl;        // Debug Line 
     // Rcpp::Rcout << "trt2 bound[2]: "<< treat2_bnd[1] << std::endl;        // Debug Line     
     
     if(std::max(treat1_bnd(0), treat2_bnd(0)) > std::min(treat1_bnd(1), treat2_bnd(1))){
         // Rcpp::Rcout << "Terminal Node: LB > UB" << std::endl;        // Debug Line    
         return true;
     }
     
     vec split_can = unique(
         clamp(
             proj_x, std::max(treat1_bnd(0), treat2_bnd(0)),
             std::min(treat1_bnd(1), treat2_bnd(1))
         )
     );
     
     if(split_can.n_elem < 1){
         // Rcpp::Rcout << "Terminal Node: Not Enough Split_candidate" << std::endl;        // Debug Line    
         return true;
     }
     
     vec split_can_final = split_can.elem(randperm(split_can.n_elem, std::min(num_random_splits, split_can.n_elem)));
     // Rcpp::Rcout << "Split_can_final: " << split_can_final << std::endl;        // Debug Line 
     
     
     // Initiate variables for the final split results
     double best_decrease = -1;
     double best_value = 0;
     rowvec x_b_center = mean(x_b,0);
     
     findBestSplitValue(nodeID, 
                        proj_x, proj_y, split_can_final,
                        best_value, best_decrease);
     
     
     // Stop if no good split found
     if (best_decrease < 0) {
         // Rcpp::Rcout << "Terminal Node: Not decrease in variance reduction" << std::endl;        // Debug Line    
         return true;
     }
     
     // Rcpp::Rcout << "Best Value" << std::endl; // Debug Line
     
     
     // 
     // Save best values
     child_nodes[nodeID]->set_coef(coef_x);
     child_nodes[nodeID]->set_value(best_value + dot(x_b_center, coef_x));
     
     // uvec L_indices_proj = find(proj_x < best_value); // debug line
     // uvec R_indices_proj = find(proj_x >= best_value);// debug line
     // Rcpp::Rcout << "[Use Proj] # of Left Child is " << L_indices_proj.n_elem << std::endl;  // debug line
     // Rcpp::Rcout << "[Use Proj] # of Right Child is " << R_indices_proj.n_elem << std::endl; // debug line
     
     // Compute Variable Importance
     vec incrmt  = n * coef_x;
     (*variable_importance) += incrmt;
     
     return false;
 }
 
 void Tree::findBestSplitValue(size_t nodeID,
                               const vec& proj_x, const vec& proj_y, const vec& split_can_final,
                               double& best_value,
                               double& best_decrease) {
     
     size_t num_unique = split_can_final.n_elem;
     
     // For loop to iterate over each unique value
     for (size_t i = 0; i < num_unique; ++i) {
         
         // Make the cut
         uvec L_indices = find(proj_x <= split_can_final[i]);
         uvec R_indices = find(proj_x > split_can_final[i]);
         size_t L_length = L_indices.n_elem;
         size_t R_length = R_indices.n_elem;
         
         double decrease = -1;
         // calculate impurity
         if(L_length > 1 && R_length > 1)
            decrease = (L_length-1)*var(proj_y.elem(L_indices)) + (R_length-1)*var(proj_y.elem(R_indices));
         
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
 
 } // namespace MOTE
 