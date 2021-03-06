#ifndef NODE_H_
#define NODE_H_


#include<limits>
#include <RcppArmadillo.h>

using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

namespace MOTE{

class Node{
public:
  Node():
  // TODO: making split_value to NA as a way to differentiate internal node and terminal node
  split_value(0), n1(0), n2(0){
  };
  
  Node(arma::rowvec& Outcome_1, arma::rowvec& Outcome_2,
       arma::vec& coefs,
       double split_value,
       size_t n1,
       size_t n2){
    this->n1 = n1;
    this->n2 = n2;
    this->Outcome_1 = Outcome_1;
    this->Outcome_2 = Outcome_2;
    this->coefs = coefs;
    this->split_value = split_value;
  }
  
  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;
  
  /*-----------------------------------------------------------------------
   Setter & Getter Functions
  #-----------------------------------------------------------------------*/
  
  void set_n(size_t n1, size_t n2) {
    this->n1 = n1;
    this->n2 = n2;
  }
  
  void set_sum(arma::rowvec sum1, arma::rowvec sum2){
    this->Outcome_1 = sum1;
    this->Outcome_2 = sum2;
  }
  
  void set_coef(arma::vec coefs){
    this->coefs = coefs;
  }
  
  void set_value(double val){
    this->split_value = val;
  }
  
  double get_value(){
    return(this->split_value);
  }
  
  arma::vec get_coefs(){
    return(this->coefs);
  }
  
  size_t get_samplesize(){
    return(this->n1 + this->n2);
  }
  
  size_t get_n1(){
    return(this->n1);
  }
  
  size_t get_n2(){
    return(this->n2);
  }
  
  rowvec get_outcome1(){
    return(this->Outcome_1);
  }
  
  rowvec get_outcome2(){
    return(this->Outcome_2);
  }
  
private:
  
  arma::vec coefs;
  
  double split_value;
  
  size_t n1;
  size_t n2;
  
  arma::rowvec Outcome_1;
  arma::rowvec Outcome_2;
};

} // namespace MOTE

#endif /* NODE_H_ */
