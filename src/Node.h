#ifndef NODE_H_
#define NODE_H_


// ?? Put more library here
#include<limits>
#include <RcppArmadillo.h>

using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

namespace MOTE{

class Node{
public:
  Node():
  // Another option for initiaing split_valu is std::numeric_limits<double>::quiet_NaN()
  leaf(false), split_value(0), n1(0), n2(0){
  };
  
  // Node(arma::vec& Outcome_1, arma::vec& Outcome_2,
  //      arma::vec& xcenter,
  //      arma::vec& coefs,
  //      bool leaf,
  //      double split_value){
  //   this->Outcome_1 = Outcome_1;
  //   this->Outcome_2 = Outcome_2;
  //   this->xcenter = xcenter;
  //   this->coefs = coefs;
  //   this->leaf = leaf;
  //   this->split_value = split_value;
  // }
  

  
  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;
  
  // ?? Add setter and getter functions
  
  void set_leaf(bool value) {
    this->leaf = value;
  }
  
  void set_n(size_t n1, size_t n2) {
    this->n1 = n1;
    this->n2 = n2;
  }
  
  void set_sum(arma::vec sum1, arma::vec sum2){
    this->Outcome_1 = sum1;
    this->Outcome_2 = sum2;
  }
  
  void set_coef(arma::vec coefs){
    this->coefs = coefs;
  }
  
  // void set_center(arma::vec center){
  //   this->xcenter = center;
  // }
  
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
  
private:
  // Terminal node=true, internal node=false
  bool leaf;
  
  // arma::vec xcenter;
  arma::vec coefs;
  
  double split_value;
  
  size_t n1;
  size_t n2;
  
  arma::vec Outcome_1;
  arma::vec Outcome_2;
};

} // namespace MOTE

#endif /* NODE_H_ */