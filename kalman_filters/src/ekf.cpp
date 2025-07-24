#include <vector>
#include <cmath>
#include <Eigen/SVD>
#include "kalman_filters/ekf.hpp"

EKF::EKF(std::shared_ptr<SystemModel> _model, const Eigen::MatrixXd& _P)
: model(_model)
, P(_P)
, P0(_P)
, x_hat(_P.rows())
{
}

EKF::~EKF() {

}

void EKF::init() {

  P = P0;
  x_hat.setZero();
  I = Eigen::MatrixXd::Identity(P.rows(), P.rows());
}

void EKF::init(const Eigen::VectorXd& x) {

  init();
  x_hat = x;
}

void EKF::predict(const Eigen::VectorXd& u) {

  x_hat = model->DynamicsModel(x_hat, u);
  Eigen::MatrixXd A_ = model->JacobDynamicsModel(x_hat, u);
  P = A_ * P * A_.transpose() + model->Q;
}

void EKF::update(const Eigen::VectorXd& y) {

  Eigen::MatrixXd C_ = model->JacobObservationModel(x_hat);
  
  // Compute innovation covariance
  Eigen::MatrixXd S = C_ * P * C_.transpose() + model->R;
  
  // Check if S is invertible using condition number
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(S);
  double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
  
  if (cond > 1e10) {
    // Matrix is nearly singular, use pseudo-inverse
    double tolerance = 1e-10 * svd.singularValues()(0);
    K = P * C_.transpose() * svd.solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));
  } else {
    // Safe to use regular inverse
    K = P * C_.transpose() * S.inverse();
  }
  
  x_hat += K * (y - model->ObservationModel(x_hat));
  P = (I - K * C_) * P;
  
  // Ensure P remains symmetric
  P = (P + P.transpose()) / 2.0;
}

const Eigen::VectorXd& EKF::get_state() const { 
  return x_hat; 
}

const Eigen::MatrixXd& EKF::get_cov() const {
  return P;
}