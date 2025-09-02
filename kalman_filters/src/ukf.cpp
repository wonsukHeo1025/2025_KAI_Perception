#include <vector>
#include <cmath>
#include <Eigen/Cholesky>
#include <Eigen/SVD>
#include "kalman_filters/ukf.hpp"

UKF::UKF(
  std::shared_ptr<SystemModel> _model, 
  const Eigen::MatrixXd& _P, 
  double scale /*= 0*/
)
: model(_model)
, P(_P)
, P0(_P)
, x_hat(P.rows())
, lambda(scale)
{
  wc.reserve(2 * P.rows() + 1);
  ws.reserve(2 * P.rows() + 1);

  ws.push_back(lambda / (P.rows() + lambda));
  wc.push_back(lambda / (P.rows() + lambda));

  for (size_t i = 1; i < 2*P.rows()+1; i++) {
    double val = 1.0 / (2.0 * (lambda + P.rows()));
    ws.push_back(val);
    wc.push_back(val);
  }
}

UKF::~UKF() {

}

void UKF::init() {

  P = P0;
  x_hat.setZero();
  I = Eigen::MatrixXd::Identity(P.rows(), P.rows());
}

void UKF::init(const Eigen::VectorXd& x) {

  init();
  x_hat = x;
}

void UKF::ut(const Eigen::VectorXd& _x, const Eigen::MatrixXd& _P, std::vector<Eigen::VectorXd>& sigmaPt) {
  
  Eigen::MatrixXd L;
  
  // Try Cholesky decomposition first
  Eigen::LLT<Eigen::MatrixXd> chol(_P);
  if (chol.info() == Eigen::Success) {
    L = chol.matrixL();
  } else {
    // Fallback to SVD if Cholesky fails
    // Add small regularization to ensure positive definiteness
    Eigen::MatrixXd P_reg = _P + 1e-6 * Eigen::MatrixXd::Identity(_P.rows(), _P.cols());
    
    // Try Cholesky again with regularization
    Eigen::LLT<Eigen::MatrixXd> chol_reg(P_reg);
    if (chol_reg.info() == Eigen::Success) {
      L = chol_reg.matrixL();
    } else {
      // If still fails, use SVD decomposition
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(P_reg, Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::VectorXd s = svd.singularValues();
      
      // Ensure all singular values are positive
      for (int i = 0; i < s.size(); ++i) {
        if (s(i) < 1e-10) s(i) = 1e-10;
      }
      
      // Reconstruct square root matrix: L = U * sqrt(S)
      L = svd.matrixU() * s.cwiseSqrt().asDiagonal();
    }
  }

  sigmaPt.clear();
  sigmaPt.reserve(2 * _P.rows() + 1);

  sigmaPt.push_back(_x);  // Changed from x_hat to _x to fix bug
  double scale = sqrt(P.rows() + lambda);
  
  for (size_t i = 0; i < _P.rows(); i++) {
    Eigen::VectorXd preSigmaPt1 = _x + scale * L.col(i);
    Eigen::VectorXd preSigmaPt2 = _x - scale * L.col(i);
    sigmaPt.push_back(preSigmaPt1);
    sigmaPt.push_back(preSigmaPt2);
  }
}

void UKF::predict(const Eigen::VectorXd& u) {

  // Use thread_local for better performance in multi-threaded environments
  thread_local std::vector<Eigen::VectorXd> sigmaPt;
  sigmaPt.clear();
  ut(x_hat, P, sigmaPt);

  Eigen::VectorXd x_new(x_hat.rows());
  x_new.setZero();
  for (size_t i = 0; i < 2*P.rows() + 1; i++) {
    sigmaPt[i] = model->DynamicsModel(sigmaPt[i], u);
    x_new += ws[i] * sigmaPt[i];
  }
  Eigen::MatrixXd p_new(P.rows(), P.rows());
  p_new.setZero();
  for (size_t i = 0; i < 2*P.rows() + 1; i++) {
    p_new += wc[i] * (sigmaPt[i] - x_new) * (sigmaPt[i] - x_new).transpose();
  }

  x_hat = x_new;
  P = p_new + model->Q;

}

void UKF::update(const Eigen::VectorXd& y) {

  // Use thread_local for better performance
  thread_local std::vector<Eigen::VectorXd> sigmaPt;
  thread_local std::vector<Eigen::VectorXd> gamma;
  
  sigmaPt.clear();
  ut(x_hat, P, sigmaPt);

  gamma.clear();
  gamma.reserve(sigmaPt.size());
  gamma.resize(sigmaPt.size());

  Eigen::VectorXd z(y.rows());
  z.setZero();
  for (size_t i = 0; i < 2*P.rows() + 1; i++) {
    gamma[i] = model->ObservationModel(sigmaPt[i]); 
    z += ws[i] * gamma[i];
  }

  Eigen::MatrixXd P_obs(z.rows(), z.rows()), P_est(P.rows(), z.rows());
  P_obs.setZero();
  P_est.setZero();
  for (size_t i = 0; i < 2*P.rows() + 1; i++) {
    P_obs += wc[i] * (gamma[i] - z) * (gamma[i] - z).transpose();
    P_est += wc[i] * (sigmaPt[i] - x_hat) * (gamma[i] - z).transpose();
  }

  // Compute innovation covariance with measurement noise
  Eigen::MatrixXd S = P_obs + model->R;
  
  // Check if S is invertible using condition number
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(S);
  double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
  
  if (cond > 1e10) {
    // Matrix is nearly singular, use pseudo-inverse
    double tolerance = 1e-10 * svd.singularValues()(0);
    K = P_est * svd.solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));
  } else {
    // Safe to use regular inverse
    K = P_est * S.inverse();
  }
  
  x_hat += K * (y - z);
  P = P - K * P_est.transpose();
  
  // Ensure P remains symmetric
  P = (P + P.transpose()) / 2.0;
}

const Eigen::VectorXd& UKF::get_state() const { 
  return x_hat; 
}

const Eigen::MatrixXd& UKF::get_cov() const {
  return P;
}