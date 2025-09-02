#pragma once

#include <Eigen/Dense>
#include <memory>
#include "system_model.hpp"

class EKF {

public:
  EKF(std::shared_ptr<SystemModel> model, const Eigen::MatrixXd& P);
  ~EKF();

  void init();
  void init(const Eigen::VectorXd& x);
  void predict(const Eigen::VectorXd& u);
  void update(const Eigen::VectorXd& y);
  const Eigen::VectorXd& get_state() const;
  const Eigen::MatrixXd& get_cov() const;

private:
  Eigen::MatrixXd P;
  std::shared_ptr<SystemModel> model;
  Eigen::MatrixXd P0;   // initial P.
  Eigen::MatrixXd K;    // kalman gain.
  Eigen::MatrixXd I;    // unit matrix.

  Eigen::VectorXd x_hat; // estimated state.
};