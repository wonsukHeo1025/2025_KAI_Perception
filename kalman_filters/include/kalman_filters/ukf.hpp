#pragma once

#include <Eigen/Dense>
#include <memory>
#include "system_model.hpp"

class UKF {
public:
  UKF(std::shared_ptr<SystemModel> model, const Eigen::MatrixXd& P, double scale = 0.0);
  ~UKF();

  void init();
  void init(const Eigen::VectorXd& x);
  void predict(const Eigen::VectorXd &u);
  void update(const Eigen::VectorXd& y);
  const Eigen::VectorXd& get_state() const;
  const Eigen::MatrixXd& get_cov() const;

private:
  void ut(const Eigen::VectorXd& _x, const Eigen::MatrixXd& _P, std::vector<Eigen::VectorXd>& sigmaPt);

private:
  Eigen::MatrixXd P;
  std::shared_ptr<SystemModel> model;
  Eigen::MatrixXd P0;   // initial P.
  Eigen::MatrixXd K;    // kalman gain.
  Eigen::MatrixXd I;    // unit matrix.

  Eigen::VectorXd x_hat; // estimated state.

  std::vector<double> wc, ws; // weight.
  double lambda;
};