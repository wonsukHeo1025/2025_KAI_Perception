#pragma once
#include "system_model.hpp"

class SystemModelA : public SystemModel {

public:
  SystemModelA (
    const Eigen::VectorXd& x0,
    const Eigen::MatrixXd& _R, // Measurement noise covariance.
    const Eigen::MatrixXd& _Q, // Process noise covariance.
    double _dt = 1.0/30.0
  )
  : SystemModel(x0, _R, _Q, _dt)
  {
  }

  virtual Eigen::VectorXd DynamicsModel(const Eigen::VectorXd &x, const Eigen::VectorXd &u) {
    // State vector: [px, py, yaw, v]
    double px = x(0);
    double py = x(1);
    double yaw = x(2);
    double v = x(3);

    // Control input: [acceleration, yaw_rate]
    double acc = u(0);
    double yaw_v = u(1);

    // Simple kinematic bicycle model
    Eigen::VectorXd newX(4);
    newX(0) = px + v*dt*cos(yaw);   // x position
    newX(1) = py + v*dt*sin(yaw);   // y position
    newX(2) = yaw + yaw_v*dt;       // yaw angle
    newX(3) = v + acc * dt;         // velocity

    return newX;
  }

  virtual Eigen::VectorXd ObservationModel(const Eigen::VectorXd &x) {

		// std::cout << "ObservationModel" << std::endl;
    static Eigen::MatrixXd h(2, 4);
    h << 1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0;

    return h * x;
  }

  virtual Eigen::MatrixXd JacobDynamicsModel(const Eigen::VectorXd &x, const Eigen::VectorXd &u) {
    double yaw = x(2);
    double v = x(3);

    Eigen::MatrixXd jF(4, 4);
    jF << 1.0, 0.0, -dt*v*sin(yaw), dt*cos(yaw),
          0.0, 1.0,  dt*v*cos(yaw), dt*sin(yaw),
          0.0, 0.0, 1.0, 0.0, 
          0.0, 0.0, 0.0, 1.0;

    return jF;
  }

  virtual Eigen::MatrixXd JacobObservationModel(const Eigen::VectorXd &x) {

    Eigen::MatrixXd jH(2, 4);
    jH << 1.0, 0.0, 0.0, 0.0,
          0.0, 1.0, 0.0, 0.0;

    return jH;
  }
};

class SystemSimulator {

public:
  SystemSimulator(std::shared_ptr<SystemModel> model);
  ~SystemSimulator();

  void init();
  void step(Eigen::VectorXd& u);

private:
  std::shared_ptr<SystemModel> model;

public:
  std::vector<Eigen::VectorXd> true_results;
  std::vector<Eigen::VectorXd> obs_results;
};