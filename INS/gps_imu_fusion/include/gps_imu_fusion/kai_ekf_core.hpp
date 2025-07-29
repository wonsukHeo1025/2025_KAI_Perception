#ifndef KAI_EKF_CORE_HPP
#define KAI_EKF_CORE_HPP

#include <stdint.h>
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <tuple>
#include <mutex>
#include <shared_mutex>
#include "gps_imu_fusion/imu_calibration_v2.hpp"

namespace kai {

struct EkfParams {

  float accel_noise = 0.05f;        
  float gyro_noise = 0.00175f;      
  float accel_bias_noise = 0.01f;   
  float gyro_bias_noise = 0.00025f; 
  float accel_bias_tau = 100.0f;    
  float gyro_bias_tau = 50.0f;      

  float gps_pos_noise_ne = 3.0f;    
  float gps_pos_noise_d = 6.0f;     
  float gps_vel_noise_ne = 0.5f;    
  float gps_vel_noise_d = 1.0f;     

  float init_pos_unc = 10.0f;       
  float init_vel_unc = 1.0f;        
  float init_att_unc = 0.34906f;    
  float init_hdg_unc = 3.14159f;    
  float init_accel_bias_unc = 0.981f; 
  float init_gyro_bias_unc = 0.01745f; 
};

struct GpsCoordinate {
  double lat;  
  double lon;  
  double alt;  
};

struct GpsVelocity {
  double vN;  
  double vE;  
  double vD;  
};

struct ImuData {
  float gyroX;  
  float gyroY;  
  float gyroZ;  
  float accX;   
  float accY;   
  float accZ;   
  float hX;     
  float hY;     
  float hZ;     
};

constexpr float GRAVITY = 9.807f;
constexpr double ECCENTRICITY_SQ = 0.0066943799901;
constexpr double EARTH_RADIUS = 6378137.0;

class KaiEkfCore {
public:
  KaiEkfCore();

  void setParameters(const EkfParams& params);
  
  bool initialized() const { return initialized_; }
  
  float getPitch_rad() const { return theta; }
  float getRoll_rad() const { return phi; }
  float getHeading_rad() const { return psi; }
  float getHeadingConstrainAngle180_rad() const { return constrainAngle180(psi); }
  double getLatitude_rad() const { return lat_ins; }
  double getLongitude_rad() const { return lon_ins; }
  double getAltitude_m() const { return alt_ins; }
  double getVelNorth_ms() const { return vn_ins; }
  double getVelEast_ms() const { return ve_ins; }
  double getVelDown_ms() const { return vd_ins; }
  float getGroundTrack_rad() const { return atan2f((float)ve_ins, (float)vn_ins); }
  float getGyroBiasX_rads() const { return gbx; }
  float getGyroBiasY_rads() const { return gby; }
  float getGyroBiasZ_rads() const { return gbz; }
  float getAccelBiasX_mss() const { return abx; }
  float getAccelBiasY_mss() const { return aby; }
  float getAccelBiasZ_mss() const { return abz; }
  
  std::tuple<float,float,float> getPitchRollYaw(float ax, float ay, float az, float hx, float hy, float hz);
  
  void ekfUpdate(uint64_t time,
                double vn, double ve, double vd,
                double lat, double lon, double alt,
                float p, float q, float r,
                float ax, float ay, float az,
                float hx, float hy, float hz);
                
  void imuUpdateEkf(uint64_t time, const ImuData& imu);
  void gpsCoordinateUpdateEkf(const GpsCoordinate& coor);
  void gpsVelocityUpdateEkf(const GpsVelocity& vel);
  void updateProcessNoiseMatrix();
  void resetCovarianceMatrix();
  
  void setImuCalibration(const ImuCalibrationData& calibData);


private:
  mutable std::shared_mutex shMutex;
  
  EkfParams params_;
  
  GpsCoordinate gpsCoor;
  GpsVelocity gpsVel;
  ImuData imuDat;
  
  bool initialized_ = false;
  uint64_t _tprev;
  float phi, theta, psi;  
  double vn_ins, ve_ins, vd_ins;
  double lat_ins, lon_ins, alt_ins;
  float Bxc, Byc;
  float abx = 0.0f, aby = 0.0f, abz = 0.0f;
  float gbx = 0.0f, gby = 0.0f, gbz = 0.0f;
  Eigen::Matrix<float,15,15> Fs = Eigen::Matrix<float,15,15>::Identity();
  Eigen::Matrix<float,15,15> PHI = Eigen::Matrix<float,15,15>::Zero();
  Eigen::Matrix<float,15,15> P = Eigen::Matrix<float,15,15>::Zero();
  Eigen::Matrix<float,15,12> Gs = Eigen::Matrix<float,15,12>::Zero();
  Eigen::Matrix<float,12,12> Rw = Eigen::Matrix<float,12,12>::Zero();
  Eigen::Matrix<float,15,15> Q = Eigen::Matrix<float,15,15>::Zero();
  Eigen::Matrix<float,3,1> grav = Eigen::Matrix<float,3,1>::Zero();
  Eigen::Matrix<float,3,1> om_ib = Eigen::Matrix<float,3,1>::Zero();
  Eigen::Matrix<float,3,1> f_b = Eigen::Matrix<float,3,1>::Zero();
  Eigen::Matrix<float,3,3> C_N2B = Eigen::Matrix<float,3,3>::Zero();
  Eigen::Matrix<float,3,3> C_B2N = Eigen::Matrix<float,3,3>::Zero();
  Eigen::Matrix<float,3,1> dx = Eigen::Matrix<float,3,1>::Zero();
  Eigen::Matrix<double,3,1> dxd = Eigen::Matrix<double,3,1>::Zero();
  Eigen::Matrix<double,3,1> estmimated_ins = Eigen::Matrix<double,3,1>::Zero();
  Eigen::Matrix<double,3,1> V_ins = Eigen::Matrix<double,3,1>::Zero();
  Eigen::Matrix<double,3,1> lla_ins = Eigen::Matrix<double,3,1>::Zero();
  Eigen::Matrix<double,3,1> V_gps = Eigen::Matrix<double,3,1>::Zero();
  Eigen::Matrix<double,3,1> lla_gps = Eigen::Matrix<double,3,1>::Zero();
  Eigen::Matrix<double,3,1> pos_ecef_ins = Eigen::Matrix<double,3,1>::Zero();
  Eigen::Matrix<double,3,1> pos_ned_ins = Eigen::Matrix<double,3,1>::Zero();
  Eigen::Matrix<double,3,1> pos_ecef_gps = Eigen::Matrix<double,3,1>::Zero();
  Eigen::Matrix<double,3,1> pos_ned_gps = Eigen::Matrix<double,3,1>::Zero();
  Eigen::Matrix<float,4,1> quat = Eigen::Matrix<float,4,1>::Zero();
  Eigen::Matrix<float,4,1> dq = Eigen::Matrix<float,4,1>::Zero();
  Eigen::Matrix<float,6,1> y = Eigen::Matrix<float,6,1>::Zero();
  Eigen::Matrix<float,6,6> R = Eigen::Matrix<float,6,6>::Zero();
  Eigen::Matrix<float,15,1> x = Eigen::Matrix<float,15,1>::Zero();
  Eigen::Matrix<float,15,6> K = Eigen::Matrix<float,15,6>::Zero();
  Eigen::Matrix<float,6,15> H = Eigen::Matrix<float,6,15>::Zero();
  Eigen::Matrix<float,3,3> skewSymmetric(Eigen::Matrix<float,3,1> v);
  
  void ekfInit(uint64_t time, 
               double vn, double ve, double vd, 
               double lat, double lon, double alt,
               float p, float q, float r, 
               float ax, float ay, float az,
               float hx, float hy, float hz);
  
  Eigen::Matrix<double,3,1> llaRate(const Eigen::Matrix<double,3,1>& V, const Eigen::Matrix<double,3,1>& lla);
  Eigen::Matrix<double,3,1> llaRate(const Eigen::Matrix<double,3,1>& V, double lat, double alt);
  Eigen::Matrix<double,3,1> lla2ecef(const Eigen::Matrix<double,3,1>& lla);
  Eigen::Matrix<double,3,1> ecef2ned(const Eigen::Matrix<double,3,1>& ecef, const Eigen::Matrix<double,3,1>& pos_ref);
  Eigen::Matrix<float,3,3> quat2dcm(const Eigen::Matrix<float,4,1>& q);
  Eigen::Matrix<float,4,1> quatMultiply(const Eigen::Matrix<float,4,1>& p, const Eigen::Matrix<float,4,1>& q);
  float constrainAngle180(float angle) const;
  float constrainAngle360(float angle) const;
  std::pair<double, double> earthRadius(double lat);
  Eigen::Matrix<float,4,1> eulerToQuaternion(float yaw, float pitch, float roll);
  std::tuple<float,float,float> quaternionToEuler(const Eigen::Matrix<float,4,1>& quat);
 
  void updateJacobianMatrix();
  void updateProcessNoiseAndCovariance(float dt);
  void updateBias(float ax, float ay, float az, float p, float q, float r);
  void update15StatesAfterKf();
  void updateMeasurementResidual();
  void updateIns();
};

}

#endif 