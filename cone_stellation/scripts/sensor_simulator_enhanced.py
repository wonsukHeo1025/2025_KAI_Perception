#!/usr/bin/env python3
"""
Enhanced Sensor Simulation Module for ConeSTELLATION
Provides realistic IMU and RTK GPS simulation for tight coupling integration

Enhanced features:
- IMU: Temperature drift, scale factor errors, axis misalignment
- GPS: WGS84/UTM conversion, RTK status transitions, DOP effects, multipath
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import utm
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf_transformations import quaternion_from_euler, euler_from_quaternion
import rclpy.time
import rclpy.duration


class RtkStatus(Enum):
    """RTK GPS fix status"""
    NO_FIX = 0
    SINGLE = 1
    FLOAT = 2
    FIX = 3


@dataclass
class EnhancedImuConfig:
    """Enhanced IMU configuration with realistic error models"""
    # Basic Allan variance parameters
    gyro_noise_density: float = 0.005      # rad/s/√Hz
    gyro_bias_stability: float = 0.1       # rad/s
    gyro_random_walk: float = 0.00001      # rad/s²/√Hz
    
    accel_noise_density: float = 0.01      # m/s²/√Hz
    accel_bias_stability: float = 0.01     # m/s²
    accel_random_walk: float = 0.0001      # m/s³/√Hz
    
    # Temperature effects
    temperature_reference: float = 25.0     # °C
    gyro_temp_coefficient: float = 0.01     # rad/s/°C
    accel_temp_coefficient: float = 0.001   # m/s²/°C
    temperature_time_constant: float = 300  # seconds
    
    # Scale factor errors (parts per million)
    gyro_scale_factor_ppm: float = 500     # ppm
    accel_scale_factor_ppm: float = 1000   # ppm
    
    # Axis misalignment (radians)
    gyro_misalignment_rad: float = 0.001   # ~0.057 degrees
    accel_misalignment_rad: float = 0.002  # ~0.115 degrees
    
    # G-sensitivity (gyro error due to acceleration)
    gyro_g_sensitivity: float = 0.0001     # rad/s/g


@dataclass
class EnhancedGpsConfig:
    """Enhanced GPS configuration with RTK modes and realistic errors"""
    # RTK mode parameters
    rtk_mode: str = "rtk_fix"              # rtk_fix, rtk_float, single, no_fix
    
    # Fix mode noise (1-sigma)
    rtk_fix_noise_h: float = 0.02          # m - horizontal
    rtk_fix_noise_v: float = 0.04          # m - vertical
    rtk_float_noise_h: float = 0.3         # m
    rtk_float_noise_v: float = 0.5         # m
    single_noise_h: float = 2.0            # m
    single_noise_v: float = 5.0            # m
    no_fix_noise_h: float = 10.0           # m
    no_fix_noise_v: float = 15.0           # m
    
    # RTK status transition parameters
    fix_loss_probability: float = 0.001     # per second
    float_to_fix_probability: float = 0.1   # per second
    single_to_float_probability: float = 0.05 # per second
    
    # DOP (Dilution of Precision) effects
    hdop_min: float = 0.8
    hdop_max: float = 2.0
    vdop_min: float = 1.0
    vdop_max: float = 3.0
    
    # Multipath parameters
    multipath_amplitude: float = 0.5        # meters
    multipath_frequency: float = 0.1        # Hz
    
    # Update rate
    update_rate: float = 10.0               # Hz


class EnhancedImuSimulator:
    """Enhanced IMU simulator with temperature, scale factors, and misalignment"""
    
    def __init__(self, config: EnhancedImuConfig):
        self.config = config
        
        # Bias states
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        
        # Temperature state
        self.temperature = config.temperature_reference
        self.ambient_temperature = config.temperature_reference
        
        # Scale factor errors (1 + error)
        self.gyro_scale_factors = 1.0 + np.random.normal(0, config.gyro_scale_factor_ppm * 1e-6, 3)
        self.accel_scale_factors = 1.0 + np.random.normal(0, config.accel_scale_factor_ppm * 1e-6, 3)
        
        # Axis misalignment matrices
        self.gyro_misalignment = self._generate_misalignment_matrix(config.gyro_misalignment_rad)
        self.accel_misalignment = self._generate_misalignment_matrix(config.accel_misalignment_rad)
        
        # Gravity reference
        self.gravity = 9.81
    
    def _generate_misalignment_matrix(self, max_angle: float) -> np.ndarray:
        """Generate random axis misalignment matrix"""
        # Small random rotations around each axis
        angles = np.random.uniform(-max_angle, max_angle, 3)
        
        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        
        return Rz @ Ry @ Rx
    
    def update_temperature(self, dt: float, ambient_temp: Optional[float] = None):
        """Update temperature state with thermal model"""
        if ambient_temp is not None:
            self.ambient_temperature = ambient_temp
        
        # Simple exponential thermal model
        tau = self.config.temperature_time_constant
        self.temperature += (self.ambient_temperature - self.temperature) * dt / tau
        
        # Add small temperature noise
        self.temperature += np.random.normal(0, 0.01)
    
    def update_bias(self, dt: float):
        """Update bias with temperature effects and random walk"""
        # Temperature-induced bias drift
        temp_drift = self.temperature - self.config.temperature_reference
        temp_bias_accel = temp_drift * self.config.accel_temp_coefficient
        temp_bias_gyro = temp_drift * self.config.gyro_temp_coefficient
        
        # Random walk
        accel_rw_sigma = self.config.accel_random_walk * np.sqrt(dt)
        gyro_rw_sigma = self.config.gyro_random_walk * np.sqrt(dt)
        
        self.accel_bias += np.random.normal(0, accel_rw_sigma, 3)
        self.gyro_bias += np.random.normal(0, gyro_rw_sigma, 3)
        
        # Add temperature bias
        self.accel_bias += temp_bias_accel * np.ones(3) * dt
        self.gyro_bias += temp_bias_gyro * np.ones(3) * dt
        
        # Clamp to stability limits
        self.accel_bias = np.clip(self.accel_bias, 
                                  -self.config.accel_bias_stability,
                                  self.config.accel_bias_stability)
        self.gyro_bias = np.clip(self.gyro_bias,
                                -self.config.gyro_bias_stability,
                                self.config.gyro_bias_stability)
    
    def generate_imu_data(self, true_accel: np.ndarray, true_angular_vel: np.ndarray,
                         true_orientation: Tuple[float, float, float, float],
                         dt: float, timestamp: rclpy.time.Time) -> Imu:
        """Generate enhanced IMU data with all error models"""
        # Update states
        self.update_temperature(dt)
        self.update_bias(dt)
        
        # Create IMU message
        imu_msg = Imu()
        imu_msg.header.stamp = timestamp.to_msg()
        imu_msg.header.frame_id = "imu_link"
        
        # Add gravity to acceleration
        # Convert quaternion to rotation matrix to get gravity in body frame
        q = true_orientation
        R = np.array([
            [1-2*(q[1]**2+q[2]**2), 2*(q[0]*q[1]-q[2]*q[3]), 2*(q[0]*q[2]+q[1]*q[3])],
            [2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[0]**2+q[2]**2), 2*(q[1]*q[2]-q[0]*q[3])],
            [2*(q[0]*q[2]-q[1]*q[3]), 2*(q[1]*q[2]+q[0]*q[3]), 1-2*(q[0]**2+q[1]**2)]
        ])
        gravity_body = R.T @ np.array([0, 0, -self.gravity])
        
        accel_with_gravity = true_accel - gravity_body
        
        # Apply scale factors
        scaled_accel = accel_with_gravity * self.accel_scale_factors
        scaled_gyro = true_angular_vel * self.gyro_scale_factors
        
        # Apply misalignment
        misaligned_accel = self.accel_misalignment @ scaled_accel
        misaligned_gyro = self.gyro_misalignment @ scaled_gyro
        
        # Add g-sensitivity (gyro error due to acceleration)
        g_sensitivity_error = self.config.gyro_g_sensitivity * misaligned_accel / self.gravity
        misaligned_gyro += g_sensitivity_error
        
        # Add bias
        biased_accel = misaligned_accel + self.accel_bias
        biased_gyro = misaligned_gyro + self.gyro_bias
        
        # Add white noise
        sampling_rate = 1.0 / dt
        accel_noise_sigma = self.config.accel_noise_density * np.sqrt(sampling_rate)
        gyro_noise_sigma = self.config.gyro_noise_density * np.sqrt(sampling_rate)
        
        noisy_accel = biased_accel + np.random.normal(0, accel_noise_sigma, 3)
        noisy_gyro = biased_gyro + np.random.normal(0, gyro_noise_sigma, 3)
        
        # Set message values
        imu_msg.linear_acceleration.x = noisy_accel[0]
        imu_msg.linear_acceleration.y = noisy_accel[1]
        imu_msg.linear_acceleration.z = noisy_accel[2]
        
        imu_msg.angular_velocity.x = noisy_gyro[0]
        imu_msg.angular_velocity.y = noisy_gyro[1]
        imu_msg.angular_velocity.z = noisy_gyro[2]
        
        # Set orientation (with small noise)
        orientation_noise = 0.001
        imu_msg.orientation.x = q[0] + np.random.normal(0, orientation_noise)
        imu_msg.orientation.y = q[1] + np.random.normal(0, orientation_noise)
        imu_msg.orientation.z = q[2] + np.random.normal(0, orientation_noise)
        imu_msg.orientation.w = q[3] + np.random.normal(0, orientation_noise)
        
        # Normalize quaternion
        quat_array = np.array([imu_msg.orientation.x, imu_msg.orientation.y,
                              imu_msg.orientation.z, imu_msg.orientation.w])
        quat_array /= np.linalg.norm(quat_array)
        imu_msg.orientation.x = quat_array[0]
        imu_msg.orientation.y = quat_array[1]
        imu_msg.orientation.z = quat_array[2]
        imu_msg.orientation.w = quat_array[3]
        
        # Set covariances
        imu_msg.orientation_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
        
        gyro_variance = gyro_noise_sigma**2
        imu_msg.angular_velocity_covariance = [
            gyro_variance, 0.0, 0.0,
            0.0, gyro_variance, 0.0,
            0.0, 0.0, gyro_variance
        ]
        
        accel_variance = accel_noise_sigma**2
        imu_msg.linear_acceleration_covariance = [
            accel_variance, 0.0, 0.0,
            0.0, accel_variance, 0.0,
            0.0, 0.0, accel_variance
        ]
        
        return imu_msg


class EnhancedGpsSimulator:
    """Enhanced GPS simulator with RTK modes and realistic effects"""
    
    def __init__(self, config: EnhancedGpsConfig, 
                 origin_lat: float = 37.5665, 
                 origin_lon: float = 126.9780):
        self.config = config
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        
        # Get UTM zone for origin
        self.utm_zone_number, self.utm_zone_letter = utm.from_latlon(origin_lat, origin_lon)[2:]
        
        # Current RTK status
        self.rtk_status = self._parse_rtk_mode(config.rtk_mode)
        
        # DOP values
        self.hdop = (config.hdop_min + config.hdop_max) / 2
        self.vdop = (config.vdop_min + config.vdop_max) / 2
        
        # Multipath phase
        self.multipath_phase = 0.0
        
        # Status transition timer
        self.time_in_current_status = 0.0
        
        print(f"Enhanced GPS initialized at {origin_lat}, {origin_lon}")
        print(f"UTM Zone: {self.utm_zone_number}{self.utm_zone_letter}")
    
    def _parse_rtk_mode(self, mode: str) -> RtkStatus:
        """Parse RTK mode string to enum"""
        mode_map = {
            "rtk_fix": RtkStatus.FIX,
            "rtk_float": RtkStatus.FLOAT,
            "single": RtkStatus.SINGLE,
            "no_fix": RtkStatus.NO_FIX
        }
        return mode_map.get(mode, RtkStatus.FIX)
    
    def update_rtk_status(self, dt: float):
        """Update RTK status with realistic transitions"""
        self.time_in_current_status += dt
        
        # Random status transitions
        if self.rtk_status == RtkStatus.FIX:
            # Can lose fix
            if np.random.random() < self.config.fix_loss_probability * dt:
                self.rtk_status = RtkStatus.FLOAT
                self.time_in_current_status = 0.0
                print(f"GPS: RTK Fix → Float")
                
        elif self.rtk_status == RtkStatus.FLOAT:
            # Can improve to fix or degrade to single
            if np.random.random() < self.config.float_to_fix_probability * dt:
                self.rtk_status = RtkStatus.FIX
                self.time_in_current_status = 0.0
                print(f"GPS: RTK Float → Fix")
            elif np.random.random() < 0.01 * dt:  # 1% per second
                self.rtk_status = RtkStatus.SINGLE
                self.time_in_current_status = 0.0
                print(f"GPS: RTK Float → Single")
                
        elif self.rtk_status == RtkStatus.SINGLE:
            # Can improve to float
            if np.random.random() < self.config.single_to_float_probability * dt:
                self.rtk_status = RtkStatus.FLOAT
                self.time_in_current_status = 0.0
                print(f"GPS: Single → RTK Float")
    
    def update_dop(self, dt: float):
        """Update DOP values with slow variation"""
        # Slow random walk for DOP values
        dop_change_rate = 0.1  # per second
        
        self.hdop += np.random.normal(0, dop_change_rate * dt)
        self.vdop += np.random.normal(0, dop_change_rate * dt)
        
        # Clamp to ranges
        self.hdop = np.clip(self.hdop, self.config.hdop_min, self.config.hdop_max)
        self.vdop = np.clip(self.vdop, self.config.vdop_min, self.config.vdop_max)
    
    def get_noise_parameters(self) -> Tuple[float, float]:
        """Get noise parameters based on current RTK status and DOP"""
        if self.rtk_status == RtkStatus.FIX:
            base_h = self.config.rtk_fix_noise_h
            base_v = self.config.rtk_fix_noise_v
        elif self.rtk_status == RtkStatus.FLOAT:
            base_h = self.config.rtk_float_noise_h
            base_v = self.config.rtk_float_noise_v
        elif self.rtk_status == RtkStatus.SINGLE:
            base_h = self.config.single_noise_h
            base_v = self.config.single_noise_v
        else:
            base_h = self.config.no_fix_noise_h
            base_v = self.config.no_fix_noise_v
        
        # Scale by DOP
        h_noise = base_h * self.hdop
        v_noise = base_v * self.vdop
        
        return h_noise, v_noise
    
    def apply_multipath(self, position: np.ndarray, dt: float) -> np.ndarray:
        """Apply multipath effects to position"""
        # Update multipath phase
        self.multipath_phase += 2 * np.pi * self.config.multipath_frequency * dt
        
        # Sinusoidal multipath error
        multipath_error = self.config.multipath_amplitude * np.sin(self.multipath_phase)
        
        # Apply mainly to horizontal position
        position[0] += multipath_error * 0.7
        position[1] += multipath_error * 0.3
        
        return position
    
    def generate_gps_data(self, true_x: float, true_y: float, true_z: float,
                         dt: float, timestamp: rclpy.time.Time) -> NavSatFix:
        """Generate enhanced GPS data with UTM conversion"""
        # Update states
        self.update_rtk_status(dt)
        self.update_dop(dt)
        
        # Get noise parameters
        h_noise, v_noise = self.get_noise_parameters()
        
        # Add noise to local position
        noisy_pos = np.array([true_x, true_y, true_z])
        noisy_pos[0] += np.random.normal(0, h_noise)
        noisy_pos[1] += np.random.normal(0, h_noise)
        noisy_pos[2] += np.random.normal(0, v_noise)
        
        # Apply multipath if not RTK fix
        if self.rtk_status != RtkStatus.FIX:
            noisy_pos = self.apply_multipath(noisy_pos, dt)
        
        # Convert local coordinates to UTM
        # Assume origin is at UTM coordinates
        origin_easting, origin_northing, _, _ = utm.from_latlon(
            self.origin_lat, self.origin_lon
        )
        
        utm_easting = origin_easting + noisy_pos[0]
        utm_northing = origin_northing + noisy_pos[1]
        
        # Convert UTM to lat/lon
        lat, lon = utm.to_latlon(
            utm_easting, utm_northing, 
            self.utm_zone_number, self.utm_zone_letter
        )
        
        # Create GPS message
        gps_msg = NavSatFix()
        gps_msg.header.stamp = timestamp.to_msg()
        gps_msg.header.frame_id = "gps_link"
        
        gps_msg.latitude = lat
        gps_msg.longitude = lon
        gps_msg.altitude = noisy_pos[2]
        
        # Set status based on RTK mode
        gps_msg.status.service = NavSatStatus.SERVICE_GPS
        
        if self.rtk_status == RtkStatus.FIX:
            gps_msg.status.status = NavSatStatus.STATUS_GBAS_FIX  # RTK fixed
        elif self.rtk_status == RtkStatus.FLOAT:
            gps_msg.status.status = NavSatStatus.STATUS_DGPS_FIX  # RTK float
        elif self.rtk_status == RtkStatus.SINGLE:
            gps_msg.status.status = NavSatStatus.STATUS_FIX       # GPS fix
        else:
            gps_msg.status.status = NavSatStatus.STATUS_NO_FIX    # No fix
        
        # Set covariance based on actual noise and status
        # For RTK Fix, use very low covariance as observed (0.00002)
        if self.rtk_status == RtkStatus.FIX:
            h_cov = 0.00002  # As observed in real RTK data
            v_cov = 0.00004
        else:
            h_cov = h_noise**2
            v_cov = v_noise**2
        
        gps_msg.position_covariance = [
            h_cov, 0.0, 0.0,
            0.0, h_cov, 0.0,
            0.0, 0.0, v_cov
        ]
        gps_msg.position_covariance_type = 2  # COVARIANCE_TYPE_DIAGONAL_KNOWN
        
        return gps_msg
    
    def generate_utm_pose(self, true_x: float, true_y: float, true_z: float,
                         true_orientation: Tuple[float, float, float, float],
                         dt: float, timestamp: rclpy.time.Time) -> PoseWithCovarianceStamped:
        """Generate UTM pose message for direct fusion"""
        # Get noise parameters
        h_noise, v_noise = self.get_noise_parameters()
        
        # Add noise
        noisy_x = true_x + np.random.normal(0, h_noise)
        noisy_y = true_y + np.random.normal(0, h_noise)
        noisy_z = true_z + np.random.normal(0, v_noise)
        
        # Create pose message
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = timestamp.to_msg()
        pose_msg.header.frame_id = "utm"
        
        pose_msg.pose.pose.position.x = noisy_x
        pose_msg.pose.pose.position.y = noisy_y
        pose_msg.pose.pose.position.z = noisy_z
        
        # Use true orientation (GPS doesn't provide orientation)
        pose_msg.pose.pose.orientation.x = true_orientation[0]
        pose_msg.pose.pose.orientation.y = true_orientation[1]
        pose_msg.pose.pose.orientation.z = true_orientation[2]
        pose_msg.pose.pose.orientation.w = true_orientation[3]
        
        # Set covariance
        cov = np.zeros(36)
        if self.rtk_status == RtkStatus.FIX:
            cov[0] = 0.00002   # x
            cov[7] = 0.00002   # y
            cov[14] = 0.00004  # z
        else:
            cov[0] = h_noise**2
            cov[7] = h_noise**2
            cov[14] = v_noise**2
        
        # Very high uncertainty for orientation (GPS doesn't measure it)
        cov[21] = 1e6  # roll
        cov[28] = 1e6  # pitch
        cov[35] = 1e6  # yaw
        
        pose_msg.pose.covariance = cov.tolist()
        
        return pose_msg


# Import the original classes for compatibility
from sensor_simulator import OdometrySimulator, SensorSimulator as OriginalSensorSimulator


class EnhancedSensorSimulator:
    """Enhanced sensor simulator with improved IMU and GPS models"""
    
    def __init__(self, imu_config: EnhancedImuConfig, gps_config: EnhancedGpsConfig,
                 odometry_config=None):
        self.imu_sim = EnhancedImuSimulator(imu_config)
        self.gps_sim = EnhancedGpsSimulator(gps_config)
        
        # Use original odometry simulator if config provided
        if odometry_config:
            self.odom_sim = OdometrySimulator(odometry_config)
        else:
            self.odom_sim = None
    
    def generate_all_sensors(self, vehicle_state: Dict, dt: float,
                           timestamp: rclpy.time.Time) -> Dict:
        """Generate all sensor data with enhanced models"""
        # Extract state
        pos = vehicle_state['position']
        orient = vehicle_state['orientation']
        lin_vel = vehicle_state['linear_velocity']
        ang_vel = vehicle_state['angular_velocity']
        lin_acc = vehicle_state.get('linear_acceleration', [0, 0, 0])
        
        # Generate quaternion
        q = quaternion_from_euler(orient[0], orient[1], orient[2])
        
        # Generate sensor data
        imu_msg = self.imu_sim.generate_imu_data(
            np.array(lin_acc), np.array(ang_vel), q, dt, timestamp
        )
        
        gps_msg = self.gps_sim.generate_gps_data(
            pos[0], pos[1], pos[2], dt, timestamp
        )
        
        utm_pose = self.gps_sim.generate_utm_pose(
            pos[0], pos[1], pos[2], q, dt, timestamp
        )
        
        result = {
            'imu': imu_msg,
            'gps': gps_msg,
            'utm_pose': utm_pose
        }
        
        # Add odometry if available
        if self.odom_sim:
            odom_msg = self.odom_sim.generate_odometry_data(
                pos[0], pos[1], orient[2],
                lin_vel[0], lin_vel[1], ang_vel[2],
                dt, timestamp
            )
            result['odometry'] = odom_msg
        
        return result


# Test function
if __name__ == "__main__":
    # Test enhanced simulators
    imu_config = EnhancedImuConfig()
    gps_config = EnhancedGpsConfig(rtk_mode="rtk_fix")
    
    sim = EnhancedSensorSimulator(imu_config, gps_config)
    
    # Test vehicle state
    vehicle_state = {
        'position': [10.0, 5.0, 0.0],
        'orientation': [0.0, 0.0, 0.1],  # roll, pitch, yaw
        'linear_velocity': [5.0, 0.0, 0.0],
        'angular_velocity': [0.0, 0.0, 0.1],
        'linear_acceleration': [0.5, 0.0, 0.0]
    }
    
    # Generate sensor data
    import rclpy
    rclpy.init()
    timestamp = rclpy.time.Time()
    
    sensors = sim.generate_all_sensors(vehicle_state, 0.01, timestamp)
    
    print(f"IMU accel: {sensors['imu'].linear_acceleration.x:.3f}, "
          f"{sensors['imu'].linear_acceleration.y:.3f}, "
          f"{sensors['imu'].linear_acceleration.z:.3f}")
    print(f"GPS: {sensors['gps'].latitude:.8f}, {sensors['gps'].longitude:.8f}")
    print(f"GPS status: {sensors['gps'].status.status}")
    print(f"UTM pose: {sensors['utm_pose'].pose.pose.position.x:.3f}, "
          f"{sensors['utm_pose'].pose.pose.position.y:.3f}")
    
    rclpy.shutdown()