#!/usr/bin/env python3
"""
Sensor Simulation Module for CC-SLAM-SYM

Provides realistic sensor data generation with noise models for:
- IMU (Inertial Measurement Unit)
- GPS (Global Positioning System)
- Odometry

Includes drift, bias, and white noise simulation for realistic sensor behavior.
"""

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
from tf_transformations import quaternion_from_euler
import rclpy.time


@dataclass
class SensorNoiseConfig:
    """Configuration for sensor noise parameters"""
    # IMU parameters (Allan Variance based)
    imu_gyro_noise_density: float = 0.005      # rad/s/√Hz
    imu_gyro_bias_stability: float = 0.1       # rad/s
    imu_gyro_random_walk: float = 0.00001      # rad/s²/√Hz
    
    imu_accel_noise_density: float = 0.01      # m/s²/√Hz
    imu_accel_bias_stability: float = 0.01     # m/s²
    imu_accel_random_walk: float = 0.0001      # m/s³/√Hz
    
    # GPS parameters
    gps_mode: str = "rtk"                      # GPS mode
    gps_rtk_fix_noise_h: float = 0.02          # m - RTK Fix horizontal
    gps_rtk_fix_noise_v: float = 0.04          # m - RTK Fix vertical  
    gps_rtk_float_noise_h: float = 0.3         # m - RTK Float horizontal
    gps_rtk_float_noise_v: float = 0.5         # m - RTK Float vertical
    gps_single_noise_h: float = 2.0            # m - Single horizontal
    gps_single_noise_v: float = 5.0            # m - Single vertical
    
    # Odometry drift parameters (x, y, theta)
    odom_drift_x_systematic: float = 0.0       # % - x축 bias (systematic error)
    odom_drift_x_random: float = 0.0           # m - x축 random noise stddev
    odom_drift_y_systematic: float = 0.0       # % - y축 bias (systematic error)
    odom_drift_y_random: float = 0.0           # m - y축 random noise stddev
    odom_drift_theta_systematic: float = 0.0   # % - theta bias (systematic error)
    odom_drift_theta_random: float = 0.0       # rad - theta random noise stddev


class ImuSimulator:
    """Simulates IMU sensor data with realistic noise and drift"""
    
    def __init__(self, config: SensorNoiseConfig):
        self.config = config
        
        # Bias states (slowly changing over time)
        self.accel_bias_x = 0.0
        self.accel_bias_y = 0.0
        self.accel_bias_z = 0.0
        self.gyro_bias_x = 0.0
        self.gyro_bias_y = 0.0
        self.gyro_bias_z = 0.0
        
        # Initial gravity vector (assuming flat ground)
        self.gravity = 9.81
    
    def update_bias(self, dt: float) -> None:
        """Update bias states using Allan variance random walk model"""
        # Acceleration bias random walk: sigma = K * sqrt(dt)
        accel_rw_sigma = self.config.imu_accel_random_walk * np.sqrt(dt)
        self.accel_bias_x += np.random.normal(0, accel_rw_sigma)
        self.accel_bias_y += np.random.normal(0, accel_rw_sigma)
        self.accel_bias_z += np.random.normal(0, accel_rw_sigma)
        
        # Gyroscope bias random walk
        gyro_rw_sigma = self.config.imu_gyro_random_walk * np.sqrt(dt)
        self.gyro_bias_x += np.random.normal(0, gyro_rw_sigma)
        self.gyro_bias_y += np.random.normal(0, gyro_rw_sigma)
        self.gyro_bias_z += np.random.normal(0, gyro_rw_sigma)
        
        # Limit bias to bias stability values
        self.accel_bias_x = np.clip(self.accel_bias_x, -self.config.imu_accel_bias_stability, 
                                   self.config.imu_accel_bias_stability)
        self.accel_bias_y = np.clip(self.accel_bias_y, -self.config.imu_accel_bias_stability,
                                   self.config.imu_accel_bias_stability)
        self.accel_bias_z = np.clip(self.accel_bias_z, -self.config.imu_accel_bias_stability,
                                   self.config.imu_accel_bias_stability)
        
        self.gyro_bias_x = np.clip(self.gyro_bias_x, -self.config.imu_gyro_bias_stability,
                                  self.config.imu_gyro_bias_stability)
        self.gyro_bias_y = np.clip(self.gyro_bias_y, -self.config.imu_gyro_bias_stability,
                                  self.config.imu_gyro_bias_stability)
        self.gyro_bias_z = np.clip(self.gyro_bias_z, -self.config.imu_gyro_bias_stability,
                                  self.config.imu_gyro_bias_stability)
    
    def generate_imu_data(self, true_accel: np.ndarray, true_angular_vel: np.ndarray, 
                         true_orientation: Tuple[float, float, float, float],
                         dt: float, timestamp: rclpy.time.Time) -> Imu:
        """
        Generate IMU message with realistic noise
        
        Args:
            true_accel: True linear acceleration [ax, ay, az] in robot frame
            true_angular_vel: True angular velocity [wx, wy, wz] in robot frame
            true_orientation: True orientation as quaternion (x, y, z, w)
            dt: Time step
            timestamp: ROS timestamp
        
        Returns:
            Imu message with noise and bias
        """
        # Update bias
        self.update_bias(dt)
        
        # Create IMU message
        imu_msg = Imu()
        imu_msg.header.stamp = timestamp.to_msg()
        imu_msg.header.frame_id = "imu_link"
        
        # Apply gravity to acceleration (IMU measures gravity)
        # In robot frame, gravity appears as upward acceleration when stationary
        gravity_in_robot = np.array([0, 0, self.gravity])
        
        # Calculate white noise from noise density
        # White noise sigma = noise_density * sqrt(sampling_rate)
        sampling_rate = 1.0 / dt  # Hz
        accel_white_noise_sigma = self.config.imu_accel_noise_density * np.sqrt(sampling_rate)
        gyro_white_noise_sigma = self.config.imu_gyro_noise_density * np.sqrt(sampling_rate)
        
        # Add bias and white noise to acceleration
        accel_with_gravity = true_accel + gravity_in_robot
        imu_msg.linear_acceleration.x = (accel_with_gravity[0] + self.accel_bias_x + 
                                        np.random.normal(0, accel_white_noise_sigma))
        imu_msg.linear_acceleration.y = (accel_with_gravity[1] + self.accel_bias_y + 
                                        np.random.normal(0, accel_white_noise_sigma))
        imu_msg.linear_acceleration.z = (accel_with_gravity[2] + self.accel_bias_z + 
                                        np.random.normal(0, accel_white_noise_sigma))
        
        # Add bias and white noise to angular velocity
        imu_msg.angular_velocity.x = (true_angular_vel[0] + self.gyro_bias_x + 
                                     np.random.normal(0, gyro_white_noise_sigma))
        imu_msg.angular_velocity.y = (true_angular_vel[1] + self.gyro_bias_y + 
                                     np.random.normal(0, gyro_white_noise_sigma))
        imu_msg.angular_velocity.z = (true_angular_vel[2] + self.gyro_bias_z + 
                                     np.random.normal(0, gyro_white_noise_sigma))
        
        # Set orientation (with small noise)
        orientation_noise = 0.001  # Very small orientation noise
        imu_msg.orientation.x = true_orientation[0] + np.random.normal(0, orientation_noise)
        imu_msg.orientation.y = true_orientation[1] + np.random.normal(0, orientation_noise)
        imu_msg.orientation.z = true_orientation[2] + np.random.normal(0, orientation_noise)
        imu_msg.orientation.w = true_orientation[3] + np.random.normal(0, orientation_noise)
        
        # Normalize quaternion
        quat_norm = np.sqrt(imu_msg.orientation.x**2 + imu_msg.orientation.y**2 + 
                           imu_msg.orientation.z**2 + imu_msg.orientation.w**2)
        imu_msg.orientation.x /= quat_norm
        imu_msg.orientation.y /= quat_norm
        imu_msg.orientation.z /= quat_norm
        imu_msg.orientation.w /= quat_norm
        
        # Set covariances (diagonal matrices) - must be exactly 9 floats
        # Orientation covariance (small fixed value)
        imu_msg.orientation_covariance = [
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.01
        ]
        
        # Angular velocity covariance (based on noise density)
        gyro_variance = gyro_white_noise_sigma**2
        imu_msg.angular_velocity_covariance = [
            float(gyro_variance), 0.0, 0.0,
            0.0, float(gyro_variance), 0.0,
            0.0, 0.0, float(gyro_variance)
        ]
        
        # Linear acceleration covariance (based on noise density)
        accel_variance = accel_white_noise_sigma**2
        imu_msg.linear_acceleration_covariance = [
            float(accel_variance), 0.0, 0.0,
            0.0, float(accel_variance), 0.0,
            0.0, 0.0, float(accel_variance)
        ]
        
        return imu_msg


class GpsSimulator:
    """Simulates GPS sensor data with realistic noise"""
    
    def __init__(self, config: SensorNoiseConfig, origin_lat: float = 37.5665, origin_lon: float = 126.9780):
        """
        Initialize GPS simulator
        
        Args:
            config: Sensor noise configuration
            origin_lat: Origin latitude (default: Seoul)
            origin_lon: Origin longitude (default: Seoul)
        """
        self.config = config
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        
        # For debugging: print origin
        print(f"GPS Simulator initialized at origin: {origin_lat}, {origin_lon}")
    
    def xy_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        """Convert local XY coordinates to latitude/longitude"""
        # Approximate conversion (good for small distances)
        lat_per_meter = 1.0 / 111111.0
        lon_per_meter = 1.0 / (111111.0 * np.cos(np.radians(self.origin_lat)))
        
        lat = self.origin_lat + y * lat_per_meter
        lon = self.origin_lon + x * lon_per_meter
        
        return lat, lon
    
    def generate_gps_data(self, true_x: float, true_y: float, true_z: float,
                         timestamp: rclpy.time.Time) -> NavSatFix:
        """
        Generate GPS NavSatFix message with RTK mode support
        
        Args:
            true_x, true_y: True position in local coordinates (meters)
            true_z: True altitude (meters)
            timestamp: ROS timestamp
        
        Returns:
            NavSatFix message with mode-appropriate GPS noise
        """
        # Determine noise levels based on GPS mode
        if self.config.gps_mode == "rtk" or self.config.gps_mode == "rtk_fix":
            h_noise = self.config.gps_rtk_fix_noise_h
            v_noise = self.config.gps_rtk_fix_noise_v
            status = 2  # DGPS/RTK fix
        elif self.config.gps_mode == "rtk_float":
            h_noise = self.config.gps_rtk_float_noise_h
            v_noise = self.config.gps_rtk_float_noise_v
            status = 2  # DGPS/RTK fix (float)
        elif self.config.gps_mode == "dgps":
            # Use intermediate values between single and RTK float
            h_noise = (self.config.gps_single_noise_h + self.config.gps_rtk_float_noise_h) / 2
            v_noise = (self.config.gps_single_noise_v + self.config.gps_rtk_float_noise_v) / 2
            status = 2  # DGPS fix
        else:  # single
            h_noise = self.config.gps_single_noise_h
            v_noise = self.config.gps_single_noise_v
            status = 1  # GPS fix
        
        # Add noise to position
        noisy_x = true_x + np.random.normal(0, h_noise)
        noisy_y = true_y + np.random.normal(0, h_noise)
        noisy_z = true_z + np.random.normal(0, v_noise)
        
        # Convert to lat/lon
        lat, lon = self.xy_to_latlon(noisy_x, noisy_y)
        
        # Create GPS message
        gps_msg = NavSatFix()
        gps_msg.header.stamp = timestamp.to_msg()
        gps_msg.header.frame_id = "gps_link"
        
        gps_msg.latitude = lat
        gps_msg.longitude = lon
        gps_msg.altitude = noisy_z
        
        # Status based on mode
        gps_msg.status.status = status
        gps_msg.status.service = 1  # GPS service
        
        # Position covariance (m²) - must be exactly 9 floats
        gps_msg.position_covariance = [
            float(h_noise**2), 0.0, 0.0,
            0.0, float(h_noise**2), 0.0,
            0.0, 0.0, float(v_noise**2)
        ]
        gps_msg.position_covariance_type = 2  # Diagonal known
        
        return gps_msg


class OdometrySimulator:
    """Simulates wheel odometry with cumulative drift"""
    
    def __init__(self, config: SensorNoiseConfig):
        self.config = config
        
        # Odometry internal state (accumulates from origin)
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0
        
        # Previous ground truth state (for calculating deltas)
        self.prev_true_x = 0.0
        self.prev_true_y = 0.0
        self.prev_true_theta = 0.0
        
        # Track total distance for drift calculation
        self.total_distance = 0.0
        self.total_rotation = 0.0
        
        # First update flag
        self.first_update = True
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def generate_odometry_data(self, true_x: float, true_y: float, true_theta: float,
                              true_vx: float, true_vy: float, true_vtheta: float,
                              dt: float, timestamp: rclpy.time.Time,
                              child_frame_id: str = "base_link") -> Odometry:
        """
        Generate Odometry message with cumulative drift and noise
        
        Args:
            true_x, true_y, true_theta: True pose
            true_vx, true_vy, true_vtheta: True velocities
            dt: Time step
            timestamp: ROS timestamp
            child_frame_id: Child frame ID
        
        Returns:
            Odometry message with drift and noise
        """
        # Initialize on first update
        if self.first_update:
            self.prev_true_x = true_x
            self.prev_true_y = true_y
            self.prev_true_theta = true_theta
            self.odom_x = true_x
            self.odom_y = true_y
            self.odom_theta = true_theta
            self.first_update = False
        
        # Calculate ground truth delta in global frame
        delta_x_global = true_x - self.prev_true_x
        delta_y_global = true_y - self.prev_true_y
        delta_theta = self._normalize_angle(true_theta - self.prev_true_theta)
        
        # Convert global delta to robot's local frame (at previous position)
        cos_prev = np.cos(self.prev_true_theta)
        sin_prev = np.sin(self.prev_true_theta)
        delta_x_local = delta_x_global * cos_prev + delta_y_global * sin_prev
        delta_y_local = -delta_x_global * sin_prev + delta_y_global * cos_prev
        
        # Calculate distance moved for drift computation
        distance_moved = np.sqrt(delta_x_local**2 + delta_y_local**2)
        
        # Apply drift to the movement (per-axis control)
        # IMPORTANT: Bias should create drift even during straight motion!
        
        # X-axis drift (전진 방향)
        if distance_moved > 0:
            # Apply systematic bias (as % of movement)
            delta_x_local *= (1.0 + self.config.odom_drift_x_systematic * 0.01)
            # Add random noise
            delta_x_local += np.random.normal(0, self.config.odom_drift_x_random)
        
        # Y-axis drift (측면 방향) - FIXED: Apply even when delta_y_local is 0!
        if distance_moved > 0:
            # Systematic bias: sideways drift proportional to forward movement
            # (like a car with bad alignment drifting sideways)
            sideways_drift = distance_moved * self.config.odom_drift_y_systematic * 0.01
            delta_y_local += sideways_drift
            # Add random noise
            delta_y_local += np.random.normal(0, self.config.odom_drift_y_random)
        
        # Theta drift (heading) - Apply proportional to movement
        if distance_moved > 0 or abs(delta_theta) > 0:
            # Systematic bias on actual rotation
            if abs(delta_theta) > 0:
                delta_theta *= (1.0 + self.config.odom_drift_theta_systematic * 0.01)
            
            # Random noise - always apply when moving
            if distance_moved > 0:
                # Heading drift per meter of travel (rad/m)
                # This simulates wheel size differences, uneven surfaces, etc.
                heading_noise_per_meter = self.config.odom_drift_theta_random
                delta_theta += np.random.normal(0, heading_noise_per_meter * distance_moved)
        
        # No additional white noise - only drift affects odometry
        
        # Update odometry state by transforming local delta to odometry frame
        # Note: Use the odometry's current heading, not ground truth
        cos_odom = np.cos(self.odom_theta)
        sin_odom = np.sin(self.odom_theta)
        self.odom_x += delta_x_local * cos_odom - delta_y_local * sin_odom
        self.odom_y += delta_x_local * sin_odom + delta_y_local * cos_odom
        self.odom_theta = self._normalize_angle(self.odom_theta + delta_theta)
        
        # Update tracking variables
        self.total_distance += distance_moved
        self.total_rotation += abs(delta_theta)
        
        # Update previous ground truth state
        self.prev_true_x = true_x
        self.prev_true_y = true_y
        self.prev_true_theta = true_theta
        
        # Debug output
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 50 == 0:
            error_x = self.odom_x - true_x
            error_y = self.odom_y - true_y
            error_theta = self._normalize_angle(self.odom_theta - true_theta)
            error_dist = np.sqrt(error_x**2 + error_y**2)
            
            # Calculate drift rates
            if self.total_distance > 0:
                lateral_drift_rate = (error_y / self.total_distance) * 100  # % of distance
                heading_drift_rate = error_theta / self.total_distance  # rad/m
            else:
                lateral_drift_rate = 0
                heading_drift_rate = 0
            
            print(f"ODOM: err={error_dist:.3f}m (x={error_x:.3f}, y={error_y:.3f}), "
                  f"θ_err={np.degrees(error_theta):.1f}°, dist={self.total_distance:.1f}m | "
                  f"lateral_drift={lateral_drift_rate:.1f}%, heading_drift={heading_drift_rate:.4f}rad/m")
        
        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp.to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = child_frame_id
        
        # Set position (no additional noise - drift only)
        odom_msg.pose.pose.position.x = self.odom_x
        odom_msg.pose.pose.position.y = self.odom_y
        odom_msg.pose.pose.position.z = 0.0
        
        # Set orientation (no additional noise - drift only)
        noisy_theta = self.odom_theta
        q = quaternion_from_euler(0, 0, noisy_theta)
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]
        
        # Set velocities (no noise - velocities come from position differentiation)
        odom_msg.twist.twist.linear.x = true_vx
        odom_msg.twist.twist.linear.y = true_vy
        odom_msg.twist.twist.angular.z = true_vtheta
        
        # Covariances (reflect actual noise and drift configuration)
        # Base covariance from configured noise levels
        pose_cov = np.zeros(36)
        
        # Position covariance based on systematic drift and random noise
        # X uncertainty: combination of systematic drift and random noise
        x_systematic_var = (self.config.odom_drift_x_systematic * 0.01 * self.total_distance)**2
        x_random_var = self.config.odom_drift_x_random**2
        pose_cov[0] = x_systematic_var + x_random_var + 0.001**2  # x uncertainty
        
        # Y uncertainty: lateral drift effects
        y_systematic_var = (self.config.odom_drift_y_systematic * 0.01 * self.total_distance)**2
        y_random_var = self.config.odom_drift_y_random**2
        pose_cov[7] = y_systematic_var + y_random_var + 0.001**2  # y uncertainty
        
        # Theta uncertainty: heading drift
        theta_systematic_var = (self.config.odom_drift_theta_systematic * 0.01 * self.total_rotation)**2
        theta_random_var = (self.config.odom_drift_theta_random * np.pi/180)**2  # Convert deg to rad
        pose_cov[35] = theta_systematic_var + theta_random_var + 0.001**2  # theta uncertainty
        
        odom_msg.pose.covariance = pose_cov.tolist()
        
        # Twist covariance (fixed small values)
        twist_cov = np.zeros(36)
        twist_cov[0] = 0.001   # vx
        twist_cov[7] = 0.001   # vy
        twist_cov[35] = 0.001  # vtheta
        odom_msg.twist.covariance = twist_cov.tolist()
        
        return odom_msg


class SensorSimulator:
    """Main sensor simulator combining IMU, GPS, and Odometry"""
    
    def __init__(self, config: SensorNoiseConfig):
        self.config = config
        self.imu_sim = ImuSimulator(config)
        self.gps_sim = GpsSimulator(config)
        self.odom_sim = OdometrySimulator(config)
    
    def generate_all_sensors(self, vehicle_state: Dict, dt: float, 
                           timestamp: rclpy.time.Time) -> Tuple[Imu, NavSatFix, Odometry]:
        """
        Generate all sensor data for current vehicle state
        
        Args:
            vehicle_state: Dictionary containing:
                - position: [x, y, z]
                - orientation: [roll, pitch, yaw]
                - linear_velocity: [vx, vy, vz]
                - angular_velocity: [wx, wy, wz]
                - linear_acceleration: [ax, ay, az]
            dt: Time step
            timestamp: ROS timestamp
        
        Returns:
            Tuple of (IMU, GPS, Odometry) messages
        """
        # Extract state
        pos = vehicle_state['position']
        orient = vehicle_state['orientation']
        lin_vel = vehicle_state['linear_velocity']
        ang_vel = vehicle_state['angular_velocity']
        lin_acc = vehicle_state.get('linear_acceleration', [0, 0, 0])
        
        # Generate quaternion from Euler angles
        q = quaternion_from_euler(orient[0], orient[1], orient[2])
        
        # Generate sensor data
        imu_msg = self.imu_sim.generate_imu_data(
            np.array(lin_acc), np.array(ang_vel), q, dt, timestamp
        )
        
        gps_msg = self.gps_sim.generate_gps_data(
            pos[0], pos[1], pos[2], timestamp
        )
        
        odom_msg = self.odom_sim.generate_odometry_data(
            pos[0], pos[1], orient[2],
            lin_vel[0], lin_vel[1], ang_vel[2],
            dt, timestamp
        )
        
        return imu_msg, gps_msg, odom_msg