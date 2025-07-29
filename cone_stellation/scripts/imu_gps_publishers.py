#!/usr/bin/env python3
"""
IMU and GPS publishers that match exact topic formats from input_topic_form.md
Publishes to /ouster/imu and /ublox_gps_node/fix + /ublox_gps_node/fix_velocity
"""

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from geometry_msgs.msg import TwistWithCovarianceStamped
from std_msgs.msg import Header
import utm
from tf_transformations import quaternion_from_euler
from sensor_simulator_enhanced import (
    EnhancedImuSimulator, EnhancedGpsSimulator, 
    EnhancedImuConfig, EnhancedGpsConfig, RtkStatus
)
from typing import Tuple
import math


class RealisticSensorPublisher(Node):
    def __init__(self):
        super().__init__('realistic_sensor_publisher')
        
        # Declare parameters
        self.declare_parameter('motion_type', 'circular')  # circular, figure8, straight, stop_and_go
        self.declare_parameter('radius', 50.0)  # meters for circular motion
        self.declare_parameter('velocity', 10.0)  # m/s
        
        # Get parameters
        self.motion_type = self.get_parameter('motion_type').value
        self.radius = self.get_parameter('radius').value
        self.velocity = self.get_parameter('velocity').value
        
        # Publishers matching exact topics from input_topic_form.md
        self.imu_pub = self.create_publisher(Imu, '/ouster/imu', 10)
        self.gps_fix_pub = self.create_publisher(NavSatFix, '/ublox_gps_node/fix', 10)
        self.gps_vel_pub = self.create_publisher(TwistWithCovarianceStamped, '/ublox_gps_node/fix_velocity', 10)
        
        # Initialize enhanced simulators
        imu_config = EnhancedImuConfig()
        gps_config = EnhancedGpsConfig()
        gps_config.rtk_mode = "rtk_fix"  # Start with RTK fix
        
        self.imu_sim = EnhancedImuSimulator(imu_config)
        
        # Seoul reference point from input_topic_form.md
        self.ref_lat = 37.5413753
        self.ref_lon = 127.0779785
        self.ref_alt = 39.5
        
        # Initialize GPS simulator with same origin
        self.gps_sim = EnhancedGpsSimulator(gps_config, self.ref_lat, self.ref_lon)
        
        # Convert to UTM for local calculations
        self.utm_x_ref, self.utm_y_ref, self.utm_zone, self.utm_letter = utm.from_latlon(self.ref_lat, self.ref_lon)
        
        # Motion state
        self.t = 0.0
        self.x = 0.0  # Local position
        self.y = 0.0
        self.theta = 0.0
        self.vx = 0.0  # Local velocity
        self.vy = 0.0
        self.vtheta = 0.0
        
        # Create timers
        self.imu_timer = self.create_timer(0.01, self.publish_imu)  # 100Hz like real data
        self.gps_timer = self.create_timer(0.125, self.publish_gps)  # ~8Hz like real data
        
        self.get_logger().info(f'Realistic sensor publisher started with {self.motion_type} motion')
        
    def update_motion(self, dt: float):
        """Update vehicle motion based on selected profile"""
        self.t += dt
        
        if self.motion_type == 'circular':
            # Circular motion
            omega = self.velocity / self.radius
            self.theta = omega * self.t
            self.x = self.radius * np.cos(self.theta)
            self.y = self.radius * np.sin(self.theta)
            self.vx = -self.radius * omega * np.sin(self.theta)
            self.vy = self.radius * omega * np.cos(self.theta)
            self.vtheta = omega
            
        elif self.motion_type == 'figure8':
            # Figure-8 motion
            omega = self.velocity / self.radius
            t = omega * self.t
            self.x = self.radius * np.sin(t)
            self.y = self.radius * np.sin(t) * np.cos(t)
            self.vx = self.radius * omega * np.cos(t)
            self.vy = self.radius * omega * (np.cos(2*t))
            self.theta = np.arctan2(self.vy, self.vx)
            self.vtheta = (self.vx * (-2*omega*np.sin(2*t)) - self.vy * (-omega*np.sin(t))) / (self.vx**2 + self.vy**2 + 1e-6)
            
        elif self.motion_type == 'straight':
            # Straight line motion
            self.x = self.velocity * self.t
            self.y = 0.0
            self.theta = 0.0
            self.vx = self.velocity
            self.vy = 0.0
            self.vtheta = 0.0
            
        elif self.motion_type == 'stop_and_go':
            # Stop and go motion (10s move, 5s stop)
            cycle_time = 15.0
            move_time = 10.0
            cycle_phase = self.t % cycle_time
            
            if cycle_phase < move_time:
                # Moving phase
                self.vx = self.velocity
                self.vy = 0.0
                self.vtheta = 0.0
                self.x += self.vx * dt
            else:
                # Stopped phase
                self.vx = 0.0
                self.vy = 0.0
                self.vtheta = 0.0
    
    def publish_imu(self):
        """Publish IMU data matching /ouster/imu format"""
        # Update motion
        dt = 0.01
        self.update_motion(dt)
        
        # Get ground truth accelerations (in body frame)
        # For circular motion: centripetal acceleration
        accel_truth = np.array([0.0, 0.0, 9.81])  # Gravity
        if self.motion_type == 'circular':
            # Add centripetal acceleration
            centripetal = self.velocity**2 / self.radius
            accel_truth[0] += centripetal * np.sin(self.theta)
            accel_truth[1] += -centripetal * np.cos(self.theta)
        
        # Angular velocity
        angular_vel_truth = np.array([0.0, 0.0, self.vtheta])
        
        # Simulate IMU measurements
        timestamp = self.get_clock().now()
        dt_sim = 0.01  # 100Hz
        
        # Get orientation quaternion
        quat = quaternion_from_euler(0, 0, self.theta)  # roll, pitch, yaw
        
        # Generate IMU data
        imu_msg = self.imu_sim.generate_imu_data(
            accel_truth,
            angular_vel_truth,
            quat,
            dt_sim,
            timestamp
        )
        
        # Override some fields to match exact real data format
        imu_msg.header.frame_id = 'os_imu'  # Match real data frame
        
        # Real Ouster IMU doesn't provide orientation
        imu_msg.orientation.x = 0.0
        imu_msg.orientation.y = 0.0
        imu_msg.orientation.z = 0.0
        imu_msg.orientation.w = 1.0
        imu_msg.orientation_covariance = [-1.0] * 9
        
        # IMU covariances are automatically calculated by the IMU simulator
        # based on the actual noise standard deviations used:
        # - Angular velocity covariance = gyro_noise_sigma^2
        # - Linear acceleration covariance = accel_noise_sigma^2
        # This ensures the published covariances match the actual noise in the data
        
        self.imu_pub.publish(imu_msg)
    
    def publish_gps(self):
        """Publish GPS fix and velocity matching /ublox_gps_node format"""
        # Update motion
        dt = 0.125
        self.update_motion(dt)
        
        # Ground truth position (local coordinates for GPS simulator)
        position_truth = np.array([self.x, self.y, self.ref_alt])
        velocity_truth = np.array([self.vx, self.vy, 0.0])
        
        # Simulate GPS measurements
        timestamp = self.get_clock().now()
        
        # Generate GPS fix data
        fix_msg = self.gps_sim.generate_gps_data(
            position_truth[0],  # x in local frame
            position_truth[1],  # y in local frame
            position_truth[2],  # z (altitude)
            dt,
            timestamp
        )
        
        # Override frame_id to match real data
        fix_msg.header.frame_id = 'gps'
        
        self.gps_fix_pub.publish(fix_msg)
        
        # Create TwistWithCovarianceStamped for velocity
        vel_msg = TwistWithCovarianceStamped()
        vel_msg.header.stamp = timestamp.to_msg()
        vel_msg.header.frame_id = 'gps'
        
        # Velocity in ENU frame with noise based on RTK status
        rtk_status = self.gps_sim.rtk_status
        if rtk_status == RtkStatus.FIX:
            vel_noise_std = 0.02
        elif rtk_status == RtkStatus.FLOAT:
            vel_noise_std = 0.1
        else:
            vel_noise_std = 0.5
        
        # Generate actual noise for each component
        vel_noise_x = np.random.normal(0, vel_noise_std)
        vel_noise_y = np.random.normal(0, vel_noise_std)
        vel_noise_z = np.random.normal(0, vel_noise_std)
        
        vel_msg.twist.twist.linear.x = velocity_truth[0] + vel_noise_x
        vel_msg.twist.twist.linear.y = velocity_truth[1] + vel_noise_y
        vel_msg.twist.twist.linear.z = velocity_truth[2] + vel_noise_z
        
        # No angular velocity from GPS
        vel_msg.twist.twist.angular.x = 0.0
        vel_msg.twist.twist.angular.y = 0.0
        vel_msg.twist.twist.angular.z = 0.0
        
        # Calculate velocity covariance from actual noise values
        # Covariance matrix diagonal = variance of actual noise
        vel_var_x = vel_noise_x ** 2
        vel_var_y = vel_noise_y ** 2
        vel_var_z = vel_noise_z ** 2
        
        # For a more accurate representation, we could use the expected variance
        # based on the noise standard deviation, which gives a more stable estimate
        expected_variance = vel_noise_std ** 2
        
        vel_msg.twist.covariance = [
            expected_variance, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, expected_variance, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, expected_variance, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, -1.0, 0.0, 0.0,  # Angular velocity not provided
            0.0, 0.0, 0.0, 0.0, -1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, -1.0
        ]
        
        self.gps_vel_pub.publish(vel_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RealisticSensorPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()