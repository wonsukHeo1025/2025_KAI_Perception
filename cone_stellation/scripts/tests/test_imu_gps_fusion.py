#!/usr/bin/env python3
"""
Test script for IMU-GPS fusion with various motion profiles
Demonstrates enhanced sensor simulators for robot_localization testing
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Header
import yaml

# Import enhanced simulators
from sensor_simulator_enhanced import (
    EnhancedImuSimulator, EnhancedGpsSimulator,
    EnhancedImuConfig, EnhancedGpsConfig,
    EnhancedSensorSimulator
)
from sensor_simulator import SensorNoiseConfig
from motion_controller import VehicleState


class MotionProfile:
    """Generate various motion profiles for testing"""
    
    @staticmethod
    def straight_line(duration: float, speed: float = 5.0, dt: float = 0.01) -> List[VehicleState]:
        """Straight line motion with constant speed"""
        states = []
        t = 0.0
        
        while t < duration:
            state = VehicleState(
                position=np.array([speed * t, 0.0, 0.0]),
                orientation=np.array([0.0, 0.0, 0.0]),
                linear_velocity=np.array([speed, 0.0, 0.0]),
                angular_velocity=np.array([0.0, 0.0, 0.0]),
                linear_acceleration=np.array([0.0, 0.0, 0.0])
            )
            states.append(state)
            t += dt
        
        return states
    
    @staticmethod
    def circular_motion(duration: float, radius: float = 20.0, 
                       angular_speed: float = 0.25, dt: float = 0.01) -> List[VehicleState]:
        """Circular motion with constant angular velocity"""
        states = []
        t = 0.0
        
        while t < duration:
            theta = angular_speed * t
            speed = radius * angular_speed
            
            state = VehicleState(
                position=np.array([
                    radius * np.cos(theta),
                    radius * np.sin(theta),
                    0.0
                ]),
                orientation=np.array([0.0, 0.0, theta + np.pi/2]),
                linear_velocity=np.array([
                    -speed * np.sin(theta),
                    speed * np.cos(theta),
                    0.0
                ]),
                angular_velocity=np.array([0.0, 0.0, angular_speed]),
                linear_acceleration=np.array([
                    -speed * angular_speed * np.cos(theta),
                    -speed * angular_speed * np.sin(theta),
                    0.0
                ])
            )
            states.append(state)
            t += dt
        
        return states
    
    @staticmethod
    def figure_eight(duration: float, size: float = 20.0, dt: float = 0.01) -> List[VehicleState]:
        """Figure-8 motion pattern"""
        states = []
        t = 0.0
        omega = 2 * np.pi / 20.0  # 20 seconds per loop
        
        while t < duration:
            # Lemniscate parametric equations
            theta = omega * t
            scale = size / (1 + np.sin(theta)**2)
            
            x = scale * np.cos(theta)
            y = scale * np.sin(theta) * np.cos(theta)
            
            # Velocities (derivatives)
            dx_dt = -scale * np.sin(theta) * omega
            dy_dt = scale * np.cos(2*theta) * omega
            
            speed = np.sqrt(dx_dt**2 + dy_dt**2)
            heading = np.arctan2(dy_dt, dx_dt)
            
            # Angular velocity (heading derivative)
            dheading_dt = (2 * scale * np.sin(2*theta) * omega**2) / (dx_dt**2 + dy_dt**2)
            
            state = VehicleState(
                position=np.array([x, y, 0.0]),
                orientation=np.array([0.0, 0.0, heading]),
                linear_velocity=np.array([dx_dt, dy_dt, 0.0]),
                angular_velocity=np.array([0.0, 0.0, dheading_dt]),
                linear_acceleration=np.array([0.0, 0.0, 0.0])  # Simplified
            )
            states.append(state)
            t += dt
        
        return states
    
    @staticmethod
    def stop_and_go(duration: float, stop_interval: float = 5.0, 
                    speed: float = 5.0, dt: float = 0.01) -> List[VehicleState]:
        """Stop and go motion pattern"""
        states = []
        t = 0.0
        x = 0.0
        
        while t < duration:
            # Determine if moving or stopped
            cycle_time = t % (2 * stop_interval)
            
            if cycle_time < stop_interval:
                # Moving phase
                v = speed
                a = 0.0
                x += v * dt
            else:
                # Stopped phase
                v = 0.0
                a = 0.0
            
            state = VehicleState(
                position=np.array([x, 0.0, 0.0]),
                orientation=np.array([0.0, 0.0, 0.0]),
                linear_velocity=np.array([v, 0.0, 0.0]),
                angular_velocity=np.array([0.0, 0.0, 0.0]),
                linear_acceleration=np.array([a, 0.0, 0.0])
            )
            states.append(state)
            t += dt
        
        return states


class ImuGpsFusionTester(Node):
    """ROS2 node for testing IMU-GPS fusion with enhanced simulators"""
    
    def __init__(self):
        super().__init__('imu_gps_fusion_tester')
        
        # Create publishers
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        self.gps_pub = self.create_publisher(NavSatFix, '/gps/fix', 10)
        self.utm_pub = self.create_publisher(PoseWithCovarianceStamped, '/gps/utm', 10)
        self.ground_truth_pub = self.create_publisher(Odometry, '/ground_truth', 10)
        
        # Load configuration
        self.declare_parameter('motion_profile', 'circular')
        self.declare_parameter('duration', 60.0)
        self.declare_parameter('publish_rate', 100.0)
        
        # Create enhanced sensor simulators
        imu_config = EnhancedImuConfig(
            gyro_noise_density=0.005,
            accel_noise_density=0.01,
            temperature_time_constant=300.0,
            gyro_scale_factor_ppm=500,
            accel_scale_factor_ppm=1000
        )
        
        gps_config = EnhancedGpsConfig(
            rtk_mode="rtk_fix",
            fix_loss_probability=0.001,
            multipath_amplitude=0.3
        )
        
        # For odometry noise
        odom_config = SensorNoiseConfig(
            odom_drift_x_systematic=0.02,
            odom_drift_y_systematic=0.02,
            odom_drift_theta_systematic=0.02
        )
        
        self.sensor_sim = EnhancedSensorSimulator(imu_config, gps_config, odom_config)
        
        # Generate motion profile
        profile_name = self.get_parameter('motion_profile').value
        duration = self.get_parameter('duration').value
        
        self.get_logger().info(f"Generating {profile_name} motion profile for {duration}s")
        
        if profile_name == 'straight':
            self.motion_states = MotionProfile.straight_line(duration)
        elif profile_name == 'circular':
            self.motion_states = MotionProfile.circular_motion(duration)
        elif profile_name == 'figure8':
            self.motion_states = MotionProfile.figure_eight(duration)
        elif profile_name == 'stop_go':
            self.motion_states = MotionProfile.stop_and_go(duration)
        else:
            self.get_logger().error(f"Unknown motion profile: {profile_name}")
            self.motion_states = []
        
        # State tracking
        self.state_index = 0
        self.start_time = self.get_clock().now()
        
        # Create timer
        rate = self.get_parameter('publish_rate').value
        self.timer = self.create_timer(1.0 / rate, self.publish_sensors)
        
        self.get_logger().info("IMU-GPS fusion tester started")
    
    def publish_sensors(self):
        """Publish sensor data for current state"""
        if self.state_index >= len(self.motion_states):
            self.get_logger().info("Motion profile completed")
            self.timer.cancel()
            return
        
        # Get current state
        state = self.motion_states[self.state_index]
        self.state_index += 1
        
        # Current timestamp
        current_time = self.get_clock().now()
        dt = 0.01  # 100Hz
        
        # Generate sensor data
        sensor_data = self.sensor_sim.generate_all_sensors(
            state.to_dict(), dt, current_time
        )
        
        # Publish all sensors
        self.imu_pub.publish(sensor_data['imu'])
        self.gps_pub.publish(sensor_data['gps'])
        self.utm_pub.publish(sensor_data['utm_pose'])
        
        # Publish ground truth
        ground_truth = Odometry()
        ground_truth.header.stamp = current_time.to_msg()
        ground_truth.header.frame_id = "odom"
        ground_truth.child_frame_id = "base_link"
        
        ground_truth.pose.pose.position.x = state.position[0]
        ground_truth.pose.pose.position.y = state.position[1]
        ground_truth.pose.pose.position.z = state.position[2]
        
        # Convert orientation to quaternion
        from tf_transformations import quaternion_from_euler
        q = quaternion_from_euler(
            state.orientation[0], 
            state.orientation[1], 
            state.orientation[2]
        )
        ground_truth.pose.pose.orientation.x = q[0]
        ground_truth.pose.pose.orientation.y = q[1]
        ground_truth.pose.pose.orientation.z = q[2]
        ground_truth.pose.pose.orientation.w = q[3]
        
        ground_truth.twist.twist.linear.x = state.linear_velocity[0]
        ground_truth.twist.twist.linear.y = state.linear_velocity[1]
        ground_truth.twist.twist.linear.z = state.linear_velocity[2]
        ground_truth.twist.twist.angular.x = state.angular_velocity[0]
        ground_truth.twist.twist.angular.y = state.angular_velocity[1]
        ground_truth.twist.twist.angular.z = state.angular_velocity[2]
        
        self.ground_truth_pub.publish(ground_truth)
        
        # Log GPS status occasionally
        if self.state_index % 100 == 0:
            gps_status = sensor_data['gps'].status.status
            status_map = {0: "NO_FIX", 1: "FIX", 2: "DGPS", 3: "GBAS"}
            self.get_logger().info(
                f"Time: {self.state_index/100:.1f}s, "
                f"GPS: {status_map.get(gps_status, 'UNKNOWN')}, "
                f"Pos: ({state.position[0]:.1f}, {state.position[1]:.1f})"
            )


def plot_motion_profiles():
    """Visualize different motion profiles"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    profiles = [
        ("Straight Line", MotionProfile.straight_line(20)),
        ("Circular", MotionProfile.circular_motion(25)),
        ("Figure-8", MotionProfile.figure_eight(40)),
        ("Stop & Go", MotionProfile.stop_and_go(20))
    ]
    
    for ax, (name, states) in zip(axes, profiles):
        positions = np.array([s.position for s in states])
        ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(name)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        # Mark start and end
        ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
        ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('/tmp/motion_profiles.png', dpi=150)
    print("Motion profiles saved to /tmp/motion_profiles.png")


def main(args=None):
    # First plot the motion profiles
    print("Generating motion profile plots...")
    plot_motion_profiles()
    
    # Then run the ROS node
    rclpy.init(args=args)
    node = ImuGpsFusionTester()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()