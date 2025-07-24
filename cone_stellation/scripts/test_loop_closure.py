#!/usr/bin/env python3
"""
Test script to verify loop closure implementation by simulating a simple loop trajectory
"""

import rclpy
from rclpy.node import Node
import numpy as np
import time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from custom_interface.msg import TrackedConeArray, TrackedCone

class LoopClosureTestNode(Node):
    def __init__(self):
        super().__init__('loop_closure_test_node')
        
        # Publishers
        self.cone_pub = self.create_publisher(
            TrackedConeArray, 
            '/tracked_cone_array', 
            10
        )
        self.odom_pub = self.create_publisher(
            Odometry,
            '/odom',
            10
        )
        
        # Parameters
        self.track_radius = 20.0  # meters
        self.cone_spacing = 3.0   # meters between cones
        self.vehicle_speed = 2.0  # m/s
        self.dt = 0.1            # 10 Hz
        
        # State
        self.time = 0.0
        self.position = np.array([self.track_radius, 0.0])  # Start at (R, 0)
        self.heading = np.pi / 2  # Start facing +Y
        
        # Generate cone positions (circular track)
        self.generate_track_cones()
        
        # Timer for simulation
        self.timer = self.create_timer(self.dt, self.timer_callback)
        
        self.get_logger().info(f"Loop closure test started with {len(self.inner_cones)} inner and {len(self.outer_cones)} outer cones")
        
    def generate_track_cones(self):
        """Generate cones for a circular track"""
        # Inner and outer radii
        inner_radius = self.track_radius - 2.0
        outer_radius = self.track_radius + 2.0
        
        # Number of cones
        inner_circumference = 2 * np.pi * inner_radius
        outer_circumference = 2 * np.pi * outer_radius
        
        n_inner = int(inner_circumference / self.cone_spacing)
        n_outer = int(outer_circumference / self.cone_spacing)
        
        # Generate cone positions
        self.inner_cones = []
        self.outer_cones = []
        
        for i in range(n_inner):
            angle = 2 * np.pi * i / n_inner
            x = inner_radius * np.cos(angle)
            y = inner_radius * np.sin(angle)
            self.inner_cones.append((x, y, 'BLUE'))  # Inner cones are blue
            
        for i in range(n_outer):
            angle = 2 * np.pi * i / n_outer
            x = outer_radius * np.cos(angle)
            y = outer_radius * np.sin(angle)
            self.outer_cones.append((x, y, 'YELLOW'))  # Outer cones are yellow
            
        self.all_cones = self.inner_cones + self.outer_cones
        
    def timer_callback(self):
        """Main simulation loop"""
        # Update vehicle position (circular motion)
        angular_velocity = self.vehicle_speed / self.track_radius
        self.heading += angular_velocity * self.dt
        
        # Normalize heading
        self.heading = np.fmod(self.heading, 2 * np.pi)
        
        # Update position on circular path
        angle_on_track = np.arctan2(self.position[1], self.position[0])
        angle_on_track += angular_velocity * self.dt
        
        self.position[0] = self.track_radius * np.cos(angle_on_track)
        self.position[1] = self.track_radius * np.sin(angle_on_track)
        
        # Add some noise to simulate real sensor
        noise_std = 0.05
        self.position += np.random.normal(0, noise_std, 2)
        
        # Publish odometry
        self.publish_odometry()
        
        # Detect and publish visible cones
        self.publish_visible_cones()
        
        self.time += self.dt
        
        # Log progress
        if int(self.time * 10) % 50 == 0:  # Every 5 seconds
            angle_degrees = np.degrees(angle_on_track) % 360
            self.get_logger().info(f"Time: {self.time:.1f}s, Angle: {angle_degrees:.1f}°, Pos: ({self.position[0]:.1f}, {self.position[1]:.1f})")
            
            # Check if we've completed a loop
            if self.time > 30.0 and angle_degrees < 10.0:
                self.get_logger().info("=== COMPLETED LOOP - EXPECTING LOOP CLOSURE ===")
        
    def publish_odometry(self):
        """Publish vehicle odometry"""
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        
        # Position
        odom_msg.pose.pose.position.x = self.position[0]
        odom_msg.pose.pose.position.y = self.position[1]
        odom_msg.pose.pose.position.z = 0.0
        
        # Orientation (quaternion from heading)
        odom_msg.pose.pose.orientation.z = np.sin(self.heading / 2)
        odom_msg.pose.pose.orientation.w = np.cos(self.heading / 2)
        
        # Velocity
        odom_msg.twist.twist.linear.x = self.vehicle_speed
        odom_msg.twist.twist.angular.z = self.vehicle_speed / self.track_radius
        
        self.odom_pub.publish(odom_msg)
        
    def publish_visible_cones(self):
        """Publish cones visible from current position"""
        max_range = 15.0  # Maximum detection range
        fov = np.pi * 0.75  # 135 degree field of view
        
        tracked_array = TrackedConeArray()
        tracked_array.header.stamp = self.get_clock().now().to_msg()
        tracked_array.header.frame_id = 'base_link'
        
        cone_id = 0
        for cone_x, cone_y, color in self.all_cones:
            # Transform to vehicle frame
            dx = cone_x - self.position[0]
            dy = cone_y - self.position[1]
            
            # Rotate to vehicle frame
            cos_h = np.cos(-self.heading)
            sin_h = np.sin(-self.heading)
            
            x_vehicle = dx * cos_h - dy * sin_h
            y_vehicle = dx * sin_h + dy * cos_h
            
            # Check if in range and FOV
            distance = np.sqrt(x_vehicle**2 + y_vehicle**2)
            angle = np.arctan2(y_vehicle, x_vehicle)
            
            if distance < max_range and abs(angle) < fov/2 and x_vehicle > 0:
                # Create tracked cone
                cone = TrackedCone()
                cone.id = cone_id  # Consistent ID for same physical cone
                cone.position.x = x_vehicle + np.random.normal(0, 0.1)  # Add measurement noise
                cone.position.y = y_vehicle + np.random.normal(0, 0.1)
                cone.position.z = 0.0
                cone.color = color
                cone.confidence = 0.9
                
                tracked_array.cones.append(cone)
                
            cone_id += 1
        
        if len(tracked_array.cones) > 0:
            self.cone_pub.publish(tracked_array)

def main(args=None):
    rclpy.init(args=args)
    node = LoopClosureTestNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()