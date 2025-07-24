#!/usr/bin/env python3
"""Quick test to verify dummy_publisher is working"""

import rclpy
from rclpy.node import Node
import subprocess
import time

class TestDummyPublisher(Node):
    def __init__(self):
        super().__init__('test_dummy_publisher')
        
        # Wait a bit for topics to appear
        time.sleep(2)
        
        # Check available topics
        result = subprocess.run(['ros2', 'topic', 'list'], capture_output=True, text=True)
        self.get_logger().info("Available topics:")
        self.get_logger().info(result.stdout)
        
        # Check TF
        result = subprocess.run(['ros2', 'run', 'tf2_ros', 'tf2_echo', 'odom', 'base_link', '--timeout', '2'], 
                              capture_output=True, text=True)
        self.get_logger().info("TF odom->base_link:")
        self.get_logger().info(result.stdout)
        if result.stderr:
            self.get_logger().error(result.stderr)
        
        # Check node list
        result = subprocess.run(['ros2', 'node', 'list'], capture_output=True, text=True)
        self.get_logger().info("Active nodes:")
        self.get_logger().info(result.stdout)

def main():
    rclpy.init()
    node = TestDummyPublisher()
    rclpy.shutdown()

if __name__ == '__main__':
    main()