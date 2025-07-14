#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
from datetime import datetime

class InterpolationVisualizer(Node):
    def __init__(self):
        super().__init__('interpolation_visualizer')
        
        # 파라미터
        self.declare_parameter('original_topic', '/ouster/points')
        self.declare_parameter('interpolated_topic', '/ouster/interpolated_points')
        
        original_topic = self.get_parameter('original_topic').get_parameter_value().string_value
        interpolated_topic = self.get_parameter('interpolated_topic').get_parameter_value().string_value
        
        # QoS 프로파일 설정 (센서 데이터용)
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 구독자
        self.original_sub = self.create_subscription(
            PointCloud2, original_topic, self.original_callback, sensor_qos)
        self.interp_sub = self.create_subscription(
            PointCloud2, interpolated_topic, self.interp_callback, sensor_qos)
        
        # 통계
        self.original_stats = None
        self.interp_stats = None
        self.frame_count = 0
        
        # 타이머
        self.timer = self.create_timer(2.0, self.print_stats)
        
        self.get_logger().info(f'Interpolation Visualizer Started')
        self.get_logger().info(f'  Original topic: {original_topic}')
        self.get_logger().info(f'  Interpolated topic: {interpolated_topic}')
    
    def calculate_stats(self, msg):
        """포인트클라우드 통계 계산"""
        stats = {
            'height': msg.height,
            'width': msg.width,
            'total_points': msg.height * msg.width,
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'frame_id': msg.header.frame_id,
            'is_dense': msg.is_dense,
            'point_step': msg.point_step,
            'row_step': msg.row_step
        }
        
        # 데이터 크기 계산
        stats['data_size_mb'] = len(msg.data) / (1024 * 1024)
        
        return stats
    
    def original_callback(self, msg):
        self.original_stats = self.calculate_stats(msg)
    
    def interp_callback(self, msg):
        self.interp_stats = self.calculate_stats(msg)
        self.frame_count += 1
    
    def print_stats(self):
        """통계 출력"""
        self.get_logger().info('='*60)
        self.get_logger().info(f'Interpolation Statistics - {datetime.now().strftime("%H:%M:%S")}')
        self.get_logger().info('='*60)
        
        if self.original_stats:
            self.get_logger().info('Original PointCloud:')
            self.get_logger().info(f'  Dimensions: {self.original_stats["height"]}x{self.original_stats["width"]}')
            self.get_logger().info(f'  Total points: {self.original_stats["total_points"]:,}')
            self.get_logger().info(f'  Data size: {self.original_stats["data_size_mb"]:.2f} MB')
            self.get_logger().info(f'  Frame ID: {self.original_stats["frame_id"]}')
        
        if self.interp_stats:
            self.get_logger().info('Interpolated PointCloud:')
            self.get_logger().info(f'  Dimensions: {self.interp_stats["height"]}x{self.interp_stats["width"]}')
            self.get_logger().info(f'  Total points: {self.interp_stats["total_points"]:,}')
            self.get_logger().info(f'  Data size: {self.interp_stats["data_size_mb"]:.2f} MB')
            
            if self.original_stats:
                # 비교 통계
                height_ratio = self.interp_stats["height"] / self.original_stats["height"]
                points_ratio = self.interp_stats["total_points"] / self.original_stats["total_points"]
                size_ratio = self.interp_stats["data_size_mb"] / self.original_stats["data_size_mb"]
                
                self.get_logger().info('Comparison:')
                self.get_logger().info(f'  Height ratio: {height_ratio:.2f}x')
                self.get_logger().info(f'  Points ratio: {points_ratio:.2f}x')
                self.get_logger().info(f'  Data size ratio: {size_ratio:.2f}x')
        
        if self.frame_count > 0:
            self.get_logger().info(f'Frames processed: {self.frame_count}')
        
        if not self.original_stats and not self.interp_stats:
            self.get_logger().warn('No data received yet...')
        
        self.get_logger().info('='*60 + '\n')

def main(args=None):
    rclpy.init(args=args)
    node = InterpolationVisualizer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()