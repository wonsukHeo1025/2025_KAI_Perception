#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
import time
from collections import defaultdict
import statistics

class InterpolationBenchmark(Node):
    def __init__(self):
        super().__init__('interpolation_benchmark')
        
        # 모니터링할 토픽 목록
        self.topics_to_monitor = {
            # 원본
            '/ouster/points': 'Original PointCloud',
            '/ouster/range_image': 'Original Range Image',
            '/ouster/intensity_image': 'Original Intensity Image',
            
            # 보간 결과
            '/ouster/test_interpolated_points': 'Test Interpolation (Linear)',
            '/ouster/spherical_interpolated_points': 'Spherical Interpolation',
            '/ouster/interpolated_points_from_images': 'Image-based Interpolation',
            '/ouster/interpolated_range_image': 'Interpolated Range Image',
            '/ouster/interpolated_intensity_image': 'Interpolated Intensity Image',
        }
        
        # 통계 저장
        self.stats = defaultdict(lambda: {
            'count': 0,
            'last_time': None,
            'intervals': [],
            'sizes': [],
            'heights': [],
            'widths': []
        })
        
        # QoS 프로파일 설정 (센서 데이터용)
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 구독자 생성
        for topic in self.topics_to_monitor:
            if 'image' in topic.lower():
                self.create_subscription(
                    Image, topic, 
                    lambda msg, t=topic: self.image_callback(msg, t), 
                    sensor_qos)
            else:
                self.create_subscription(
                    PointCloud2, topic, 
                    lambda msg, t=topic: self.pointcloud_callback(msg, t), 
                    sensor_qos)
        
        # 리포트 타이머
        self.timer = self.create_timer(5.0, self.print_report)
        
        self.get_logger().info('Interpolation Benchmark Started')
        self.get_logger().info(f'Monitoring {len(self.topics_to_monitor)} topics')
    
    def pointcloud_callback(self, msg, topic):
        current_time = time.time()
        stats = self.stats[topic]
        
        # 시간 간격 계산
        if stats['last_time'] is not None:
            interval = (current_time - stats['last_time']) * 1000  # ms
            stats['intervals'].append(interval)
            if len(stats['intervals']) > 100:  # 최근 100개만 유지
                stats['intervals'].pop(0)
        
        stats['last_time'] = current_time
        stats['count'] += 1
        stats['sizes'].append(len(msg.data) / (1024 * 1024))  # MB
        stats['heights'].append(msg.height)
        stats['widths'].append(msg.width)
        
        # 유지 관리
        if len(stats['sizes']) > 100:
            stats['sizes'].pop(0)
            stats['heights'].pop(0)
            stats['widths'].pop(0)
    
    def image_callback(self, msg, topic):
        current_time = time.time()
        stats = self.stats[topic]
        
        # 시간 간격 계산
        if stats['last_time'] is not None:
            interval = (current_time - stats['last_time']) * 1000  # ms
            stats['intervals'].append(interval)
            if len(stats['intervals']) > 100:
                stats['intervals'].pop(0)
        
        stats['last_time'] = current_time
        stats['count'] += 1
        stats['sizes'].append(len(msg.data) / (1024 * 1024))  # MB
        stats['heights'].append(msg.height)
        stats['widths'].append(msg.width)
        
        # 유지 관리
        if len(stats['sizes']) > 100:
            stats['sizes'].pop(0)
            stats['heights'].pop(0)
            stats['widths'].pop(0)
    
    def print_report(self):
        self.get_logger().info('='*80)
        self.get_logger().info('INTERPOLATION BENCHMARK REPORT')
        self.get_logger().info('='*80)
        
        # 활성 토픽 분류
        active_topics = []
        inactive_topics = []
        
        for topic, name in self.topics_to_monitor.items():
            if self.stats[topic]['count'] > 0:
                active_topics.append((topic, name))
            else:
                inactive_topics.append((topic, name))
        
        # 활성 토픽 리포트
        if active_topics:
            self.get_logger().info('\nACTIVE TOPICS:')
            self.get_logger().info('-'*80)
            self.get_logger().info(f'{"Topic":<45} {"Hz":>6} {"Latency":>10} {"Size":>8} {"Dims":>12}')
            self.get_logger().info('-'*80)
            
            for topic, name in active_topics:
                stats = self.stats[topic]
                
                if stats['intervals']:
                    avg_interval = statistics.mean(stats['intervals'])
                    hz = 1000.0 / avg_interval if avg_interval > 0 else 0
                    latency_str = f"{avg_interval:.1f} ms"
                else:
                    hz = 0
                    latency_str = "N/A"
                
                if stats['sizes']:
                    avg_size = statistics.mean(stats['sizes'])
                    size_str = f"{avg_size:.2f} MB"
                else:
                    size_str = "N/A"
                
                if stats['heights'] and stats['widths']:
                    h = int(statistics.mode(stats['heights']))
                    w = int(statistics.mode(stats['widths']))
                    dims_str = f"{h}x{w}"
                else:
                    dims_str = "N/A"
                
                self.get_logger().info(
                    f'{name:<45} {hz:>6.1f} {latency_str:>10} {size_str:>8} {dims_str:>12}'
                )
        
        # 비활성 토픽
        if inactive_topics:
            self.get_logger().info('\nINACTIVE TOPICS:')
            for topic, name in inactive_topics:
                self.get_logger().info(f'  - {name} ({topic})')
        
        # 성능 비교
        self.get_logger().info('\n' + '='*80)
        self.get_logger().info('PERFORMANCE COMPARISON:')
        self.get_logger().info('-'*80)
        
        # 원본 대비 비교
        if '/ouster/points' in [t for t, _ in active_topics]:
            original_stats = self.stats['/ouster/points']
            if original_stats['intervals']:
                original_hz = 1000.0 / statistics.mean(original_stats['intervals'])
                
                comparisons = []
                for topic, name in active_topics:
                    if 'interpolated' in topic and 'points' in topic:
                        stats = self.stats[topic]
                        if stats['intervals']:
                            hz = 1000.0 / statistics.mean(stats['intervals'])
                            ratio = hz / original_hz if original_hz > 0 else 0
                            latency = statistics.mean(stats['intervals']) - statistics.mean(original_stats['intervals'])
                            comparisons.append((name, ratio, latency))
                
                if comparisons:
                    for name, ratio, latency in sorted(comparisons, key=lambda x: x[1], reverse=True):
                        self.get_logger().info(
                            f'{name}: {ratio:.2%} of original speed, +{latency:.1f}ms latency'
                        )
        
        self.get_logger().info('='*80 + '\n')

def main(args=None):
    rclpy.init(args=args)
    node = InterpolationBenchmark()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()