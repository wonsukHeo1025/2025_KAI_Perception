#!/usr/bin/env python3
"""
filc 기능 테스트 스크립트
Ouster 토픽 및 카메라 토픽의 존재 여부를 확인하고,
색상 할당 및 보간 기능을 테스트합니다.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
import time
import subprocess
import signal
import sys

class filcTester(Node):
    def __init__(self):
        super().__init__('filc_tester')
        
        self.topics_status = {
            '/ouster/points': False,
            '/ouster/range_image': False,
            '/ouster/intensity_image': False,
            '/filc/colored_points': False,
            '/ouster/interpolated_range_image': False,
            '/ouster/interpolated_intensity_image': False
        }
        
        self.subscribers = {}
        self.create_subscribers()
        
        # 5초 후 결과 출력
        self.timer = self.create_timer(5.0, self.print_status)
        
        self.get_logger().info("filc 기능 테스트 시작...")
        self.get_logger().info("5초 동안 토픽 상태를 확인합니다...")

    def create_subscribers(self):
        """토픽 구독자 생성"""
        for topic in self.topics_status.keys():
            if 'points' in topic:
                self.subscribers[topic] = self.create_subscription(
                    PointCloud2, topic, 
                    lambda msg, t=topic: self.topic_callback(t, msg), 1)
            else:  # image topics
                self.subscribers[topic] = self.create_subscription(
                    Image, topic, 
                    lambda msg, t=topic: self.topic_callback(t, msg), 1)

    def topic_callback(self, topic_name, msg):
        """토픽 메시지 수신 시 호출"""
        if not self.topics_status[topic_name]:
            self.topics_status[topic_name] = True
            self.get_logger().info(f"✓ {topic_name} 토픽 확인됨")

    def print_status(self):
        """현재 상태 출력"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("filc 토픽 상태 리포트")
        self.get_logger().info("="*60)
        
        # 입력 토픽 상태
        self.get_logger().info("입력 토픽:")
        input_topics = ['/ouster/points', '/ouster/range_image', '/ouster/intensity_image']
        for topic in input_topics:
            status = "✓ 활성" if self.topics_status[topic] else "✗ 비활성"
            self.get_logger().info(f"  {topic}: {status}")
        
        # 출력 토픽 상태
        self.get_logger().info("\n출력 토픽:")
        output_topics = ['/filc/colored_points', '/ouster/interpolated_range_image', 
                        '/ouster/interpolated_intensity_image']
        for topic in output_topics:
            status = "✓ 활성" if self.topics_status[topic] else "✗ 비활성"
            self.get_logger().info(f"  {topic}: {status}")
        
        # 종합 평가
        self.get_logger().info("\n종합 평가:")
        if self.topics_status['/ouster/points']:
            self.get_logger().info("✓ 기본 LiDAR 데이터 입력 정상")
        else:
            self.get_logger().warn("✗ LiDAR 포인트 클라우드가 수신되지 않음")
        
        if self.topics_status['/filc/colored_points']:
            self.get_logger().info("✓ 색상 할당된 포인트 클라우드 출력 정상")
        else:
            self.get_logger().warn("✗ 색상 할당 기능이 작동하지 않음")
        
        has_range = self.topics_status['/ouster/range_image']
        has_intensity = self.topics_status['/ouster/intensity_image']
        if has_range and has_intensity:
            self.get_logger().info("✓ 보간 기능 입력 데이터 정상")
            if (self.topics_status['/ouster/interpolated_range_image'] and 
                self.topics_status['/ouster/interpolated_intensity_image']):
                self.get_logger().info("✓ 보간 기능 출력 정상")
            else:
                self.get_logger().warn("✗ 보간 기능이 비활성화되어 있거나 작동하지 않음")
        else:
            self.get_logger().warn("✗ Ouster 이미지 토픽 없음 - 보간 기능 사용 불가")
        
        self.get_logger().info("="*60)
        
        # 노드 종료
        rclpy.shutdown()

def signal_handler(sig, frame):
    """시그널 핸들러"""
    print("\n테스트가 중단되었습니다.")
    rclpy.shutdown()
    sys.exit(0)

def check_filc_node_status():
    """filc 노드 실행 상태 확인"""
    try:
        result = subprocess.run(['ros2', 'node', 'list'], 
                               capture_output=True, text=True, check=True)
        nodes = result.stdout.strip().split('\n')
        
        filc_nodes = [node for node in nodes if 'filc' in node.lower()]
        
        if filc_nodes:
            print("✓ filc 노드들이 실행 중입니다:")
            for node in filc_nodes:
                print(f"  - {node}")
            return True
        else:
            print("✗ filc 노드가 실행되지 않았습니다.")
            print("다음 명령으로 filc을 실행하세요:")
            print("  ros2 launch filc interpolation_demo.launch.py")
            return False
            
    except subprocess.CalledProcessError:
        print("✗ ROS2 명령 실행 실패. ROS2 환경이 설정되어 있는지 확인하세요.")
        return False

def main():
    """메인 함수"""
    signal.signal(signal.SIGINT, signal_handler)
    
    print("filc 기능 테스트 스크립트")
    print("="*50)
    
    # filc 노드 상태 확인
    if not check_filc_node_status():
        print("\n먼저 filc 노드를 실행해주세요.")
        return
    
    print("\n토픽 모니터링을 시작합니다...")
    
    # ROS2 초기화 및 노드 실행
    rclpy.init()
    
    try:
        tester = filcTester()
        rclpy.spin(tester)
    except KeyboardInterrupt:
        print("\n테스트가 중단되었습니다.")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main() 