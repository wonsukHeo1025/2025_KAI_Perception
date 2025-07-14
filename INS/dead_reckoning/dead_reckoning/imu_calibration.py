#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Imu
import numpy as np
import json
import os
import time
import signal
import sys
from datetime import datetime

class IMUCalibration(Node):
    def __init__(self):
        super().__init__('imu_calibration')
        
        # QoS 프로파일 설정 (Best Effort)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # IMU 구독자 설정
        self.subscription = self.create_subscription(
            Imu,
            '/ouster/imu',
            self.imu_callback,
            qos_profile
        )
        
        # 데이터 저장용 리스트
        self.accel_data = []
        self.gyro_data = []
        
        # 캘리브레이션 상태
        self.is_collecting = True  # 자동으로 시작
        self.start_time = time.time()
        
        # 캘리브레이션 결과
        self.accel_bias = np.array([0.0, 0.0, 0.0])
        self.gyro_bias = np.array([0.0, 0.0, 0.0])
        self.gravity_magnitude = 9.81
        
        # 진행률 표시를 위한 변수
        self.last_progress_time = time.time()
        self.progress_interval = 5.0  # 5초마다 진행률 표시
        
        self.get_logger().info('IMU 정적 캘리브레이션을 시작합니다...')
        self.get_logger().info('데이터 수집 중... (Ctrl+C로 중단 및 결과 확인)')
        self.get_logger().warn('센서가 수평하고 정지된 상태인지 확인하세요!')
        
    def imu_callback(self, msg):
        if self.is_collecting:
            # 가속도 데이터 (m/s²)
            accel = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])
            
            # 각속도 데이터 (rad/s)
            gyro = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])
            
            self.accel_data.append(accel)
            self.gyro_data.append(gyro)
            
            # 진행률 표시
            current_time = time.time()
            if current_time - self.last_progress_time >= self.progress_interval:
                elapsed_time = current_time - self.start_time
                sample_count = len(self.accel_data)
                
                self.get_logger().info(f"경과 시간: {elapsed_time:.1f}초 | 샘플 수: {sample_count}")
                self.last_progress_time = current_time
    
    def calculate_bias(self):
        """바이어스 계산"""
        if len(self.accel_data) == 0:
            self.get_logger().error("수집된 데이터가 없습니다.")
            return False
            
        accel_array = np.array(self.accel_data)
        gyro_array = np.array(self.gyro_data)
        
        # 자이로스코프 바이어스 (평균값)
        self.gyro_bias = np.mean(gyro_array, axis=0)
        
        # 가속도계 바이어스 계산
        accel_mean = np.mean(accel_array, axis=0)
        
        # 중력 방향 추정 (가장 큰 성분이 중력 방향)
        gravity_axis = np.argmax(np.abs(accel_mean))
        gravity_sign = np.sign(accel_mean[gravity_axis])
        
        # 중력 벡터 생성
        gravity_vector = np.zeros(3)
        gravity_vector[gravity_axis] = gravity_sign * self.gravity_magnitude
        
        # 가속도계 바이어스 = 측정값 - 이론적 중력값
        self.accel_bias = accel_mean - gravity_vector
        
        # 통계 정보 계산
        accel_std = np.std(accel_array, axis=0)
        gyro_std = np.std(gyro_array, axis=0)
        
        # 결과 출력
        total_time = time.time() - self.start_time
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("IMU 정적 캘리브레이션 결과")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"총 수집 시간: {total_time:.1f}초")
        self.get_logger().info(f"수집된 샘플 수: {len(self.accel_data)}")
        self.get_logger().info(f"평균 샘플링 레이트: {len(self.accel_data)/total_time:.1f} Hz")
        self.get_logger().info(f"감지된 중력 방향: {'X' if gravity_axis==0 else 'Y' if gravity_axis==1 else 'Z'}")
        self.get_logger().info(f"측정된 중력 크기: {np.linalg.norm(accel_mean):.4f} m/s²")
        self.get_logger().info(f"이론적 중력 크기: {self.gravity_magnitude:.4f} m/s²")
        
        self.get_logger().info("가속도계 바이어스 (m/s²):")
        self.get_logger().info(f"   X: {self.accel_bias[0]:8.5f} ± {accel_std[0]:.5f}")
        self.get_logger().info(f"   Y: {self.accel_bias[1]:8.5f} ± {accel_std[1]:.5f}")
        self.get_logger().info(f"   Z: {self.accel_bias[2]:8.5f} ± {accel_std[2]:.5f}")
        
        self.get_logger().info("자이로스코프 바이어스 (rad/s):")
        self.get_logger().info(f"   X: {self.gyro_bias[0]:8.6f} ± {gyro_std[0]:.6f}")
        self.get_logger().info(f"   Y: {self.gyro_bias[1]:8.6f} ± {gyro_std[1]:.6f}")
        self.get_logger().info(f"   Z: {self.gyro_bias[2]:8.6f} ± {gyro_std[2]:.6f}")
        self.get_logger().info("=" * 70)
        
        return True
    
    def save_calibration(self, filename=None):
        """캘리브레이션 결과 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"imu_calibration_{timestamp}.json"
        
        # 패키지의 config 디렉토리에 저장
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
        os.makedirs(config_dir, exist_ok=True)
        filepath = os.path.join(config_dir, filename)
        
        total_time = time.time() - self.start_time
        
        calibration_data = {
            'timestamp': datetime.now().isoformat(),
            'collection_duration': total_time,
            'sample_count': len(self.accel_data),
            'sampling_rate': len(self.accel_data) / total_time,
            'accel_bias': {
                'x': float(self.accel_bias[0]),
                'y': float(self.accel_bias[1]),
                'z': float(self.accel_bias[2])
            },
            'gyro_bias': {
                'x': float(self.gyro_bias[0]),
                'y': float(self.gyro_bias[1]),
                'z': float(self.gyro_bias[2])
            },
            'gravity_magnitude': float(self.gravity_magnitude),
            'statistics': {
                'accel_std': np.std(np.array(self.accel_data), axis=0).tolist(),
                'gyro_std': np.std(np.array(self.gyro_data), axis=0).tolist(),
                'measured_gravity': float(np.linalg.norm(np.mean(np.array(self.accel_data), axis=0)))
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            self.get_logger().info(f"캘리브레이션 결과가 저장되었습니다: {filepath}")
            return True
        except Exception as e:
            self.get_logger().error(f"저장 실패: {e}")
            return False

def signal_handler(signum, frame, calibration_node):
    """Ctrl+C 신호 처리"""
    calibration_node.get_logger().info("\n사용자 중단 신호를 받았습니다. 캘리브레이션을 완료합니다...")
    calibration_node.is_collecting = False
    
    if calibration_node.calculate_bias():
        calibration_node.save_calibration()
        calibration_node.get_logger().info("캘리브레이션이 성공적으로 완료되었습니다!")
    else:
        calibration_node.get_logger().error("캘리브레이션 실패!")
    
    calibration_node.destroy_node()
    rclpy.shutdown()
    sys.exit(0)

def main():
    rclpy.init()
    
    try:
        calibration_node = IMUCalibration()
        
        # Ctrl+C 신호 처리기 등록
        signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, calibration_node))
        
        calibration_node.get_logger().info("IMU 토픽 연결을 기다리는 중...")
        
        # 노드 실행
        rclpy.spin(calibration_node)
        
    except KeyboardInterrupt:
        pass  # signal_handler에서 처리
    except Exception as e:
        if 'calibration_node' in locals():
            calibration_node.get_logger().error(f"예상치 못한 오류: {e}")
            calibration_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 