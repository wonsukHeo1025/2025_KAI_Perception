#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Imu
import numpy as np
import json
import os
import time
import threading
from datetime import datetime
import sys

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
            '/imu/data',
            self.imu_callback,
            qos_profile
        )
        
        # 데이터 저장용 리스트
        self.accel_data = []
        self.gyro_data = []
        
        # 캘리브레이션 상태
        self.is_collecting = False
        self.collection_start_time = None
        self.collection_duration = 0
        
        # 캘리브레이션 결과
        self.accel_bias = np.array([0.0, 0.0, 0.0])
        self.gyro_bias = np.array([0.0, 0.0, 0.0])
        self.gravity_magnitude = 9.81
        
        self.get_logger().info('IMU 캘리브레이션 노드가 시작되었습니다.')
        
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
            
            # 수집 시간 체크
            elapsed_time = time.time() - self.collection_start_time
            if elapsed_time >= self.collection_duration:
                self.stop_collection()
    
    def start_collection(self, duration):
        """데이터 수집 시작"""
        self.accel_data = []
        self.gyro_data = []
        self.is_collecting = True
        self.collection_start_time = time.time()
        self.collection_duration = duration
        
        self.get_logger().info(f"{duration}초 동안 IMU 데이터를 수집합니다...")
        self.get_logger().warn("IMU를 수평하게 놓고 움직이지 마세요!")
        
    def stop_collection(self):
        """데이터 수집 중지"""
        self.is_collecting = False
        
        if len(self.accel_data) > 0:
            self.get_logger().info(f"데이터 수집 완료! ({len(self.accel_data)}개 샘플)")
            self.calculate_bias()
        else:
            self.get_logger().error("수집된 데이터가 없습니다.")
    
    def calculate_bias(self):
        """바이어스 계산"""
        if len(self.accel_data) == 0:
            return
            
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
        self.get_logger().info("=" * 60)
        self.get_logger().info("캘리브레이션 결과")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"수집된 샘플 수: {len(self.accel_data)}")
        self.get_logger().info(f"수집 시간: {self.collection_duration}초")
        self.get_logger().info(f"감지된 중력 방향: {'X' if gravity_axis==0 else 'Y' if gravity_axis==1 else 'Z'}")
        self.get_logger().info(f"중력 크기: {np.linalg.norm(accel_mean):.3f} m/s²")
        
        self.get_logger().info("가속도계 바이어스 (m/s²):")
        self.get_logger().info(f"   X: {self.accel_bias[0]:8.5f} ± {accel_std[0]:.5f}")
        self.get_logger().info(f"   Y: {self.accel_bias[1]:8.5f} ± {accel_std[1]:.5f}")
        self.get_logger().info(f"   Z: {self.accel_bias[2]:8.5f} ± {accel_std[2]:.5f}")
        
        self.get_logger().info("자이로스코프 바이어스 (rad/s):")
        self.get_logger().info(f"   X: {self.gyro_bias[0]:8.5f} ± {gyro_std[0]:.5f}")
        self.get_logger().info(f"   Y: {self.gyro_bias[1]:8.5f} ± {gyro_std[1]:.5f}")
        self.get_logger().info(f"   Z: {self.gyro_bias[2]:8.5f} ± {gyro_std[2]:.5f}")
        self.get_logger().info("=" * 60)
    
    def save_calibration(self, filename=None):
        """캘리브레이션 결과 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"imu_calibration_{timestamp}.json"
        
        # 패키지의 config 디렉토리에 저장
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
        os.makedirs(config_dir, exist_ok=True)
        filepath = os.path.join(config_dir, filename)
        
        calibration_data = {
            'timestamp': datetime.now().isoformat(),
            'collection_duration': self.collection_duration,
            'sample_count': len(self.accel_data),
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
            'gravity_magnitude': float(self.gravity_magnitude)
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            self.get_logger().info(f"캘리브레이션 결과가 저장되었습니다: {filepath}")
            return True
        except Exception as e:
            self.get_logger().error(f"저장 실패: {e}")
            return False

    def print_menu(self):
        """메뉴 출력"""
        self.get_logger().info("=" * 60)
        self.get_logger().info("IMU 캘리브레이션 도구")
        self.get_logger().info("=" * 60)
        self.get_logger().info("사용 방법:")
        self.get_logger().info("   1. IMU 센서를 수평한 표면에 놓으세요")
        self.get_logger().info("   2. 센서가 완전히 정지된 상태에서 캘리브레이션을 시작하세요")
        self.get_logger().info("   3. 캘리브레이션 중에는 센서를 움직이지 마세요")
        self.get_logger().info("")
        self.get_logger().info("키 명령:")
        self.get_logger().info("   1: 10초 캘리브레이션")
        self.get_logger().info("   2: 30초 캘리브레이션")
        self.get_logger().info("   3: 1분 캘리브레이션")
        self.get_logger().info("   4: 5분 캘리브레이션")
        self.get_logger().info("   5: 30분 캘리브레이션")
        self.get_logger().info("   6: 1시간 캘리브레이션")
        self.get_logger().info("   s: 캘리브레이션 결과 저장")
        self.get_logger().info("   q: 종료")
        self.get_logger().info("=" * 60)

def main():
    rclpy.init()
    
    try:
        calibration_node = IMUCalibration()
        
        calibration_node.get_logger().info("IMU 캘리브레이션 도구를 시작합니다...")
        calibration_node.get_logger().info("IMU 토픽 연결을 기다리는 중...")
        
        # 스핀 스레드 시작
        spin_thread = threading.Thread(target=rclpy.spin, args=(calibration_node,))
        spin_thread.daemon = True
        spin_thread.start()
        
        # 잠시 대기 후 메뉴 출력
        time.sleep(1)
        calibration_node.print_menu()
        
        # 시간 매핑
        time_mapping = {
            '1': 10,      # 10초
            '2': 30,      # 30초
            '3': 60,      # 1분
            '4': 300,     # 5분
            '5': 1800,    # 30분
            '6': 3600     # 1시간
        }
        
        while rclpy.ok():
            try:
                # 간단한 입력 방식 사용
                user_input = input("\n명령을 입력하세요 (1-6, s, q): ").strip()
                
                if not user_input:
                    continue
                    
                key = user_input[0].lower()
                calibration_node.get_logger().info(f"입력된 명령: {key}")
                
                if key == 'q':
                    calibration_node.get_logger().info("캘리브레이션 도구를 종료합니다.")
                    break
                elif key == 's':
                    if len(calibration_node.accel_data) > 0:
                        calibration_node.save_calibration()
                    else:
                        calibration_node.get_logger().warn("저장할 캘리브레이션 데이터가 없습니다.")
                elif key in time_mapping:
                    if not calibration_node.is_collecting:
                        duration = time_mapping[key]
                        duration_str = {
                            10: "10초", 30: "30초", 60: "1분", 
                            300: "5분", 1800: "30분", 3600: "1시간"
                        }[duration]
                        
                        calibration_node.get_logger().info(f"{duration_str} 캘리브레이션을 시작합니다.")
                        calibration_node.get_logger().warn("센서를 수평하게 놓고 움직이지 마세요!")
                        calibration_node.get_logger().info("3초 후 시작됩니다...")
                        
                        for i in range(3, 0, -1):
                            calibration_node.get_logger().info(f"   {i}...")
                            time.sleep(1)
                        
                        calibration_node.start_collection(duration)
                        
                        # 진행률 표시
                        calibration_node.get_logger().info("데이터 수집 중...")
                        last_update = time.time()
                        
                        while calibration_node.is_collecting:
                            current_time = time.time()
                            if current_time - last_update >= 2.0:  # 2초마다 업데이트
                                elapsed = current_time - calibration_node.collection_start_time
                                progress = (elapsed / duration) * 100
                                remaining = duration - elapsed
                                
                                calibration_node.get_logger().info(f"   진행률: {progress:.1f}% | 남은 시간: {remaining:.1f}초 | 샘플: {len(calibration_node.accel_data)}")
                                last_update = current_time
                            
                            time.sleep(0.5)
                        
                    else:
                        calibration_node.get_logger().warn("이미 캘리브레이션이 진행 중입니다.")
                else:
                    calibration_node.get_logger().warn(f"알 수 없는 명령: {key}")
                    calibration_node.get_logger().info("도움말을 보려면 메뉴를 확인하세요.")
                    
            except EOFError:
                break
            except KeyboardInterrupt:
                break
    
    except KeyboardInterrupt:
        if 'calibration_node' in locals():
            calibration_node.get_logger().warn("사용자에 의해 중단되었습니다.")
    
    finally:
        if 'calibration_node' in locals():
            calibration_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 