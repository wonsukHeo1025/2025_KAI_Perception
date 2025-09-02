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
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

class ImprovedIMUCalibration(Node):
    def __init__(self):
        super().__init__('improved_imu_calibration')
        
        # QoS 프로파일 설정
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
        
        # 데이터 저장
        self.accel_data = []
        self.gyro_data = []
        self.timestamps = []
        
        # 캘리브레이션 상태
        self.is_collecting = True
        self.start_timestamp = None  # 첫 번째 메시지 타임스탬프
        self.last_timestamp = None   # 마지막 메시지 타임스탬프
        
        # 진행률 표시 (메시지 타임스탬프 기준)
        self.last_progress_timestamp = None
        self.progress_interval = 5.0  # 5초마다 진행률 표시
        
        # 캘리브레이션 결과
        self.accel_bias = np.array([0.0, 0.0, 0.0])
        self.gyro_bias = np.array([0.0, 0.0, 0.0])
        self.accel_scale = np.array([1.0, 1.0, 1.0])
        self.accel_cross_coupling = np.zeros((3, 3))
        self.gravity_magnitude = 9.81
        
        # Allan variance 결과
        self.allan_accel = {}
        self.allan_gyro = {}
        
        self.get_logger().info('고급 IMU 캘리브레이션을 시작합니다...')
        self.get_logger().info('Allan variance 분석 및 로버스트 추정 포함')
        self.get_logger().info('데이터 수집 중... (Ctrl+C로 중단)')
        
    def imu_callback(self, msg):
        if self.is_collecting:
            # 타임스탬프 저장 (메시지 헤더 기준)
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.timestamps.append(timestamp)
            
            # 첫 번째 메시지 타임스탬프 기록
            if self.start_timestamp is None:
                self.start_timestamp = timestamp
                self.last_progress_timestamp = timestamp
                self.get_logger().info(f'첫 번째 메시지 타임스탬프: {timestamp:.3f}초')
            
            self.last_timestamp = timestamp
            
            # 가속도/각속도 데이터
            accel = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])
            
            gyro = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])
            
            self.accel_data.append(accel)
            self.gyro_data.append(gyro)
            
            # 진행률 표시 (메시지 타임스탬프 기준)
            if timestamp - self.last_progress_timestamp >= self.progress_interval:
                elapsed_time = timestamp - self.start_timestamp
                sample_count = len(self.accel_data)
                
                # 샘플링 레이트 계산 (메시지 타임스탬프 기준)
                if elapsed_time > 0:
                    sampling_rate = sample_count / elapsed_time
                else:
                    sampling_rate = 0
                
                # 실시간 통계
                if sample_count > 100:
                    recent_accel = np.array(self.accel_data[-100:])
                    current_gravity = np.linalg.norm(np.mean(recent_accel, axis=0))
                    self.get_logger().info(f"경과: {elapsed_time:.1f}초 | 샘플: {sample_count} | 샘플링레이트: {sampling_rate:.1f}Hz | 현재 중력: {current_gravity:.4f}m/s²")
                
                self.last_progress_timestamp = timestamp
    
    def robust_bias_estimation(self):
        """로버스트 바이어스 추정 (이상치 제거)"""
        if len(self.accel_data) < 100:
            self.get_logger().error("충분한 데이터가 없습니다.")
            return False
            
        accel_array = np.array(self.accel_data)
        gyro_array = np.array(self.gyro_data)
        
        self.get_logger().info("로버스트 바이어스 추정을 시작합니다...")
        
        # 1. 이상치 제거 (IQR 방법)
        accel_clean, gyro_clean = self.remove_outliers(accel_array, gyro_array)
        
        # 2. 자이로스코프 바이어스 (중앙값 기반)
        self.gyro_bias = np.median(gyro_clean, axis=0)
        
        # 3. 가속도계 바이어스 (중력 보상)
        accel_mean = np.median(accel_clean, axis=0)
        
        # 중력 방향 추정 개선 (최대 성분 + 부호)
        gravity_magnitude = np.linalg.norm(accel_mean)
        gravity_unit = accel_mean / gravity_magnitude
        
        # 이론적 중력 벡터 (가장 가까운 축 방향)
        axis_candidates = np.array([[1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1]])
        dots = np.abs(np.dot(axis_candidates, gravity_unit))
        best_axis_idx = np.argmax(dots)
        theoretical_gravity = axis_candidates[best_axis_idx] * self.gravity_magnitude
        
        self.accel_bias = accel_mean - theoretical_gravity
        
        self.get_logger().info(f"감지된 중력 방향: {axis_candidates[best_axis_idx]}")
        self.get_logger().info(f"중력 정렬 정확도: {dots[best_axis_idx]:.4f}")
        
        return True
    
    def remove_outliers(self, accel_data, gyro_data, factor=2.0):
        """IQR 방법으로 이상치 제거"""
        def remove_outliers_1d(data, factor):
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
            return (data >= lower_bound) & (data <= upper_bound)
        
        # 각 축별로 이상치 마스크 생성
        accel_mask = np.ones(len(accel_data), dtype=bool)
        gyro_mask = np.ones(len(gyro_data), dtype=bool)
        
        for i in range(3):
            accel_mask &= remove_outliers_1d(accel_data[:, i], factor)
            gyro_mask &= remove_outliers_1d(gyro_data[:, i], factor)
        
        # 합집합 마스크
        combined_mask = accel_mask & gyro_mask
        
        outliers_removed = len(accel_data) - np.sum(combined_mask)
        self.get_logger().info(f"이상치 제거: {outliers_removed}개 ({outliers_removed/len(accel_data)*100:.1f}%)")
        
        return accel_data[combined_mask], gyro_data[combined_mask]
    
    def allan_variance_analysis(self):
        """Allan Variance 분석"""
        if len(self.accel_data) < 1000:
            self.get_logger().warn("Allan variance 분석을 위한 데이터가 부족합니다.")
            return
        
        self.get_logger().info("Allan Variance 분석을 수행합니다...")
        
        # 샘플링 레이트 계산
        if len(self.timestamps) > 1:
            dt_array = np.diff(self.timestamps)
            dt_median = np.median(dt_array)
            fs = 1.0 / dt_median
        else:
            fs = 100.0  # 기본값
        
        self.get_logger().info(f"추정 샘플링 레이트: {fs:.1f} Hz")
        
        # Allan variance 계산
        self.allan_accel = self.compute_allan_variance(np.array(self.accel_data), fs, "가속도계")
        self.allan_gyro = self.compute_allan_variance(np.array(self.gyro_data), fs, "자이로스코프")
        
    def compute_allan_variance(self, data, fs, sensor_name):
        """Allan Variance 계산"""
        # 클러스터 시간 (tau) 범위 설정
        max_samples = len(data) // 10  # 최대 1/10까지
        taus = np.logspace(0, np.log10(max_samples), 20).astype(int)
        taus = np.unique(taus)
        
        allan_vars = np.zeros((len(taus), 3))
        
        for i, tau in enumerate(taus):
            if tau * 2 >= len(data):
                break
                
            # Allan variance 계산
            for axis in range(3):
                # 클러스터 평균 계산
                n_clusters = len(data) // tau
                clusters = data[:n_clusters*tau, axis].reshape(n_clusters, tau)
                cluster_means = np.mean(clusters, axis=1)
                
                # Allan variance
                if len(cluster_means) > 1:
                    diffs = np.diff(cluster_means)
                    allan_vars[i, axis] = np.var(diffs) / 2.0
        
        # 결과 저장
        valid_idx = np.any(allan_vars > 0, axis=1)
        result = {
            'tau': taus[valid_idx] / fs,  # 초 단위
            'sigma': np.sqrt(allan_vars[valid_idx]),
            'bias_stability': np.min(np.sqrt(allan_vars[valid_idx]), axis=0) if np.any(valid_idx) else np.zeros(3),
            'sensor_name': sensor_name
        }
        
        # 로그 출력
        self.get_logger().info(f"{sensor_name} Allan Variance 결과:")
        for axis, axis_name in enumerate(['X', 'Y', 'Z']):
            bs = result['bias_stability'][axis]
            unit = "m/s²" if "가속도" in sensor_name else "rad/s"
            self.get_logger().info(f"   {axis_name}축 Bias Stability: {bs:.2e} {unit}")
        
        return result
    
    def temperature_compensation_analysis(self):
        """온도 보상 분석 (기본 구현)"""
        # 실제로는 온도 센서 데이터가 필요하지만, 여기서는 시간 기반 드리프트 분석
        if len(self.accel_data) < 1000:
            return
        
        self.get_logger().info("시간 기반 드리프트 분석을 수행합니다...")
        
        accel_array = np.array(self.accel_data)
        gyro_array = np.array(self.gyro_data)
        
        # 시간 윈도우별 평균 계산
        window_size = len(accel_array) // 10
        n_windows = len(accel_array) // window_size
        
        accel_drift = np.zeros((n_windows, 3))
        gyro_drift = np.zeros((n_windows, 3))
        
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size
            
            accel_drift[i] = np.mean(accel_array[start_idx:end_idx], axis=0)
            gyro_drift[i] = np.mean(gyro_array[start_idx:end_idx], axis=0)
        
        # 드리프트 계수 계산
        time_points = np.arange(n_windows)
        accel_drift_rates = np.zeros(3)
        gyro_drift_rates = np.zeros(3)
        
        for axis in range(3):
            # 선형 회귀로 드리프트 레이트 계산
            slope_a, _, _, _, _ = stats.linregress(time_points, accel_drift[:, axis])
            slope_g, _, _, _, _ = stats.linregress(time_points, gyro_drift[:, axis])
            
            accel_drift_rates[axis] = slope_a
            gyro_drift_rates[axis] = slope_g
        
        self.get_logger().info("시간 드리프트 분석:")
        self.get_logger().info(f"가속도계 드리프트 레이트: {accel_drift_rates}")
        self.get_logger().info(f"자이로스코프 드리프트 레이트: {gyro_drift_rates}")
    
    def print_comprehensive_results(self):
        """종합 결과 출력"""
        if self.start_timestamp is not None and self.last_timestamp is not None:
            total_time = self.last_timestamp - self.start_timestamp
        else:
            total_time = 0
        
        self.get_logger().info("=" * 80)
        self.get_logger().info("고급 IMU 캘리브레이션 종합 결과")
        self.get_logger().info("=" * 80)
        
        # 기본 정보 (메시지 타임스탬프 기준)
        self.get_logger().info(f"총 데이터 기간: {total_time:.1f}초 (메시지 타임스탬프 기준)")
        self.get_logger().info(f"수집된 샘플: {len(self.accel_data)}개")
        if total_time > 0:
            avg_sampling_rate = len(self.accel_data) / total_time
            self.get_logger().info(f"평균 샘플링 레이트: {avg_sampling_rate:.1f} Hz")
        
        # 바이어스 결과
        self.get_logger().info("\n가속도계 바이어스 (m/s²):")
        self.get_logger().info(f"   X: {self.accel_bias[0]:8.5f}")
        self.get_logger().info(f"   Y: {self.accel_bias[1]:8.5f}")
        self.get_logger().info(f"   Z: {self.accel_bias[2]:8.5f}")
        
        self.get_logger().info("\n자이로스코프 바이어스 (rad/s):")
        self.get_logger().info(f"   X: {self.gyro_bias[0]:8.6f}")
        self.get_logger().info(f"   Y: {self.gyro_bias[1]:8.6f}")
        self.get_logger().info(f"   Z: {self.gyro_bias[2]:8.6f}")
        
        # Allan variance 결과
        if 'bias_stability' in self.allan_accel:
            self.get_logger().info("\nAllan Variance Bias Stability:")
            self.get_logger().info(f"가속도계: {self.allan_accel['bias_stability']}")
            self.get_logger().info(f"자이로스코프: {self.allan_gyro['bias_stability']}")
        
        self.get_logger().info("=" * 80)
    
    def save_comprehensive_results(self, filename=None):
        """종합 결과 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"improved_imu_calibration_{timestamp}.json"
        
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
        os.makedirs(config_dir, exist_ok=True)
        filepath = os.path.join(config_dir, filename)
        
        # 메시지 타임스탬프 기준 시간 계산
        if self.start_timestamp is not None and self.last_timestamp is not None:
            total_time = self.last_timestamp - self.start_timestamp
        else:
            total_time = 0
        
        # 통계 계산
        accel_array = np.array(self.accel_data)
        gyro_array = np.array(self.gyro_data)
        
        calibration_data = {
            'timestamp': datetime.now().isoformat(),
            'calibration_type': 'improved_robust',
            'collection_info': {
                'duration': total_time,  # 메시지 타임스탬프 기준
                'sample_count': len(self.accel_data),
                'sampling_rate': len(self.accel_data) / total_time if total_time > 0 else 0,
                'start_timestamp': self.start_timestamp,
                'end_timestamp': self.last_timestamp
            },
            'bias_estimation': {
                'accel_bias': {
                    'x': float(self.accel_bias[0]),
                    'y': float(self.accel_bias[1]),
                    'z': float(self.accel_bias[2])
                },
                'gyro_bias': {
                    'x': float(self.gyro_bias[0]),
                    'y': float(self.gyro_bias[1]),
                    'z': float(self.gyro_bias[2])
                }
            },
            'statistics': {
                'accel_std': np.std(accel_array, axis=0).tolist(),
                'gyro_std': np.std(gyro_array, axis=0).tolist(),
                'accel_median': np.median(accel_array, axis=0).tolist(),
                'gyro_median': np.median(gyro_array, axis=0).tolist(),
                'measured_gravity': float(np.linalg.norm(np.median(accel_array, axis=0)))
            },
            'allan_variance': {
                'accel': self.allan_accel,
                'gyro': self.allan_gyro
            },
            'gravity_magnitude': float(self.gravity_magnitude)
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(calibration_data, f, indent=2, default=str)
            self.get_logger().info(f"고급 캘리브레이션 결과 저장: {filepath}")
            return True
        except Exception as e:
            self.get_logger().error(f"저장 실패: {e}")
            return False

def signal_handler(signum, frame, calibration_node):
    """Ctrl+C 신호 처리"""
    calibration_node.get_logger().info("\n캘리브레이션을 완료합니다...")
    calibration_node.is_collecting = False
    
    # 고급 분석 수행
    if calibration_node.robust_bias_estimation():
        calibration_node.allan_variance_analysis()
        calibration_node.temperature_compensation_analysis()
        calibration_node.print_comprehensive_results()
        calibration_node.save_comprehensive_results()
        calibration_node.get_logger().info("고급 캘리브레이션이 성공적으로 완료되었습니다!")
    else:
        calibration_node.get_logger().error("캘리브레이션 실패!")
    
    calibration_node.destroy_node()
    rclpy.shutdown()
    sys.exit(0)

def main():
    rclpy.init()
    
    try:
        calibration_node = ImprovedIMUCalibration()
        
        # Ctrl+C 신호 처리기 등록
        signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, calibration_node))
        
        calibration_node.get_logger().info("IMU 토픽 연결 대기 중...")
        
        # 노드 실행
        rclpy.spin(calibration_node)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if 'calibration_node' in locals():
            calibration_node.get_logger().error(f"오류: {e}")
            calibration_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 