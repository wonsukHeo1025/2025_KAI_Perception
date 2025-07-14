#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion, Vector3
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster
import tf2_geometry_msgs
import numpy as np
import math
import json
import os
import glob
from scipy.spatial.transform import Rotation


class DeadReckoningNode(Node):
    def __init__(self):
        super().__init__('dead_reckoning_node')
        
        # 캘리브레이션 데이터 로드
        self.accel_bias = np.array([0.0, 0.0, 0.0])
        self.gyro_bias = np.array([0.0, 0.0, 0.0])
        self.gravity_magnitude = 9.81
        self.load_calibration()
        
        # 중력 초기화 관련 변수들 (캘리브레이션된 데이터용)
        self.initial_gravity_vector_calibrated = None
        self.initial_gravity_vector_raw = None
        self.is_initialized = False
        self.gravity_samples_calibrated = []
        self.gravity_samples_raw = []
        self.required_samples = 200  # 초기화에 필요한 샘플 수 (약 2초)
        
        # QoS 프로파일 설정 (Ouster IMU와 호환되도록)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 구독자
        self.imu_subscription = self.create_subscription(
            Imu,
            '/ouster/imu',
            self.imu_callback,
            qos_profile
        )
        
        # 발행자
        self.path_publisher_calibrated = self.create_publisher(Path, '/dead_reckoning/path_calibrated', 10)
        self.path_publisher_raw = self.create_publisher(Path, '/dead_reckoning/path_raw', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, '/dead_reckoning/markers', 10)
        
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # 캘리브레이션된 상태 변수들
        self.position_calibrated = np.array([0.0, 0.0, 0.0])
        self.velocity_calibrated = np.array([0.0, 0.0, 0.0])
        self.orientation_calibrated = np.array([1.0, 0.0, 0.0, 0.0])
        
        # 원본 상태 변수들
        self.position_raw = np.array([0.0, 0.0, 0.0])
        self.velocity_raw = np.array([0.0, 0.0, 0.0])
        self.orientation_raw = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Path 메시지들
        self.path_msg_calibrated = Path()
        self.path_msg_calibrated.header.frame_id = "map"
        self.path_msg_raw = Path()
        self.path_msg_raw.header.frame_id = "map"
        
        # 이전 시간
        self.last_time = None
        
        # 타이머 (마커 발행용)
        self.marker_timer = self.create_timer(0.1, self.publish_markers)
        
        # 통계 정보
        self.sample_count = 0
        self.max_drift_distance_calibrated = 0.0
        self.max_drift_distance_raw = 0.0
        
        self.get_logger().info('Dead Reckoning Node가 시작되었습니다.')
        self.get_logger().info('/ouster/imu 토픽을 구독하고 있습니다.')
        self.get_logger().info(f'초기화를 위해 {self.required_samples}개 샘플을 수집합니다...')
        self.get_logger().info('캘리브레이션 적용 vs 원본 데이터 비교 모드입니다.')

    def load_calibration(self):
        """최신 캘리브레이션 파일을 로드합니다."""
        try:
            # config 디렉토리에서 캘리브레이션 파일 찾기
            config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
            calibration_files = glob.glob(os.path.join(config_dir, 'improved_imu_calibration_*.json'))
            
            if not calibration_files:
                self.get_logger().warn('캘리브레이션 파일을 찾을 수 없습니다. 기본값을 사용합니다.')
                return
            
            # 가장 최신 파일 선택
            latest_file = max(calibration_files, key=os.path.getctime)
            
            with open(latest_file, 'r') as f:
                calibration_data = json.load(f)
            
            # 새로운 improved calibration 형식에 맞춰 바이어스 데이터 로드
            if 'bias_estimation' in calibration_data:
                # 새로운 형식 (improved_calibration.py)
                self.accel_bias = np.array([
                    calibration_data['bias_estimation']['accel_bias']['x'],
                    calibration_data['bias_estimation']['accel_bias']['y'],
                    calibration_data['bias_estimation']['accel_bias']['z']
                ])
                
                self.gyro_bias = np.array([
                    calibration_data['bias_estimation']['gyro_bias']['x'],
                    calibration_data['bias_estimation']['gyro_bias']['y'],
                    calibration_data['bias_estimation']['gyro_bias']['z']
                ])
                
                self.get_logger().info('새로운 improved calibration 형식을 감지했습니다.')
            else:
                # 기존 형식 (하위 호환성)
                self.accel_bias = np.array([
                    calibration_data['accel_bias']['x'],
                    calibration_data['accel_bias']['y'],
                    calibration_data['accel_bias']['z']
                ])
                
                self.gyro_bias = np.array([
                    calibration_data['gyro_bias']['x'],
                    calibration_data['gyro_bias']['y'],
                    calibration_data['gyro_bias']['z']
                ])
                
                self.get_logger().info('기존 캘리브레이션 형식을 감지했습니다.')
            
            self.gravity_magnitude = calibration_data.get('gravity_magnitude', 9.81)
            
            # 추가 정보 출력 (improved calibration에서 제공하는 경우)
            if 'collection_info' in calibration_data:
                duration = calibration_data['collection_info'].get('duration', 0)
                sample_count = calibration_data['collection_info'].get('sample_count', 0)
                sampling_rate = calibration_data['collection_info'].get('sampling_rate', 0)
                self.get_logger().info(f'캘리브레이션 정보: 지속시간 {duration:.1f}초, 샘플 {sample_count}개, 샘플링레이트 {sampling_rate:.1f}Hz')
            
            # Allan variance 정보 출력 (있는 경우)
            if 'allan_variance' in calibration_data:
                allan_accel = calibration_data['allan_variance'].get('accel', {})
                allan_gyro = calibration_data['allan_variance'].get('gyro', {})
                
                if 'bias_stability' in allan_accel:
                    self.get_logger().info(f'가속도계 Bias Stability: {allan_accel["bias_stability"]}')
                if 'bias_stability' in allan_gyro:
                    self.get_logger().info(f'자이로스코프 Bias Stability: {allan_gyro["bias_stability"]}')
            
            self.get_logger().info(f'캘리브레이션 데이터를 로드했습니다: {os.path.basename(latest_file)}')
            self.get_logger().info(f'가속도계 바이어스: [{self.accel_bias[0]:.5f}, {self.accel_bias[1]:.5f}, {self.accel_bias[2]:.5f}]')
            self.get_logger().info(f'자이로스코프 바이어스: [{self.gyro_bias[0]:.5f}, {self.gyro_bias[1]:.5f}, {self.gyro_bias[2]:.5f}]')
            
        except KeyError as e:
            self.get_logger().error(f'캘리브레이션 파일 구조 오류: {e}')
            self.get_logger().warn('기본 바이어스 값을 사용합니다.')
        except Exception as e:
            self.get_logger().error(f'캘리브레이션 로드 실패: {e}')
            self.get_logger().warn('기본 바이어스 값을 사용합니다.')

    def initialize_gravity_direction(self, accel_data_calibrated, accel_data_raw):
        """초기 몇 초간의 데이터로 센서 프레임에서의 중력 방향을 결정합니다."""
        self.gravity_samples_calibrated.append(accel_data_calibrated.copy())
        self.gravity_samples_raw.append(accel_data_raw.copy())
        
        # 진행률 표시
        if len(self.gravity_samples_calibrated) % 50 == 0:  # 50샘플마다 진행률 표시
            progress = len(self.gravity_samples_calibrated) / self.required_samples * 100
            self.get_logger().info(f'초기화 진행률: {progress:.1f}% ({len(self.gravity_samples_calibrated)}/{self.required_samples})')
        
        if len(self.gravity_samples_calibrated) >= self.required_samples:
            # 캘리브레이션된 데이터의 중력 벡터
            gravity_mean_calibrated = np.mean(self.gravity_samples_calibrated, axis=0)
            self.initial_gravity_vector_calibrated = gravity_mean_calibrated
            
            # 원본 데이터의 중력 벡터
            gravity_mean_raw = np.mean(self.gravity_samples_raw, axis=0)
            self.initial_gravity_vector_raw = gravity_mean_raw
            
            # 중력 벡터의 크기 확인
            gravity_magnitude_calibrated = np.linalg.norm(self.initial_gravity_vector_calibrated)
            gravity_magnitude_raw = np.linalg.norm(self.initial_gravity_vector_raw)
            
            self.is_initialized = True
            
            self.get_logger().info('=== 중력 초기화 완료 ===')
            self.get_logger().info(f'캘리브레이션된 중력 벡터: [{self.initial_gravity_vector_calibrated[0]:.4f}, {self.initial_gravity_vector_calibrated[1]:.4f}, {self.initial_gravity_vector_calibrated[2]:.4f}]')
            self.get_logger().info(f'캘리브레이션된 중력 크기: {gravity_magnitude_calibrated:.4f} m/s²')
            self.get_logger().info(f'원본 중력 벡터: [{self.initial_gravity_vector_raw[0]:.4f}, {self.initial_gravity_vector_raw[1]:.4f}, {self.initial_gravity_vector_raw[2]:.4f}]')
            self.get_logger().info(f'원본 중력 크기: {gravity_magnitude_raw:.4f} m/s²')
            self.get_logger().info(f'예상 중력 크기: {self.gravity_magnitude:.4f} m/s²')
            self.get_logger().info('데드 레코닝을 시작합니다...')
            
            # 메모리 절약을 위해 샘플 데이터 삭제
            del self.gravity_samples_calibrated
            del self.gravity_samples_raw

    def imu_callback(self, msg):
        # 메시지 헤더의 타임스탬프를 rclpy.time.Time 객체로 변환
        current_event_time = rclpy.time.Time.from_msg(msg.header.stamp)
        
        # 원본 IMU 데이터 추출
        angular_velocity_raw = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        
        linear_acceleration_raw = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        
        # 캘리브레이션 적용된 데이터
        angular_velocity_calibrated = angular_velocity_raw - self.gyro_bias
        linear_acceleration_calibrated = linear_acceleration_raw - self.accel_bias
        
        # 초기화가 안 되었으면 중력 방향 초기화
        if not self.is_initialized:
            self.initialize_gravity_direction(linear_acceleration_calibrated, linear_acceleration_raw)
            return
        
        # 시간 차이 계산 및 유효성 검사
        if self.last_time is None:
            self.last_time = current_event_time
            return
        
        dt_duration = current_event_time - self.last_time
        dt = dt_duration.nanoseconds / 1e9
        
        # dt 유효성 검사
        if dt <= 0:
            self.get_logger().warn(
                f'dt ({dt:.4f}s)가 0 또는 음수입니다. 현재 메시지 시간: {current_event_time.nanoseconds/1e9:.4f}s, '
                f'이전 메시지 시간: {self.last_time.nanoseconds/1e9:.4f}s. 이 IMU 데이터를 건너뜁니다.'
            )
            self.last_time = current_event_time
            return
        
        # dt가 너무 큰 경우
        max_reasonable_dt = 1.0 
        if dt > max_reasonable_dt:
            self.get_logger().warn(
                f'dt ({dt:.4f}s)가 설정된 최대값 ({max_reasonable_dt:.1f}s)을 초과했습니다. 이 IMU 데이터를 건너뜁니다.'
            )
            self.last_time = current_event_time
            return

        # 캘리브레이션된 데이터 처리
        self.update_orientation(angular_velocity_calibrated, dt, is_calibrated=True)
        accel_without_gravity_calibrated = linear_acceleration_calibrated - self.initial_gravity_vector_calibrated
        world_acceleration_calibrated = self.rotate_vector_by_quaternion(accel_without_gravity_calibrated, self.orientation_calibrated)
        self.velocity_calibrated += world_acceleration_calibrated * dt
        self.position_calibrated += self.velocity_calibrated * dt
        
        # 원본 데이터 처리
        self.update_orientation(angular_velocity_raw, dt, is_calibrated=False)
        accel_without_gravity_raw = linear_acceleration_raw - self.initial_gravity_vector_raw
        world_acceleration_raw = self.rotate_vector_by_quaternion(accel_without_gravity_raw, self.orientation_raw)
        self.velocity_raw += world_acceleration_raw * dt
        self.position_raw += self.velocity_raw * dt
        
        # 드리프트 통계 업데이트
        current_drift_calibrated = np.linalg.norm(self.position_calibrated)
        if current_drift_calibrated > self.max_drift_distance_calibrated:
            self.max_drift_distance_calibrated = current_drift_calibrated
        
        current_drift_raw = np.linalg.norm(self.position_raw)
        if current_drift_raw > self.max_drift_distance_raw:
            self.max_drift_distance_raw = current_drift_raw
        
        # TF 발행 (2개의 서로 다른 프레임)
        self.publish_tf_calibrated(current_event_time)
        self.publish_tf_raw(current_event_time)
        
        # Path 업데이트 및 발행
        self.update_path_calibrated(current_event_time)
        self.update_path_raw(current_event_time)
        
        # 로그 출력 (개선된 정보 포함)
        self.sample_count += 1
        log_time_seconds = current_event_time.nanoseconds / 1e9
        current_int_seconds = int(log_time_seconds)
        if current_int_seconds % 10 == 0:  # 10초마다 로깅
            if not hasattr(self, '_imu_cb_last_log_sec') or self._imu_cb_last_log_sec != current_int_seconds:
                velocity_magnitude_calibrated = np.linalg.norm(self.velocity_calibrated)
                velocity_magnitude_raw = np.linalg.norm(self.velocity_raw)
                self.get_logger().info(
                    f'캘리브레이션: 위치[{self.position_calibrated[0]:.2f}, {self.position_calibrated[1]:.2f}, {self.position_calibrated[2]:.2f}] '
                    f'속도{velocity_magnitude_calibrated:.3f}m/s 드리프트{self.max_drift_distance_calibrated:.2f}m'
                )
                self.get_logger().info(
                    f'원본 데이터: 위치[{self.position_raw[0]:.2f}, {self.position_raw[1]:.2f}, {self.position_raw[2]:.2f}] '
                    f'속도{velocity_magnitude_raw:.3f}m/s 드리프트{self.max_drift_distance_raw:.2f}m | 샘플: {self.sample_count}'
                )
                self._imu_cb_last_log_sec = current_int_seconds
        
        # 다음 콜백을 위해 last_time 업데이트
        self.last_time = current_event_time

    def update_orientation(self, angular_velocity, dt, is_calibrated=True):
        """각속도를 이용해 자세를 업데이트합니다."""
        # 각속도 크기
        omega_magnitude = np.linalg.norm(angular_velocity)
        
        if omega_magnitude > 1e-6:  # 작은 값 체크
            # 회전축
            axis = angular_velocity / omega_magnitude
            
            # 회전각
            angle = omega_magnitude * dt
            
            # 회전 쿼터니언 생성
            rotation_quat = np.array([
                math.cos(angle / 2),
                axis[0] * math.sin(angle / 2),
                axis[1] * math.sin(angle / 2),
                axis[2] * math.sin(angle / 2)
            ])
            
            # 현재 자세에 회전 적용
            if is_calibrated:
                self.orientation_calibrated = self.multiply_quaternions(self.orientation_calibrated, rotation_quat)
                self.orientation_calibrated = self.orientation_calibrated / np.linalg.norm(self.orientation_calibrated)
            else:
                self.orientation_raw = self.multiply_quaternions(self.orientation_raw, rotation_quat)
                self.orientation_raw = self.orientation_raw / np.linalg.norm(self.orientation_raw)

    def rotate_vector_by_quaternion(self, vector, quaternion):
        """쿼터니언을 이용해 벡터를 회전시킵니다."""
        # 벡터를 쿼터니언으로 변환 [0, x, y, z]
        vector_quat = np.array([0.0, vector[0], vector[1], vector[2]])
        
        # q * v * q_conjugate
        q_conjugate = np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])
        
        temp = self.multiply_quaternions(quaternion, vector_quat)
        result_quat = self.multiply_quaternions(temp, q_conjugate)
        
        return result_quat[1:4]  # x, y, z 성분만 반환

    def multiply_quaternions(self, q1, q2):
        """두 쿼터니언을 곱합니다."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def publish_tf_calibrated(self, timestamp):
        """캘리브레이션된 TF를 발행합니다."""
        t = TransformStamped()
        t.header.stamp = timestamp.to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link_calibrated"
        
        # 위치
        t.transform.translation.x = self.position_calibrated[0]
        t.transform.translation.y = self.position_calibrated[1]
        t.transform.translation.z = self.position_calibrated[2]
        
        # 자세
        t.transform.rotation.w = self.orientation_calibrated[0]
        t.transform.rotation.x = self.orientation_calibrated[1]
        t.transform.rotation.y = self.orientation_calibrated[2]
        t.transform.rotation.z = self.orientation_calibrated[3]
        
        self.tf_broadcaster.sendTransform(t)

    def publish_tf_raw(self, timestamp):
        """원본 데이터 TF를 발행합니다."""
        t = TransformStamped()
        t.header.stamp = timestamp.to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link_raw"
        
        # 위치
        t.transform.translation.x = self.position_raw[0]
        t.transform.translation.y = self.position_raw[1]
        t.transform.translation.z = self.position_raw[2]
        
        # 자세
        t.transform.rotation.w = self.orientation_raw[0]
        t.transform.rotation.x = self.orientation_raw[1]
        t.transform.rotation.y = self.orientation_raw[2]
        t.transform.rotation.z = self.orientation_raw[3]
        
        self.tf_broadcaster.sendTransform(t)

    def update_path_calibrated(self, timestamp):
        """캘리브레이션된 Path를 업데이트하고 발행합니다."""
        pose = PoseStamped()
        pose.header.stamp = timestamp.to_msg()
        pose.header.frame_id = "map"
        
        pose.pose.position.x = self.position_calibrated[0]
        pose.pose.position.y = self.position_calibrated[1]
        pose.pose.position.z = self.position_calibrated[2]
        
        pose.pose.orientation.w = self.orientation_calibrated[0]
        pose.pose.orientation.x = self.orientation_calibrated[1]
        pose.pose.orientation.y = self.orientation_calibrated[2]
        pose.pose.orientation.z = self.orientation_calibrated[3]
        
        self.path_msg_calibrated.poses.append(pose)
        
        # Path가 너무 길어지지 않도록 제한
        if len(self.path_msg_calibrated.poses) > 2000:
            self.path_msg_calibrated.poses = self.path_msg_calibrated.poses[-2000:]
        
        self.path_msg_calibrated.header.stamp = timestamp.to_msg()
        self.path_publisher_calibrated.publish(self.path_msg_calibrated)

    def update_path_raw(self, timestamp):
        """원본 데이터 Path를 업데이트하고 발행합니다."""
        pose = PoseStamped()
        pose.header.stamp = timestamp.to_msg()
        pose.header.frame_id = "map"
        
        pose.pose.position.x = self.position_raw[0]
        pose.pose.position.y = self.position_raw[1]
        pose.pose.position.z = self.position_raw[2]
        
        pose.pose.orientation.w = self.orientation_raw[0]
        pose.pose.orientation.x = self.orientation_raw[1]
        pose.pose.orientation.y = self.orientation_raw[2]
        pose.pose.orientation.z = self.orientation_raw[3]
        
        self.path_msg_raw.poses.append(pose)
        
        # Path가 너무 길어지지 않도록 제한
        if len(self.path_msg_raw.poses) > 2000:
            self.path_msg_raw.poses = self.path_msg_raw.poses[-2000:]
        
        self.path_msg_raw.header.stamp = timestamp.to_msg()
        self.path_publisher_raw.publish(self.path_msg_raw)

    def publish_markers(self):
        """원점, 좌표축, 현재 위치들을 표시하는 마커를 발행합니다."""
        marker_array = MarkerArray()
        
        # 원점 마커 (흰색 구)
        origin_marker = Marker()
        origin_marker.header.frame_id = "map"
        origin_marker.header.stamp = self.get_clock().now().to_msg()
        origin_marker.ns = "origin"
        origin_marker.id = 0
        origin_marker.type = Marker.SPHERE
        origin_marker.action = Marker.ADD
        origin_marker.pose.position.x = 0.0
        origin_marker.pose.position.y = 0.0
        origin_marker.pose.position.z = 0.0
        origin_marker.pose.orientation.w = 1.0
        origin_marker.scale.x = 0.2
        origin_marker.scale.y = 0.2
        origin_marker.scale.z = 0.2
        origin_marker.color.r = 1.0
        origin_marker.color.g = 1.0
        origin_marker.color.b = 1.0
        origin_marker.color.a = 1.0
        marker_array.markers.append(origin_marker)
        
        # 캘리브레이션된 현재 위치 마커 (초록색 구)
        if self.is_initialized:
            current_pos_marker_calibrated = Marker()
            current_pos_marker_calibrated.header.frame_id = "map"
            current_pos_marker_calibrated.header.stamp = self.get_clock().now().to_msg()
            current_pos_marker_calibrated.ns = "current_position_calibrated"
            current_pos_marker_calibrated.id = 0
            current_pos_marker_calibrated.type = Marker.SPHERE
            current_pos_marker_calibrated.action = Marker.ADD
            current_pos_marker_calibrated.pose.position.x = self.position_calibrated[0]
            current_pos_marker_calibrated.pose.position.y = self.position_calibrated[1]
            current_pos_marker_calibrated.pose.position.z = self.position_calibrated[2]
            current_pos_marker_calibrated.pose.orientation.w = self.orientation_calibrated[0]
            current_pos_marker_calibrated.pose.orientation.x = self.orientation_calibrated[1]
            current_pos_marker_calibrated.pose.orientation.y = self.orientation_calibrated[2]
            current_pos_marker_calibrated.pose.orientation.z = self.orientation_calibrated[3]
            current_pos_marker_calibrated.scale.x = 0.15
            current_pos_marker_calibrated.scale.y = 0.15
            current_pos_marker_calibrated.scale.z = 0.15
            current_pos_marker_calibrated.color.r = 0.0
            current_pos_marker_calibrated.color.g = 1.0  # 초록색 (캘리브레이션됨)
            current_pos_marker_calibrated.color.b = 0.0
            current_pos_marker_calibrated.color.a = 1.0
            marker_array.markers.append(current_pos_marker_calibrated)
            
            # 원본 데이터 현재 위치 마커 (빨간색 구)
            current_pos_marker_raw = Marker()
            current_pos_marker_raw.header.frame_id = "map"
            current_pos_marker_raw.header.stamp = self.get_clock().now().to_msg()
            current_pos_marker_raw.ns = "current_position_raw"
            current_pos_marker_raw.id = 0
            current_pos_marker_raw.type = Marker.SPHERE
            current_pos_marker_raw.action = Marker.ADD
            current_pos_marker_raw.pose.position.x = self.position_raw[0]
            current_pos_marker_raw.pose.position.y = self.position_raw[1]
            current_pos_marker_raw.pose.position.z = self.position_raw[2]
            current_pos_marker_raw.pose.orientation.w = self.orientation_raw[0]
            current_pos_marker_raw.pose.orientation.x = self.orientation_raw[1]
            current_pos_marker_raw.pose.orientation.y = self.orientation_raw[2]
            current_pos_marker_raw.pose.orientation.z = self.orientation_raw[3]
            current_pos_marker_raw.scale.x = 0.15
            current_pos_marker_raw.scale.y = 0.15
            current_pos_marker_raw.scale.z = 0.15
            current_pos_marker_raw.color.r = 1.0  # 빨간색 (원본)
            current_pos_marker_raw.color.g = 0.0
            current_pos_marker_raw.color.b = 0.0
            current_pos_marker_raw.color.a = 1.0
            marker_array.markers.append(current_pos_marker_raw)
        
        # X축 (빨간색)
        x_axis = Marker()
        x_axis.header.frame_id = "map"
        x_axis.header.stamp = self.get_clock().now().to_msg()
        x_axis.ns = "axes"
        x_axis.id = 1
        x_axis.type = Marker.ARROW
        x_axis.action = Marker.ADD
        x_axis.pose.position.x = 0.0
        x_axis.pose.position.y = 0.0
        x_axis.pose.position.z = 0.0
        x_axis.pose.orientation.w = 1.0
        x_axis.scale.x = 1.0  # 길이
        x_axis.scale.y = 0.05  # 두께
        x_axis.scale.z = 0.05
        x_axis.color.r = 1.0
        x_axis.color.g = 0.0
        x_axis.color.b = 0.0
        x_axis.color.a = 1.0
        marker_array.markers.append(x_axis)
        
        # Y축 (초록색)
        y_axis = Marker()
        y_axis.header.frame_id = "map"
        y_axis.header.stamp = self.get_clock().now().to_msg()
        y_axis.ns = "axes"
        y_axis.id = 2
        y_axis.type = Marker.ARROW
        y_axis.action = Marker.ADD
        y_axis.pose.position.x = 0.0
        y_axis.pose.position.y = 0.0
        y_axis.pose.position.z = 0.0
        # Y축을 위한 90도 회전 (Z축 기준)
        y_axis.pose.orientation.w = 0.707
        y_axis.pose.orientation.x = 0.0
        y_axis.pose.orientation.y = 0.0
        y_axis.pose.orientation.z = 0.707
        y_axis.scale.x = 1.0
        y_axis.scale.y = 0.05
        y_axis.scale.z = 0.05
        y_axis.color.r = 0.0
        y_axis.color.g = 1.0
        y_axis.color.b = 0.0
        y_axis.color.a = 1.0
        marker_array.markers.append(y_axis)
        
        # Z축 (파란색)
        z_axis = Marker()
        z_axis.header.frame_id = "map"
        z_axis.header.stamp = self.get_clock().now().to_msg()
        z_axis.ns = "axes"
        z_axis.id = 3
        z_axis.type = Marker.ARROW
        z_axis.action = Marker.ADD
        z_axis.pose.position.x = 0.0
        z_axis.pose.position.y = 0.0
        z_axis.pose.position.z = 0.0
        # Z축을 위한 -90도 회전 (Y축 기준)
        z_axis.pose.orientation.w = 0.707
        z_axis.pose.orientation.x = 0.0
        z_axis.pose.orientation.y = -0.707
        z_axis.pose.orientation.z = 0.0
        z_axis.scale.x = 1.0
        z_axis.scale.y = 0.05
        z_axis.scale.z = 0.05
        z_axis.color.r = 0.0
        z_axis.color.g = 0.0
        z_axis.color.b = 1.0
        z_axis.color.a = 1.0
        marker_array.markers.append(z_axis)
        
        # 캘리브레이션된 드리프트 원 표시 (초록색, 반투명)
        if self.is_initialized and self.max_drift_distance_calibrated > 0.01:
            drift_circle_calibrated = Marker()
            drift_circle_calibrated.header.frame_id = "map"
            drift_circle_calibrated.header.stamp = self.get_clock().now().to_msg()
            drift_circle_calibrated.ns = "drift_visualization_calibrated"
            drift_circle_calibrated.id = 0
            drift_circle_calibrated.type = Marker.CYLINDER
            drift_circle_calibrated.action = Marker.ADD
            drift_circle_calibrated.pose.position.x = 0.0
            drift_circle_calibrated.pose.position.y = 0.0
            drift_circle_calibrated.pose.position.z = 0.01
            drift_circle_calibrated.pose.orientation.w = 1.0
            drift_circle_calibrated.scale.x = self.max_drift_distance_calibrated * 2
            drift_circle_calibrated.scale.y = self.max_drift_distance_calibrated * 2
            drift_circle_calibrated.scale.z = 0.01
            drift_circle_calibrated.color.r = 0.0
            drift_circle_calibrated.color.g = 1.0
            drift_circle_calibrated.color.b = 0.0
            drift_circle_calibrated.color.a = 0.3  # 반투명
            marker_array.markers.append(drift_circle_calibrated)
        
        # 원본 드리프트 원 표시 (빨간색, 반투명)
        if self.is_initialized and self.max_drift_distance_raw > 0.01:
            drift_circle_raw = Marker()
            drift_circle_raw.header.frame_id = "map"
            drift_circle_raw.header.stamp = self.get_clock().now().to_msg()
            drift_circle_raw.ns = "drift_visualization_raw"
            drift_circle_raw.id = 0
            drift_circle_raw.type = Marker.CYLINDER
            drift_circle_raw.action = Marker.ADD
            drift_circle_raw.pose.position.x = 0.0
            drift_circle_raw.pose.position.y = 0.0
            drift_circle_raw.pose.position.z = 0.02  # 캘리브레이션 원보다 약간 위
            drift_circle_raw.pose.orientation.w = 1.0
            drift_circle_raw.scale.x = self.max_drift_distance_raw * 2
            drift_circle_raw.scale.y = self.max_drift_distance_raw * 2
            drift_circle_raw.scale.z = 0.01
            drift_circle_raw.color.r = 1.0
            drift_circle_raw.color.g = 0.0
            drift_circle_raw.color.b = 0.0
            drift_circle_raw.color.a = 0.3  # 반투명
            marker_array.markers.append(drift_circle_raw)
        
        self.marker_publisher.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = DeadReckoningNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('노드가 종료됩니다.')
        if node.is_initialized:
            node.get_logger().info(f'=== 최종 드리프트 비교 통계 ===')
            node.get_logger().info(f'총 처리 샘플: {node.sample_count}')
            node.get_logger().info(f'캘리브레이션된 최대 드리프트: {node.max_drift_distance_calibrated:.4f}m')
            node.get_logger().info(f'원본 데이터 최대 드리프트: {node.max_drift_distance_raw:.4f}m')
            
            improvement_ratio = node.max_drift_distance_raw / node.max_drift_distance_calibrated if node.max_drift_distance_calibrated > 0 else float('inf')
            node.get_logger().info(f'드리프트 개선 비율: {improvement_ratio:.2f}x')
            
            node.get_logger().info(f'캘리브레이션 최종 위치: [{node.position_calibrated[0]:.4f}, {node.position_calibrated[1]:.4f}, {node.position_calibrated[2]:.4f}]')
            node.get_logger().info(f'원본 데이터 최종 위치: [{node.position_raw[0]:.4f}, {node.position_raw[1]:.4f}, {node.position_raw[2]:.4f}]')
            node.get_logger().info(f'캘리브레이션 최종 속도: {np.linalg.norm(node.velocity_calibrated):.6f} m/s')
            node.get_logger().info(f'원본 데이터 최종 속도: {np.linalg.norm(node.velocity_raw):.6f} m/s')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()