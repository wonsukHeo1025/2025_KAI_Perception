import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
import numpy as np
from scipy.spatial.transform import Rotation
import traceback # 예외 로깅을 위해 추가

# 메시지 타입 임포트
from custom_interface.msg import ModifiedFloat32MultiArray
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3 # For angular_velocity, linear_acceleration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer

# 칼만 필터 라이브러리
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

# --- Track 클래스 (UKF 적용) ---
class Track:
    def __init__(self, track_id, initial_position_xyz, color, dt, T_imu_to_sensor,
                 P_initial=5.0, R_measurement=0.5, Q_process_diag_pos=0.1):
        self.track_id = track_id
        self.dt = dt
        self.T_imu_to_sensor = np.array(T_imu_to_sensor).reshape(4, 4)
        self.R_imu_to_sensor = self.T_imu_to_sensor[:3, :3]
        self.t_imu_to_sensor = self.T_imu_to_sensor[:3, 3]

        dim_x = 3
        dim_z = 2
        points = MerweScaledSigmaPoints(n=dim_x, alpha=0.1, beta=2.0, kappa=(3.0 - dim_x))

        # --- fx 함수를 인스턴스 메서드가 아닌 일반 함수 또는 staticmethod로 정의 ---
        # filterpy가 self를 어떻게 처리할지 불확실하므로, 필요한 정보(R_imu_to_sensor)를
        # fx_args를 통해 전달하는 방식으로 변경 시도.
        # 또는 fx를 Track 클래스 외부에 정의할 수도 있음.
        # 여기서는 staticmethod로 변경 시도
        # self.ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=self.dt,
        #                                  fx=self.fx, hx=self.hx, points=points)
        self.ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=self.dt,
                                         fx=Track.static_fx, hx=self.hx, points=points)


        self.ukf.x = np.array(initial_position_xyz, dtype=np.float64)
        self.ukf.P = np.eye(dim_x) * P_initial
        self.ukf.R = np.eye(dim_z) * R_measurement
        self.ukf.Q = np.diag([Q_process_diag_pos] * dim_x)

        # 색상 관련 변수 (기존 코드 유지)
        self.color_history = []
        self.color_counts = {"unknown": 0, "blue cone": 0, "red cone": 0, "yellow cone": 0}
        self.max_history_size = 20
        self.color_confidence_threshold = 3
        self.definite_color = None
        self.add_color_to_history(color.lower())

        self.missed_detections = 0

    # !--- fx 함수를 staticmethod로 변경하고 시그니처 수정 ---!
    @staticmethod
    def static_fx(x, dt, fx_args=None):
        """
        상태 전이 함수 (Static Method).
        센서의 움직임(IMU)을 기반으로 다음 스텝의 센서 좌표계에서
        정지된 콘의 예상 위치를 계산합니다.
        x: 현재 상태 [x, y, z] (센서 좌표계 기준)
        dt: 시간 간격
        fx_args: (R_imu_to_sensor, omega_imu_vec, accel_imu_vec) 튜플
        """
        # fx_args 튜플에서 필요한 값들 추출
        R_imu_to_sensor, omega_imu_vec, accel_imu_vec = fx_args
        
        # 1. IMU 각속도를 센서 좌표계로 변환
        omega_sensor = R_imu_to_sensor @ omega_imu_vec
        # accel_sensor = R_imu_to_sensor @ accel_imu_vec # 참고용

        # 2. 센서의 회전 계산
        rotation_vector = omega_sensor * dt
        rotation_angle = np.linalg.norm(rotation_vector)

        # 회전 보상 행렬 계산 (센서 회전의 역변환)
        if rotation_angle > 1e-9:
            # scipy 사용: 센서 좌표계가 dt동안 회전한 변환의 역변환
            R_compensation = Rotation.from_rotvec(-rotation_vector).as_matrix()
        else: # 회전이 거의 없음
            R_compensation = np.eye(3)

        # 3. 콘의 위치 예측 (회전 보상만 적용)
        return R_compensation @ x

    def hx(self, x):
        """
        측정 함수 (Measurement Function).
        상태 벡터 [x, y, z]에서 측정값 [x, y]를 추출합니다.
        """
        return x[0:2]

    def predict(self, dt, imu_msg):
        """UKF 예측 단계 실행"""
        use_angular_vel = imu_msg.angular_velocity_covariance[0] != -1.0

        omega_imu = np.zeros(3, dtype=np.float64)
        if use_angular_vel:
             omega_imu = np.array([imu_msg.angular_velocity.x,
                                   imu_msg.angular_velocity.y,
                                   imu_msg.angular_velocity.z], dtype=np.float64)

        accel_imu = np.array([imu_msg.linear_acceleration.x,
                              imu_msg.linear_acceleration.y,
                              imu_msg.linear_acceleration.z], dtype=np.float64)

        # !--- fx_args를 튜플(tuple)로 전달 ---!
        # static_fx에 필요한 추가 인자들을 순서대로 넣음
        fx_args_tuple = (self.R_imu_to_sensor, omega_imu, accel_imu)
        self.ukf.predict(dt=dt, fx_args=fx_args_tuple)

    # --- update, add_color_to_history, get_predicted_position_xy, get_smoothed_color 메서드는 이전과 동일 ---
    def update(self, measurement_xy, color):
        """UKF 업데이트 단계 실행 및 색상 처리"""
        z = np.array(measurement_xy, dtype=np.float64)
        self.ukf.update(z)

        # 색상 처리 로직 (기존 코드 유지)
        color_lower = color.lower()
        self.add_color_to_history(color_lower)
        if self.definite_color is None:
            for cone_color, count in self.color_counts.items():
                if (cone_color != "unknown" and
                    count >= self.color_confidence_threshold and
                    cone_color in ["blue cone", "red cone", "yellow cone"]):
                    self.definite_color = cone_color
                    break
        self.missed_detections = 0

    def add_color_to_history(self, color):
        """색상 히스토리 관리 (기존 코드 유지)"""
        color_lower = color.lower()
        self.color_history.append(color_lower)
        if len(self.color_history) > self.max_history_size:
            old_color = self.color_history.pop(0)
            if old_color in self.color_counts:
                self.color_counts[old_color] = max(0, self.color_counts[old_color] - 1) # 음수 방지
        if color_lower in self.color_counts:
            self.color_counts[color_lower] += 1
        else:
            self.color_counts[color_lower] = 1

    def get_predicted_position_xy(self):
        """필터링된 XY 위치 반환"""
        return self.ukf.x[0:2]

    def get_smoothed_color(self):
        """안정화된 색상 반환 (기존 코드 개선)"""
        if self.definite_color is not None:
            return self.definite_color.capitalize()

        best_color = "unknown"
        best_count = 0
        for color, count in self.color_counts.items():
            if color != "unknown" and count > best_count:
                best_color = color
                best_count = count

        if best_color != "unknown":
            return best_color.capitalize()
        else:
            return "Unknown"


# --- ConeTracker 노드 (변경 없음, 이전 코드 사용) ---
class ConeTracker(Node):
    def __init__(self):
        super().__init__('cone_tracker_ukf')

        # 파라미터 선언 (기본값 설정)
        self.declare_parameter('max_missed_detections', 9)
        self.declare_parameter('distance_threshold', 0.7)
        self.declare_parameter('cone_z_offset', -0.6)
        self.declare_parameter('ukf.P_initial', 5.0)
        self.declare_parameter('ukf.R_measurement', 0.5)
        self.declare_parameter('ukf.Q_process_diag_pos', 0.1)
        self.declare_parameter('fixed_dt', 0.056)

        default_transform = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        self.declare_parameter('imu_to_sensor_transform', default_transform)

        # 파라미터 값 가져오기
        self.max_missed_detections = self.get_parameter('max_missed_detections').value
        self.distance_threshold = self.get_parameter('distance_threshold').value
        self.cone_z_offset = self.get_parameter('cone_z_offset').value
        self.P_initial = self.get_parameter('ukf.P_initial').value
        self.R_measurement = self.get_parameter('ukf.R_measurement').value
        self.Q_process_diag_pos = self.get_parameter('ukf.Q_process_diag_pos').value
        self.fixed_dt = self.get_parameter('fixed_dt').value

        imu_transform_list = self.get_parameter('imu_to_sensor_transform').value
        if len(imu_transform_list) == 16:
             self.T_imu_to_sensor = np.array(imu_transform_list, dtype=np.float64).reshape(4, 4)
        else:
             self.get_logger().error(f"Invalid 'imu_to_sensor_transform' length: {len(imu_transform_list)}. Using identity.")
             self.T_imu_to_sensor = np.eye(4, dtype=np.float64)

        self.get_logger().info(f"Cone Z Offset: {self.cone_z_offset}")
        self.get_logger().info(f"IMU to Sensor Transform:\n{self.T_imu_to_sensor}")
        self.get_logger().info(f"Distance Threshold: {self.distance_threshold}")
        self.get_logger().info(f"Max Missed Detections: {self.max_missed_detections}")
        self.get_logger().info(f"Fixed DT: {self.fixed_dt if self.fixed_dt > 0 else 'Using message timestamps'}")

        self.add_on_set_parameters_callback(self.parameters_callback)

        # QoS 설정
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 구독자 설정 (message_filters 사용)
        self.cone_sub = Subscriber(self, ModifiedFloat32MultiArray, '/fused_sorted_cones', qos_profile=qos_profile)
        self.imu_sub = Subscriber(self, Imu, '/ouster/imu', qos_profile=qos_profile)

        # 시간 동기화기
        self.ts = ApproximateTimeSynchronizer(
            [self.cone_sub, self.imu_sub],
            queue_size=15,
            slop=0.1
        )
        self.ts.registerCallback(self.listener_callback)

        # 발행자 설정
        self.publisher_ = self.create_publisher(
            ModifiedFloat32MultiArray,
            '/fused_sorted_cones_ukf',
            qos_profile)

        self.tracks = {}
        self.next_track_id = 0
        self.last_time_stamp = None

        self.get_logger().info('Cone Tracker UKF node initialized.')

    def parameters_callback(self, params):
        """파라미터 변경 시 호출될 콜백 함수"""
        result = SetParametersResult(successful=True)
        for param in params:
            param_name = param.name
            param_value = param.value
            try:
                if param_name == 'max_missed_detections':
                    self.max_missed_detections = param_value
                elif param_name == 'distance_threshold':
                    self.distance_threshold = param_value
                elif param_name == 'cone_z_offset':
                    self.cone_z_offset = param_value
                elif param_name == 'ukf.P_initial':
                    self.P_initial = param_value
                    # 새로 생성되는 트랙에 적용됨
                elif param_name == 'ukf.R_measurement':
                    self.R_measurement = param_value
                    # 기존 트랙 R 업데이트
                    for track_id in list(self.tracks.keys()): # Iterate over a copy of keys
                         if track_id in self.tracks:
                             self.tracks[track_id].ukf.R = np.eye(self.tracks[track_id].ukf.dim_z) * self.R_measurement
                elif param_name == 'ukf.Q_process_diag_pos':
                    self.Q_process_diag_pos = param_value
                    # 기존 트랙 Q 업데이트
                    for track_id in list(self.tracks.keys()): # Iterate over a copy of keys
                         if track_id in self.tracks:
                             self.tracks[track_id].ukf.Q = np.diag([self.Q_process_diag_pos] * self.tracks[track_id].ukf.dim_x)
                elif param_name == 'fixed_dt':
                    self.fixed_dt = param_value
                elif param_name == 'imu_to_sensor_transform':
                    if len(param_value) == 16:
                        self.T_imu_to_sensor = np.array(param_value, dtype=np.float64).reshape(4, 4)
                        # 기존 트랙 변환 행렬 업데이트
                        for track_id in list(self.tracks.keys()): # Iterate over a copy of keys
                             if track_id in self.tracks:
                                 track = self.tracks[track_id]
                                 track.T_imu_to_sensor = self.T_imu_to_sensor
                                 track.R_imu_to_sensor = self.T_imu_to_sensor[:3, :3]
                                 track.t_imu_to_sensor = self.T_imu_to_sensor[:3, 3]
                    else:
                        self.get_logger().error(f"Invalid 'imu_to_sensor_transform' length during update: {len(param_value)}. Not applied.")
                        result.successful = False
                else:
                     self.get_logger().warn(f"Ignoring update for unknown parameter: {param_name}")
                     continue # 다음 파라미터로 이동

                self.get_logger().info(f"Updated parameter '{param_name}' to {param_value}")

            except Exception as e:
                 # 파라미터 업데이트 중 트랙이 삭제되는 경우 KeyError 발생 가능
                 self.get_logger().error(f"Failed to update parameter '{param_name}' or apply to tracks: {e}")
                 result.successful = False # 실패로 처리

        return result

    def listener_callback(self, cone_msg, imu_msg):
        """동기화된 콘과 IMU 메시지 처리"""
        current_time = self.get_clock().now()
        current_time_stamp = current_time.to_msg()
        current_time_sec = current_time_stamp.sec + current_time_stamp.nanosec / 1e9

        # dt 계산
        if self.fixed_dt > 0:
            dt = self.fixed_dt
        elif self.last_time_stamp is not None:
            last_time_sec = self.last_time_stamp.sec + self.last_time_stamp.nanosec / 1e9
            dt = current_time_sec - last_time_sec
            if dt <= 0 or dt > 1.0:
                self.get_logger().warn(f"Unusual dt calculated ({dt:.4f}s). Using fixed_dt or default.")
                dt = self.fixed_dt if self.fixed_dt > 0 else 0.05
        else:
            dt = self.fixed_dt if self.fixed_dt > 0 else 0.05 # 첫 프레임

        self.last_time_stamp = current_time_stamp

        # 콘 메시지 데이터 추출
        num_detections = len(cone_msg.data) // 2
        detections = []
        colors = cone_msg.class_names
        for i in range(num_detections):
            x = cone_msg.data[2 * i]
            y = cone_msg.data[2 * i + 1]
            detections.append((x, y))

        # 1. 모든 트랙에 대해 예측 단계 실행
        # Iterate over a copy of keys in case tracks are deleted during prediction (less likely but safer)
        for track_id in list(self.tracks.keys()):
            if track_id in self.tracks: # Check if track still exists
                try:
                    self.tracks[track_id].predict(dt, imu_msg)
                except Exception as e:
                    self.get_logger().error(f"Error during prediction for track {track_id}: {e}")
                    # Consider deleting the problematic track or handling the error
                    # del self.tracks[track_id]


        # 2. 데이터 연관 (Data Association) - Nearest Neighbor (개선 가능)
        matched_indices = set()
        assigned_tracks = set()
        association_pairs = []

        if num_detections > 0 and len(self.tracks) > 0:
            track_ids = list(self.tracks.keys()) # Get current track IDs
            cost_matrix = np.full((num_detections, len(track_ids)), self.distance_threshold * 1.1) # Use track_ids length

            for i, det_xy in enumerate(detections):
                for j, track_id in enumerate(track_ids):
                    if track_id in self.tracks: # Check track exists
                        track = self.tracks[track_id]
                        pred_pos_xy = track.get_predicted_position_xy()
                        dist = np.linalg.norm(np.array(det_xy) - pred_pos_xy)
                        if dist < self.distance_threshold:
                            cost_matrix[i, j] = dist

            # 간단 매칭 (개선: Hungarian Algorithm 등 사용)
            for i in range(num_detections):
                 if len(track_ids) > 0: # Check if there are tracks to match against
                     valid_costs = cost_matrix[i, :]
                     min_cost_idx = np.argmin(valid_costs)
                     min_cost = valid_costs[min_cost_idx]

                     if min_cost < self.distance_threshold:
                         track_id = track_ids[min_cost_idx]

                         # 다른 detection에 이미 더 좋게 할당되었는지 확인
                         already_assigned_better = False
                         for det_idx_other, tid, dist_other in association_pairs:
                             if tid == track_id and dist_other <= min_cost:
                                 already_assigned_better = True
                                 break
                         if not already_assigned_better:
                             # 기존 할당 제거하고 새로 추가
                             association_pairs = [p for p in association_pairs if p[1] != track_id]
                             association_pairs.append((i, track_id, min_cost))
                             matched_indices.add(i)
                             assigned_tracks.add(track_id)


        # 3. 업데이트 및 트랙 관리
        # 매칭된 트랙 업데이트
        for det_idx, track_id, _ in association_pairs:
            if track_id in self.tracks: # Check track exists
                 measurement_xy = detections[det_idx]
                 color = colors[det_idx]
                 try:
                     self.tracks[track_id].update(measurement_xy, color)
                 except Exception as e:
                     self.get_logger().error(f"Error during update for track {track_id}: {e}")
                     # Consider deleting the problematic track
                     # if track_id in self.tracks: del self.tracks[track_id]


        # 매칭되지 않은 트랙: missed_detections 증가 및 삭제
        track_ids_to_delete = []
        # Iterate over a copy of keys
        for track_id in list(self.tracks.keys()):
            if track_id in self.tracks: # Check track exists
                if track_id not in assigned_tracks:
                    track = self.tracks[track_id]
                    track.missed_detections += 1
                    if track.missed_detections > self.max_missed_detections:
                        track_ids_to_delete.append(track_id)
                        self.get_logger().info(f"Track {track_id} deleted (max missed)")

        for track_id in track_ids_to_delete:
            if track_id in self.tracks: # Check before deleting
                del self.tracks[track_id]

        # 매칭되지 않은 detection: 새 트랙 생성
        for i in range(num_detections):
            if i not in matched_indices:
                det_xy = detections[i]
                # color 리스트 길이 확인 추가
                if i < len(colors):
                    color = colors[i]
                    initial_pos_xyz = [det_xy[0], det_xy[1], self.cone_z_offset]
                    try:
                        new_track = Track(self.next_track_id, initial_pos_xyz, color, dt,
                                        self.T_imu_to_sensor, self.P_initial, self.R_measurement, self.Q_process_diag_pos)
                        self.tracks[self.next_track_id] = new_track
                        self.get_logger().info(f"New track {self.next_track_id} created at {det_xy}")
                        self.next_track_id += 1
                    except Exception as e:
                         self.get_logger().error(f"Error creating new track: {e}")
                else:
                    self.get_logger().warn(f"Detection index {i} out of range for colors list (len={len(colors)}). Skipping new track creation.")


        # 4. 결과 발행 (필터링된 XY 좌표)
        filtered_msg = ModifiedFloat32MultiArray()
        filtered_msg.header = cone_msg.header
        # 레이아웃 복사 시 주의: deepcopy가 필요할 수 있으나, 여기서는 필수 필드만 재설정
        filtered_msg.layout.dim = [] # 이전 레이아웃 초기화
        filtered_msg.layout.data_offset = 0

        output_data = []
        output_colors = []

        # Iterate over a copy of keys
        sorted_track_ids = sorted(list(self.tracks.keys()))
        for track_id in sorted_track_ids:
            if track_id in self.tracks: # Check track exists
                track = self.tracks[track_id]
                pos_xy = track.get_predicted_position_xy()
                color = track.get_smoothed_color()
                output_data.extend(pos_xy.tolist())
                output_colors.append(color)

        filtered_msg.data = output_data
        filtered_msg.class_names = output_colors

        # 레이아웃 정보 설정
        from std_msgs.msg import MultiArrayDimension
        dim = MultiArrayDimension()
        dim.label = "tracked_cones"
        dim.size = len(output_colors)
        dim.stride = len(output_data) # Should be 2 * size if data is [x1,y1,x2,y2,...]
        filtered_msg.layout.dim.append(dim)

        self.publisher_.publish(filtered_msg)


# --- main 함수 (변경 없음, 이전 코드 사용) ---
def main(args=None):
    rclpy.init(args=args)
    cone_tracker = None # 초기화 추가
    try:
        cone_tracker = ConeTracker()
        rclpy.spin(cone_tracker)
    except KeyboardInterrupt:
        if cone_tracker:
             cone_tracker.get_logger().info('KeyboardInterrupt, shutting down.')
    except Exception as e:
        logger = rclpy.logging.get_logger("cone_tracker_ukf_main")
        logger.error(f"Unhandled exception in main loop: {e}")
        logger.error(traceback.format_exc()) # 트레이스백 로깅
    finally:
        # 노드 파괴 및 종료 로직 강화
        if cone_tracker is not None:
             if rclpy.ok(): # Check if context is still valid
                 try:
                     cone_tracker.destroy_node()
                 except Exception as destroy_e:
                     rclpy.logging.get_logger("cone_tracker_ukf_main").error(f"Error destroying node: {destroy_e}")

        if rclpy.ok(): # Check if context is still valid before shutting down
             try:
                 rclpy.shutdown()
             except Exception as shutdown_e:
                 rclpy.logging.get_logger("cone_tracker_ukf_main").error(f"Error shutting down RCLPY: {shutdown_e}")


if __name__ == '__main__':
    main()