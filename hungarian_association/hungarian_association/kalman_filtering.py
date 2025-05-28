import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
import numpy as np
from scipy.spatial.transform import Rotation
import traceback

# 메시지 타입 임포트
from custom_interface.msg import ModifiedFloat32MultiArray
from custom_interface.msg import TrackedCone, TrackedConeArray
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer

# 칼만 필터 라이브러리
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

# UKF를 사용한 트랙 클래스 (3D 측정값 처리)
class Track:
    def __init__(self, track_id, initial_position_xyz, color, dt, T_imu_to_sensor,
                 P_initial_pos=5.0, P_initial_vel=1.0, R_measurement=0.5,
                 Q_process_diag_pos=0.1, Q_process_diag_vel=0.5):
        self.track_id = track_id
        self.initial_dt = dt
        
        # IMU 프레임에서 센서 프레임으로의 변환 행렬 저장
        # T_imu_to_sensor: 4x4 동차 변환 행렬
        # [ R_imu_to_sensor  t_imu_to_sensor ]
        # [       0                1         ]
        self.T_imu_to_sensor = np.array(T_imu_to_sensor).reshape(4, 4)
        self.R_imu_to_sensor = self.T_imu_to_sensor[:3, :3]  # 3x3 회전 행렬
        self.t_imu_to_sensor = self.T_imu_to_sensor[:3, 3]   # 3x1 이동 벡터

        # 상태 벡터: [콘_x, 콘_y, 콘_z, 센서_vx, 센서_vy, 센서_vz]
        dim_x = 6
        # 측정값은 3D [x, y, z]
        dim_z = 3
        points = MerweScaledSigmaPoints(n=dim_x, alpha=0.1, beta=2.0, kappa=(3.0 - dim_x))

        self.ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=self.initial_dt,
                                         fx=Track.static_fx, hx=self.hx, points=points)

        # 초기 상태
        self.ukf.x = np.zeros(dim_x, dtype=np.float64)
        self.ukf.x[0:3] = np.array(initial_position_xyz, dtype=np.float64)

        # 공분산 행렬
        self.ukf.P = np.diag([P_initial_pos]*3 + [P_initial_vel]*3)
        self.ukf.R = np.eye(dim_z) * R_measurement
        self.ukf.Q = np.diag([Q_process_diag_pos]*3 + [Q_process_diag_vel]*3)

        # 색상 관련 변수
        self.color_history = []
        self.color_counts = {"unknown": 0, "blue cone": 0, "red cone": 0, "yellow cone": 0}
        self.max_history_size = 20
        self.color_confidence_threshold = 3
        self.definite_color = None
        self.add_color_to_history(color.lower())

        self.missed_detections = 0

    @staticmethod
    def static_fx(x, dt, fx_args=None):
        """
        상태 전이 함수
        IMU 데이터를 기반으로 다음 상태 [콘_위치, 센서_속도]를 예측
        
        x: 현재 상태 [px, py, pz, vx, vy, vz]
        dt: 시간 간격
        fx_args: (R_imu_to_sensor, omega_imu_vec, accel_imu_vec) 튜플
        """
        if fx_args is None:
             print("[Track.static_fx] 경고: fx_args가 None입니다!")
             return x

        R_imu_to_sensor, omega_imu_vec, accel_imu_vec = fx_args
        current_pos_cone = x[0:3]
        current_vel_sensor = x[3:6]

        # 1. IMU 측정값을 센서 프레임으로 변환
        omega_sensor = R_imu_to_sensor @ omega_imu_vec
        accel_sensor = R_imu_to_sensor @ accel_imu_vec

        # 2. dt 동안의 센서 회전 계산
        rotation_vector = omega_sensor * dt
        try:
            R_delta = Rotation.from_rotvec(rotation_vector).as_matrix()
        except ValueError:
            R_delta = np.eye(3)

        # 역회전 (k+1 프레임에서 k 프레임으로 변환)
        R_compensation = R_delta.T

        # 3. k+1 시점의 센서 속도 예측 (k+1 프레임에서 표현)
        vel_kplus1_in_k = current_vel_sensor + accel_sensor * dt * -1
        predicted_vel_sensor = R_compensation @ vel_kplus1_in_k

        # 4. k+1 시점의 콘 위치 예측 (k+1 프레임에서 표현)
        delta_pos_sensor_in_k = current_vel_sensor * dt + 0.5 * accel_sensor * dt**2
        predicted_pos_cone = R_compensation @ (current_pos_cone - delta_pos_sensor_in_k)

        # 5. 예측된 상태 벡터 결합
        predicted_x = np.concatenate((predicted_pos_cone, predicted_vel_sensor))

        return predicted_x

    def hx(self, x):
        """
        측정 함수. 상태 벡터에서 [콘_x, 콘_y, 콘_z] 추출
        상태 x: [px, py, pz, vx, vy, vz]
        """
        return x[0:3]

    def predict(self, dt, imu_msg):
        """
        UKF 예측 단계 실행
        
        Args:
            dt: 마지막 예측 이후 시간 간격
            imu_msg: 각속도와 선형 가속도를 포함한 IMU 메시지
        """
        # IMU 메시지에서 각속도 추출
        use_angular_vel = imu_msg.angular_velocity_covariance[0] != -1.0
        omega_imu = np.zeros(3, dtype=np.float64)
        if use_angular_vel:
             omega_imu = np.array([imu_msg.angular_velocity.x,
                                   imu_msg.angular_velocity.y,
                                   imu_msg.angular_velocity.z], dtype=np.float64)

        # IMU 메시지에서 선형 가속도 추출
        use_linear_accel = imu_msg.linear_acceleration_covariance[0] != -1.0
        accel_imu = np.zeros(3, dtype=np.float64)
        if use_linear_accel:
            accel_imu = np.array([imu_msg.linear_acceleration.x,
                                  imu_msg.linear_acceleration.y,
                                  imu_msg.linear_acceleration.z], dtype=np.float64)
            
            # 중력 보정 구현
            use_orientation = imu_msg.orientation_covariance[0] != -1.0
            
            if use_orientation:
                q = np.array([imu_msg.orientation.x, 
                              imu_msg.orientation.y,
                              imu_msg.orientation.z,
                              imu_msg.orientation.w], dtype=np.float64)
                
                if np.linalg.norm(q) > 0.99:
                    gravity_world = np.array([0.0, 0.0, 9.81], dtype=np.float64)
                    
                    try:
                        r = Rotation.from_quat(q)
                        gravity_imu = r.apply(gravity_world)
                        accel_imu = accel_imu - gravity_imu
                    except Exception as e:
                        print(f"[Track.predict] 중력 보정 오류: {e}")
                else:
                    print("[Track.predict] 유효하지 않은 쿼터니언 - 간단한 중력 보정 사용")
                    accel_imu[2] = 0.0
            else:
                accel_imu[2] = 0.0

        # UKF 예측 단계에 IMU 데이터 전달
        fx_args_tuple = (self.R_imu_to_sensor, omega_imu, accel_imu)
        self.ukf.predict(dt=dt, fx_args=fx_args_tuple)

    def update(self, measurement_xyz, color):
        """UKF 업데이트 단계 실행 및 색상 처리 (3D 측정값 사용)"""
        z = np.asarray(measurement_xyz, dtype=np.float64)
        if z.shape != (3,):
            print(f"[Track.update 오류] 트랙 {self.track_id}: 잘못된 측정값 형태 {z.shape}. 예상: (3,).")
            return

        self.ukf.update(z)

        # 색상 처리 로직
        color_lower = color.lower()
        self.add_color_to_history(color_lower)
        if self.definite_color is None:
            counts = [(c, self.color_counts[c]) for c in ["blue cone", "red cone", "yellow cone"] if c in self.color_counts]
            if counts:
                 counts.sort(key=lambda item: item[1], reverse=True)
                 if counts[0][1] >= self.color_confidence_threshold:
                      self.definite_color = counts[0][0]

        self.missed_detections = 0

    def add_color_to_history(self, color):
        """색상 히스토리 관리"""
        color_lower = color.lower()
        self.color_history.append(color_lower)
        if len(self.color_history) > self.max_history_size:
            old_color = self.color_history.pop(0)
            if old_color in self.color_counts:
                self.color_counts[old_color] = max(0, self.color_counts[old_color] - 1)
        self.color_counts[color_lower] = self.color_counts.get(color_lower, 0) + 1

    def get_predicted_position_xyz(self):
        """필터링된 XYZ 위치 반환"""
        return self.ukf.x[0:3]

    def get_smoothed_color(self):
        """안정화된 색상 반환"""
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


# 콘 트래커 노드 (3D 입력/출력 처리)
class ConeTracker(Node):
    def __init__(self):
        super().__init__('cone_tracker_ukf')

        # 파라미터 선언
        self.declare_parameter('max_missed_detections', 4) 
        self.declare_parameter('distance_threshold', 0.7)
        self.declare_parameter('ukf.P_initial_pos', 0.001)
        self.declare_parameter('ukf.P_initial_vel', 100.0)
        self.declare_parameter('ukf.R_measurement', 0.1)
        self.declare_parameter('ukf.Q_process_diag_pos', 0.1)
        self.declare_parameter('ukf.Q_process_diag_vel', 0.1)
        self.declare_parameter('fixed_dt', 0.056)
        
        # IMU 프레임에서 센서 프레임으로의 변환 (os_imu -> os_sensor)
        default_transform = [
            1.0, 0.0, 0.0, 0.006253,    # X_sensor = X_imu + 6.253 mm (m 단위로 변환)
            0.0, 1.0, 0.0, -0.011775,   # Y_sensor = Y_imu - 11.775 mm (m 단위로 변환)
            0.0, 0.0, 1.0, 0.007645,    # Z_sensor = Z_imu + 7.645 mm (m 단위로 변환)
            0.0, 0.0, 0.0, 1.0
        ]
        self.declare_parameter('imu_to_sensor_transform', default_transform)

        # 파라미터 가져오기
        self.max_missed_detections = self.get_parameter('max_missed_detections').value
        self.distance_threshold = self.get_parameter('distance_threshold').value
        self.P_initial_pos = self.get_parameter('ukf.P_initial_pos').value
        self.P_initial_vel = self.get_parameter('ukf.P_initial_vel').value
        self.R_measurement = self.get_parameter('ukf.R_measurement').value
        self.Q_process_diag_pos = self.get_parameter('ukf.Q_process_diag_pos').value
        self.Q_process_diag_vel = self.get_parameter('ukf.Q_process_diag_vel').value
        self.fixed_dt = self.get_parameter('fixed_dt').value

        imu_transform_list = self.get_parameter('imu_to_sensor_transform').value
        try:
             self.T_imu_to_sensor = np.array(imu_transform_list, dtype=np.float64).reshape(4, 4)
        except ValueError as e:
             self.get_logger().error(f"잘못된 'imu_to_sensor_transform' 형식: {e}. 단위 행렬 사용.")
             self.T_imu_to_sensor = np.eye(4, dtype=np.float64)

        # 파라미터 로깅
        self.get_logger().info("--- 콘 트래커 UKF (3D 입력) 파라미터 ---")
        self.get_logger().info(f" 거리 임계값 (3D): {self.distance_threshold}")
        self.get_logger().info(f" 최대 검출 누락 횟수: {self.max_missed_detections}")
        self.get_logger().info(f" UKF 파라미터:")
        self.get_logger().info(f"   P_initial_pos: {self.P_initial_pos}")
        self.get_logger().info(f"   P_initial_vel: {self.P_initial_vel}")
        self.get_logger().info(f"   R_measurement (축별): {self.R_measurement}")
        self.get_logger().info(f"   Q_process_diag_pos: {self.Q_process_diag_pos}")
        self.get_logger().info(f"   Q_process_diag_vel: {self.Q_process_diag_vel}")
        self.get_logger().info(f" 고정 dt: {self.fixed_dt}")
        self.get_logger().info(f" IMU에서 센서로의 변환 (os_imu to os_sensor):")
        
        t_mm = self.T_imu_to_sensor[:3, 3] * 1000.0  # 표시를 위해 mm로 변환
        self.get_logger().info(f"   [ {self.T_imu_to_sensor[0,0]:.1f}  {self.T_imu_to_sensor[0,1]:.1f}  {self.T_imu_to_sensor[0,2]:.1f}  {t_mm[0]:.3f} mm ]")
        self.get_logger().info(f"   [ {self.T_imu_to_sensor[1,0]:.1f}  {self.T_imu_to_sensor[1,1]:.1f}  {self.T_imu_to_sensor[1,2]:.1f}  {t_mm[1]:.3f} mm ]")
        self.get_logger().info(f"   [ {self.T_imu_to_sensor[2,0]:.1f}  {self.T_imu_to_sensor[2,1]:.1f}  {self.T_imu_to_sensor[2,2]:.1f}  {t_mm[2]:.3f} mm ]")
        self.get_logger().info(f"   [ {self.T_imu_to_sensor[3,0]:.1f}  {self.T_imu_to_sensor[3,1]:.1f}  {self.T_imu_to_sensor[3,2]:.1f}  {self.T_imu_to_sensor[3,3]:.1f} ]")
        self.get_logger().info("---------------------------------------------")

        self.add_on_set_parameters_callback(self.parameters_callback)

        # QoS 설정
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 구독자 - '/fused_sorted_cones' 입력 토픽은 이제 X, Y, Z 좌표를 포함해야 함
        self.cone_sub = Subscriber(self, ModifiedFloat32MultiArray, '/fused_sorted_cones', qos_profile=qos_profile)
        self.imu_sub = Subscriber(self, Imu, '/ouster/imu', qos_profile=qos_profile)

        # 시간 동기화
        self.ts = ApproximateTimeSynchronizer(
            [self.cone_sub, self.imu_sub],
            queue_size=20,
            slop=0.15
        )
        self.ts.registerCallback(self.listener_callback)

        # 발행자 - '/fused_sorted_cones_ukf' 출력 토픽은 TrackedConeArray를 발행
        self.publisher_ = self.create_publisher(
            TrackedConeArray,
            '/fused_sorted_cones_ukf',
            qos_profile)

        self.tracks = {}
        self.next_track_id = 0
        self.last_time_stamp = None

        self.get_logger().info('3D 입력을 위한 콘 트래커 UKF 노드 초기화 완료.')

    def parameters_callback(self, params):
        """파라미터 콜백 - 기존 트랙의 R 및 Q 업데이트"""
        result = SetParametersResult(successful=True)
        param_dict = {p.name: p.value for p in params}

        try:
            # 노드 파라미터 업데이트
            if 'max_missed_detections' in param_dict: self.max_missed_detections = param_dict['max_missed_detections']
            if 'distance_threshold' in param_dict: self.distance_threshold = param_dict['distance_threshold']
            if 'ukf.P_initial_pos' in param_dict: self.P_initial_pos = param_dict['ukf.P_initial_pos']
            if 'ukf.P_initial_vel' in param_dict: self.P_initial_vel = param_dict['ukf.P_initial_vel']
            if 'ukf.R_measurement' in param_dict: self.R_measurement = param_dict['ukf.R_measurement']
            if 'ukf.Q_process_diag_pos' in param_dict: self.Q_process_diag_pos = param_dict['ukf.Q_process_diag_pos']
            if 'ukf.Q_process_diag_vel' in param_dict: self.Q_process_diag_vel = param_dict['ukf.Q_process_diag_vel']
            if 'fixed_dt' in param_dict: self.fixed_dt = param_dict['fixed_dt']
            if 'imu_to_sensor_transform' in param_dict:
                try:
                     new_transform = np.array(param_dict['imu_to_sensor_transform'], dtype=np.float64).reshape(4, 4)
                     self.T_imu_to_sensor = new_transform
                     self.get_logger().info("'imu_to_sensor_transform' 업데이트됨")
                except ValueError as e:
                     self.get_logger().error(f"업데이트 중 잘못된 'imu_to_sensor_transform': {e}. 적용되지 않음.")
                     result.successful = False

            # 기존 트랙에 변경 사항 적용
            if result.successful:
                track_ids = list(self.tracks.keys())
                for track_id in track_ids:
                    if track_id in self.tracks:
                        track = self.tracks[track_id]
                        if 'ukf.R_measurement' in param_dict:
                             track.ukf.R = np.eye(track.ukf.dim_z) * self.R_measurement
                        if 'ukf.Q_process_diag_pos' in param_dict or 'ukf.Q_process_diag_vel' in param_dict:
                             track.ukf.Q = np.diag([self.Q_process_diag_pos]*3 + [self.Q_process_diag_vel]*3)
                        if 'imu_to_sensor_transform' in param_dict:
                             track.T_imu_to_sensor = self.T_imu_to_sensor
                             track.R_imu_to_sensor = self.T_imu_to_sensor[:3, :3]
                             track.t_imu_to_sensor = self.T_imu_to_sensor[:3, 3]

        except Exception as e:
             self.get_logger().error(f"파라미터 콜백 처리 중 오류: {e}")
             self.get_logger().error(traceback.format_exc())
             result.successful = False

        # 최종 상태 로깅
        if result.successful:
             self.get_logger().info("파라미터 업데이트 성공.")
        else:
             self.get_logger().warn("하나 이상의 파라미터에 대한 업데이트 실패.")

        return result

    def listener_callback(self, cone_msg, imu_msg):
        """동기화된 3D 콘 및 IMU 메시지 처리"""
        current_time = self.get_clock().now()
        current_time_stamp = current_time.to_msg()
        current_time_sec = current_time.nanoseconds / 1e9

        # dt 계산
        dt = self.fixed_dt
        if self.fixed_dt <= 0:
            if self.last_time_stamp is not None:
                 last_time_sec = self.last_time_stamp.sec + self.last_time_stamp.nanosec / 1e9
                 dt = current_time_sec - last_time_sec
                 if dt <= 1e-4 or dt > 1.0:
                     self.get_logger().warn(f"비정상적인 dt 계산됨 ({dt:.4f}s). 조정 또는 기본값 사용.")
                     dt = np.clip(dt, 0.01, 0.2)
            else:
                 dt = 0.05  # 첫 프레임을 위한 기본 dt

        self.last_time_stamp = current_time_stamp

        # 3D 콘 데이터 추출 (X, Y, Z)
        detections = []
        colors = cone_msg.class_names
        num_data_elements = len(cone_msg.data)
        values_per_cone = 3  # X, Y, Z

        if num_data_elements % values_per_cone != 0:
            self.get_logger().error(
                f"입력 콘 데이터 길이({num_data_elements})가 {values_per_cone}의 배수가 아닙니다. 프레임 건너뜀."
            )
            return

        num_detections = num_data_elements // values_per_cone

        if len(colors) != num_detections:
             self.get_logger().error(f"검출 수({num_detections})와 색상 수({len(colors)}) 불일치. 프레임 건너뜀.")
             return

        for i in range(num_detections):
            idx = i * values_per_cone
            x = cone_msg.data[idx + 0]
            y = cone_msg.data[idx + 1]
            z = cone_msg.data[idx + 2]
            detections.append((x, y, z))

        # 1. 예측 단계
        track_ids_predict = list(self.tracks.keys())
        for track_id in track_ids_predict:
            if track_id in self.tracks:
                try:
                    self.tracks[track_id].predict(dt, imu_msg)
                except Exception as e:
                    self.get_logger().error(f"트랙 {track_id} 예측 중 오류: {e}\n{traceback.format_exc()}")

        # 2. 데이터 연관 (3D 거리 사용)
        matched_indices = set()
        assigned_tracks = set()
        association_pairs = []

        if num_detections > 0 and len(self.tracks) > 0:
            track_ids_assoc = list(self.tracks.keys())
            cost_matrix = np.full((num_detections, len(track_ids_assoc)), self.distance_threshold + 0.1)

            detections_np = np.array(detections, dtype=np.float64)

            for i, det_xyz in enumerate(detections):
                det_xyz_np = detections_np[i]
                for j, track_id in enumerate(track_ids_assoc):
                    if track_id in self.tracks:
                        track = self.tracks[track_id]
                        pred_pos_xyz = track.get_predicted_position_xyz()
                        dist = np.linalg.norm(det_xyz_np - pred_pos_xyz)
                        if dist < self.distance_threshold:
                            cost_matrix[i, j] = dist

            # 간단한 그리디 매칭 (3D 비용 행렬 사용)
            possible_matches = []
            for i in range(num_detections):
                 for j in range(len(track_ids_assoc)):
                      if cost_matrix[i, j] < self.distance_threshold:
                           possible_matches.append((cost_matrix[i, j], i, j))

            possible_matches.sort()  # 거리순 정렬

            matched_det_indices_assoc = set()
            matched_track_indices_assoc = set()

            for dist, det_idx, track_idx in possible_matches:
                 if det_idx not in matched_det_indices_assoc and track_idx not in matched_track_indices_assoc:
                      track_id = track_ids_assoc[track_idx]
                      association_pairs.append((det_idx, track_id, dist))
                      matched_indices.add(det_idx)
                      assigned_tracks.add(track_id)
                      matched_det_indices_assoc.add(det_idx)
                      matched_track_indices_assoc.add(track_idx)

        # 3. 매치된 트랙 업데이트 및 매치되지 않은 트랙/검출 처리
        for det_idx, track_id, _ in association_pairs:
            if track_id in self.tracks:
                 measurement_xyz = detections[det_idx]
                 color = colors[det_idx]
                 try:
                     self.tracks[track_id].update(measurement_xyz, color)
                 except Exception as e:
                     self.get_logger().error(f"트랙 {track_id} 업데이트 중 오류: {e}\n{traceback.format_exc()}")

        # 매치되지 않은 트랙 처리 (너무 자주 누락된 경우 삭제)
        track_ids_to_delete = []
        track_ids_unmatched = list(self.tracks.keys())
        for track_id in track_ids_unmatched:
            if track_id in self.tracks and track_id not in assigned_tracks:
                 track = self.tracks[track_id]
                 track.missed_detections += 1
                 if track.missed_detections > self.max_missed_detections:
                     track_ids_to_delete.append(track_id)
                     self.get_logger().info(f"트랙 {track_id} 삭제됨 (누락 {track.missed_detections} > {self.max_missed_detections})")

        for track_id in track_ids_to_delete:
            if track_id in self.tracks: del self.tracks[track_id]

        # 매치되지 않은 검출 처리 (새 트랙 생성)
        for i in range(num_detections):
            if i not in matched_indices:
                det_xyz = detections[i]
                color = colors[i]
                initial_pos_xyz = list(det_xyz)
                try:
                    new_track = Track(self.next_track_id, initial_pos_xyz, color, dt,
                                      self.T_imu_to_sensor,
                                      P_initial_pos=self.P_initial_pos,
                                      P_initial_vel=self.P_initial_vel,
                                      R_measurement=self.R_measurement,
                                      Q_process_diag_pos=self.Q_process_diag_pos,
                                      Q_process_diag_vel=self.Q_process_diag_vel)
                    self.tracks[self.next_track_id] = new_track
                    pos_str = f"[{initial_pos_xyz[0]:.3f}, {initial_pos_xyz[1]:.3f}, {initial_pos_xyz[2]:.3f}]"
                    self.get_logger().info(f"새 트랙 {self.next_track_id} 생성됨 위치: {pos_str}, 색상: {color}")
                    self.next_track_id += 1
                except Exception as e:
                     self.get_logger().error(f"검출 {i}에 대한 새 트랙 생성 중 오류: {e}\n{traceback.format_exc()}")

        # 결과 발행 (TrackedConeArray 사용하여 필터링된 XYZ)
        tracked_cones_msg = TrackedConeArray()
        tracked_cones_msg.header = cone_msg.header

        cones_list = []
        sorted_track_ids = sorted(list(self.tracks.keys()))
        for track_id in sorted_track_ids:
            if track_id in self.tracks:
                track = self.tracks[track_id]
                pos_xyz = track.get_predicted_position_xyz()
                color = track.get_smoothed_color()

                cone = TrackedCone()
                cone.track_id = track.track_id
                cone.position.x = pos_xyz[0]
                cone.position.y = pos_xyz[1]
                cone.position.z = pos_xyz[2]
                cone.color = color
                cones_list.append(cone)

        tracked_cones_msg.cones = cones_list

        self.publisher_.publish(tracked_cones_msg)

def main(args=None):
    rclpy.init(args=args)
    cone_tracker = None
    try:
        cone_tracker = ConeTracker()
        rclpy.spin(cone_tracker)
    except KeyboardInterrupt:
        if cone_tracker: cone_tracker.get_logger().info('KeyboardInterrupt, 종료 중.')
    except Exception as e:
        logger = rclpy.logging.get_logger("cone_tracker_ukf_main")
        logger.error(f"메인 루프에서 처리되지 않은 예외: {e}\n{traceback.format_exc()}")
    finally:
        if cone_tracker and rclpy.ok():
             try: cone_tracker.destroy_node()
             except Exception as destroy_e: rclpy.logging.get_logger("cone_tracker_ukf_main").error(f"노드 제거 중 오류: {destroy_e}")
        if rclpy.ok():
             try: rclpy.shutdown()
             except Exception as shutdown_e: rclpy.logging.get_logger("cone_tracker_ukf_main").error(f"RCLPY 종료 중 오류: {shutdown_e}")

if __name__ == '__main__':
    main()