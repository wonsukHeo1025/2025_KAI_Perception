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
from filterpy.common import Q_discrete_white_noise # Keep for potential use

# --- Track 클래스 (UKF 적용, 3D Measurement) ---
class Track:
    # Added P_initial_vel, Q_process_diag_vel parameters
    # R_measurement now applies to a 3x3 matrix
    def __init__(self, track_id, initial_position_xyz, color, dt, T_imu_to_sensor,
                 P_initial_pos=5.0, P_initial_vel=1.0, R_measurement=0.5,
                 Q_process_diag_pos=0.1, Q_process_diag_vel=0.5):
        self.track_id = track_id
        self.initial_dt = dt
        # Store the transformation matrix from IMU frame to sensor frame
        # T_imu_to_sensor is a 4x4 homogeneous transformation matrix: 
        # [ R_imu_to_sensor  t_imu_to_sensor ]
        # [       0                1         ]
        self.T_imu_to_sensor = np.array(T_imu_to_sensor).reshape(4, 4)
        self.R_imu_to_sensor = self.T_imu_to_sensor[:3, :3]  # 3x3 rotation matrix
        self.t_imu_to_sensor = self.T_imu_to_sensor[:3, 3]   # 3x1 translation vector

        # State Vector: [cone_x, cone_y, cone_z, sensor_vx, sensor_vy, sensor_vz]
        dim_x = 6
        # --- MODIFIED: Measurement is now 3D [x, y, z] ---
        dim_z = 3
        points = MerweScaledSigmaPoints(n=dim_x, alpha=0.1, beta=2.0, kappa=(3.0 - dim_x))

        self.ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=self.initial_dt,
                                         fx=Track.static_fx, hx=self.hx, points=points)

        # Initial State
        self.ukf.x = np.zeros(dim_x, dtype=np.float64)
        self.ukf.x[0:3] = np.array(initial_position_xyz, dtype=np.float64) # Use full XYZ

        # Covariance Matrices
        self.ukf.P = np.diag([P_initial_pos]*3 + [P_initial_vel]*3)
        # --- MODIFIED: R is now 3x3 ---
        self.ukf.R = np.eye(dim_z) * R_measurement
        self.ukf.Q = np.diag([Q_process_diag_pos]*3 + [Q_process_diag_vel]*3)


        # Color related variables (unchanged)
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
        Augmented State Transition Function (Static Method).
        Predicts the next state [cone_pos_sensor, sensor_vel_sensor]
        based on sensor motion (IMU).

        x: Current state [px, py, pz, vx, vy, vz]
        dt: Time interval
        fx_args: (R_imu_to_sensor, omega_imu_vec, accel_imu_vec) tuple
            - R_imu_to_sensor: 3x3 rotation matrix from IMU to sensor frame
            - omega_imu_vec: Angular velocity in IMU frame [wx, wy, wz]
            - accel_imu_vec: Linear acceleration in IMU frame [ax, ay, az]
        """
        if fx_args is None:
             # Should not happen if called correctly from predict
             print("[Track.static_fx] Warning: fx_args is None!") # Add logging
             return x # Predict no change

        R_imu_to_sensor, omega_imu_vec, accel_imu_vec = fx_args
        current_pos_cone = x[0:3]
        current_vel_sensor = x[3:6]

        # 1. Transform IMU readings to Sensor Frame
        # Apply rotation from IMU frame to sensor frame
        # Note: Translation is not needed for angular velocity and acceleration as they are vectors, not points
        omega_sensor = R_imu_to_sensor @ omega_imu_vec
        accel_sensor = R_imu_to_sensor @ accel_imu_vec

        # 2. Calculate Sensor Rotation during dt
        rotation_vector = omega_sensor * dt
        # Use scipy Rotation
        try:
            R_delta = Rotation.from_rotvec(rotation_vector).as_matrix()
        except ValueError: # Handle potential zero rotation vector issues if norm is exactly zero before check
            R_delta = np.eye(3)

        # Inverse rotation (transforms from new frame k+1 back to old frame k)
        R_compensation = R_delta.T

        # 3. Predict Sensor Velocity at k+1 (expressed in frame k+1)
        vel_kplus1_in_k = current_vel_sensor + accel_sensor * dt * -1 # -1을 곱해서 상대운동 효과를 내긴 했는데 일단 뭐가 문제인지는 더 파 봐야함...
        predicted_vel_sensor = R_compensation @ vel_kplus1_in_k

        # 4. Predict Cone Position at k+1 (expressed in frame k+1)
        delta_pos_sensor_in_k = current_vel_sensor * dt + 0.5 * accel_sensor * dt**2
        predicted_pos_cone = R_compensation @ (current_pos_cone - delta_pos_sensor_in_k)

        # 5. Combine into the predicted state vector
        predicted_x = np.concatenate((predicted_pos_cone, predicted_vel_sensor))

        return predicted_x

    # --- MODIFIED: Measurement Function hx extracts 3D position ---
    def hx(self, x):
        """
        Measurement Function. Extracts [cone_x, cone_y, cone_z] from the state vector.
        State x is [px, py, pz, vx, vy, vz]
        """
        return x[0:3] # Return the x, y, z position of the cone

    def predict(self, dt, imu_msg):
        """
        UKF 예측 단계 실행
        
        Args:
            dt: Time delta since last prediction
            imu_msg: IMU message containing angular velocity and linear acceleration data
        """
        # Extract angular velocity from IMU message if available
        use_angular_vel = imu_msg.angular_velocity_covariance[0] != -1.0
        omega_imu = np.zeros(3, dtype=np.float64)
        if use_angular_vel:
             # Angular velocity in IMU frame
             omega_imu = np.array([imu_msg.angular_velocity.x,
                                   imu_msg.angular_velocity.y,
                                   imu_msg.angular_velocity.z], dtype=np.float64)

        # Extract linear acceleration from IMU message if available
        use_linear_accel = imu_msg.linear_acceleration_covariance[0] != -1.0
        accel_imu = np.zeros(3, dtype=np.float64)
        if use_linear_accel:
            # Linear acceleration in IMU frame
            accel_imu = np.array([imu_msg.linear_acceleration.x,
                                  imu_msg.linear_acceleration.y,
                                  imu_msg.linear_acceleration.z], dtype=np.float64)
            
            # 중력 보정 구현
            # 1. IMU의 orientation이 유효한지 확인
            use_orientation = imu_msg.orientation_covariance[0] != -1.0
            
            if use_orientation:
                # IMU의 orientation 정보를 사용하여 중력 벡터를 IMU 프레임으로 변환
                q = np.array([imu_msg.orientation.x, 
                              imu_msg.orientation.y,
                              imu_msg.orientation.z,
                              imu_msg.orientation.w], dtype=np.float64)
                
                if np.linalg.norm(q) > 0.99:  # 유효한 quaternion인지 확인
                    # 전역 프레임에서 중력 벡터 (ENU 좌표계 기준)
                    gravity_world = np.array([0.0, 0.0, 9.81], dtype=np.float64)
                    
                    # Quaternion을 Rotation 객체로 변환
                    try:
                        r = Rotation.from_quat(q)
                        # 글로벌 프레임의 중력 벡터를 IMU 프레임으로 변환
                        gravity_imu = r.apply(gravity_world)
                        # 측정된 가속도에서 중력 성분 제거
                        accel_imu = accel_imu - gravity_imu
                    except Exception as e:
                        print(f"[Track.predict] Gravity compensation error: {e}")
                else:
                    # 유효하지 않은 quaternion인 경우 대체 방법 사용
                    # 간단한 대체 방법: z 방향 가속도를 0으로 설정 (또는 다른 보정 방법)
                    print("[Track.predict] Invalid quaternion - using simplified gravity compensation")
                    accel_imu[2] = 0.0  # z방향 가속도는 없다고 가정
            else:
                # Orientation 정보가 없는 경우 간단한 대체 방법 적용
                print("[Track.predict] No orientation data - using simplified gravity compensation")
                accel_imu[2] = 0.0  # z방향 가속도는 없다고 가정

        # Pass IMU data to the UKF prediction step
        # R_imu_to_sensor will transform the IMU measurements to the sensor frame
        # The transformation applies the specific os_imu to os_sensor transform matrix:
        # [ 1  0  0  0.006253 ]  // X_sensor = X_imu + 6.253 mm
        # [ 0  1  0 -0.011775 ]  // Y_sensor = Y_imu - 11.775 mm
        # [ 0  0  1  0.007645 ]  // Z_sensor = Z_imu + 7.645 mm
        # [ 0  0  0  1        ]
        fx_args_tuple = (self.R_imu_to_sensor, omega_imu, accel_imu)
        self.ukf.predict(dt=dt, fx_args=fx_args_tuple)

    # --- MODIFIED: update takes 3D measurement ---
    def update(self, measurement_xyz, color):
        """UKF 업데이트 단계 실행 및 색상 처리 (using 3D measurement)"""
        # Ensure measurement is a NumPy array
        z = np.asarray(measurement_xyz, dtype=np.float64)
        if z.shape != (3,):
            # Add error logging here if needed
            print(f"[Track.update Error] Track {self.track_id}: Invalid measurement shape {z.shape}. Expected (3,).")
            return # Skip update if measurement shape is wrong

        self.ukf.update(z) # Pass the 3D measurement z

        # Color processing logic (unchanged)
        color_lower = color.lower()
        self.add_color_to_history(color_lower)
        if self.definite_color is None:
            # Simplified definite color logic
            counts = [(c, self.color_counts[c]) for c in ["blue cone", "red cone", "yellow cone"] if c in self.color_counts]
            if counts:
                 counts.sort(key=lambda item: item[1], reverse=True)
                 if counts[0][1] >= self.color_confidence_threshold:
                      self.definite_color = counts[0][0]

        self.missed_detections = 0

    def add_color_to_history(self, color):
        """색상 히스토리 관리 (unchanged)"""
        color_lower = color.lower()
        self.color_history.append(color_lower)
        if len(self.color_history) > self.max_history_size:
            old_color = self.color_history.pop(0)
            if old_color in self.color_counts:
                self.color_counts[old_color] = max(0, self.color_counts[old_color] - 1)
        # Use setdefault for cleaner counting
        self.color_counts[color_lower] = self.color_counts.get(color_lower, 0) + 1


    # --- MODIFIED: get predicted XYZ position ---
    def get_predicted_position_xyz(self):
        """필터링된 XYZ 위치 반환 (Extracts from the 6D state)"""
        return self.ukf.x[0:3]

    def get_smoothed_color(self):
        """안정화된 색상 반환 (unchanged)"""
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
             # Ensure "Unknown" is returned if no other color dominates or history is empty/only unknown
             return "Unknown"


# --- ConeTracker 노드 (Handling 3D Input/Output) ---
class ConeTracker(Node):
    def __init__(self):
        super().__init__('cone_tracker_ukf')

        # Parameters (R_measurement now applies to 3x3)
        self.declare_parameter('max_missed_detections', 4)
        self.declare_parameter('distance_threshold', 0.7) # Threshold now applies to 3D distance
        # cone_z_offset might be less critical if input Z is reliable, but keep for potential use/fallback
        self.declare_parameter('cone_z_offset', -0.6)
        self.declare_parameter('ukf.P_initial_pos', 5.0)
        self.declare_parameter('ukf.P_initial_vel', 1.0)
        self.declare_parameter('ukf.R_measurement', 0.5) # Uncertainty per axis (x, y, z)
        self.declare_parameter('ukf.Q_process_diag_pos', 0.1)
        self.declare_parameter('ukf.Q_process_diag_vel', 0.5)
        self.declare_parameter('fixed_dt', 0.056)
        # Transform from IMU frame to sensor frame (os_imu to os_sensor)
        # Convert millimeters to meters (divide by 1000)
        default_transform = [
            1.0, 0.0, 0.0, 0.006253,    # X_sensor = X_imu + 6.253 mm (converted to m)
            0.0, 1.0, 0.0, -0.011775,   # Y_sensor = Y_imu - 11.775 mm (converted to m)
            0.0, 0.0, 1.0, 0.007645,    # Z_sensor = Z_imu + 7.645 mm (converted to m)
            0.0, 0.0, 0.0, 1.0
        ]
        self.declare_parameter('imu_to_sensor_transform', default_transform)

        # Get parameters
        self.max_missed_detections = self.get_parameter('max_missed_detections').value
        self.distance_threshold = self.get_parameter('distance_threshold').value
        self.cone_z_offset = self.get_parameter('cone_z_offset').value
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
             self.get_logger().error(f"Invalid 'imu_to_sensor_transform' format: {e}. Using identity.")
             self.T_imu_to_sensor = np.eye(4, dtype=np.float64)

        # Log parameters
        self.get_logger().info("--- Cone Tracker UKF (3D Input) Parameters ---")
        self.get_logger().info(f" Distance Threshold (3D): {self.distance_threshold}")
        self.get_logger().info(f" Max Missed Detections: {self.max_missed_detections}")
        self.get_logger().info(f" Cone Z Offset: {self.cone_z_offset}")
        self.get_logger().info(f" UKF Parameters:")
        self.get_logger().info(f"   P_initial_pos: {self.P_initial_pos}")
        self.get_logger().info(f"   P_initial_vel: {self.P_initial_vel}")
        self.get_logger().info(f"   R_measurement (per axis): {self.R_measurement}")
        self.get_logger().info(f"   Q_process_diag_pos: {self.Q_process_diag_pos}")
        self.get_logger().info(f"   Q_process_diag_vel: {self.Q_process_diag_vel}")
        self.get_logger().info(f" Fixed dt: {self.fixed_dt}")
        self.get_logger().info(f" IMU to Sensor Transform (os_imu to os_sensor):")
        # Format the transformation matrix nicely for logging
        t_mm = self.T_imu_to_sensor[:3, 3] * 1000.0  # Convert to mm for display
        self.get_logger().info(f"   [ {self.T_imu_to_sensor[0,0]:.1f}  {self.T_imu_to_sensor[0,1]:.1f}  {self.T_imu_to_sensor[0,2]:.1f}  {t_mm[0]:.3f} mm ]")
        self.get_logger().info(f"   [ {self.T_imu_to_sensor[1,0]:.1f}  {self.T_imu_to_sensor[1,1]:.1f}  {self.T_imu_to_sensor[1,2]:.1f}  {t_mm[1]:.3f} mm ]")
        self.get_logger().info(f"   [ {self.T_imu_to_sensor[2,0]:.1f}  {self.T_imu_to_sensor[2,1]:.1f}  {self.T_imu_to_sensor[2,2]:.1f}  {t_mm[2]:.3f} mm ]")
        self.get_logger().info(f"   [ {self.T_imu_to_sensor[3,0]:.1f}  {self.T_imu_to_sensor[3,1]:.1f}  {self.T_imu_to_sensor[3,2]:.1f}  {self.T_imu_to_sensor[3,3]:.1f} ]")
        self.get_logger().info("---------------------------------------------")

        self.add_on_set_parameters_callback(self.parameters_callback)

        # QoS (unchanged)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers (unchanged topics, QoS)
        # INPUT topic '/fused_sorted_cones' now expected to have X, Y, Z
        self.cone_sub = Subscriber(self, ModifiedFloat32MultiArray, '/fused_sorted_cones', qos_profile=qos_profile)
        self.imu_sub = Subscriber(self, Imu, '/ouster/imu', qos_profile=qos_profile)

        # Time Synchronizer (unchanged)
        self.ts = ApproximateTimeSynchronizer(
            [self.cone_sub, self.imu_sub],
            queue_size=20,
            slop=0.15
        )
        self.ts.registerCallback(self.listener_callback)

        # Publisher (OUTPUT topic '/fused_sorted_cones_ukf' will now publish X, Y, Z)
        self.publisher_ = self.create_publisher(
            ModifiedFloat32MultiArray,
            '/fused_sorted_cones_ukf',
            qos_profile)

        self.tracks = {}
        self.next_track_id = 0
        self.last_time_stamp = None

        self.get_logger().info('Cone Tracker UKF node initialized for 3D input.')

    def parameters_callback(self, params):
        """Parameter callback - updates R and Q for existing tracks."""
        result = SetParametersResult(successful=True)
        param_dict = {p.name: p.value for p in params}

        try:
            # Update node parameters (as before)
            if 'max_missed_detections' in param_dict: self.max_missed_detections = param_dict['max_missed_detections']
            if 'distance_threshold' in param_dict: self.distance_threshold = param_dict['distance_threshold']
            if 'cone_z_offset' in param_dict: self.cone_z_offset = param_dict['cone_z_offset']
            if 'ukf.P_initial_pos' in param_dict: self.P_initial_pos = param_dict['ukf.P_initial_pos']
            if 'ukf.P_initial_vel' in param_dict: self.P_initial_vel = param_dict['ukf.P_initial_vel']
            if 'ukf.R_measurement' in param_dict: self.R_measurement = param_dict['ukf.R_measurement']
            if 'ukf.Q_process_diag_pos' in param_dict: self.Q_process_diag_pos = param_dict['ukf.Q_process_diag_pos']
            if 'ukf.Q_process_diag_vel' in param_dict: self.Q_process_diag_vel = param_dict['ukf.Q_process_diag_vel']
            if 'fixed_dt' in param_dict: self.fixed_dt = param_dict['fixed_dt']
            if 'imu_to_sensor_transform' in param_dict:
                # Validation during update
                try:
                     new_transform = np.array(param_dict['imu_to_sensor_transform'], dtype=np.float64).reshape(4, 4)
                     self.T_imu_to_sensor = new_transform
                     self.get_logger().info("Updated 'imu_to_sensor_transform'")
                except ValueError as e:
                     self.get_logger().error(f"Invalid 'imu_to_sensor_transform' during update: {e}. Not applied.")
                     result.successful = False

            # Apply changes to existing tracks
            if result.successful: # Only proceed if parameters are valid so far
                track_ids = list(self.tracks.keys())
                for track_id in track_ids:
                    if track_id in self.tracks:
                        track = self.tracks[track_id]
                        if 'ukf.R_measurement' in param_dict:
                             # R is now 3x3
                             track.ukf.R = np.eye(track.ukf.dim_z) * self.R_measurement
                        if 'ukf.Q_process_diag_pos' in param_dict or 'ukf.Q_process_diag_vel' in param_dict:
                             # Q is 6x6 block diagonal
                             track.ukf.Q = np.diag([self.Q_process_diag_pos]*3 + [self.Q_process_diag_vel]*3)
                        if 'imu_to_sensor_transform' in param_dict:
                             # Update transform components in track
                             track.T_imu_to_sensor = self.T_imu_to_sensor
                             track.R_imu_to_sensor = self.T_imu_to_sensor[:3, :3]
                             track.t_imu_to_sensor = self.T_imu_to_sensor[:3, 3]

        except Exception as e:
             self.get_logger().error(f"Failed during parameters callback processing: {e}")
             self.get_logger().error(traceback.format_exc())
             result.successful = False

        # Log final status
        if result.successful:
             self.get_logger().info("Parameter update successful.")
        else:
             self.get_logger().warn("Parameter update failed for one or more parameters.")

        return result


    def listener_callback(self, cone_msg, imu_msg):
        """Handles synchronized 3D cone and IMU messages."""
        current_time = self.get_clock().now()
        current_time_stamp = current_time.to_msg()
        current_time_sec = current_time.nanoseconds / 1e9 # Use floating point seconds

        # dt calculation (using floating point seconds)
        dt = self.fixed_dt
        if self.fixed_dt <= 0:
            if self.last_time_stamp is not None:
                 last_time_sec = self.last_time_stamp.sec + self.last_time_stamp.nanosec / 1e9
                 dt = current_time_sec - last_time_sec
                 if dt <= 1e-4 or dt > 1.0:
                     self.get_logger().warn(f"Unusual dt calculated ({dt:.4f}s). Clamping or using default.")
                     # Option: Use previous dt, default, or clamp
                     dt = np.clip(dt, 0.01, 0.2) # Clamp dt to a reasonable range
                     # Or use default: dt = 0.05
            else:
                 dt = 0.05 # Default dt for the first frame

        self.last_time_stamp = current_time_stamp # Store the Time message

        # --- MODIFIED: Extract 3D cone data (X, Y, Z) ---
        detections = []
        colors = cone_msg.class_names
        num_data_elements = len(cone_msg.data)
        values_per_cone = 3 # X, Y, Z

        if num_data_elements % values_per_cone != 0:
            self.get_logger().error(
                f"Input cone data length ({num_data_elements}) is not a multiple of {values_per_cone}. Skipping frame."
            )
            return

        num_detections = num_data_elements // values_per_cone

        if len(colors) != num_detections:
             self.get_logger().error(f"Mismatch between number of detections ({num_detections}) and number of colors ({len(colors)}). Skipping frame.")
             return

        for i in range(num_detections):
            idx = i * values_per_cone
            x = cone_msg.data[idx + 0]
            y = cone_msg.data[idx + 1]
            z = cone_msg.data[idx + 2]
            detections.append((x, y, z)) # Store as 3-tuple

        # 1. Predict Step (unchanged logic)
        track_ids_predict = list(self.tracks.keys())
        for track_id in track_ids_predict:
            if track_id in self.tracks:
                try:
                    self.tracks[track_id].predict(dt, imu_msg)
                except Exception as e:
                    self.get_logger().error(f"Error during prediction for track {track_id}: {e}\n{traceback.format_exc()}")


        # --- MODIFIED: Data Association using 3D distance ---
        matched_indices = set()
        assigned_tracks = set()
        association_pairs = []

        if num_detections > 0 and len(self.tracks) > 0:
            track_ids_assoc = list(self.tracks.keys())
            cost_matrix = np.full((num_detections, len(track_ids_assoc)), self.distance_threshold + 0.1)

            # Convert detections to numpy array for easier slicing if needed
            detections_np = np.array(detections, dtype=np.float64)

            for i, det_xyz in enumerate(detections): # Use the 3D detection
                det_xyz_np = detections_np[i] # Use numpy version for norm calc
                for j, track_id in enumerate(track_ids_assoc):
                    if track_id in self.tracks:
                        track = self.tracks[track_id]
                        # Get the predicted 3D position
                        pred_pos_xyz = track.get_predicted_position_xyz()
                        # Calculate 3D Euclidean distance
                        dist = np.linalg.norm(det_xyz_np - pred_pos_xyz)
                        if dist < self.distance_threshold:
                            cost_matrix[i, j] = dist

            # Simple Greedy Matching (using 3D cost matrix)
            possible_matches = []
            for i in range(num_detections):
                 for j in range(len(track_ids_assoc)):
                      if cost_matrix[i, j] < self.distance_threshold:
                           possible_matches.append((cost_matrix[i, j], i, j)) # (distance, det_idx, track_idx)

            possible_matches.sort() # Sort by distance

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


        # 3. Update matched tracks & Handle unmatched tracks/detections
        # --- MODIFIED: Update with 3D measurement ---
        for det_idx, track_id, _ in association_pairs:
            if track_id in self.tracks:
                 measurement_xyz = detections[det_idx] # Get the 3D tuple
                 color = colors[det_idx]
                 try:
                     self.tracks[track_id].update(measurement_xyz, color) # Pass 3D measurement
                 except Exception as e:
                     self.get_logger().error(f"Error during update for track {track_id}: {e}\n{traceback.format_exc()}")


        # Handle unmatched tracks (delete if missed too often)
        track_ids_to_delete = []
        track_ids_unmatched = list(self.tracks.keys())
        for track_id in track_ids_unmatched:
            if track_id in self.tracks and track_id not in assigned_tracks:
                 track = self.tracks[track_id]
                 track.missed_detections += 1
                 if track.missed_detections > self.max_missed_detections:
                     track_ids_to_delete.append(track_id)
                     self.get_logger().info(f"Track {track_id} deleted (missed {track.missed_detections} > {self.max_missed_detections})")

        for track_id in track_ids_to_delete:
            if track_id in self.tracks: del self.tracks[track_id]

        # Handle unmatched detections (create new tracks)
        # --- MODIFIED: Use detected XYZ for initial position ---
        for i in range(num_detections):
            if i not in matched_indices:
                det_xyz = detections[i] # Get the full 3D detection
                color = colors[i]
                # Use the detected XYZ directly as the initial position
                initial_pos_xyz = list(det_xyz)
                try:
                    # Create Track instance with all required parameters
                    new_track = Track(self.next_track_id, initial_pos_xyz, color, dt,
                                      self.T_imu_to_sensor,
                                      P_initial_pos=self.P_initial_pos,
                                      P_initial_vel=self.P_initial_vel,
                                      R_measurement=self.R_measurement,
                                      Q_process_diag_pos=self.Q_process_diag_pos,
                                      Q_process_diag_vel=self.Q_process_diag_vel)
                    self.tracks[self.next_track_id] = new_track
                    # Log with 3 digits of precision for position
                    pos_str = f"[{initial_pos_xyz[0]:.3f}, {initial_pos_xyz[1]:.3f}, {initial_pos_xyz[2]:.3f}]"
                    self.get_logger().info(f"New track {self.next_track_id} created at {pos_str}, color {color}")
                    self.next_track_id += 1
                except Exception as e:
                     self.get_logger().error(f"Error creating new track for detection {i}: {e}\n{traceback.format_exc()}")


        # --- MODIFIED: Publish Results (Filtered XYZ) ---
        filtered_msg = ModifiedFloat32MultiArray()
        filtered_msg.header = cone_msg.header # Preserve timestamp/frame_id from input cones
        filtered_msg.layout.dim = []
        filtered_msg.layout.data_offset = 0

        output_data = []
        output_colors = []

        sorted_track_ids = sorted(list(self.tracks.keys()))
        for track_id in sorted_track_ids:
            if track_id in self.tracks:
                track = self.tracks[track_id]
                pos_xyz = track.get_predicted_position_xyz() # Get filtered XYZ
                color = track.get_smoothed_color()
                output_data.extend(pos_xyz.tolist()) # Add [x, y, z]
                output_colors.append(color)

        filtered_msg.data = output_data
        filtered_msg.class_names = output_colors

        # Update layout info for 3D data
        from std_msgs.msg import MultiArrayDimension
        num_tracked_cones = len(output_colors)
        if num_tracked_cones > 0:
             dim_cones = MultiArrayDimension()
             dim_cones.label = "tracked_cones"
             dim_cones.size = num_tracked_cones
             dim_cones.stride = len(output_data) # Total number of floats
             filtered_msg.layout.dim.append(dim_cones)

             dim_coords = MultiArrayDimension()
             dim_coords.label = "xyz" # Label indicates 3D coords
             dim_coords.size = 3       # We have 3 values per cone
             dim_coords.stride = 3     # Each cone block has size 3
             filtered_msg.layout.dim.append(dim_coords)
        else: # Handle case of no tracks
             dim_cones = MultiArrayDimension()
             dim_cones.label = "tracked_cones"
             dim_cones.size = 0
             dim_cones.stride = 0
             filtered_msg.layout.dim.append(dim_cones)
             # Optionally add the xyz dim even if size is 0
             # dim_coords = MultiArrayDimension(label="xyz", size=3, stride=3)
             # filtered_msg.layout.dim.append(dim_coords)


        self.publisher_.publish(filtered_msg)


# --- main 함수 (unchanged) ---
def main(args=None):
    rclpy.init(args=args)
    cone_tracker = None
    try:
        cone_tracker = ConeTracker()
        rclpy.spin(cone_tracker)
    except KeyboardInterrupt:
        if cone_tracker: cone_tracker.get_logger().info('KeyboardInterrupt, shutting down.')
    except Exception as e:
        logger = rclpy.logging.get_logger("cone_tracker_ukf_main")
        logger.error(f"Unhandled exception in main loop: {e}\n{traceback.format_exc()}")
    finally:
        if cone_tracker and rclpy.ok():
             try: cone_tracker.destroy_node()
             except Exception as destroy_e: rclpy.logging.get_logger("cone_tracker_ukf_main").error(f"Error destroying node: {destroy_e}")
        if rclpy.ok():
             try: rclpy.shutdown()
             except Exception as shutdown_e: rclpy.logging.get_logger("cone_tracker_ukf_main").error(f"Error shutting down RCLPY: {shutdown_e}")


if __name__ == '__main__':
    main()