#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PointStamped, Point, Quaternion, Vector3, TransformStamped
from custom_interface.msg import ModifiedFloat32MultiArray
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import json
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# --- Color Definitions (same as visualizer) ---
COLOR_MAP = {
    "red":      ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8),
    "red cone": ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8),
    "yellow":   ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8),
    "yellow cone":ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8),
    "blue":     ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8),
    "blue cone":ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8),
    "unknown":  ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8), # Green for Unknown
}
DEFAULT_COLOR = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.8) # Gray for unmapped classes

class ConeMapperNode(Node):
    """
    Subscribes to filtered cone detections (relative to sensor),
    transforms them into the map frame using TF, maintains a simple map,
    and publishes map markers for RViz.
    
    Frame structure:
    - map: Global fixed frame
    - odom: Odometry frame (initially coincident with map)
    - os_sensor: Sensor base frame (센서 바닥 기준)
    - os_lidar: LiDAR sensor center 
    """
    def __init__(self):
        super().__init__('cone_mapper_node')

        # --- Parameters ---
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('input_topic', '/fused_sorted_cones_ukf')
        self.declare_parameter('marker_topic', '/mapped_cones_markers')
        self.declare_parameter('association_threshold', 0.7) # Max distance (meters) to associate detection with existing landmark
        self.declare_parameter('marker_scale', [0.3, 0.3, 0.3]) # x, y, z scale for markers
        self.declare_parameter('save_map_on_shutdown', False)
        self.declare_parameter('load_map_on_startup', False)
        self.declare_parameter('map_file_path', 'cone_map.json')
        self.declare_parameter('publish_fallback_tf', False)

        self.map_frame = self.get_parameter('map_frame').value
        input_topic = self.get_parameter('input_topic').value
        marker_topic = self.get_parameter('marker_topic').value
        self.assoc_threshold = self.get_parameter('association_threshold').value
        marker_scale_list = self.get_parameter('marker_scale').value
        self.save_map = self.get_parameter('save_map_on_shutdown').value
        self.load_map = self.get_parameter('load_map_on_startup').value
        self.map_file = self.get_parameter('map_file_path').value
        self.publish_fallback = self.get_parameter('publish_fallback_tf').value

        if len(marker_scale_list) == 3:
            self.marker_scale = Vector3(x=marker_scale_list[0], y=marker_scale_list[1], z=marker_scale_list[2])
        else:
            self.get_logger().warn("Invalid 'marker_scale' param length. Using [0.3, 0.3, 0.3].")
            self.marker_scale = Vector3(x=0.3, y=0.3, z=0.3)

        if self.load_map:
            self._load_map_from_file()

        # --- TF Listener ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Map State ---
        # Dictionary: landmark_id -> {'position': np.array([x,y,z]), 'color': ColorRGBA, 'class_name': str}
        self.landmarks = {}
        self.next_landmark_id = 0
        self.published_marker_ids = set() # Track IDs published in the last cycle
        
        # TF 에러 추적을 위한 변수
        self._tf_error_count = 0
        self._tf_error_count_by_frame = {}

        # --- QoS 프로필 설정 ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # --- Publisher ---
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)

        # --- Subscriber ---
        self.cone_sub = self.create_subscription(
            ModifiedFloat32MultiArray,
            input_topic,
            self._cones_callback,
            qos_profile=sensor_qos
        )

        # tf_static_broadcaster 추가
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        
        # 필요한 경우 fallback TF 게시
        if self.publish_fallback:
            self._publish_fallback_tf()

        self.get_logger().info("Cone Mapper Node started.")
        self.get_logger().info(f"Mapping cones to frame: '{self.map_frame}'")
        self.get_logger().info(f"Subscribing to cone topic: '{input_topic}'")
        self.get_logger().info(f"Publishing map markers to: '{marker_topic}'")
        self.get_logger().info("Using BEST_EFFORT reliability for sensor data")
        self.get_logger().info("Expecting sensor frame os_sensor or os_lidar")

    def _get_color_for_class(self, class_name: str) -> ColorRGBA:
        """Returns the corresponding color for a given class name."""
        return COLOR_MAP.get(class_name.lower(), DEFAULT_COLOR)

    def _cones_callback(self, msg: ModifiedFloat32MultiArray):
        """Processes cone detections, transforms to map frame, updates map, publishes markers."""
        detections_sensor = [] # List of (pos_xyz_np, class_name) in sensor frame
        sensor_frame = msg.header.frame_id

        if not sensor_frame:
            self.get_logger().warn("Input cone message has empty frame_id. Cannot transform. Skipping.")
            return

        # --- 1. Parse Input Message (Expect 3D format) ---
        try:
            if len(msg.layout.dim) == 2 and msg.layout.dim[1].size == 3:
                num_cones = msg.layout.dim[0].size
                stride = msg.layout.dim[1].stride # Should be 3

                if num_cones * stride > len(msg.data) or num_cones != len(msg.class_names):
                     raise ValueError("Data size or class name mismatch with layout.")

                for i in range(num_cones):
                    idx = i * stride
                    pos_np = np.array([msg.data[idx], msg.data[idx+1], msg.data[idx+2]], dtype=np.float64)
                    detections_sensor.append((pos_np, msg.class_names[i]))

            # Add fallback parsing if needed
            elif len(msg.data) > 0:
                 self.get_logger().warn("Input message might not be in expected 3D layout. Trying simple XYZ parse.")
                 if len(msg.data) % 3 == 0 and len(msg.data)//3 == len(msg.class_names):
                     num_cones = len(msg.data) // 3
                     for i in range(num_cones):
                          idx = i * 3
                          pos_np = np.array([msg.data[idx], msg.data[idx+1], msg.data[idx+2]], dtype=np.float64)
                          detections_sensor.append((pos_np, msg.class_names[i]))
                 else:
                     raise ValueError("Fallback parse failed: Data length not multiple of 3 or class name mismatch.")
            else:
                # No data or invalid layout
                self.get_logger().debug("Received empty or unparseable cone message.")
                # Still publish markers (will handle deleting old ones)

        except Exception as e:
            self.get_logger().error(f"Error parsing input cone message: {e}. Skipping.")
            return

        # --- 2. Get Transform from Sensor to Map ---
        try:
            transform_map_sensor = self.tf_buffer.lookup_transform(
                self.map_frame,      
                sensor_frame,        
                msg.header.stamp,    
                timeout=Duration(seconds=0.5)
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            # 프레임별 에러 카운터 업데이트
            if sensor_frame not in self._tf_error_count_by_frame:
                self._tf_error_count_by_frame[sensor_frame] = 0
            
            self._tf_error_count_by_frame[sensor_frame] += 1
            self._tf_error_count += 1
            
            # 10회마다 한 번만 로그 출력
            if self._tf_error_count_by_frame[sensor_frame] % 10 == 1:
                self.get_logger().warn(f"TF lookup failed for frame '{sensor_frame}': {e}")
                self.get_logger().warn(f"Ensure IMU odometry node is running with base_frame='os_sensor'")
                if self.publish_fallback is False:
                    self.get_logger().warn("Consider setting 'publish_fallback_tf' to true")
            
            return

        # Transform 찾음 - 에러 카운터 리셋
        if sensor_frame in self._tf_error_count_by_frame and self._tf_error_count_by_frame[sensor_frame] > 0:
            self.get_logger().info(f"Successfully found transform for {sensor_frame} after {self._tf_error_count_by_frame[sensor_frame]} attempts!")
            self._tf_error_count_by_frame[sensor_frame] = 0

        # --- 3. Process Detections: Transform, Associate, Update Map ---
        current_map_associations = {} # landmark_id -> list of matched detection positions in map frame

        for pos_sensor_np, class_name in detections_sensor:
            # Create PointStamped for transformation
            pt_sensor = PointStamped()
            pt_sensor.header.frame_id = sensor_frame
            pt_sensor.header.stamp = msg.header.stamp # Use detection time
            pt_sensor.point.x = pos_sensor_np[0]
            pt_sensor.point.y = pos_sensor_np[1]
            pt_sensor.point.z = pos_sensor_np[2]

            # Transform point to map frame
            try:
                pt_map = tf2_geometry_msgs.do_transform_point(pt_sensor, transform_map_sensor)
                pos_map_np = np.array([pt_map.point.x, pt_map.point.y, pt_map.point.z])
            except Exception as e:
                self.get_logger().error(f"Error transforming point: {e}")
                continue # Skip this detection

            # Data Association (Nearest Neighbor in Map Frame)
            best_match_id = -1
            min_dist = self.assoc_threshold
            
            for lm_id, landmark in self.landmarks.items():
                dist = np.linalg.norm(pos_map_np - landmark['position'])
                
                # 같은 색상이면 연관성 점수 가중치 부여 (거리를 약간 줄여줌)
                if landmark['class_name'].lower() == class_name.lower() and class_name.lower() != "unknown":
                    dist *= 0.8  # 같은 색상에 가중치 부여
                    
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = lm_id

            # Map Update
            if best_match_id != -1:
                # Matched existing landmark
                if best_match_id not in current_map_associations:
                    current_map_associations[best_match_id] = []
                current_map_associations[best_match_id].append(pos_map_np)
                # Simple color update: if landmark was unknown, update it
                if self.landmarks[best_match_id]['class_name'].lower() == "unknown" and class_name.lower() != "unknown":
                    self.landmarks[best_match_id]['class_name'] = class_name
                    self.landmarks[best_match_id]['color'] = self._get_color_for_class(class_name)
            else:
                # New landmark
                new_id = self.next_landmark_id
                self.landmarks[new_id] = {
                    'position': pos_map_np,
                    'color': self._get_color_for_class(class_name),
                    'class_name': class_name
                }
                self.next_landmark_id += 1
                if new_id not in current_map_associations:
                    current_map_associations[new_id] = []
                current_map_associations[new_id].append(pos_map_np) # Associate with itself

        # --- Refine positions of matched landmarks ---
        for lm_id, observed_positions in current_map_associations.items():
            if lm_id in self.landmarks and len(observed_positions) > 0:
                # Simple average filter (could be more sophisticated)
                current_pos = self.landmarks[lm_id]['position']
                avg_observed = np.mean(np.array(observed_positions), axis=0)
                # Moving average alpha (e.g., 0.5)
                alpha = 0.5
                self.landmarks[lm_id]['position'] = alpha * avg_observed + (1.0 - alpha) * current_pos


        # --- 4. Publish Markers ---
        marker_array = MarkerArray()
        current_marker_ids = set()

        for lm_id, landmark in self.landmarks.items():
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = msg.header.stamp # Use cone message time for consistency
            marker.ns = "mapped_cones"
            marker.id = lm_id
            marker.type = Marker.SPHERE # Or CUBE, CYLINDER
            marker.action = Marker.ADD

            marker.pose.position = Point(x=landmark['position'][0], y=landmark['position'][1], z=landmark['position'][2])
            marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0) # Default orientation

            marker.scale = self.marker_scale
            marker.color = landmark['color']

            marker.lifetime = Duration(seconds=0).to_msg() # Persist indefinitely

            marker_array.markers.append(marker)
            current_marker_ids.add(lm_id)

        # Add DELETE markers for landmarks that disappeared
        ids_to_delete = self.published_marker_ids - current_marker_ids
        for del_id in ids_to_delete:
            delete_marker = Marker()
            delete_marker.header.frame_id = self.map_frame
            delete_marker.header.stamp = msg.header.stamp
            delete_marker.ns = "mapped_cones"
            delete_marker.id = del_id
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        # Update the set of published IDs for the next cycle
        self.published_marker_ids = current_marker_ids

        if marker_array.markers:
            self.marker_pub.publish(marker_array)

    def _load_map_from_file(self):
        """맵 데이터를 파일에서 불러옵니다."""
        try:
            with open(self.map_file, 'r') as f:
                data = json.load(f)
                
                # JSON에서 position 데이터가 리스트로 표현되었을 때 NumPy 배열로 변환
                landmarks_dict = {}
                for k, v in data.items():
                    if 'position' in v and isinstance(v['position'], list):
                        v['position'] = np.array(v['position'], dtype=np.float64)
                    landmarks_dict[int(k)] = v
                
                self.landmarks = landmarks_dict
                if self.landmarks:
                    self.next_landmark_id = max(self.landmarks.keys()) + 1
                    self.get_logger().info(f"Loaded {len(self.landmarks)} landmarks from {self.map_file}")
        except FileNotFoundError:
            self.get_logger().warn(f"Map file {self.map_file} not found. Starting with empty map.")
        except Exception as e:
            self.get_logger().error(f"Error loading map from file: {e}")

    def _save_map_to_file(self):
        """맵 데이터를 파일에 저장합니다."""
        if not self.save_map or not self.landmarks:
            return
            
        try:
            # NumPy 배열을 JSON 직렬화 가능한 리스트로 변환
            landmarks_dict = {}
            for k, v in self.landmarks.items():
                landmarks_dict[k] = v.copy()
                if 'position' in v and isinstance(v['position'], np.ndarray):
                    landmarks_dict[k]['position'] = v['position'].tolist()
            
            with open(self.map_file, 'w') as f:
                json.dump(landmarks_dict, f, indent=2)
            self.get_logger().info(f"Saved {len(self.landmarks)} landmarks to {self.map_file}")
        except Exception as e:
            self.get_logger().error(f"Error saving map to file: {e}")

    def _publish_fallback_tf(self):
        """임시 TF 연결 게시 (문제 해결용)"""
        self.get_logger().warn("Publishing fallback static TF. This is a temporary solution!")
        
        # map -> os_sensor 직접 연결 (임시 해결책)
        static_transform = TransformStamped()
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = self.map_frame
        static_transform.child_frame_id = 'os_sensor'  # 로봇 베이스 프레임
        static_transform.transform.translation.x = 0.0
        static_transform.transform.translation.y = 0.0
        static_transform.transform.translation.z = 0.0
        static_transform.transform.rotation.x = 0.0
        static_transform.transform.rotation.y = 0.0
        static_transform.transform.rotation.z = 0.0
        static_transform.transform.rotation.w = 1.0
        self.tf_static_broadcaster.sendTransform(static_transform)
        
        # os_sensor -> os_lidar 연결 (정확한 변환 행렬 적용)
        lidar_transform = TransformStamped()
        lidar_transform.header.stamp = self.get_clock().now().to_msg()
        lidar_transform.header.frame_id = 'os_sensor'
        lidar_transform.child_frame_id = 'os_lidar'
        
        # 변환 행렬 적용 (회전 매트릭스를 쿼터니언으로 변환)
        # X축과 Y축을 180도 회전 (X_sensor = -X_lidar, Y_sensor = -Y_lidar)
        # 이는 Z축 기준 180도 회전과 같음
        lidar_transform.transform.rotation.x = 0.0
        lidar_transform.transform.rotation.y = 0.0
        lidar_transform.transform.rotation.z = 1.0  # Z축 기준 180도 회전
        lidar_transform.transform.rotation.w = 0.0
        
        # 위치 오프셋 적용 (mm를 m로 변환)
        lidar_transform.transform.translation.x = 0.0
        lidar_transform.transform.translation.y = 0.0
        lidar_transform.transform.translation.z = 0.038195  # 38.195mm
        
        self.tf_static_broadcaster.sendTransform(lidar_transform)
        
        # os_sensor -> os_imu 연결 (정확한 변환 행렬 적용)
        imu_transform = TransformStamped()
        imu_transform.header.stamp = self.get_clock().now().to_msg()
        imu_transform.header.frame_id = 'os_sensor'
        imu_transform.child_frame_id = 'os_imu'
        
        # os_imu -> os_sensor의 역변환 (센서→IMU는 IMU→센서의 역)
        # 회전은 없음 (단위 회전)
        imu_transform.transform.rotation.x = 0.0
        imu_transform.transform.rotation.y = 0.0
        imu_transform.transform.rotation.z = 0.0
        imu_transform.transform.rotation.w = 1.0
        
        # 위치 오프셋 적용 (mm를 m로 변환, 역방향이므로 부호 반전)
        imu_transform.transform.translation.x = -0.006253   # -6.253mm
        imu_transform.transform.translation.y = 0.011775    # 11.775mm
        imu_transform.transform.translation.z = -0.007645   # -7.645mm
        
        self.tf_static_broadcaster.sendTransform(imu_transform)
        
        self.get_logger().warn("Published fallback transforms with exact sensor offsets:")
        self.get_logger().warn("- map -> os_sensor (identity)")
        self.get_logger().warn("- os_sensor -> os_lidar (180° Z rotation, 38.195mm Z offset)")
        self.get_logger().warn("- os_sensor -> os_imu (offsets: X=-6.253mm, Y=11.775mm, Z=-7.645mm)")

    def destroy_node(self):
        """노드가 종료될 때 맵을 저장합니다."""
        if self.save_map:
            self._save_map_to_file()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ConeMapperNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node:
            node.get_logger().error(f"Unhandled exception: {e}")
            import traceback
            node.get_logger().error(traceback.format_exc())
        else: 
            print(f"Exception before node init: {e}")
    finally:
        if node: 
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()