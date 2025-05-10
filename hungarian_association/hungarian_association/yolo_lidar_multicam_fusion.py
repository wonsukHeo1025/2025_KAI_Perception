import os
import cv2
import yaml
import numpy as np
import rclpy
from typing import Tuple, List, Optional, Dict, Any

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer
from scipy.optimize import linear_sum_assignment
from hungarian_association.config_utils import load_hungarian_config

from yolo_msgs.msg import DetectionArray
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension
from custom_interface.msg import ModifiedFloat32MultiArray

def parse_multi_extrinsics_from_file(yaml_path: str) -> Dict[str, np.ndarray]:
    """여러 카메라의 외부 파라미터 행렬을 단일 YAML 파일에서 불러옵니다."""
    all_extrinsics = {}
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        for cam_id, cam_data in data.items():
            if 'extrinsic_matrix' in cam_data:
                matrix_list = cam_data['extrinsic_matrix']
                T = np.array(matrix_list, dtype=np.float64)
                all_extrinsics[cam_id] = T
            else:
                rclpy.logging.get_logger('yaml_parser').warn(
                    f"Camera ID '{cam_id}' in {yaml_path} missing 'extrinsic_matrix' key."
                )
        if not all_extrinsics:
            raise ValueError(f"No extrinsic matrices found or parsed from {yaml_path}")
        return all_extrinsics
    except Exception as e:
        rclpy.logging.get_logger('yaml_parser').error(f"Failed to parse multi-camera extrinsic matrices from {yaml_path}: {e}")
        raise

def parse_multi_intrinsics_from_file(yaml_path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """여러 카메라의 내부 파라미터를 단일 YAML 파일에서 불러옵니다."""
    all_intrinsics = {}
    try:
        with open(yaml_path, 'r') as f:
            calib_data_full = yaml.safe_load(f)
        for cam_id, calib_data in calib_data_full.items():
            if 'camera_matrix' in calib_data and 'distortion_coefficients' in calib_data:
                cam_mat_data = calib_data['camera_matrix']['data']
                camera_matrix = np.array(cam_mat_data, dtype=np.float64).reshape((3, 3))
                dist_data = calib_data['distortion_coefficients']['data']
                dist_coeffs = np.array(dist_data, dtype=np.float64).reshape((1, -1))
                all_intrinsics[cam_id] = (camera_matrix, dist_coeffs)
            else:
                rclpy.logging.get_logger('yaml_parser').warn(
                    f"Camera ID '{cam_id}' in {yaml_path} missing 'camera_matrix' or 'distortion_coefficients'."
                )
        if not all_intrinsics:
            raise ValueError(f"No intrinsic parameters found or parsed from {yaml_path}")
        return all_intrinsics
    except Exception as e:
        rclpy.logging.get_logger('yaml_parser').error(f"Failed to parse multi-camera intrinsic parameters from {yaml_path}: {e}")
        raise

class YoloLidarFusion(Node):
    def __init__(self):
        super().__init__('hungarian_association_node')

        try:
            self.config = load_hungarian_config()
            if self.config is None:
                raise ValueError("Configuration loading returned None.")
        except Exception as e:
            self.get_logger().fatal(f"CRITICAL: Failed to load hungarian_association configuration. Error: {e}. Shutting down.")
            raise SystemExit("Failed to load critical configuration.") from e

        hungarian_config = self.config.get('hungarian_association', {})

        self.max_matching_distance = hungarian_config.get('max_matching_distance')
        if self.max_matching_distance is None:
             self.get_logger().error("Parameter 'max_matching_distance' not found in config. Using default 25.0")
             self.max_matching_distance = 25.0
        else:
             self.get_logger().info(f"Max matching distance (for projected points): {self.max_matching_distance}")

        calib_config = hungarian_config.get('calibration', {})
        config_folder = calib_config.get('config_folder', '')
        extrinsic_file_name = calib_config.get('camera_extrinsic_calibration', '')
        intrinsic_file_name = calib_config.get('camera_intrinsic_calibration', '')

        if not config_folder or not extrinsic_file_name or not intrinsic_file_name:
             self.get_logger().fatal("CRITICAL: Calibration file paths or config folder not fully specified. Shutting down.")
             raise SystemExit("Missing critical calibration path configuration.")

        extrinsic_yaml_path = os.path.join(config_folder, extrinsic_file_name)
        intrinsic_yaml_path = os.path.join(config_folder, intrinsic_file_name)

        try:
            # lidar_frame -> camera_frame 변환 행렬
            raw_cam_extrinsics_lidar_to_cam = parse_multi_extrinsics_from_file(extrinsic_yaml_path)
            self.cam_intrinsics = parse_multi_intrinsics_from_file(intrinsic_yaml_path)
            self.get_logger().info(f"Loaded RAW extrinsics (lidar_frame to cam_frame) for cameras: {list(raw_cam_extrinsics_lidar_to_cam.keys())}")
            self.get_logger().info(f"Loaded intrinsics for cameras: {list(self.cam_intrinsics.keys())}")
        except Exception as e:
            self.get_logger().fatal(f"CRITICAL: Failed to load multi-camera calibrations. Error: {e}. Shutting down.")
            raise SystemExit("Failed to load multi-camera calibrations.") from e

        # 센서 좌표계(os_sensor) -> 라이다 좌표계 변환 행렬
        self.T_sensor_to_lidar = np.array(
            [[-1,  0,  0,   0      ],
             [ 0, -1,  0,   0      ],
             [ 0,  0,  1,  -0.038195], # Z축 오프셋 값
             [ 0,  0,  0,   1      ]], dtype=np.float64)
        self.get_logger().info(f"Defined T_sensor_to_lidar (NEEDS VERIFICATION FOR YOUR SETUP):\\n{self.T_sensor_to_lidar}")
        
        # 센서 -> 카메라 변환
        self.final_cam_transforms = {}

        self.camera_configs = hungarian_config.get('cameras', [])
        if not self.camera_configs:
            self.get_logger().fatal("CRITICAL: No cameras defined in the 'cameras' section of the config. Shutting down.")
            raise SystemExit("No cameras configured.")

        for cam_conf in self.camera_configs:
            cam_id = cam_conf.get('id')
            T_lidar_to_cam = raw_cam_extrinsics_lidar_to_cam.get(cam_id)

            if T_lidar_to_cam is None or cam_id not in self.cam_intrinsics:
                self.get_logger().fatal(f"CRITICAL: Calibration data (extrinsic or intrinsic) not found for configured camera ID '{cam_id}'. Shutting down.")
                raise SystemExit(f"Missing calibration for camera '{cam_id}'.")

            # 최종 변환 행렬 계산: T_sensor_to_cam = T_lidar_to_cam @ T_sensor_to_lidar
            # P_cam = T_lidar_to_cam @ T_sensor_to_lidar @ P_sensor
            self.final_cam_transforms[cam_id] = T_lidar_to_cam @ self.T_sensor_to_lidar
            self.get_logger().info(f"Calculated final T_sensor_to_cam for camera '{cam_id}':\\n{self.final_cam_transforms[cam_id]}")

        cones_topic = hungarian_config.get('cones_topic', "/sorted_cones_time")
        output_topic = hungarian_config.get('output_topic', "/fused_sorted_cones")

        qos_config = hungarian_config.get('qos', {})
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=qos_config.get('history_depth', 1)
        )

        self.cones_sub = Subscriber(self, ModifiedFloat32MultiArray, cones_topic, qos_profile=best_effort_qos)
        
        self.detection_subs = []
        for cam_conf in self.camera_configs:
            cam_id = cam_conf['id']
            topic = cam_conf['detections_topic']
            self.detection_subs.append(Subscriber(self, DetectionArray, topic, qos_profile=best_effort_qos))
            self.get_logger().info(f"Subscribing to detections for camera '{cam_id}' on topic: {topic}")

        self.get_logger().info(f"Subscribing to cones topic: {cones_topic}")
        self.get_logger().info(f"Publishing fused output to: {output_topic}")

        subscribers_list = [self.cones_sub] + self.detection_subs
        self.ats = ApproximateTimeSynchronizer(
            subscribers_list,
            queue_size=qos_config.get('sync_queue_size', 10),
            slop=qos_config.get('sync_slop', 0.1) # 단위: 초
        )
        self.ats.registerCallback(self.multi_camera_hungarian_callback)

        self.coord_pub = self.create_publisher(
            ModifiedFloat32MultiArray,
            output_topic,
            qos_profile=best_effort_qos
        )

        self.unmatched_class_name = "Unknown"
        self.get_logger().info(f"Unmatched LiDAR cones will be labeled as: '{self.unmatched_class_name}'")
        self.get_logger().info('YoloLidarFusion node initialized successfully with corrected transforms.')


    @staticmethod
    def convert_yolo_msg_to_array(yolo_msg: DetectionArray) -> np.ndarray:
        boxes = []
        if not yolo_msg.detections:
            return np.empty((0, 4))
        for detection in yolo_msg.detections:
            cx = detection.bbox.center.position.x
            cy = detection.bbox.center.position.y
            w = detection.bbox.size.x
            h = detection.bbox.size.y
            if w <= 0 or h <= 0:
                rclpy.logging.get_logger('yolo_converter').warn(
                    f"Received invalid bbox dimensions (w={w}, h={h}). Skipping detection."
                )
                continue
            boxes.append([cx, cy, w, h])
        return np.array(boxes, dtype=np.float32)


    def project_lidar_for_matching(self, 
                                   cones_xyz_sensor_frame: np.ndarray, # 입력: sensor_frame 기준 LiDAR 포인트
                                   T_sensor_to_cam: np.ndarray,       # 변환 행렬: sensor_frame -> cam_frame
                                   camera_matrix: np.ndarray,
                                   dist_coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        num_points = cones_xyz_sensor_frame.shape[0]
        if num_points == 0:
            return np.empty((0, 2)), np.empty((0,), dtype=int)

        cones_xyz_sensor_h = np.hstack((cones_xyz_sensor_frame, np.ones((num_points, 1), dtype=np.float32)))
        
        # P_cam_row = P_sensor_row @ T_sensor_to_cam.T
        cones_cam_h = cones_xyz_sensor_h @ T_sensor_to_cam.T
        cones_cam_all = cones_cam_h[:, :3] # 카메라 좌표계의 3D 포인트 (N,3)

        # 카메라 전방 포인트 필터링 (Z_cam > 0)
        valid_indices_for_projection = np.where(cones_cam_all[:, 2] > 1e-3)[0]
        if len(valid_indices_for_projection) == 0:
             self.get_logger().debug("No LiDAR points were in front of the camera for projection.")
             return np.empty((0, 2)), np.empty((0,), dtype=int)

        cones_cam_projectable = cones_cam_all[valid_indices_for_projection]
        original_indices_of_projected = valid_indices_for_projection

        try:
            # rvec, tvec은 T_sensor_to_cam에 포함되어 있으므로 0으로 설정
            cone_image_points, _ = cv2.projectPoints(
                cones_cam_projectable.astype(np.float64), # (M,3)
                np.zeros((3,1), dtype=np.float64), # rvec
                np.zeros((3,1), dtype=np.float64), # tvec
                camera_matrix.astype(np.float64),
                dist_coeffs.astype(np.float64)
            )
            cone_image_points = cone_image_points.reshape(-1, 2) # (M,2)
        except cv2.error as e:
             self.get_logger().error(f"cv2.projectPoints failed during projection: {e}")
             return np.empty((0, 2)), np.empty((0,), dtype=int)
        
        self.get_logger().debug(f"Successfully projected {len(cone_image_points)} points for matching.")
        return cone_image_points, original_indices_of_projected


    def compute_cost_matrix(self, yolo_bboxes: np.ndarray, cone_image_points: np.ndarray) -> np.ndarray:
        # yolo_bboxes: (P, 4) [cx, cy, w, h]
        # cone_image_points: (M, 2) [u, v]
        # 반환: cost_matrix (P, M) - cost_matrix[i,j]는 i번째 bbox와 j번째 투영된 콘 간의 비용
        num_boxes = yolo_bboxes.shape[0]
        num_cones = cone_image_points.shape[0]

        if num_boxes == 0 or num_cones == 0:
             return np.full((num_boxes, num_cones), self.max_matching_distance + 1.0)

        cost_matrix = np.zeros((num_boxes, num_cones))

        for i in range(num_boxes):
            box_center_x = yolo_bboxes[i, 0]
            box_center_y = yolo_bboxes[i, 1]
            distances = np.linalg.norm(cone_image_points - [box_center_x, box_center_y], axis=1)
            cost_matrix[i, :] = np.where(distances < self.max_matching_distance, distances, self.max_matching_distance + 1.0)
        
        return cost_matrix


    def multi_camera_hungarian_callback(self, cone_msg: ModifiedFloat32MultiArray, *yolo_msgs: DetectionArray):
        try:
            if len(yolo_msgs) != len(self.camera_configs):
                self.get_logger().error(
                    f"Mismatch in received YOLO messages ({len(yolo_msgs)}) "
                    f"and configured cameras ({len(self.camera_configs)}). Skipping callback."
                )
                return

            self.get_logger().debug(f"Received synchronized messages. Cones timestamp: {cone_msg.header.stamp}")

            cone_data = np.array(cone_msg.data, dtype=np.float32)
            num_points = 0
            # 입력 LiDAR 포인트는 os_sensor 프레임 기준
            cones_xyz_all_sensor_frame = np.empty((0, 3), dtype=np.float32)

            if len(cone_msg.layout.dim) >= 2 and cone_msg.layout.dim[1].size == 3:
                num_points = cone_msg.layout.dim[0].size
                expected_size = num_points * 3
                if cone_data.size == expected_size and num_points > 0:
                    cones_xyz_all_sensor_frame = cone_data.reshape(num_points, 3)
                elif num_points > 0:
                     self.get_logger().error(f"Cone data size ({cone_data.size}) mismatch with layout. Skipping.")
                     return
            else:
                 self.get_logger().error(f"Input cone layout invalid or not XYZ. Skipping.")
                 return

            filtered_msg = ModifiedFloat32MultiArray()
            filtered_msg.header = cone_msg.header
            filtered_msg.layout.dim.append(MultiArrayDimension(label="cones", size=num_points, stride=num_points * 3))
            filtered_msg.layout.dim.append(MultiArrayDimension(label="coords", size=3, stride=3))
            filtered_msg.data = cone_data.tolist() # 원본 LiDAR 데이터 사용
            filtered_msg.class_names = [self.unmatched_class_name] * num_points

            if num_points == 0:
                self.get_logger().info("Received empty cone message. Publishing empty fused message.")
                self.coord_pub.publish(filtered_msg)
                return

            # {original_lidar_idx: (cost, class_name, cam_id)}
            lidar_point_final_matches: Dict[int, Tuple[float, str, str]] = {} 
            total_yolo_boxes_processed = 0
            total_raw_matches_before_resolution = 0

            for i, cam_conf in enumerate(self.camera_configs):
                cam_id = cam_conf['id']
                yolo_msg_current_cam = yolo_msgs[i]
                
                self.get_logger().debug(f"Processing camera: {cam_id}")

                # sensor_frame -> camera_frame
                T_sensor_to_current_cam = self.final_cam_transforms.get(cam_id)
                intrinsics_current = self.cam_intrinsics.get(cam_id)

                if T_sensor_to_current_cam is None or intrinsics_current is None:
                    self.get_logger().warn(f"Transform or intrinsic data missing for camera {cam_id} in callback. Skipping this camera.")
                    continue
                
                current_cam_matrix, current_dist_coeffs = intrinsics_current
                yolo_bboxes_current_cam = self.convert_yolo_msg_to_array(yolo_msg_current_cam)
                
                num_yolo_boxes_current_cam = yolo_bboxes_current_cam.shape[0]
                total_yolo_boxes_processed += num_yolo_boxes_current_cam

                if num_yolo_boxes_current_cam == 0:
                    self.get_logger().debug(f"No YOLO boxes from camera {cam_id}.")
                    continue

                cone_image_points, original_indices_of_projected = self.project_lidar_for_matching(
                    cones_xyz_all_sensor_frame, # 입력: sensor 프레임 기준
                    T_sensor_to_current_cam,    # 변환: sensor -> cam
                    current_cam_matrix,
                    current_dist_coeffs
                )
                num_projected = cone_image_points.shape[0]

                if num_projected == 0:
                    self.get_logger().debug(f"No LiDAR points projectable for camera {cam_id}.")
                    continue
                
                cost_matrix = self.compute_cost_matrix(yolo_bboxes_current_cam, cone_image_points)
                
                # yolo_indices: YOLO 박스 인덱스, projected_cone_indices: 투영된 콘 인덱스
                yolo_indices, projected_cone_indices = linear_sum_assignment(cost_matrix)
                
                current_cam_matches_count = 0
                for yolo_idx, proj_cone_idx in zip(yolo_indices, projected_cone_indices):
                    cost = cost_matrix[yolo_idx, proj_cone_idx]
                    if cost < self.max_matching_distance:
                        original_lidar_idx = original_indices_of_projected[proj_cone_idx]
                        
                        if not (0 <= yolo_idx < len(yolo_msg_current_cam.detections)):
                            self.get_logger().warn(f"Matched YOLO index {yolo_idx} out of bounds for camera {cam_id}. Skipping this match.")
                            continue

                        class_name = yolo_msg_current_cam.detections[yolo_idx].class_name
                        current_cam_matches_count += 1
                        total_raw_matches_before_resolution +=1

                        # 가장 비용이 낮은 매칭으로 업데이트
                        if original_lidar_idx not in lidar_point_final_matches or \
                           cost < lidar_point_final_matches[original_lidar_idx][0]:
                            lidar_point_final_matches[original_lidar_idx] = (cost, class_name, cam_id)
                
                self.get_logger().debug(f"Camera {cam_id}: {num_projected} points projectable, {current_cam_matches_count} matched within threshold.")

            num_final_assigned_matches = 0
            for k_lidar_idx, (cost, class_name, cam_id) in lidar_point_final_matches.items():
                if 0 <= k_lidar_idx < num_points: 
                    filtered_msg.class_names[k_lidar_idx] = class_name
                    num_final_assigned_matches +=1
                else:
                    self.get_logger().warn(f"Lidar index {k_lidar_idx} from matches out of bounds ({num_points}).")

            self.coord_pub.publish(filtered_msg)
            self.get_logger().info(
                f'Published {num_points} cones. '
                f'{num_final_assigned_matches} matched with YOLO (from any camera), '
                f'{num_points - num_final_assigned_matches} labeled as {self.unmatched_class_name}. '
                f'({total_yolo_boxes_processed} total YOLO boxes processed, '
                f'{total_raw_matches_before_resolution} raw matches before resolution).'
            )

        except Exception as e:
            self.get_logger().error(f'Error in multi_camera_hungarian_callback: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())


def main(args=None):
    rclpy.init(args=args)
    try:
        yolo_lidar_fusion_node = YoloLidarFusion()
        rclpy.spin(yolo_lidar_fusion_node)
    # KeyboardInterrupt 처리
    except (SystemExit, KeyboardInterrupt) as e:
         if isinstance(e, KeyboardInterrupt):
             rclpy.logging.get_logger('main').info('Keyboard Interrupt (SIGINT) received. Shutting down...')
         else:
             rclpy.logging.get_logger('main').fatal(f"Node initialization or spinning failed: {e}")
    finally:
        if 'yolo_lidar_fusion_node' in locals() and hasattr(yolo_lidar_fusion_node, 'destroy_node') and rclpy.ok():
            yolo_lidar_fusion_node.destroy_node()
        if rclpy.ok():
             rclpy.shutdown()
        print("YoloLidarFusion node shutdown complete.")

if __name__ == '__main__':
    main()