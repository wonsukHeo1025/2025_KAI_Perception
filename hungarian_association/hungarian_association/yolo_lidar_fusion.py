import os
import cv2
import yaml
import numpy as np
import rclpy
from typing import Tuple, List, Optional, Dict

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer
from scipy.optimize import linear_sum_assignment
# Assuming config_utils is available in the specified package
from hungarian_association.config_utils import load_hungarian_config

from yolo_msgs.msg import DetectionArray
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension
from custom_interface.msg import ModifiedFloat32MultiArray # Make sure this matches your actual interface package

def load_extrinsic_matrix(yaml_path: str) -> np.ndarray:
    """YAML 파일에서 외부 변환 행렬을 로드합니다."""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        matrix_list = data['extrinsic_matrix']
        T = np.array(matrix_list, dtype=np.float64)
        return T
    except Exception as e:
        rclpy.logging.get_logger('yaml_loader').error(f"Failed to load extrinsic matrix from {yaml_path}: {e}")
        raise

def load_camera_calibration(yaml_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """YAML 파일에서 카메라 캘리브레이션 정보를 로드합니다."""
    try:
        with open(yaml_path, 'r') as f:
            calib_data = yaml.safe_load(f)
        cam_mat_data = calib_data['camera_matrix']['data']
        camera_matrix = np.array(cam_mat_data, dtype=np.float64).reshape((3, 3))
        dist_data = calib_data['distortion_coefficients']['data']
        dist_coeffs = np.array(dist_data, dtype=np.float64).reshape((1, -1))
        return camera_matrix, dist_coeffs
    except Exception as e:
        rclpy.logging.get_logger('yaml_loader').error(f"Failed to load camera calibration from {yaml_path}: {e}")
        raise

class YoloLidarFusion(Node):
    def __init__(self):
        super().__init__('hungarian_association_node')

        # 설정 로드 (이전 버전의 강화된 버전)
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
             self.get_logger().error("Parameter 'max_matching_distance' not found in config. Using default 5.0")
             self.max_matching_distance = 5.0
        else:
             self.get_logger().info(f"Max matching distance (for projected points): {self.max_matching_distance}")

        calib_config = hungarian_config.get('calibration', {})
        config_folder = calib_config.get('config_folder', '')
        extrinsic_file = calib_config.get('camera_extrinsic_calibration', '')
        intrinsic_file = calib_config.get('camera_intrinsic_calibration', '')

        if not config_folder or not extrinsic_file or not intrinsic_file:
             self.get_logger().fatal("CRITICAL: Calibration file paths or config folder not fully specified in config. Shutting down.")
             raise SystemExit("Missing critical calibration path configuration.")

        extrinsic_yaml = os.path.join(config_folder, extrinsic_file)
        self.T_lidar_to_cam = load_extrinsic_matrix(extrinsic_yaml)

        camera_yaml = os.path.join(config_folder, intrinsic_file)
        self.camera_matrix, self.dist_coeffs = load_camera_calibration(camera_yaml)

        self.get_logger().info("Loaded extrinsic:\n{}".format(self.T_lidar_to_cam))
        self.get_logger().info("Camera matrix:\n{}".format(self.camera_matrix))
        self.get_logger().info("Distortion coeffs:\n{}".format(self.dist_coeffs))

        cones_topic = hungarian_config.get('cones_topic', "/sorted_cones_time")
        boxes_topic = hungarian_config.get('boxes_topic', "/detections")
        output_topic = hungarian_config.get('output_topic', "/fused_sorted_cones")

        self.get_logger().info(f"Subscribing to cones topic: {cones_topic}")
        self.get_logger().info(f"Subscribing to boxes topic: {boxes_topic}")
        self.get_logger().info(f"Publishing fused output to: {output_topic}")

        qos_config = hungarian_config.get('qos', {})
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=qos_config.get('history_depth', 1)
        )

        self.cones_sub = Subscriber(self, ModifiedFloat32MultiArray, cones_topic, qos_profile=best_effort_qos)
        self.boxes_sub = Subscriber(self, DetectionArray, boxes_topic, qos_profile=best_effort_qos)

        self.ats = ApproximateTimeSynchronizer(
            [self.cones_sub, self.boxes_sub],
            queue_size=qos_config.get('sync_queue_size', 10),
            slop=qos_config.get('sync_slop', 0.1)
        )
        self.ats.registerCallback(self.hungarian_callback)

        self.coord_pub = self.create_publisher(
            ModifiedFloat32MultiArray,
            output_topic,
            qos_profile=best_effort_qos
        )

        self.unmatched_class_name = "Unknown"
        self.get_logger().info(f"Unmatched LiDAR cones will be labeled as: '{self.unmatched_class_name}'")
        self.get_logger().info('YoloLidarFusion node initialized successfully')


    @staticmethod
    def convert_yolo_msg_to_array(yolo_msg: DetectionArray) -> np.ndarray:
        """DetectionArray 메시지를 numpy 배열 [cx, cy, w, h]로 변환합니다."""
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


    def project_lidar_for_matching(self, cones_xyz_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        매칭 목적으로만 LiDAR 점(XYZ)을 이미지 평면에 투영합니다.
        투영 전에 카메라 뒤의 점을 필터링합니다.

        Args:
            cones_xyz_all (np.ndarray): 모든 입력 LiDAR 점을 포함하는 (N, 3) 형태의 배열

        Returns:
            Tuple[np.ndarray, np.ndarray]:
            - cone_image_points: 카메라 앞에 있는 콘의 투영된 2D 점을 포함하는 (M, 2) 배열
            - original_indices_of_projected: cone_image_points의 점에 해당하는 원래 인덱스(0 ~ N-1)를
                                              포함하는 (M,) 배열
                                              투영 가능한 점이 없으면 빈 배열을 반환합니다.
        """
        num_points = cones_xyz_all.shape[0]
        if num_points == 0:
            return np.empty((0, 2)), np.empty((0,), dtype=int)

        # 모든 점을 동차 좌표로 변환 (N, 4)
        cones_xyz_h = np.hstack((cones_xyz_all, np.ones((num_points, 1), dtype=np.float32)))

        # 모든 점을 LiDAR에서 카메라 좌표계로 변환 (N, 4)
        cones_cam_h = cones_xyz_h @ self.T_lidar_to_cam.T
        cones_cam_all = cones_cam_h[:, :3] # 비동차 3D 점 추출 (N, 3)

        # 투영을 위한 필터링
        # 카메라 앞에 있는 점들의 인덱스 찾기
        valid_indices_for_projection = np.where(cones_cam_all[:, 2] > 1e-3)[0]

        if len(valid_indices_for_projection) == 0:
             self.get_logger().debug("No LiDAR points were in front of the camera for projection.")
             return np.empty((0, 2)), np.empty((0,), dtype=int)

        # 투영을 위해 카메라 앞의 점만 선택
        cones_cam_projectable = cones_cam_all[valid_indices_for_projection]
        # 이 투영 가능한 점들의 원래 인덱스 추적
        original_indices_of_projected = valid_indices_for_projection

        # 유효한 점만 이미지 평면으로 투영
        try:
            cone_image_points, _ = cv2.projectPoints(
                cones_cam_projectable.astype(np.float64), # 투영 가능한 점만 사용 (M, 3)
                np.zeros((3,1), dtype=np.float64), # rvec
                np.zeros((3,1), dtype=np.float64), # tvec
                self.camera_matrix.astype(np.float64),
                self.dist_coeffs.astype(np.float64)
            )
            cone_image_points = cone_image_points.reshape(-1, 2) # 결과 형태 (M, 1, 2) -> (M, 2)
        except cv2.error as e:
             self.get_logger().error(f"cv2.projectPoints failed during projection for matching: {e}")
             return np.empty((0, 2)), np.empty((0,), dtype=int) # 투영 실패 시 빈 배열 반환

        self.get_logger().debug(f"Successfully projected {len(cone_image_points)} points for matching.")

        # 2D 점과 해당 원래 인덱스 반환
        return cone_image_points, original_indices_of_projected


    def compute_cost_matrix(self, yolo_bboxes: np.ndarray, cone_image_points: np.ndarray) -> np.ndarray:
        """
        YOLO 박스 중심과 투영된 콘 점 사이의 유클리드 거리에 기반한
        비용 행렬을 계산합니다.
        """
        num_boxes = yolo_bboxes.shape[0]
        num_cones = cone_image_points.shape[0]

        if num_boxes == 0 or num_cones == 0:
             # 호출자가 예상하는 형태 반환, 비용은 임계값을 통과하지 않도록 보장
             return np.full((num_boxes, num_cones), self.max_matching_distance + 1.0)

        cost_matrix = np.zeros((num_boxes, num_cones))

        for i in range(num_boxes):
            center_x = yolo_bboxes[i, 0]
            center_y = yolo_bboxes[i, 1]
            distances = np.linalg.norm(cone_image_points - [center_x, center_y], axis=1)
            cost_matrix[i, :] = np.where(distances < self.max_matching_distance, distances, self.max_matching_distance + 1.0)

        return cost_matrix


    def hungarian_callback(self, cone_msg: ModifiedFloat32MultiArray, yolo_msg: DetectionArray):
        """
        동기화된 LiDAR 콘과 YOLO 탐지를 처리합니다.
        모든 입력 LiDAR 콘을 발행합니다.
        투영과 헝가리안 매칭은 투영 가능하고 매치되는 콘에 YOLO 클래스 이름을 할당하는 데만 사용됩니다.
        매치되지 않는 콘은 'Unknown'으로 표시됩니다.
        """
        try:
            self.get_logger().debug(f"Received synchronized messages. Cones timestamp: {cone_msg.header.stamp}, YOLO timestamp: {yolo_msg.header.stamp}")

            # 1. 모든 원본 LiDAR 콘 데이터(XYZ) 추출
            cone_data = np.array(cone_msg.data, dtype=np.float32)
            num_points = 0
            cones_xyz_all = np.empty((0, 3), dtype=np.float32)

            if len(cone_msg.layout.dim) >= 2 and cone_msg.layout.dim[1].size == 3:
                num_points = cone_msg.layout.dim[0].size
                expected_size = num_points * 3
                if cone_data.size == expected_size and num_points > 0:
                    cones_xyz_all = cone_data.reshape(num_points, 3)
                elif num_points > 0: # 크기 불일치
                     self.get_logger().error(
                        f"Cone data size ({cone_data.size}) mismatch with layout "
                        f"({num_points} cones * 3 values = {expected_size}). Skipping callback."
                     )
                     return # 데이터가 손상된 경우 처리 건너뜀
                # num_points가 0이면 cones_xyz_all은 빈 상태로 유지, 아래에서 처리
            else:
                 self.get_logger().error(
                    f"Input cone layout invalid or not XYZ. Got dim: {cone_msg.layout.dim}. Skipping callback."
                 )
                 return # 레이아웃이 잘못된 경우 처리 건너뜀

            # 2. 출력 메시지 구조 준비 - 모든 입력 콘 포함
            filtered_msg = ModifiedFloat32MultiArray()
            filtered_msg.header = cone_msg.header # 타임스탬프와 frame_id 유지
            # 레이아웃은 총 입력 콘 수를 반영
            filtered_msg.layout.dim.append(MultiArrayDimension(label="cones", size=num_points, stride=num_points * 3))
            filtered_msg.layout.dim.append(MultiArrayDimension(label="coords", size=3, stride=3)) # X, Y, Z
            filtered_msg.data = cone_data.tolist() # 모든 입력 데이터 직접 복사
            filtered_msg.class_names = [self.unmatched_class_name] * num_points # 모두 Unknown으로 초기화

            # 3. 입력 콘이 없는 경우 처리
            if num_points == 0:
                self.get_logger().info("Received empty cone message. Publishing empty fused message.")
                self.coord_pub.publish(filtered_msg) # 올바르게 구조화된 빈 메시지 발행
                return

            # 4. YOLO 데이터 변환
            yolo_bboxes = self.convert_yolo_msg_to_array(yolo_msg)
            num_yolo_boxes = yolo_bboxes.shape[0]

            # 5. YOLO 박스가 존재하는 경우에만 투영 및 매칭 시도
            match_dict_orig_idx: Dict[int, int] = {} # {original_cone_idx: yolo_box_idx}
            num_projected = 0
            num_actual_matches = 0

            if num_yolo_boxes > 0:
                # 매칭만을 위한 LiDAR 점 투영
                cone_image_points, original_indices_of_projected = self.project_lidar_for_matching(cones_xyz_all)
                num_projected = cone_image_points.shape[0]

                if num_projected > 0:
                    # 투영된 점을 사용하여 비용 행렬 계산
                    cost_matrix = self.compute_cost_matrix(yolo_bboxes, cone_image_points)

                    # 헝가리안 알고리즘 실행
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)

                    # 원래 콘 인덱스를 YOLO 인덱스에 매핑하는 조회 사전 구축
                    for i, j in zip(row_ind, col_ind):
                        # i = yolo_box_index, j = projected_cone_index
                        if cost_matrix[i, j] < self.max_matching_distance:
                            # 이 투영된 점에 해당하는 원래 인덱스 찾기
                            original_idx = original_indices_of_projected[j]
                            # 원래 인덱스를 키로 사용하여 매치 저장
                            match_dict_orig_idx[original_idx] = i
                            num_actual_matches += 1
                    self.get_logger().debug(f'Matching done: {num_projected} points projectable, {num_actual_matches} matched within threshold.')
                else:
                    self.get_logger().debug('No LiDAR points were projectable for matching.')
            else:
                self.get_logger().debug('No YOLO boxes received, skipping matching.')


            # 6. 매칭 결과에 따라 클래스 이름 할당
            # 모든 원래 콘(0 ~ num_points-1)을 반복
            for k in range(num_points):
                if k in match_dict_orig_idx:
                    # 이 콘은 매치되었음, YOLO 클래스 이름 가져오기
                    yolo_idx = match_dict_orig_idx[k]
                    # yolo_idx 유효성 검사 (드물게 발생)
                    if 0 <= yolo_idx < len(yolo_msg.detections):
                         filtered_msg.class_names[k] = yolo_msg.detections[yolo_idx].class_name
                    else:
                         self.get_logger().warn(f"Matched YOLO index {yolo_idx} out of bounds for {len(yolo_msg.detections)} detections. Keeping cone {k} as Unknown.")
                # 그 외: 초기화된 대로 'Unknown'으로 유지

            # 7. 업데이트된 클래스 이름으로 모든 콘을 포함하는 메시지 발행
            self.coord_pub.publish(filtered_msg)
            self.get_logger().info(
                f'Published {num_points} cones. '
                f'{num_actual_matches} matched with YOLO, {num_points - num_actual_matches} labeled as {self.unmatched_class_name}.'
                f' ({num_projected} were candidates for matching).'
            )

        except Exception as e:
            self.get_logger().error(f'Error in hungarian_callback: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())


def main(args=None):
    rclpy.init(args=args)
    try:
        hungarian_association_node = YoloLidarFusion()
        rclpy.spin(hungarian_association_node)
    except (SystemExit, Exception) as e:
         rclpy.logging.get_logger('main').fatal(f"Node initialization or spinning failed: {e}")
    finally:
        if 'hungarian_association_node' in locals() and rclpy.ok():
            hungarian_association_node.destroy_node()
        if rclpy.ok():
             rclpy.shutdown()
        print("YoloLidarFusion node shutdown complete.")


if __name__ == '__main__':
    main()