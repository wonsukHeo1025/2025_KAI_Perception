import os
import cv2
import yaml
import numpy as np
import rclpy
from typing import Tuple, List, Optional

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer
from scipy.optimize import linear_sum_assignment
from hungarian_association.config_utils import load_hungarian_config

from yolo_msgs.msg import DetectionArray
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension
from custom_interface.msg import ModifiedFloat32MultiArray




def load_extrinsic_matrix(yaml_path: str) -> np.ndarray:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    matrix_list = data['extrinsic_matrix']
    T = np.array(matrix_list, dtype=np.float64)
    return T

def load_camera_calibration(yaml_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(yaml_path, 'r') as f:
        calib_data = yaml.safe_load(f)
    cam_mat_data = calib_data['camera_matrix']['data']
    camera_matrix = np.array(cam_mat_data, dtype=np.float64)
    dist_data = calib_data['distortion_coefficients']['data']
    dist_coeffs = np.array(dist_data, dtype=np.float64).reshape((1, -1))
    return camera_matrix, dist_coeffs

class YoloLidarFusion(Node):
    def __init__(self):
        super().__init__('hungarian_association_node')
        
        # Load the configuration from hungarian_association package
        self.config = load_hungarian_config()
        if self.config is None:
            self.get_logger().error("Failed to load hungarian_association configuration.")
            return
            
        # Get parameters from config with defaults as fallback
        hungarian_config = self.config.get('hungarian_association', {})
        
        # Set up max matching distance from config only
        self.max_matching_distance = hungarian_config.get('max_matching_distance', 5.0)
        self.get_logger().info(f"Max matching distance: {self.max_matching_distance}")
        
        # Get calibration file paths
        calib_config = hungarian_config.get('calibration', {})
        config_folder = calib_config.get('config_folder', '')
        extrinsic_file = calib_config.get('camera_extrinsic_calibration', '')
        intrinsic_file = calib_config.get('camera_intrinsic_calibration', '')
        
        # Load extrinsic and intrinsic calibrations
        extrinsic_yaml = os.path.join(config_folder, extrinsic_file)
        self.T_lidar_to_cam = load_extrinsic_matrix(extrinsic_yaml)

        camera_yaml = os.path.join(config_folder, intrinsic_file)
        self.camera_matrix, self.dist_coeffs = load_camera_calibration(camera_yaml)

        self.get_logger().info("Loaded extrinsic:\n{}".format(self.T_lidar_to_cam))
        self.get_logger().info("Camera matrix:\n{}".format(self.camera_matrix))
        self.get_logger().info("Distortion coeffs:\n{}".format(self.dist_coeffs))

        # Get topic names from config
        cones_topic = hungarian_config.get('cones_topic', "/sorted_cones_time")
        boxes_topic = hungarian_config.get('boxes_topic', "/detections")
        output_topic = hungarian_config.get('output_topic', "/fused_sorted_cones")
        
        self.get_logger().info(f"Subscribing to cones topic: {cones_topic}")
        self.get_logger().info(f"Subscribing to boxes topic: {boxes_topic}")

        # QoS settings from config
        qos_config = hungarian_config.get('qos', {})
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=qos_config.get('history_depth', 1)
        )

        # message_filters를 이용해 2개 토픽을 동기화
        self.cones_sub = Subscriber(self, ModifiedFloat32MultiArray, cones_topic, qos_profile=best_effort_qos)
        self.boxes_sub = Subscriber(self, DetectionArray, boxes_topic, qos_profile=best_effort_qos)

        # Approximate time synchronization
        self.ats = ApproximateTimeSynchronizer(
            [self.cones_sub, self.boxes_sub],
            queue_size=qos_config.get('sync_queue_size', 10),
            slop=qos_config.get('sync_slop', 0.1)
        )
        
        self.ats.registerCallback(self.hungarian_callback)

        # Color mapping for visualization (BGR format)
        self.color_mapping = {
            "red cone": (0, 0, 255),   # Red
            "yellow cone":  (0, 255, 255), # Yellow
            "blue cone":    (255, 0, 0),   # Blue
            "Unknown":      (0, 255, 0)    # Green (default)
        }

        # Publisher for fused coordinates
        self.coord_pub = self.create_publisher(
            ModifiedFloat32MultiArray, 
            output_topic,
            qos_profile=best_effort_qos
        )

        self.get_logger().info('YoloLidarFusion node initialized')

    @staticmethod
    def convert_yolo_msg_to_array(yolo_msg):
        """Convert DetectionArray message to numpy array."""
        boxes = []
        for detection in yolo_msg.detections:
            boxes.append([
                detection.bbox.center.position.x,
                detection.bbox.center.position.y,
                detection.bbox.size.x,
                detection.bbox.size.y
            ])
        return np.array(boxes)

    def convert_cone_msg_to_array(self, cone_msg):
        """Convert ModifiedFloat32MultiArray message (containing X, Y, Z) 
           to numpy array and project to image plane."""
        cone_data = np.array(cone_msg.data, dtype=np.float32)
        
        if cone_data.size == 0:
            self.get_logger().warn("Empty cones data.")
            return np.array([]), np.array([])
        
        # Get number of points (cones) from layout
        if len(cone_msg.layout.dim) < 1:
             self.get_logger().error("Cone layout dimension is missing or invalid.")
             return np.array([]), np.array([])
        num_points = cone_msg.layout.dim[0].size
        
        # --- MODIFIED: Check for 3 values (X, Y, Z) per point ---
        expected_size = num_points * 3 
        if expected_size != cone_data.size:
            self.get_logger().error(
                f"Cone data size ({cone_data.size}) does not match layout "
                f"({num_points} cones * 3 values = {expected_size})."
            )
            return np.array([]), np.array([])
        
        if num_points == 0: # Handle case where num_points is 0 but data might be weird
            return np.array([]), np.array([])

        # --- MODIFIED: Reshape to (N, 3) array directly ---
        cones_xyz = cone_data.reshape(num_points, 3) 
        
        # Convert to homogeneous coordinates (still Nx4)
        cones_xyz_h = np.hstack((cones_xyz, np.ones((cones_xyz.shape[0], 1), dtype=np.float32)))
        
        # Transform from LiDAR to camera coordinate system
        # Output is Nx4, take first 3 columns for 3D points in camera frame
        cones_cam_h = cones_xyz_h @ self.T_lidar_to_cam.T 
        cones_cam = cones_cam_h[:, :3] 
        
        # Filter out points behind the camera (negative Z in camera coords)
        valid_indices = cones_cam[:, 2] > 0 # Keep only points with Z > 0
        cones_cam_valid = cones_cam[valid_indices]
        original_indices_valid = np.arange(num_points)[valid_indices] # Keep track of original indices of valid points

        if cones_cam_valid.shape[0] == 0:
             self.get_logger().debug("No valid cone points after camera transformation/filtering.")
             return np.array([]), np.array([])
             
        # Project valid points to image plane
        rvec = np.zeros((3,1), dtype=np.float64) # Assuming no rotation relative to camera frame for projection
        tvec = np.zeros((3,1), dtype=np.float64) # Assuming no translation relative to camera frame for projection
        
        # cv2.projectPoints expects Nx3 input
        cone_image_points, _ = cv2.projectPoints(
            cones_cam_valid.astype(np.float64), # Use only valid points
            rvec, tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        cone_image_points = cone_image_points.reshape(-1, 2) # Reshape to Nx2
            
        self.get_logger().debug(f"Projected {len(cone_image_points)} valid cones to image plane")
            
        # Return the 2D projected points and their corresponding original indices
        return cone_image_points, original_indices_valid

    def compute_cost_matrix(self, yolo_bboxes, cone_points):
        num_boxes = yolo_bboxes.shape[0]
        num_cones = cone_points.shape[0]
        cost_matrix = np.zeros((num_boxes, num_cones))
        
        # Fill the cost matrix with the Euclidean distances
        for i in range(num_boxes):
            # Calculate the center of the i-th bounding box
            center_x = yolo_bboxes[i, 0]
            center_y = yolo_bboxes[i, 1]
            for j in range(num_cones):
                distance = np.linalg.norm([
                    center_x - cone_points[j, 0],
                    center_y - cone_points[j, 1]
                ])
                # Penalize matches beyond maximum distance
                cost_matrix[i, j] = distance if distance < self.max_matching_distance else 1e6
        
        # Pad the cost matrix to make it square
        if num_boxes < num_cones:
            # Set cost for dummy YOLO boxes
            dummy_rows = np.full((num_cones - num_boxes, num_cones), 1e6)
            cost_matrix = np.vstack((cost_matrix, dummy_rows))
        elif num_boxes > num_cones:
            # Set cost for dummy LiDAR points
            dummy_cols = np.full((num_boxes, num_boxes - num_cones), 0.0)
            cost_matrix = np.hstack((cost_matrix, dummy_cols))
        
        return cost_matrix

    def hungarian_callback(self, cone_msg, yolo_msg):
        """Process synchronized YOLO and LiDAR cone detections (with Z coordinate)."""
        try:
            # Convert messages to NumPy arrays
            yolo_bboxes = self.convert_yolo_msg_to_array(yolo_msg)
            
            # Project cone points (X, Y, Z) to image plane (X_img, Y_img)
            # and get original indices of the *valid* projected points
            cone_image_points, original_indices = self.convert_cone_msg_to_array(cone_msg)
            
            # 매칭된 콘만 저장할 새로운 메시지 생성 (Prepare output message)
            filtered_msg = ModifiedFloat32MultiArray()
            filtered_msg.header = cone_msg.header # Preserve timestamp and frame_id
            # Prepare layout for Nx3 output
            filtered_msg.layout.dim.append(MultiArrayDimension()) # Rows (cones)
            filtered_msg.layout.dim.append(MultiArrayDimension()) # Cols (X, Y, Z)
            filtered_msg.layout.dim[0].label = "cones"
            filtered_msg.layout.dim[1].label = "coords"
            filtered_msg.layout.dim[1].size = 3 # Statically set to 3 (X, Y, Z)
            filtered_msg.layout.dim[1].stride = 3 # Each cone data block is 3 floats
            # dim[0].size and dim[0].stride will be set after matching
            
            filtered_msg.class_names = [] # Initialize empty list for class names
            filtered_msg.data = [] # Initialize empty list for data [x0, y0, z0, x1, y1, z1, ...]
            
            if len(yolo_bboxes) == 0 or len(cone_image_points) == 0:
                self.get_logger().warn('ZERO detections in YOLO or valid LiDAR cones after projection')
                # Publish empty message but with correct layout structure
                filtered_msg.layout.dim[0].size = 0
                filtered_msg.layout.dim[0].stride = 0
                self.coord_pub.publish(filtered_msg) 
                return
            
            # 코스트 매트릭스 계산 및 매칭 (using 2D projected points)
            cost_matrix = self.compute_cost_matrix(yolo_bboxes, cone_image_points)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # 매칭된 포인트만 필터링
            num_matched = 0
            for i, j in zip(row_ind, col_ind):
                # Ensure indices are valid for original arrays and cost is within threshold
                if (i < len(yolo_bboxes) and j < len(cone_image_points) and 
                    cost_matrix[i, j] < self.max_matching_distance):
                    
                    # Get the ORIGINAL index of the matched cone from the input cone_msg
                    original_idx = original_indices[j] 
                    
                    # --- MODIFIED: Extract original X, Y, AND Z using correct stride (3) ---
                    base_idx = original_idx * 3
                    if base_idx + 2 < len(cone_msg.data): # Safety check
                        original_x = cone_msg.data[base_idx + 0]
                        original_y = cone_msg.data[base_idx + 1]
                        original_z = cone_msg.data[base_idx + 2] # Get the Z value
                        
                        # --- MODIFIED: Add X, Y, Z to the filtered message data ---
                        filtered_msg.data.extend([original_x, original_y, original_z])
                        filtered_msg.class_names.append(yolo_msg.detections[i].class_name)
                        num_matched += 1
                    else:
                         self.get_logger().warn(f"Calculated index {base_idx+2} out of bounds for cone_msg.data (size {len(cone_msg.data)}) for original_idx {original_idx}. Skipping match.")

            # --- MODIFIED: Update final layout dimensions for the output message ---
            filtered_msg.layout.dim[0].size = num_matched # Number of matched cones
            filtered_msg.layout.dim[0].stride = num_matched * 3 # Total number of float values in data
            
            # 필터링된 메시지 발행 (Publish the fused X, Y, Z coordinates with class names)
            self.coord_pub.publish(filtered_msg)
            
            self.get_logger().info(
                f'Published {num_matched} matched cones (X, Y, Z) out of '
                f'{len(cone_image_points)} projected LiDAR detections and {len(yolo_bboxes)} YOLO detections.'
            )
            
        except Exception as e:
            self.get_logger().error(f'Error in hungarian_callback: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

def main(args=None):
    rclpy.init(args=args)
    hungarian_association_node = YoloLidarFusion()
    try:
        rclpy.spin(hungarian_association_node)
    except KeyboardInterrupt:
        pass
    finally:
        hungarian_association_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()