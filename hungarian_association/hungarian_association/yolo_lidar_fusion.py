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

# --- Utility Functions (Unchanged from previous version) ---
def load_extrinsic_matrix(yaml_path: str) -> np.ndarray:
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

        # --- Configuration Loading (Robust version from previous answer) ---
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
        """Convert DetectionArray message to numpy array [cx, cy, w, h]."""
        # (Unchanged from previous robust version)
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


    # --- MODIFIED: Renamed and changed purpose ---
    def project_lidar_for_matching(self, cones_xyz_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects LiDAR points (XYZ) onto the image plane *only for matching purposes*.
        Filters points behind the camera *before* projection.

        Args:
            cones_xyz_all (np.ndarray): Array of shape (N, 3) containing ALL input LiDAR points.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
            - cone_image_points: (M, 2) array of projected 2D points for cones in front of camera.
            - original_indices_of_projected: (M,) array containing the *original* indices (from 0 to N-1)
                                             corresponding to the points in cone_image_points.
                                             Returns empty arrays if no points are projectable.
        """
        num_points = cones_xyz_all.shape[0]
        if num_points == 0:
            return np.empty((0, 2)), np.empty((0,), dtype=int)

        # Convert all points to homogeneous coordinates (N, 4)
        cones_xyz_h = np.hstack((cones_xyz_all, np.ones((num_points, 1), dtype=np.float32)))

        # Transform all points from LiDAR to camera coordinate system (N, 4)
        cones_cam_h = cones_xyz_h @ self.T_lidar_to_cam.T
        cones_cam_all = cones_cam_h[:, :3] # Extract non-homogeneous 3D points (N, 3)

        # --- Filtering *only for projection* ---
        # Find indices of points strictly in front of the camera
        valid_indices_for_projection = np.where(cones_cam_all[:, 2] > 1e-3)[0] # Get indices

        if len(valid_indices_for_projection) == 0:
             self.get_logger().debug("No LiDAR points were in front of the camera for projection.")
             return np.empty((0, 2)), np.empty((0,), dtype=int)

        # Select only the points in front of the camera for projection
        cones_cam_projectable = cones_cam_all[valid_indices_for_projection]
        # Keep track of the original indices of these projectable points
        original_indices_of_projected = valid_indices_for_projection # These are the original indices

        # Project only the valid points to the image plane
        try:
            cone_image_points, _ = cv2.projectPoints(
                cones_cam_projectable.astype(np.float64), # Use only projectable points (M, 3)
                np.zeros((3,1), dtype=np.float64), # rvec
                np.zeros((3,1), dtype=np.float64), # tvec
                self.camera_matrix.astype(np.float64),
                self.dist_coeffs.astype(np.float64)
            )
            cone_image_points = cone_image_points.reshape(-1, 2) # Result shape (M, 1, 2) -> (M, 2)
        except cv2.error as e:
             self.get_logger().error(f"cv2.projectPoints failed during projection for matching: {e}")
             return np.empty((0, 2)), np.empty((0,), dtype=int) # Return empty if projection fails

        self.get_logger().debug(f"Successfully projected {len(cone_image_points)} points for matching.")

        # Return 2D points and their corresponding ORIGINAL indices
        return cone_image_points, original_indices_of_projected


    def compute_cost_matrix(self, yolo_bboxes: np.ndarray, cone_image_points: np.ndarray) -> np.ndarray:
        """
        Computes the cost matrix based on Euclidean distance between
        YOLO box centers and *projected* cone points.
        (Unchanged from previous robust version)
        """
        num_boxes = yolo_bboxes.shape[0]
        num_cones = cone_image_points.shape[0]

        if num_boxes == 0 or num_cones == 0:
             # Return shape expected by caller, cost ensures no match passes threshold
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
        Processes synchronized LiDAR cones and YOLO detections.
        PUBLISHES ALL input LiDAR cones.
        Uses projection and Hungarian matching ONLY to assign YOLO class names
        to cones that are projectable and match; others get 'Unknown'.
        """
        try:
            self.get_logger().debug(f"Received synchronized messages. Cones timestamp: {cone_msg.header.stamp}, YOLO timestamp: {yolo_msg.header.stamp}")

            # 1. Extract ALL original LiDAR cone data (XYZ)
            cone_data = np.array(cone_msg.data, dtype=np.float32)
            num_points = 0
            cones_xyz_all = np.empty((0, 3), dtype=np.float32)

            if len(cone_msg.layout.dim) >= 2 and cone_msg.layout.dim[1].size == 3:
                num_points = cone_msg.layout.dim[0].size
                expected_size = num_points * 3
                if cone_data.size == expected_size and num_points > 0:
                    cones_xyz_all = cone_data.reshape(num_points, 3)
                elif num_points > 0: # Size mismatch
                     self.get_logger().error(
                        f"Cone data size ({cone_data.size}) mismatch with layout "
                        f"({num_points} cones * 3 values = {expected_size}). Skipping callback."
                     )
                     return # Skip processing if data is corrupt
                # If num_points is 0, cones_xyz_all remains empty, handled below
            else:
                 self.get_logger().error(
                    f"Input cone layout invalid or not XYZ. Got dim: {cone_msg.layout.dim}. Skipping callback."
                 )
                 return # Skip processing if layout is wrong

            # 2. Prepare output message structure - will contain ALL input cones
            filtered_msg = ModifiedFloat32MultiArray()
            filtered_msg.header = cone_msg.header # Preserve timestamp and frame_id
            # Layout reflects the total number of input cones
            filtered_msg.layout.dim.append(MultiArrayDimension(label="cones", size=num_points, stride=num_points * 3))
            filtered_msg.layout.dim.append(MultiArrayDimension(label="coords", size=3, stride=3)) # X, Y, Z
            filtered_msg.data = cone_data.tolist() # Directly copy all input data
            filtered_msg.class_names = [self.unmatched_class_name] * num_points # Initialize all as Unknown

            # 3. Handle case of no input cones
            if num_points == 0:
                self.get_logger().info("Received empty cone message. Publishing empty fused message.")
                self.coord_pub.publish(filtered_msg) # Publish the correctly structured empty message
                return

            # 4. Convert YOLO data
            yolo_bboxes = self.convert_yolo_msg_to_array(yolo_msg)
            num_yolo_boxes = yolo_bboxes.shape[0]

            # 5. Attempt Projection and Matching *only if* YOLO boxes exist
            match_dict_orig_idx: Dict[int, int] = {} # {original_cone_idx: yolo_box_idx}
            num_projected = 0
            num_actual_matches = 0

            if num_yolo_boxes > 0:
                # Project LiDAR points *only for matching*
                cone_image_points, original_indices_of_projected = self.project_lidar_for_matching(cones_xyz_all)
                num_projected = cone_image_points.shape[0]

                if num_projected > 0:
                    # Compute cost matrix using projected points
                    cost_matrix = self.compute_cost_matrix(yolo_bboxes, cone_image_points)

                    # Run Hungarian algorithm
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)

                    # Build the lookup dictionary mapping ORIGINAL cone index to YOLO index
                    for i, j in zip(row_ind, col_ind):
                        # i = yolo_box_index, j = projected_cone_index
                        if cost_matrix[i, j] < self.max_matching_distance:
                            # Find the original index corresponding to this projected point
                            original_idx = original_indices_of_projected[j]
                            # Store the match using the original index as the key
                            match_dict_orig_idx[original_idx] = i
                            num_actual_matches += 1
                    self.get_logger().debug(f'Matching done: {num_projected} points projectable, {num_actual_matches} matched within threshold.')
                else:
                    self.get_logger().debug('No LiDAR points were projectable for matching.')
            else:
                self.get_logger().debug('No YOLO boxes received, skipping matching.')


            # 6. Assign class names based on matching results
            # Iterate through ALL original cones (0 to num_points-1)
            for k in range(num_points):
                if k in match_dict_orig_idx:
                    # This cone was matched, get the YOLO class name
                    yolo_idx = match_dict_orig_idx[k]
                    # Safety check for yolo_idx validity (should be rare)
                    if 0 <= yolo_idx < len(yolo_msg.detections):
                         filtered_msg.class_names[k] = yolo_msg.detections[yolo_idx].class_name
                    else:
                         self.get_logger().warn(f"Matched YOLO index {yolo_idx} out of bounds for {len(yolo_msg.detections)} detections. Keeping cone {k} as Unknown.")
                # Else: it remains 'Unknown' as initialized

            # 7. Publish the message containing ALL cones with updated class names
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


# --- Main execution (Unchanged from previous robust version) ---
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