import os
import rclpy
from rclpy.node import Node
from typing import Tuple, Dict, List, Union, Any
import cv2
import numpy as np
import yaml
import threading
from queue import Queue, Empty
import time
from numba import jit, prange

from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor, ParameterType, IntegerRange
from custom_interface.msg import ModifiedFloat32MultiArray
from yolo_msgs.msg import DetectionArray

# 새로운 다중 카메라 설정 파서 가져오기
from ros2_camera_lidar_fusion.read_multi_yaml import (
    extract_multi_configuration,
    load_extrinsic_matrices,
    load_camera_calibrations
)

# --- project_boxes_cones_points.py 에서 가져온 최적화된 함수들 --- #
@jit(nopython=True, parallel=True, cache=True)
def transform_points_to_homogeneous(points: np.ndarray) -> np.ndarray:
    n_points = points.shape[0]
    points_h = np.zeros((n_points, 4), dtype=np.float64)
    for i in prange(n_points):
        points_h[i, 0] = points[i, 0]
        points_h[i, 1] = points[i, 1]
        points_h[i, 2] = points[i, 2]
        points_h[i, 3] = 1.0
    return points_h

@jit(nopython=True, parallel=True, cache=True)
def batch_matrix_multiply(points_h: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    n_points = points_h.shape[0]
    result = np.zeros((n_points, 4), dtype=np.float64)
    for i in prange(n_points):
        for j in range(4):
            for k in range(4):
                result[i, j] += points_h[i, k] * transform_matrix[k, j]
    return result

@jit(nopython=True, parallel=True, cache=True)
def filter_points_in_front(points_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_points = points_cam.shape[0]
    mask = np.zeros(n_points, dtype=np.bool_)
    for i in prange(n_points):
        mask[i] = points_cam[i, 2] > 0.0
    count = 0
    for i in range(n_points):
        if mask[i]:
            count += 1
    valid_points = np.zeros((count, 3), dtype=np.float64)
    valid_indices = np.zeros(count, dtype=np.int32)
    idx = 0
    for i in range(n_points):
        if mask[i]:
            valid_points[idx, 0] = points_cam[i, 0]
            valid_points[idx, 1] = points_cam[i, 1]
            valid_points[idx, 2] = points_cam[i, 2]
            valid_indices[idx] = i
            idx += 1
    return valid_points, valid_indices

@jit(nopython=True, parallel=True, cache=True)
def project_points_to_image_fast(points_3d: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    n_points = points_3d.shape[0]
    image_points = np.zeros((n_points, 2), dtype=np.float64)
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # 매우 작은 Z값을 피하기 위한 임계값 (epsilon)
    epsilon = 1e-9 # 더 작은 값 또는 상황에 맞게 조절

    for i in prange(n_points):
        if points_3d[i, 2] > epsilon: # 기존: points_3d[i, 2] > 1e-6
            x = points_3d[i, 0] / points_3d[i, 2]
            y = points_3d[i, 1] / points_3d[i, 2]
            image_points[i, 0] = fx * x + cx
            image_points[i, 1] = fy * y + cy
        else:
            # 유효하지 않은 투영 결과에 대해 NaN 할당 (NumPy에서 처리 용이)
            image_points[i, 0] = np.nan
            image_points[i, 1] = np.nan
    return image_points

def pointcloud2_to_xyz_array_fast(cloud_msg: PointCloud2, skip_rate: int = 1) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if cloud_msg.height == 0 or cloud_msg.width == 0:
        return np.zeros((0, 3), dtype=np.float32), None, None

    field_map = {field.name: field for field in cloud_msg.fields}
    x_field = field_map.get('x') or field_map.get('point_x')
    y_field = field_map.get('y') or field_map.get('point_y')
    z_field = field_map.get('z') or field_map.get('point_z')
    intensity_field = field_map.get('intensity')
    range_field = field_map.get('range')

    if not all([x_field, y_field, z_field]):
        print(f"Pointcloud parsing error: Missing required XYZ fields.")
        return np.zeros((0, 3), dtype=np.float32), None, None

    parsed_fields_info = []
    if x_field: parsed_fields_info.append((x_field.name, np.float32, x_field.offset))
    if y_field: parsed_fields_info.append((y_field.name, np.float32, y_field.offset))
    if z_field: parsed_fields_info.append((z_field.name, np.float32, z_field.offset))
    
    has_intensity = False
    if intensity_field and intensity_field.datatype == 7: # float32
        parsed_fields_info.append((intensity_field.name, np.float32, intensity_field.offset))
        has_intensity = True
    elif intensity_field:
        print(f"Warning: Intensity field '{intensity_field.name}' found but has unhandled datatype {intensity_field.datatype}. Expected 7 (float32).")
        intensity_field = None 
        
    has_range = False
    if range_field and range_field.datatype == 6: # uint32
        parsed_fields_info.append((range_field.name, np.uint32, range_field.offset))
        has_range = True
    elif range_field:
        print(f"Warning: Range field '{range_field.name}' found but has unhandled datatype {range_field.datatype}. Expected 6 (uint32).")
        range_field = None 

    parsed_fields_info.sort(key=lambda f: f[2])
    
    dtype_build_list = []
    last_offset = 0
    for name, dtype_val, offset in parsed_fields_info:
        padding_bytes = offset - last_offset
        if padding_bytes < 0:
            print(f"Pointcloud parsing error: Negative padding bytes for field {name}.")
            return np.zeros((0,3), dtype=np.float32), None, None
        if padding_bytes > 0:
            dtype_build_list.append( ('', f'V{padding_bytes}'))
        dtype_build_list.append((name, dtype_val))
        last_offset = offset + np.dtype(dtype_val).itemsize
        
    remaining_bytes = cloud_msg.point_step - last_offset
    if remaining_bytes < 0:
        print(f"Pointcloud parsing error: Negative remaining bytes. Point_step {cloud_msg.point_step} is too small.")
        return np.zeros((0,3), dtype=np.float32), None, None
    if remaining_bytes > 0:
        dtype_build_list.append(('', f'V{remaining_bytes}'))

    try:
        final_dtype = np.dtype(dtype_build_list)
    except Exception as e:
        print(f"Pointcloud parsing error: Failed to create numpy dtype: {e}.")
        return np.zeros((0,3), dtype=np.float32), None, None
        
    raw_data = np.frombuffer(cloud_msg.data, dtype=final_dtype)

    if skip_rate > 1:
        raw_data = raw_data[::skip_rate]

    points_xyz = np.zeros((raw_data.shape[0], 3), dtype=np.float32)
    points_xyz[:, 0] = raw_data[x_field.name]
    points_xyz[:, 1] = raw_data[y_field.name]
    points_xyz[:, 2] = raw_data[z_field.name]

    intensity_values = None
    if has_intensity and intensity_field.name in raw_data.dtype.names:
        intensity_values = raw_data[intensity_field.name].astype(np.float32)

    range_values = None
    if has_range and range_field.name in raw_data.dtype.names:
        range_values = raw_data[range_field.name].astype(np.float32) / 1000.0 # mm to m

    return points_xyz, intensity_values, range_values
# --- 가져온 함수들 끝 --- #

class CameraProcessor:
    def __init__(self, 
                 name: str,
                 node_logger: rclpy.impl.rcutils_logger.RcutilsLogger,
                 camera_matrix: np.ndarray, 
                 dist_coeffs: np.ndarray, 
                 T_lidar_to_cam: np.ndarray, 
                 T_sensor_to_lidar: np.ndarray,
                 bridge: CvBridge,
                 create_publisher_func: callable,
                 input_topic: str,
                 output_topic: str,
                 colorization_mode: str,
                 min_value_display: float,
                 max_value_display: float,
                 lidar_frame_id: str,
                 raw_lidar_frame_id: str,
                 processed_lidar_frame_id: str
                 ):
        self.name = name
        self.logger = node_logger
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.T_lidar_to_cam = T_lidar_to_cam
        self.T_sensor_to_lidar = T_sensor_to_lidar
        self.T_sensor_to_cam = self.T_lidar_to_cam @ self.T_sensor_to_lidar
        self.bridge = bridge
        self.create_publisher = create_publisher_func

        self.colorization_mode = colorization_mode
        self.min_value_display = min_value_display
        self.max_value_display = max_value_display
        self.lidar_frame_id = lidar_frame_id
        self.raw_lidar_frame_id = raw_lidar_frame_id
        self.processed_lidar_frame_id = processed_lidar_frame_id
        
        self.color_mapping = {
            "red cone": (0, 0, 255),   # 빨강
            "yellow cone":  (0, 255, 255), # 노랑
            "blue cone":    (255, 0, 0),   # 파랑
            "Unknown":      (0, 255, 0)    # 초록 (기본)
        }
        
        self.pub_image = self.create_publisher(Image, output_topic, 10)
        
        self.image_queue = Queue(maxsize=5)
        self.latest_image_msg_lock = threading.Lock()
        self.latest_image_msg = None
        
        self.sensor_data_lock = threading.Lock()
        self.current_lidar_points = None
        self.current_lidar_intensity = None
        self.current_lidar_range = None
        self.current_cones_msg = None
        self.current_boxes_msg = None
        self.last_processed_sensor_timestamp = None

        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        self.logger.info(f"CameraProcessor '{self.name}' initialized for image topic '{input_topic}'. Publishing to '{output_topic}'.")

    def update_ros_params(self, color_mode, min_val, max_val):
        self.colorization_mode = color_mode
        self.min_value_display = min_val
        self.max_value_display = max_val
        self.logger.info(f"Processor '{self.name}' ROS params updated: Mode={color_mode}, Min={min_val}, Max={max_val}")

    def _get_rainbow_colors_array(self, values: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        if values is None or values.size == 0:
            return np.array([], dtype=np.uint8).reshape(0, 3)
        if min_val >= max_val:
            return np.full((values.shape[0], 3), (128, 128, 128), dtype=np.uint8)
        normalized_values = np.clip((values - min_val) / (max_val - min_val), 0.0, 1.0)
        scaled_values = (normalized_values * 255).astype(np.uint8)
        colormap_input = scaled_values.reshape(-1, 1)
        colored_bgr_array = cv2.applyColorMap(colormap_input, cv2.COLORMAP_JET)
        return colored_bgr_array.reshape(-1, 3)

    def image_callback(self, image_msg: Image):
        with self.latest_image_msg_lock:
            self.latest_image_msg = image_msg

    def update_sensor_data(self, lidar_points, lidar_intensity, lidar_range, cones_msg, boxes_msg, sensor_timestamp):
        with self.sensor_data_lock:
            self.current_lidar_points = lidar_points
            self.current_lidar_intensity = lidar_intensity
            self.current_lidar_range = lidar_range
            self.current_cones_msg = cones_msg
            self.current_boxes_msg = boxes_msg
            self.last_processed_sensor_timestamp = sensor_timestamp

    def _processing_loop(self):
        while rclpy.ok():
            img_to_process = None
            lidar_pts, lidar_int, lidar_rng, cones, boxes, timestamp_to_process = None, None, None, None, None, None

            with self.latest_image_msg_lock:
                if self.latest_image_msg is not None:
                    img_to_process = self.latest_image_msg
            
            with self.sensor_data_lock:
                if self.current_lidar_points is not None and self.last_processed_sensor_timestamp is not None:
                    lidar_pts = self.current_lidar_points
                    lidar_int = self.current_lidar_intensity
                    lidar_rng = self.current_lidar_range
                    cones = self.current_cones_msg
                    boxes = self.current_boxes_msg
                    timestamp_to_process = self.last_processed_sensor_timestamp
            
            if img_to_process and lidar_pts is not None and timestamp_to_process is not None:
                self.process_frame(img_to_process, lidar_pts, lidar_int, lidar_rng, cones, boxes)
            
            time.sleep(0.01)

    def process_frame(self, image_msg: Image, 
                      input_cloud_xyz: np.ndarray | None,
                      input_cloud_intensity: np.ndarray | None,
                      input_cloud_range: np.ndarray | None,
                      cones_msg: ModifiedFloat32MultiArray | None, 
                      boxes_msg: DetectionArray | None):
        try:
            start_process_time = time.perf_counter()
            if not image_msg or not image_msg.data:
                self.logger.warn(f"[{self.name}] 빈 이미지 메시지, 처리 생략")
                return

            try:
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            except Exception as e:
                self.logger.error(f"[{self.name}] 이미지 변환 오류: {e}")
                return

            h_img, w_img = cv_image.shape[:2]

            if input_cloud_xyz is not None and input_cloud_xyz.shape[0] > 0:
                T_cloud_to_cam_selected = None
                use_intensity_range_color = False
                default_point_color = (0, 255, 0)

                if self.lidar_frame_id == self.raw_lidar_frame_id:
                    self.logger.debug(f"[{self.name}] Processing as RAW LiDAR ({self.lidar_frame_id})")
                    T_cloud_to_cam_selected = self.T_lidar_to_cam
                    use_intensity_range_color = True
                elif self.lidar_frame_id == self.processed_lidar_frame_id:
                    self.logger.debug(f"[{self.name}] Processing as PROCESSED cloud ({self.lidar_frame_id}) from os_sensor")
                    T_cloud_to_cam_selected = self.T_sensor_to_cam
                    use_intensity_range_color = False
                    default_point_color = (0, 255, 0)
                else:
                    self.logger.warn(f"[{self.name}] Unknown lidar_frame_id: '{self.lidar_frame_id}'. Defaulting to raw LiDAR pipeline (T_lidar_to_cam). Check 'lidar_frame_id' in config.")
                    T_cloud_to_cam_selected = self.T_lidar_to_cam
                    use_intensity_range_color = True

                input_cloud_xyz_f64 = input_cloud_xyz.astype(np.float64)
                input_cloud_h = transform_points_to_homogeneous(input_cloud_xyz_f64)
                
                cloud_cam_h = batch_matrix_multiply(input_cloud_h, T_cloud_to_cam_selected.T)
                cloud_cam_all = cloud_cam_h[:, :3]
                
                cloud_cam_front, valid_indices_cloud = filter_points_in_front(cloud_cam_all)
                
                if cloud_cam_front.shape[0] > 0:
                    if np.any(self.dist_coeffs != 0):
                        rvec = np.zeros((3,1), dtype=np.float64)
                        tvec = np.zeros((3,1), dtype=np.float64)
                        cloud_image_points, _ = cv2.projectPoints(cloud_cam_front, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                        cloud_image_points = cloud_image_points.reshape(-1, 2)
                    else:
                        cloud_image_points = project_points_to_image_fast(cloud_cam_front, self.camera_matrix)

                    # NaN/inf 값 필터링 및 유효한 인덱스 가져오기
                    valid_projection_mask = ~np.isnan(cloud_image_points).any(axis=1) & ~np.isinf(cloud_image_points).any(axis=1)
                    cloud_image_points_filtered = cloud_image_points[valid_projection_mask]
                    
                    # point_colors_bgr도 필터링된 인덱스에 맞춰줘야 함
                    # 먼저 전체 크기로 point_colors_bgr를 계산한 후, valid_projection_mask로 필터링

                    point_colors_bgr_full = None # 필터링 전 전체 포인트에 대한 색상 배열
                    if use_intensity_range_color:
                        data_to_colorize = None
                        # valid_indices_cloud는 filter_points_in_front 에서 나온 인덱스
                        # input_cloud_xyz, input_cloud_intensity 등은 필터링 전의 원본 크기
                        if self.colorization_mode == 'intensity' and input_cloud_intensity is not None and input_cloud_intensity.shape[0] == input_cloud_xyz.shape[0]:
                            # cloud_cam_front에 해당하는 intensity 값을 가져와야 함 -> valid_indices_cloud 사용
                            data_to_colorize = input_cloud_intensity[valid_indices_cloud]
                        elif self.colorization_mode == 'range' and input_cloud_range is not None and input_cloud_range.shape[0] == input_cloud_xyz.shape[0]:
                            data_to_colorize = input_cloud_range[valid_indices_cloud]

                        if data_to_colorize is not None and data_to_colorize.size > 0:
                            # data_to_colorize는 cloud_cam_front와 동일한 길이
                            point_colors_bgr_full = self._get_rainbow_colors_array(data_to_colorize, self.min_value_display, self.max_value_display)
                        else:
                            point_colors_bgr_full = np.full((cloud_cam_front.shape[0], 3), default_point_color, dtype=np.uint8)
                    else:
                        point_colors_bgr_full = np.full((cloud_cam_front.shape[0], 3), default_point_color, dtype=np.uint8)
                    
                    # 유효한 프로젝션에 대한 색상만 선택
                    point_colors_bgr = point_colors_bgr_full[valid_projection_mask]

                    # 필터링된 이미지 좌표 사용
                    if cloud_image_points_filtered.shape[0] == 0:
                        # self.logger.debug(f"[{self.name}] No valid projected points after NaN/inf filtering.")
                        pass # 다음 로직으로 (콘 처리 등)
                    else:
                        u_centers = np.round(cloud_image_points_filtered[:, 0]).astype(np.int32)
                        v_centers = np.round(cloud_image_points_filtered[:, 1]).astype(np.int32)
                        
                        dv_offsets, du_offsets = np.mgrid[-1:2, -1:2] 
                        dv_flat = dv_offsets.flatten()
                        du_flat = du_offsets.flatten()

                        for dv, du in zip(dv_flat, du_flat):
                            current_v = v_centers + dv
                            current_u = u_centers + du
                            mask = (current_v >= 0) & (current_v < h_img) & (current_u >= 0) & (current_u < w_img)
                            if np.any(mask):
                                # cv_image에 색칠할 때도 필터링된 색상 배열 사용
                                cv_image[current_v[mask], current_u[mask]] = point_colors_bgr[mask]
            
            if cones_msg is not None and cones_msg.data:
                cone_data_arr = np.array(cones_msg.data, dtype=np.float32)
                if cone_data_arr.size > 0 and cone_data_arr.size % 3 == 0:
                    num_cones = cone_data_arr.size // 3
                    cones_xyz_sensor = cone_data_arr.reshape(num_cones, 3)
                    self.logger.debug(f"[{self.name}] Received {num_cones} cones. First 3 sensor coords: {cones_xyz_sensor[:3]}")

                    cones_xyz_sensor_h = transform_points_to_homogeneous(cones_xyz_sensor.astype(np.float64))
                    cones_cam_h = batch_matrix_multiply(cones_xyz_sensor_h, self.T_sensor_to_cam.T)
                    self.logger.debug(f"[{self.name}] Cones after T_sensor_to_cam (first 3 homogeneous): {cones_cam_h[:3]}")
                    
                    cones_cam_all = cones_cam_h[:, :3]
                    self.logger.debug(f"[{self.name}] Cones in CAM frame before filtering (first 3): {cones_cam_all[:3]}")
                    
                    cones_cam_front, valid_indices_cones = filter_points_in_front(cones_cam_all)
                    self.logger.debug(f"[{self.name}] Num cones before T_sensor_to_cam: {cones_xyz_sensor.shape[0]}")
                    self.logger.debug(f"[{self.name}] Num cones in CAM frame (all): {cones_cam_all.shape[0]}")
                    self.logger.debug(f"[{self.name}] Num cones in CAM frame (front): {cones_cam_front.shape[0]}, Valid indices count: {len(valid_indices_cones)}")
                    self.logger.debug(f"[{self.name}] Cones in CAM frame AFTER filtering (first 3): {cones_cam_front[:3]}")
                    
                    if cones_cam_front.shape[0] > 0:
                        if np.any(self.dist_coeffs != 0):
                            rvec_cone = np.zeros((3,1), dtype=np.float64)
                            tvec_cone = np.zeros((3,1), dtype=np.float64)
                            cone_image_points, _ = cv2.projectPoints(cones_cam_front, rvec_cone, tvec_cone, self.camera_matrix, self.dist_coeffs)
                            cone_image_points = cone_image_points.reshape(-1, 2)
                        else:
                            cone_image_points = project_points_to_image_fast(cones_cam_front, self.camera_matrix)
                        
                        self.logger.debug(f"[{self.name}] Projected cone image points (first 3): {cone_image_points[:3]}")
                        self.logger.debug(f"[{self.name}] Image dimensions (H, W): ({h_img}, {w_img})")
                        
                        has_class_names = hasattr(cones_msg, 'class_names') and len(cones_msg.class_names) == num_cones
                        if has_class_names:
                            original_class_names = cones_msg.class_names
                            filtered_class_names = [original_class_names[i] for i in valid_indices_cones]
                        else:
                            filtered_class_names = ["Unknown"] * cones_cam_front.shape[0]
                        self.logger.debug(f"[{self.name}] Number of filtered class names: {len(filtered_class_names)}")

                        for i, (u, v) in enumerate(cone_image_points):
                            u_int, v_int = int(round(u)), int(round(v))
                            if 0 <= u_int < w_img and 0 <= v_int < h_img:
                                class_name = filtered_class_names[i] if i < len(filtered_class_names) else "Unknown"
                                color = self.color_mapping.get(class_name, (0, 255, 0))
                                cv2.circle(cv_image, (u_int, v_int), 5, color, -1)
                                cv2.circle(cv_image, (u_int, v_int), 7, (255, 255, 255), 1)
                                
                                if i < cones_cam_front.shape[0]:
                                    depth_z = cones_cam_front[i, 2]
                                    depth_text = f"{depth_z:.2f}m"
                                    cv2.putText(cv_image, depth_text, (u_int + 10, v_int + 5), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            out_msg.header = image_msg.header
            self.pub_image.publish(out_msg)

        except Exception as e:
            self.logger.error(f"[{self.name}] error in process_frame: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

class DualCameraFusionNode(Node):
    def __init__(self):
        super().__init__('dual_camera_fusion_node')
        self.get_logger().info("DualCameraFusionNode initializing...")
        
        self.config_file = extract_multi_configuration()
        if self.config_file is None:
            self.get_logger().fatal("Failed to extract multi_configuration.yaml. Shutting down.")
            rclpy.shutdown()
            return

        self.declare_parameter('colorization_mode', 'intensity',
                                ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('min_value_display', 0.0,
                                ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('max_value_display', 100.0,
                                ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('skip_rate', 1, 
                                ParameterDescriptor(description='Rate to skip lidar points for processing', 
                                                    type=ParameterType.PARAMETER_INTEGER, 
                                                    integer_range=[IntegerRange(from_value=1, to_value=10, step=1)]))
        
        self.colorization_mode = self.get_parameter('colorization_mode').value
        self.min_value_display = self.get_parameter('min_value_display').value
        self.max_value_display = self.get_parameter('max_value_display').value
        self.skip_rate = self.get_parameter('skip_rate').value
        self.add_on_set_parameters_callback(self.parameters_callback)

        config_folder = self.config_file['general']['config_folder']
        extrinsic_yaml_path = os.path.join(config_folder, self.config_file['general']['camera_extrinsic_calibration'])
        self.T_lidar_to_cam_dict = load_extrinsic_matrices(extrinsic_yaml_path)
        intrinsic_yaml_path = os.path.join(config_folder, self.config_file['general']['camera_intrinsic_calibration'])
        self.camera_calibrations = load_camera_calibrations(intrinsic_yaml_path)

        self.T_sensor_to_lidar_static = np.array([
            [-1,  0,  0,  0       ],
            [ 0, -1,  0,  0       ],
            [ 0,  0,  1, -0.038195],
            [ 0,  0,  0,  1       ]
        ], dtype=np.float64)

        self.bridge = CvBridge()
        self.camera_processors: Dict[str, CameraProcessor] = {}

        self.sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2
        )

        # 설정 파일에서 LiDAR 정보 읽기
        lidar_config = self.config_file.get('lidar', {}) # lidar 섹션 자체가 없을 경우 대비
        self.lidar_input_topic = lidar_config.get('lidar_topic', '/ouster/points') # 기본값
        # YAML 파일의 키 'frame_id'를 사용하도록 수정
        self.current_lidar_frame_id = lidar_config.get('frame_id', 'os_lidar') 
        self.get_logger().info(f"Using LiDAR topic: '{self.lidar_input_topic}' with frame_id: '{self.current_lidar_frame_id}'")

        # 비교를 위한 표준 프레임 ID들 (YAML 키 이름 확인 필요)
        # YAML에 'raw_lidar_comparison_frame_id' 와 'processed_lidar_comparison_frame_id' 가 정의되어 있다고 가정
        self.RAW_LIDAR_FRAME = lidar_config.get('raw_lidar_comparison_frame_id', 'os_lidar') 
        self.PROCESSED_LIDAR_FRAME = lidar_config.get('processed_lidar_comparison_frame_id', 'os_sensor')
        self.get_logger().info(f"Comparison frames: RAW_LIDAR_FRAME='{self.RAW_LIDAR_FRAME}', PROCESSED_LIDAR_FRAME='{self.PROCESSED_LIDAR_FRAME}'")

        for cam_id, cam_config in self.config_file.get('cameras', {}).items():
            if cam_id in self.T_lidar_to_cam_dict and cam_id in self.camera_calibrations:
                cam_matrix, dist_coeffs = self.camera_calibrations[cam_id]
                t_lidar_to_cam = self.T_lidar_to_cam_dict[cam_id]
                
                processor = CameraProcessor(
                    name=cam_config.get('name', cam_id),
                    node_logger=self.get_logger(),
                    camera_matrix=cam_matrix,
                    dist_coeffs=dist_coeffs,
                    T_lidar_to_cam=t_lidar_to_cam,
                    T_sensor_to_lidar=self.T_sensor_to_lidar_static,
                    bridge=self.bridge,
                    create_publisher_func=self.create_publisher,
                    input_topic=cam_config['image_topic'],
                    output_topic=cam_config['projected_topic'],
                    colorization_mode=self.colorization_mode,
                    min_value_display=self.min_value_display,
                    max_value_display=self.max_value_display,
                    lidar_frame_id=self.current_lidar_frame_id,
                    raw_lidar_frame_id=self.RAW_LIDAR_FRAME,
                    processed_lidar_frame_id=self.PROCESSED_LIDAR_FRAME
                )
                self.camera_processors[cam_id] = processor
                
                self.create_subscription(
                    Image, 
                    cam_config['image_topic'], 
                    processor.image_callback,
                    qos_profile=self.image_qos
                )
                self.get_logger().info(f"Initialized and subscribed for {cam_id}")
            else:
                self.get_logger().warn(f"Skipping {cam_id}: Missing calibration or extrinsic data.")

        self.lidar_sub = self.create_subscription(
            PointCloud2, 
            self.lidar_input_topic,
            self.lidar_callback, 
            qos_profile=self.sensor_qos
        )
        self.cones_sub = self.create_subscription(
            ModifiedFloat32MultiArray, 
            "/sorted_cones_time",
            self.cones_callback, 
            qos_profile=self.sensor_qos
        )

        self.shared_data_lock = threading.Lock()
        self.latest_lidar_points = None
        self.latest_lidar_intensity = None
        self.latest_lidar_range = None
        self.latest_cones_msg = None
        self.latest_sensor_timestamp = None
        
        self.get_logger().info("DualCameraFusionNode initialized successfully.")

    def parameters_callback(self, params: List[rclpy.parameter.Parameter]):
        success = True
        param_changed_for_processors = False
        for param in params:
            if param.name == 'colorization_mode':
                self.colorization_mode = param.value
                param_changed_for_processors = True
            elif param.name == 'min_value_display':
                self.min_value_display = param.value
                param_changed_for_processors = True
            elif param.name == 'max_value_display':
                self.max_value_display = param.value
                param_changed_for_processors = True
            elif param.name == 'skip_rate':
                self.skip_rate = param.value
            else:
                pass
        
        if param_changed_for_processors:
            for proc in self.camera_processors.values():
                proc.update_ros_params(self.colorization_mode, self.min_value_display, self.max_value_display)
            self.get_logger().info("Node colorization parameters updated and propagated to processors.")
        
        return SetParametersResult(successful=success)

    def detection_callback(self, msg: DetectionArray, processor: CameraProcessor):
        current_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        processor.update_sensor_data(None, None, None, None, msg, current_timestamp)

    def lidar_callback(self, msg: PointCloud2):
        points, intensity, range_data = pointcloud2_to_xyz_array_fast(msg, skip_rate=self.skip_rate)
        current_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.shared_data_lock:
            self.latest_lidar_points = points
            self.latest_lidar_intensity = intensity
            self.latest_lidar_range = range_data
            if self.latest_sensor_timestamp is None or current_timestamp > self.latest_sensor_timestamp:
                self.latest_sensor_timestamp = current_timestamp
        self._distribute_sensor_data()

    def cones_callback(self, msg: ModifiedFloat32MultiArray):
        current_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.shared_data_lock:
            self.latest_cones_msg = msg
            if self.latest_sensor_timestamp is None or current_timestamp > self.latest_sensor_timestamp:
                self.latest_sensor_timestamp = current_timestamp
        self._distribute_sensor_data()

    def _distribute_sensor_data(self):
        lidar_pts, lidar_int, lidar_rng, cones_data, main_timestamp = None, None, None, None, None
        
        with self.shared_data_lock:
            if self.latest_sensor_timestamp is None:
                return 

            main_timestamp = self.latest_sensor_timestamp
            lidar_pts = self.latest_lidar_points
            lidar_int = self.latest_lidar_intensity
            lidar_rng = self.latest_lidar_range
            cones_data = self.latest_cones_msg
        
        for cam_id, processor in self.camera_processors.items():
            processor.update_sensor_data(lidar_pts, lidar_int, lidar_rng, 
                                         cones_data, processor.current_boxes_msg, 
                                         main_timestamp)

def main(args=None):
    rclpy.init(args=args)
    node = DualCameraFusionNode()
    if not rclpy.ok():
        return

    num_threads = len(node.camera_processors) * 2 + 4
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=num_threads)
    executor.add_node(node)
    
    node.get_logger().info(f"DualCameraFusionNode starting with {num_threads} executor threads.")
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down DualCameraFusionNode by user interrupt.")
    except Exception as e:
        node.get_logger().fatal(f"Unhandled exception in executor: {str(e)}")
        import traceback
        node.get_logger().error(traceback.format_exc())
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None and rclpy.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("DualCameraFusionNode has been shut down.")

if __name__ == "__main__":
    main() 