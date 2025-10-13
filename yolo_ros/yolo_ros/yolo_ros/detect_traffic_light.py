#! /usr/bin/env python
# -*- coding:utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
import json
import os
from types import SimpleNamespace
from rcl_interfaces.msg import ParameterDescriptor

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class TrafficLightDetector(Node):
    """
    YOLO 또는 수동으로 정의된 ROI를 기반으로 신호등 색상을 감지하는 통합 노드.
    HSV 색상 범위를 트랙바로 실시간 조절하고, GUI 창 표시 여부를 파라미터로 제어하는 기능이 통합되었습니다.
    """
    GUI_HEIGHT = 6
    GREEN_TOP_MAX_RATIO = 0.05  # 최대 허용 상단 점등 비율 (전체 영역 대비)
    GREEN_BOTTOM_MIN_RATIO = 0.4  # 최소 요구 하단 점등 비율 (전체 영역 대비)
    CONSENSUS_REQUIRED_FRAMES = 2  # YOLO와 규칙 기반 결과 일치 시 필요한 연속 프레임 수
    CONFLICT_REQUIRED_FRAMES = 10  # 결과가 불일치할 때 규칙 기반 결과가 유지되어야 하는 연속 프레임 수
    DEFAULT_BINARY_THRESHOLD = 140  # 이진화 기본 임계값
    MIN_CLUSTER_WIDTH_PIXELS = 40   # 유효한 클러스터로 인정하기 위한 최소 너비
    MIN_CLUSTER_HEIGHT_PIXELS = 40  # 유효한 클러스터로 인정하기 위한 최소 높이
    MIN_CLUSTER_CIRCULARITY = 0.75  # contour area 대비 최소 원형 유사도 비율
    MIN_RULE_HEIGHT = 10  # 규칙 기반 판별을 수행하기 위한 최소 ROI 높이
    MIN_RULE_WIDTH = 6   # 규칙 기반 판별을 수행하기 위한 최소 ROI 너비
    MIN_RULE_AREA = 160  # 규칙 기반 판별을 수행하기 위한 최소 ROI 면적

    def __init__(self):
        super().__init__('traffic_light_detector')
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.script_dir = script_dir
        self.ROI_FILE = os.path.join(script_dir, "rois.json")

        # --- 파라미터 선언 ---
        self.declare_parameter('roi_mode', 'manual', ParameterDescriptor(description='ROI 감지 모드: \'yolo\' 또는 \'manual\''))
        self.declare_parameter('pixel_threshold', 200, ParameterDescriptor(description='(Manual 모드용) 색상 검출을 위한 최소 픽셀 수 임계값'))
        self.declare_parameter('show_camera_windows', True, ParameterDescriptor(description='메인 카메라 창 표시 여부'))
        self.declare_parameter('show_control_windows', True, ParameterDescriptor(description='색상 제어 창 표시 여부'))
        self.declare_parameter('show_mask_windows', True, ParameterDescriptor(description='마스크 시각화 창 표시 여부'))
        default_model_path = os.path.join(self.script_dir, "models", "yolov10n_lightonly_251002.pt")
        self.declare_parameter('yolo_model_path', default_model_path, ParameterDescriptor(description='/models/yolov10n_lightonly_251002.pt YOLO 모델 파일 경로'))
        self.declare_parameter('yolo_confidence_threshold', 0.5, ParameterDescriptor(description='YOLO 탐지를 채택할 최소 신뢰도 (0.0~1.0)'))
        self.declare_parameter('debug_mode', True, ParameterDescriptor(description='초록불 트리거를 비활성화하고 디버그 시각화를 활성화합니다.'))
        
        # --- 파라미터 값 읽어오기 ---
        self.roi_mode = self.get_parameter('roi_mode').get_parameter_value().string_value
        self.threshold_pixels = self.get_parameter('pixel_threshold').get_parameter_value().integer_value
        self.show_camera_windows = self.get_parameter('show_camera_windows').get_parameter_value().bool_value
        self.show_control_windows = self.get_parameter('show_control_windows').get_parameter_value().bool_value
        self.show_mask_windows = self.get_parameter('show_mask_windows').get_parameter_value().bool_value
        self.configured_yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        yolo_conf_param = self.get_parameter('yolo_confidence_threshold').get_parameter_value().double_value
        self.yolo_confidence_threshold = max(0.0, min(yolo_conf_param, 1.0))
        self.debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value

        if self.roi_mode == 'yolo':
            # YOLO 모드에서는 카메라 창만 사용하므로 보조 창을 강제로 비활성화한다.
            self.show_control_windows = False
            self.show_mask_windows = False
        elif self.roi_mode == 'manual':
            # Manual 모드 역시 규칙 기반 탐지를 사용하므로 추가 제어 창이 필요 없다.
            self.show_control_windows = False
            self.show_mask_windows = False

        self.bridge = CvBridge()
        self.yolo_model = None
        self.resolved_yolo_model_path = None
        self.yolo_target_classes = {'green light', 'red light', 'unknown light'}
        self.green_client = self.create_client(Trigger, '/green')
        self.virtual_green_service = self.create_service(Trigger, '/virtual_green', self.virtual_green_callback)
        
        self.mission_triggered = False; self.is_calling_green_service = False
        self.retry_green_service_timer = None
        self.gui_is_first_print = True; self.gui_state = "Waiting..."
        self.gui_service_status = "Idle"; self.gui_last_error = "None"
        self.gui_confidence = "N/A"
        self.last_final_color = "Unknown"
        self.consensus_candidate = None; self.consensus_count = 0
        self.conflict_candidate = None; self.conflict_count = 0
        self.last_primary_color = "Unknown"; self.last_secondary_color = "Unknown"
        self.last_yolo_confidence = None
        self.last_rule_blob_summary = "RuleBlob:N/A"
        self.last_decision_source = "None"
        self.debug_window_names = set()
        self.binary_trackbar_initialized = False
        self.rule_binary_threshold = self.DEFAULT_BINARY_THRESHOLD
        self.rule_ema_alpha = 0.2
        self.rule_red_ema = 0.0
        self.rule_green_ema = 0.0
        
        if self.roi_mode == 'yolo': self.init_yolo_mode()
        elif self.roi_mode == 'manual': self.init_manual_mode()
        else:
            self.get_logger().fatal(f"Invalid roi_mode: '{self.roi_mode}'. Shutting down.")
            self.create_timer(0.1, self.destroy_node); return

        self.get_logger().info(f'TrafficLightDetector node started in \'{self.roi_mode}\' mode.')

    def init_yolo_mode(self):
        self.gui_detection_status = f"YOLO-Based | Thr:{self.yolo_confidence_threshold:.2f}"
        if self.show_camera_windows:
            cv2.namedWindow("Camera 1")

        if YOLO is None:
            self.get_logger().fatal("Ultralytics YOLO 패키지를 찾을 수 없어 'yolo' 모드를 사용할 수 없습니다.")
            self.create_timer(0.1, self.destroy_node); return

        if self.yolo_model is None:
            model_path = self.resolve_yolo_model_path()
            if model_path is None:
                self.get_logger().fatal("유효한 YOLO 모델 경로를 찾을 수 없습니다. 'yolo_model_path' 파라미터와 models 디렉터리를 확인하세요.")
                self.create_timer(0.1, self.destroy_node); return
            try:
                self.yolo_model = YOLO(model_path)
                self.resolved_yolo_model_path = model_path
                self.get_logger().info(f"Loaded YOLO model from '{model_path}'")
            except Exception as exc:
                self.get_logger().fatal(f"YOLO 모델 로드에 실패했습니다: {exc}")
                self.create_timer(0.1, self.destroy_node); return

        self.image_subscriber = self.create_subscription(
            CompressedImage,
            '/usb_cam_1/image_raw/compressed',
            self.yolo_image_callback,
            10
        )

    def init_manual_mode(self):
        self.gui_detection_status = self.decorate_status("Manual (rule-based)")
        self.rois = {1: None}; self.roi_points = {1: []}
        self.temp_roi_end_point = {1: None}; self.edit_mode = False
        self.load_rois_from_file()
        self.image_subscriber = self.create_subscription(
            CompressedImage,
            '/usb_cam_1/image_raw/compressed',
            self.manual_image_callback,
            10
        )
        if self.show_camera_windows:
            cv2.namedWindow("Camera 1")
            cv2.setMouseCallback("Camera 1", self.mouse_callback, 1)

    def yolo_image_callback(self, img_msg):
        if self.mission_triggered or not rclpy.ok(): return

        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(img_msg, 'bgr8')
        except Exception as exc:
            self.get_logger().error(f"Compressed image 변환 실패: {exc}")
            return

        results = self.run_yolo_inference(frame)
        detections = self.extract_valid_detections(frame, results, camera_id=1)
        best_detection = self.find_best_traffic_light_detection(detections)

        primary_color = 'Unknown'
        secondary_color = 'Unknown'
        rule_metrics = None

        if best_detection:
            self.get_logger().info(
                f"YOLO detected: '{best_detection.class_name}' with confidence {best_detection.score:.2f}")

            if 'green light' in best_detection.class_name:
                primary_color = 'Green'
            elif 'red light' in best_detection.class_name:
                primary_color = 'Red'
            else:
                primary_color = 'Unknown'

            rule_metrics = self.evaluate_rule_based_color(
                frame,
                best_detection.bbox,
                source_label='Camera 1'
            )
            secondary_color = rule_metrics.get('color', 'Unknown')
            self.last_yolo_confidence = best_detection.score

            if self.debug_mode:
                self.update_debug_rule_visuals(rule_metrics)
        else:
            self.last_yolo_confidence = None
            if self.debug_mode:
                self.update_debug_rule_visuals({'source_label': 'Camera 1'})

        self.last_primary_color = primary_color
        self.last_secondary_color = secondary_color
        self.last_rule_blob_summary = self._format_rule_blob_summary(rule_metrics)

        final_color, decision_source = self.update_final_color_decision(primary_color, secondary_color)

        if final_color == 'Green':
            self.trigger_green_mission()

        yolo_conf_str = f"YOLO:{self.last_yolo_confidence:.2f}" if self.last_yolo_confidence is not None else "YOLO:N/A"
        self.gui_confidence = f"{yolo_conf_str} | {self.last_rule_blob_summary}"

        if decision_source == 'consensus':
            decision_status = f"Final(consensus)"
        elif decision_source == 'rule_override':
            decision_status = f"Final(rule override)"
        else:
            if primary_color == secondary_color and primary_color in {'Red', 'Green'}:
                decision_status = (
                    f"Consensus pending {self.consensus_count}/{self.CONSENSUS_REQUIRED_FRAMES}"
                )
            elif secondary_color in {'Red', 'Green'} and primary_color != secondary_color:
                decision_status = (
                    f"Rule override pending {self.conflict_count}/{self.CONFLICT_REQUIRED_FRAMES}"
                )
            else:
                decision_status = "Awaiting stable detection"

        self.gui_state = final_color
        detection_summary = [
            "YOLO Mode",
            f"1차:{primary_color}",
            f"2차:{secondary_color}",
            decision_status,
            f"Thr:{self.yolo_confidence_threshold:.2f}"
        ]
        self.gui_detection_status = self.decorate_status(" | ".join(detection_summary))
        self.update_gui()

        self.get_logger().debug(
            f"Detections -> primary:{primary_color}, secondary:{secondary_color}, final:{final_color}, "
            f"cons_cnt:{self.consensus_count}, conf_cnt:{self.conflict_count}")

        if rclpy.ok() and self.show_camera_windows:
            self.draw_yolo_results(frame, detections)
            cv2.imshow("Camera 1", frame)
            cv2.waitKey(1)

    def manual_image_callback(self, img_msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(img_msg, 'bgr8')
        except Exception as exc:
            self.get_logger().error(f"Compressed image 변환 실패: {exc}")
            return
        self.process_and_update(frame, self.rois.get(1))

    def process_and_update(self, frame, roi_tuple):
        if self.mission_triggered or not rclpy.ok(): return

        res = self.process_single_stream(frame, roi_tuple, 1)

        if rclpy.ok() and self.show_camera_windows:
            self.visualizing(res['result_img'], res['roi'], 1, res['result'])
            key = cv2.waitKey(1) & 0xFF
            if self.roi_mode == 'manual' and key == ord('q'):
                self.edit_mode = not self.edit_mode
                self.get_logger().info(f'Mode changed to: {"ROI Edit" if self.edit_mode else "Color Detection"}')

        if self.debug_mode:
            metrics = res['result'].get('metrics')
            if metrics:
                self.update_debug_rule_visuals(metrics)
            else:
                self.update_debug_rule_visuals({'source_label': 'Camera 1'})

        metrics = res['result'].get('metrics')
        best_color = res['result'].get('color', 'Unknown')

        self.last_primary_color = best_color
        self.last_secondary_color = best_color
        self.last_yolo_confidence = None
        self.last_rule_blob_summary = self._format_rule_blob_summary(metrics)

        final_color, decision_source = self.update_final_color_decision(best_color, best_color)

        if final_color == 'Green':
            self.trigger_green_mission()

        self.gui_confidence = self.last_rule_blob_summary

        if decision_source == 'consensus':
            decision_status = "Final(consensus)"
        else:
            if best_color in {'Red', 'Green'}:
                decision_status = f"Consensus pending {self.consensus_count}/{self.CONSENSUS_REQUIRED_FRAMES}"
            else:
                decision_status = "Awaiting stable detection"

        roi_color = res['result'].get('color', 'Unknown')
        detection_summary = ["Manual Mode", f"ROI1:{roi_color}", decision_status]
        self.gui_detection_status = self.decorate_status(" | ".join(detection_summary))
        self.gui_state = final_color
        self.update_gui()

    def process_single_stream(self, frame, roi_tuple, stream_id):
        if self.roi_mode == 'manual' and roi_tuple is None:
            h, w, _ = frame.shape
            roi_size = int(min(h, w) * 0.2)
            cx, cy = w // 2, h // 2
            roi_tuple = (cx - roi_size // 2, cy - roi_size // 2, roi_size, roi_size)
            self.rois[stream_id] = roi_tuple

        result_img = frame.copy()
        box_found = roi_tuple is not None
        detection_result = {'color': 'Unknown', 'metrics': None, 'bbox': None}

        if box_found:
            x, y, w, h = roi_tuple
            bbox = (x, y, x + w, y + h)
            source_label = f"Camera {stream_id}"
            metrics = self.evaluate_rule_based_color(frame, bbox, source_label=source_label)
            detection_result['color'] = metrics.get('color', 'Unknown')
            detection_result['metrics'] = metrics
            detection_result['bbox'] = bbox

        return {
            'result': detection_result,
            'result_img': result_img,
            'roi': roi_tuple,
            'box_found': box_found,
            'stream_id': stream_id
        }

    def resolve_yolo_model_path(self):
        candidates = []
        if self.configured_yolo_model_path:
            if os.path.isabs(self.configured_yolo_model_path):
                candidates.append(self.configured_yolo_model_path)
            else:
                rel_path = self.configured_yolo_model_path.lstrip('/\\')
                candidates.append(os.path.join(self.script_dir, self.configured_yolo_model_path))
                candidates.append(os.path.join(self.script_dir, rel_path))
        default_path = os.path.join(self.script_dir, "models", "yolov10n_lightonly_251002.pt")
        candidates.append(default_path)
        for path in candidates:
            if path and os.path.exists(path): return path
        return None

    def run_yolo_inference(self, frame):
        if self.yolo_model is None: return None
        try:
            return self.yolo_model(frame, verbose=False)[0]
        except Exception as exc:
            self.get_logger().error(f"YOLO 추론 실패: {exc}")
            return None

    def find_best_traffic_light_detection(self, detections):
        """검증을 통과한 탐지 중에서 가장 높은 신뢰도를 반환합니다."""
        if not detections: return None
        return max(detections, key=lambda det: det.score)

    def draw_yolo_results(self, frame, detections):
        """프레임에 검증된 YOLO 탐지 결과를 시각화합니다."""
        if not detections: return
        color_map = {
            'green light': (0, 255, 0),
            'red light': (0, 0, 255),
            'unknown light': (255, 255, 0)
        }
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = color_map.get(det.class_name, (255, 0, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name}: {det.score:.2f}"
            cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def extract_valid_detections(self, frame, yolo_results, camera_id):
        """YOLO 결과에서 필요한 클래스만 추려 이중 검증을 통과한 탐지를 반환합니다."""
        detections = []
        if yolo_results is None or not getattr(yolo_results, 'boxes', None):
            return detections

        names = yolo_results.names
        height, width = frame.shape[:2]
        for box in yolo_results.boxes:
            class_idx = int(box.cls[0])
            try:
                class_name = names[class_idx] if isinstance(names, (list, tuple)) else names.get(class_idx, str(class_idx))
            except Exception:
                class_name = str(class_idx)
            if class_name not in self.yolo_target_classes:
                continue

            score = float(box.conf[0])
            if score < self.yolo_confidence_threshold:
                continue

            box_xywh = box.xywh[0].detach().cpu().numpy()
            x_center, y_center, w, h = box_xywh
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            is_valid = True
            if class_name == 'green light':
                is_valid = self.is_green_light_active(frame, (x1, y1, x2, y2))
                if not is_valid:
                    self.get_logger().debug(
                        f"Rejected green light detection due to guard check (score={score:.2f}).")

            if is_valid:
                detections.append(
                    SimpleNamespace(
                        class_name=class_name,
                        score=score,
                        bbox=(x1, y1, x2, y2),
                        camera_id=camera_id
                    )
                )

        return detections

    def is_green_light_active(self, frame, bbox):
        """green light 탐지 결과가 실제 점등 상태인지 이진화 기반으로 검증합니다."""
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        h, w = binary.shape
        if h < 2 or w < 2:
            return False

        split_idx = h // 2
        top_half = binary[:split_idx, :]
        bottom_half = binary[split_idx:, :]

        total_pixels = h * w
        top_ratio = float(np.count_nonzero(top_half)) / total_pixels
        bottom_ratio = float(np.count_nonzero(bottom_half)) / total_pixels

        return top_ratio <= self.GREEN_TOP_MAX_RATIO and bottom_ratio >= self.GREEN_BOTTOM_MIN_RATIO

    def _update_rule_color_ema(self, top_detected, bottom_detected):
        """Update exponential moving averages for top(red) and bottom(green) detections."""
        alpha = self.rule_ema_alpha
        beta = 1.0 - alpha
        target_red = 1.0 if top_detected else 0.0
        target_green = 1.0 if bottom_detected else 0.0
        self.rule_red_ema = beta * self.rule_red_ema + alpha * target_red
        self.rule_green_ema = beta * self.rule_green_ema + alpha * target_green
        self.rule_red_ema = max(0.0, min(1.0, self.rule_red_ema))
        self.rule_green_ema = max(0.0, min(1.0, self.rule_green_ema))

    def evaluate_rule_based_color(self, frame, bbox, source_label=None):
        """Simplified rule-based detector that looks for a single bright blob per half."""

        def build_metrics():
            return {
                'color': 'Unknown',
                'threshold_used': None,
                'top_binary': None,
                'bottom_binary': None,
                'full_binary': None,
                'source_label': source_label,
                'top_blob_count': 0,
                'bottom_blob_count': 0,
                'top_blob_detected': False,
                'bottom_blob_detected': False,
                'blob_size_threshold': (self.MIN_CLUSTER_WIDTH_PIXELS, self.MIN_CLUSTER_HEIGHT_PIXELS),
                'top_clusters': [],
                'bottom_clusters': [],
                'red_ema': self.rule_red_ema,
                'green_ema': self.rule_green_ema
            }

        if bbox is None:
            self._update_rule_color_ema(False, False)
            return build_metrics()

        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            self._update_rule_color_ema(False, False)
            return build_metrics()

        h, w = roi.shape[:2]
        if (
            h < self.MIN_RULE_HEIGHT or
            w < self.MIN_RULE_WIDTH or
            (h * w) < self.MIN_RULE_AREA
        ):
            self._update_rule_color_ema(False, False)
            return build_metrics()

        if h < 2:
            self._update_rule_color_ema(False, False)
            return build_metrics()

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        split_idx = h // 2
        if split_idx <= 0 or split_idx >= h:
            return build_metrics()

        threshold = int(max(0, min(255, self.rule_binary_threshold)))
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

        top_binary = binary[:split_idx, :]
        bottom_binary = binary[split_idx:, :]

        def detect_circular_clusters(binary_half, y_offset):
            if binary_half.size == 0:
                return []
            contours, _ = cv2.findContours(binary_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            clusters = []
            for contour in contours:
                if len(contour) < 5:
                    continue
                area = cv2.contourArea(contour)
                if area <= 0:
                    continue
                x, y, bw, bh = cv2.boundingRect(contour)
                if bw < self.MIN_CLUSTER_WIDTH_PIXELS or bh < self.MIN_CLUSTER_HEIGHT_PIXELS:
                    continue
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                if radius <= 0:
                    continue
                circle_area = np.pi * (radius ** 2)
                if circle_area <= 0:
                    continue
                circularity = area / circle_area
                if circularity < self.MIN_CLUSTER_CIRCULARITY:
                    continue
                absolute_bbox = (int(x), int(y + y_offset), int(bw), int(bh))
                clusters.append({
                    'bbox': absolute_bbox,
                    'circularity': float(circularity)
                })
            return clusters

        top_clusters = detect_circular_clusters(top_binary, 0)
        bottom_clusters = detect_circular_clusters(bottom_binary, split_idx)

        top_detected = len(top_clusters) > 0
        bottom_detected = len(bottom_clusters) > 0

        self._update_rule_color_ema(top_detected, bottom_detected)

        ema_margin = 0.1
        ema_threshold = 0.4
        color = 'Unknown'
        if (self.rule_red_ema - self.rule_green_ema) >= ema_margin and self.rule_red_ema >= ema_threshold:
            color = 'Red'
        elif (self.rule_green_ema - self.rule_red_ema) >= ema_margin and self.rule_green_ema >= ema_threshold:
            color = 'Green'

        metrics = build_metrics()
        metrics.update({
            'color': color,
            'threshold_used': threshold,
            'top_binary': top_binary,
            'bottom_binary': bottom_binary,
            'full_binary': binary,
            'top_blob_count': len(top_clusters),
            'bottom_blob_count': len(bottom_clusters),
            'top_blob_detected': top_detected,
            'bottom_blob_detected': bottom_detected,
            'blob_size_threshold': (self.MIN_CLUSTER_WIDTH_PIXELS, self.MIN_CLUSTER_HEIGHT_PIXELS),
            'top_clusters': top_clusters,
            'bottom_clusters': bottom_clusters,
            'red_ema': self.rule_red_ema,
            'green_ema': self.rule_green_ema
        })

        preview_label = source_label or 'ROI'
        if metrics['full_binary'] is not None:
            preview_img = cv2.cvtColor(metrics['full_binary'], cv2.COLOR_GRAY2BGR)
            cv2.putText(preview_img, f'{preview_label}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            self.update_binary_preview(preview_img)

        return metrics

    def _format_rule_blob_summary(self, metrics):
        if not metrics or 'top_blob_detected' not in metrics or 'bottom_blob_detected' not in metrics:
            return "RuleBlob:N/A"
        top_flag = 'Y' if metrics.get('top_blob_detected') else 'N'
        bottom_flag = 'Y' if metrics.get('bottom_blob_detected') else 'N'
        top_count = metrics.get('top_blob_count')
        bottom_count = metrics.get('bottom_blob_count')
        count_suffix = ""
        if isinstance(top_count, int) and isinstance(bottom_count, int):
            count_suffix = f" ({top_count}/{bottom_count})"
        red_ema = metrics.get('red_ema', self.rule_red_ema)
        green_ema = metrics.get('green_ema', self.rule_green_ema)
        return f"RuleBlob:T{top_flag}/B{bottom_flag}{count_suffix} | EMA R:{red_ema:.2f}/G:{green_ema:.2f}"

    def update_final_color_decision(self, primary_color, secondary_color):
        """1/2차 결과를 누적해 최종 신호등 색상을 결정합니다."""
        decision_source = None

        consensus = (
            primary_color == secondary_color and
            primary_color in {'Red', 'Green'}
        )

        if consensus:
            if self.consensus_candidate == primary_color:
                self.consensus_count += 1
            else:
                self.consensus_candidate = primary_color
                self.consensus_count = 1

            if self.consensus_count >= self.CONSENSUS_REQUIRED_FRAMES:
                decision_source = 'consensus'
        else:
            self.consensus_candidate = None
            self.consensus_count = 0

        conflict = (
            secondary_color in {'Red', 'Green'} and
            primary_color != secondary_color
        )

        if conflict:
            if self.conflict_candidate == secondary_color:
                self.conflict_count += 1
            else:
                self.conflict_candidate = secondary_color
                self.conflict_count = 1

            if self.conflict_count >= self.CONFLICT_REQUIRED_FRAMES:
                decision_source = 'rule_override'
        else:
            self.conflict_candidate = None
            self.conflict_count = 0

        if decision_source:
            new_color = secondary_color if decision_source == 'rule_override' else primary_color
            if self.last_final_color != new_color:
                self.get_logger().info(
                    f"Final traffic light decision updated to {new_color} ({decision_source}).")
            self.last_final_color = new_color
            self.last_decision_source = decision_source
            # 새 결정을 내렸으면 카운터를 초기화해 다음 변화를 기다린다.
            self.consensus_candidate = None
            self.consensus_count = 0
            self.conflict_candidate = None
            self.conflict_count = 0
        else:
            self.last_decision_source = 'None'

        final_color = self.last_final_color if self.last_final_color else 'Unknown'
        return final_color, decision_source

    def update_debug_rule_visuals(self, rule_metrics):
        if not self.debug_mode:
            return

        label = 'ROI'
        full_binary = None
        threshold_used = None
        top_detected = False
        bottom_detected = False
        top_blob_count = None
        bottom_blob_count = None
        blob_size_threshold = None
        red_ema = self.rule_red_ema
        green_ema = self.rule_green_ema

        if rule_metrics:
            label = rule_metrics.get('source_label') or label
            full_binary = rule_metrics.get('full_binary')
            threshold_used = rule_metrics.get('threshold_used')
            top_detected = bool(rule_metrics.get('top_blob_detected'))
            bottom_detected = bool(rule_metrics.get('bottom_blob_detected'))
            top_blob_count = rule_metrics.get('top_blob_count')
            bottom_blob_count = rule_metrics.get('bottom_blob_count')
            blob_size_threshold = rule_metrics.get('blob_size_threshold')
            red_ema = rule_metrics.get('red_ema', red_ema)
            green_ema = rule_metrics.get('green_ema', green_ema)
        else:
            label = 'Idle'

        if full_binary is None:
            full_display = np.zeros((200, 100), dtype=np.uint8)
        else:
            full_display = full_binary

        if len(full_display.shape) == 2:
            display_bgr = cv2.cvtColor(full_display, cv2.COLOR_GRAY2BGR)
        else:
            display_bgr = full_display.copy()

        cv2.putText(display_bgr, f'{label}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

        info_img = np.zeros((240, 420, 3), dtype=np.uint8)
        top_clusters = rule_metrics.get('top_clusters', []) if rule_metrics else []
        bottom_clusters = rule_metrics.get('bottom_clusters', []) if rule_metrics else []
        for cluster in top_clusters:
            x, y, bw, bh = cluster.get('bbox', (0, 0, 0, 0))
            cv2.rectangle(display_bgr, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        for cluster in bottom_clusters:
            x, y, bw, bh = cluster.get('bbox', (0, 0, 0, 0))
            cv2.rectangle(display_bgr, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        top_circularities = [cluster.get('circularity') for cluster in top_clusters]
        bottom_circularities = [cluster.get('circularity') for cluster in bottom_clusters]
        lines = [
            f"Source: {label}",
            f"Thr Used: {threshold_used}" if threshold_used is not None else "Thr Used: N/A",
            f"Top Blob: {'Yes' if top_detected else 'No'} ({top_blob_count if top_blob_count is not None else 0})",
            f"Bottom Blob: {'Yes' if bottom_detected else 'No'} ({bottom_blob_count if bottom_blob_count is not None else 0})",
            (
                f"Blob Size Thr: {blob_size_threshold[0]}x{blob_size_threshold[1]}px"
                if isinstance(blob_size_threshold, (tuple, list)) and len(blob_size_threshold) == 2
                else "Blob Size Thr: N/A"
            ),
            f"Top Circ: {np.mean(top_circularities):.2f}" if top_circularities else "Top Circ: N/A",
            f"Bottom Circ: {np.mean(bottom_circularities):.2f}" if bottom_circularities else "Bottom Circ: N/A",
            f"Red EMA: {red_ema:.2f}",
            f"Green EMA: {green_ema:.2f}"
        ]
        for idx, text in enumerate(lines):
            cv2.putText(info_img, text, (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

        window_binary = f'Debug Binary ({label})'
        window_info = f'Debug Rule Info ({label})'
        cv2.imshow(window_binary, display_bgr)
        cv2.imshow(window_info, info_img)
        self.debug_window_names.update({window_binary, window_info})

    def update_binary_preview(self, preview_img):
        window_name = 'Binary Preview'
        if not self.binary_trackbar_initialized:
            cv2.namedWindow(window_name)
            self.debug_window_names.add(window_name)
            cv2.createTrackbar(
                'Bin Thr',
                window_name,
                int(self.rule_binary_threshold),
                255,
                lambda v: self._on_binary_threshold_trackbar(v)
            )
            self.binary_trackbar_initialized = True
        cv2.imshow(window_name, preview_img)

    def _on_binary_threshold_trackbar(self, value):
        self.rule_binary_threshold = max(0, min(255, int(value)))

    def update_gui(self):
        if not rclpy.ok(): return
        if not self.gui_is_first_print: sys.stdout.write(f"\x1b[{self.GUI_HEIGHT}A"); sys.stdout.write("\x1b[J")
        self.gui_is_first_print = False
        print((f"--- Traffic Light Detector ({self.roi_mode.capitalize()}) ---\n"
               f"  Detection   : {self.gui_detection_status}\n"
               f"  State       : {self.gui_state}\n"
               f"  Confidence  : {self.gui_confidence}\n"
               f"  Service     : {self.gui_service_status}\n"
               f"  Last Error  : {self.gui_last_error}\n"
               f"-------------------------------------"), flush=True)

    def decorate_status(self, status_text):
        if self.debug_mode and not status_text.startswith('[DEBUG]'):
            return f"[DEBUG] {status_text}"
        return status_text

    def visualizing(self, frame, roi_tuple, stream_id, detection_result):
        if roi_tuple:
            x,y,w,h = roi_tuple
            detected_color = detection_result.get('color', 'Unknown')
            if self.roi_mode == 'manual' and self.edit_mode: color = (0, 255, 255)
            elif detected_color != 'Unknown': color = (0, 255, 0) if detected_color == "Green" else (0, 0, 255)
            else: color = (255, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        if self.roi_mode == 'manual' and self.edit_mode:
            cv2.putText(frame, "ROI EDIT MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if self.roi_points.get(stream_id) and self.temp_roi_end_point.get(stream_id):
                p1,p2 = self.roi_points[stream_id][0], self.temp_roi_end_point[stream_id]
                cv2.rectangle(frame, p1, p2, (0, 255, 255), 1)
        cv2.imshow(f"Camera {stream_id}", frame)

    def mouse_callback(self, event, x, y, flags, stream_id):
        if self.roi_mode != 'manual' or not self.edit_mode: return
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.roi_points.get(stream_id):
                self.roi_points[stream_id] = [(x, y)]; self.temp_roi_end_point[stream_id] = (x, y)
            else:
                x1,y1=self.roi_points[stream_id][0]; x2,y2=x,y
                start_x,end_x=min(x1,x2),max(x1,x2); start_y,end_y=min(y1,y2),max(y1,y2)
                if end_x > start_x and end_y > start_y:
                    self.rois[stream_id] = (start_x, start_y, end_x - start_x, end_y - start_y)
                    self.save_rois_to_file()
                self.roi_points[stream_id] = []; self.temp_roi_end_point[stream_id] = None
        elif event == cv2.EVENT_MOUSEMOVE and self.roi_points.get(stream_id):
            self.temp_roi_end_point[stream_id] = (x, y)

    def load_rois_from_file(self):
        if not os.path.exists(self.ROI_FILE):
            return

        try:
            with open(self.ROI_FILE, 'r') as file:
                data = json.load(file)

            roi_value = None
            if isinstance(data, dict):
                roi_value = data.get('1', data.get(1))

            if isinstance(roi_value, (list, tuple)) and len(roi_value) == 4:
                self.rois[1] = tuple(roi_value)
            else:
                self.rois[1] = None

            self.get_logger().info(f'Successfully loaded ROIs from {self.ROI_FILE}')
        except Exception as exc:
            self.get_logger().error(f'Failed to load ROIs: {exc}')

    def save_rois_to_file(self):
        try:
            roi_value = self.rois.get(1)
            if isinstance(roi_value, tuple):
                roi_value = list(roi_value)
            payload = {'1': roi_value if isinstance(roi_value, list) else None}
            with open(self.ROI_FILE, 'w') as file:
                json.dump(payload, file, indent=4)
            self.get_logger().info(f'Successfully saved ROIs to {self.ROI_FILE}')
        except Exception as exc:
            self.get_logger().error(f'Failed to save ROIs: {exc}')
    def virtual_green_callback(self, request, response):
        if self.mission_triggered: response.success=False; response.message='Mission already triggered.'; return response
        self.get_logger().info('<<<<< VIRTUAL GREEN LIGHT triggered by service call! >>>>>')
        self.trigger_green_mission(); response.success=True; response.message='Triggering green mission.'; return response
    def trigger_green_mission(self):
        if self.debug_mode:
            self.get_logger().info('Debug mode enabled; skipping /green trigger.')
            self.gui_service_status = "Debug (skip trigger)"
            self.gui_last_error = "None"
            self.update_gui()
            return

        if self.mission_triggered: return
        self.mission_triggered = True; self.gui_service_status = "Pending..."; self.update_gui()
        if self.retry_green_service_timer is None: self.retry_green_service_timer = self.create_timer(1.0, self.green_service_call_tick)
    def green_service_call_tick(self):
        if not self.green_client.service_is_ready(): self.gui_service_status="Failed"; self.gui_last_error="Server not ready."; self.update_gui(); return
        if self.is_calling_green_service: return
        self.gui_service_status="Calling..."; self.gui_last_error="None"; self.update_gui(); self.is_calling_green_service=True
        future = self.green_client.call_async(Trigger.Request()); future.add_done_callback(self.green_service_response_callback)
    def green_service_response_callback(self, future):
        self.is_calling_green_service = False
        try:
            response = future.result()
            if response.success:
                self.gui_service_status = "Succeeded"; self.gui_last_error = "None"; self.get_logger().info('\n/green service call successful! Shutting down node.')
                if self.retry_green_service_timer is not None: self.retry_green_service_timer.cancel()
                rclpy.shutdown()
            else: self.gui_service_status = "Failed"; self.gui_last_error = f'Server returned False: {response.message}'
        except Exception as e: self.gui_service_status = "Failed"; self.gui_last_error = f'Exception: {e}'
        self.update_gui()
    def cleanup(self):
        if self.retry_green_service_timer is not None: self.retry_green_service_timer.cancel()
        if self.show_camera_windows or bool(self.debug_window_names):
            cv2.destroyAllWindows(); cv2.waitKey(1)
        print("\n--- Cleanup Process Finished ---")

def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetector()
    try: rclpy.spin(node)
    except KeyboardInterrupt: node.get_logger().info('\nKeyboardInterrupt detected, shutting down.')
    finally:
        if rclpy.ok() and node.context.ok(): node.cleanup(); node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
