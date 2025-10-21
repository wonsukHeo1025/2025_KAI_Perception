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
import os
from types import SimpleNamespace
from rcl_interfaces.msg import ParameterDescriptor

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class TrafficLightDetector(Node):
    """
    YOLO 기반 신호등 감지 노드.
    모델이 탐지한 신호등 클래스를 EMA로 평활화하여 최종 상태를 결정합니다.
    """
    GUI_HEIGHT = 6
    MODEL_EMA_ALPHA = 0.3
    MODEL_EMA_THRESHOLD = 0.6
    MODEL_EMA_MARGIN = 0.1
    MODEL_EMA_DECAY = 0.1

    def __init__(self):
        super().__init__('traffic_light_detector')

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.script_dir = script_dir

        self.declare_parameter('show_camera_windows', True, ParameterDescriptor(description='메인 카메라 창 표시 여부'))
        self.declare_parameter('debug_mode', False, ParameterDescriptor(description='초록불 트리거를 비활성화하고 디버그 시각화를 활성화합니다.'))
        default_model_path = os.path.join(self.script_dir, "models", "yolov10n_GRU.pt")
        self.declare_parameter('yolo_model_path', default_model_path, ParameterDescriptor(description='/models/yolov10n_GRU.pt YOLO 모델 파일 경로'))
        self.declare_parameter('yolo_confidence_threshold', 0.7, ParameterDescriptor(description='YOLO 탐지를 채택할 최소 신뢰도 (0.0~1.0)'))

        self.show_camera_windows = self.get_parameter('show_camera_windows').get_parameter_value().bool_value
        self.debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value
        self.configured_yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        yolo_conf_param = self.get_parameter('yolo_confidence_threshold').get_parameter_value().double_value
        self.yolo_confidence_threshold = max(0.0, min(yolo_conf_param, 1.0))

        self.bridge = CvBridge()
        self.yolo_model = None
        self.resolved_yolo_model_path = None
        self.yolo_target_classes = {'green light', 'red light', 'unknown light'}
        self.green_client = self.create_client(Trigger, '/green')
        self.virtual_green_service = self.create_service(Trigger, '/virtual_green', self.virtual_green_callback)

        self.gui_mode_label = "YOLO"
        self.mission_triggered = False
        self.is_calling_green_service = False
        self.retry_green_service_timer = None
        self.gui_is_first_print = True
        self.gui_state = "Waiting..."
        self.gui_detection_status = "Initializing..."
        self.gui_service_status = "Idle"
        self.gui_last_error = "None"
        self.gui_confidence = "N/A"
        self.model_red_ema = 0.0
        self.model_green_ema = 0.0
        self.last_model_ema_summary = "ModelEMA:R0.00/G0.00"
        self.last_yolo_confidence = None

        if self.show_camera_windows:
            cv2.namedWindow("Camera 1")

        self.init_yolo_mode()
        self.get_logger().info('TrafficLightDetector node started in YOLO mode.')


    def init_yolo_mode(self):
        self.gui_detection_status = f"YOLO | Idle | Thr:{self.yolo_confidence_threshold:.2f}"

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
        self.yolo_subscription = self.create_subscription(
            CompressedImage,
            '/usb_cam_1/image_raw/compressed',
            self.yolo_image_callback,
            10
        )

    def yolo_image_callback(self, img_msg):
        if self.mission_triggered or not rclpy.ok():
            return

        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(img_msg, 'bgr8')
        except Exception as exc:
            self.get_logger().error(f"Compressed image 변환 실패: {exc}")
            return

        yolo_results = self.run_yolo_inference(frame)
        best_detection = self.find_best_traffic_light_detection(yolo_results)

        detection_color = 'Unknown'
        detection_confidence = None

        if best_detection:
            detection_color = self._map_class_to_color(best_detection.class_name)
            detection_confidence = best_detection.score
            self.last_yolo_confidence = detection_confidence
            self.get_logger().info(
                f"YOLO detected: '{best_detection.class_name}' ({detection_color}) with confidence {detection_confidence:.2f}")
        else:
            self.last_yolo_confidence = None

        final_color = self._integrate_detection_with_ema(detection_color, detection_confidence)

        if final_color == 'Green':
            self.trigger_green_mission()

        if self.last_yolo_confidence is not None:
            yolo_conf = f"YOLO:{self.last_yolo_confidence:.2f}"
        else:
            yolo_conf = "YOLO:N/A"
        self.gui_confidence = f"{yolo_conf} | {self.last_model_ema_summary}"

        detection_state = "Detected" if final_color in {'Red', 'Green'} else "Not Detected"
        self.gui_state = final_color
        self.gui_detection_status = (
            f"YOLO ({detection_state}) | Raw:{detection_color} | EMA:{final_color} | Thr:{self.yolo_confidence_threshold:.2f}"
        )
        self.update_gui()

        if rclpy.ok() and self.show_camera_windows:
            self.draw_yolo_results(frame, yolo_results)
            cv2.imshow("Camera 1", frame)
            cv2.waitKey(1)

    def resolve_yolo_model_path(self):
        candidates = []
        if self.configured_yolo_model_path:
            if os.path.isabs(self.configured_yolo_model_path):
                candidates.append(self.configured_yolo_model_path)
            else:
                rel_path = self.configured_yolo_model_path.lstrip('/\\')
                candidates.append(os.path.join(self.script_dir, self.configured_yolo_model_path))
                candidates.append(os.path.join(self.script_dir, rel_path))
        default_path = os.path.join(self.script_dir, "models", "yolov10n_GRU.pt")
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

    def find_best_traffic_light_detection(self, yolo_results):
        """YOLO 결과에서 가장 신뢰도 높은 신호등 객체를 찾습니다."""
        if yolo_results is None or not getattr(yolo_results, 'boxes', None): return None
        names = yolo_results.names
        best_detection = None
        best_score = self.yolo_confidence_threshold
        for box in yolo_results.boxes:
            class_idx = int(box.cls[0])
            try:
                class_name = names[class_idx] if isinstance(names, (list, tuple)) else names.get(class_idx, str(class_idx))
            except Exception:
                class_name = str(class_idx)
            if class_name not in self.yolo_target_classes: continue
            score = float(box.conf[0])
            if score < self.yolo_confidence_threshold: continue
            if best_detection is None or score > best_score:
                best_detection = SimpleNamespace(class_name=class_name, score=score)
                best_score = score
        return best_detection

    def draw_yolo_results(self, frame, yolo_results):
        """프레임에 YOLO 탐지 결과를 시각화합니다."""
        if yolo_results is None or not getattr(yolo_results, 'boxes', None): return
        names = yolo_results.names
        for box in yolo_results.boxes:
            class_idx = int(box.cls[0])
            try:
                class_name = names[class_idx] if isinstance(names, (list, tuple)) else names.get(class_idx, str(class_idx))
            except Exception:
                class_name = str(class_idx)
            if class_name not in self.yolo_target_classes: continue
            score = float(box.conf[0])
            if score < self.yolo_confidence_threshold: continue
            box_xywh = box.xywh[0].detach().cpu().numpy()
            x_center, y_center, w, h = box_xywh
            x1 = int(x_center - w / 2); y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2); y2 = int(y_center + h / 2)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame.shape[1] - 1, x2); y2 = min(frame.shape[0] - 1, y2)
            if x2 <= x1 or y2 <= y1: continue
            color_map = {
                'green light': (0, 255, 0),
                'red light': (0, 0, 255),
                'unknown light': (255, 255, 0)
            }
            color = color_map.get(class_name, (255, 0, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {score:.2f}"
            cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _map_class_to_color(self, class_name):
        if not class_name:
            return 'Unknown'
        lowered = class_name.lower()
        if 'green' in lowered:
            return 'Green'
        if 'red' in lowered:
            return 'Red'
        return 'Unknown'

    def _update_model_color_ema(self, color, confidence):
        confidence = max(0.0, min(1.0, confidence if confidence is not None else 0.0))
        alpha = self.MODEL_EMA_ALPHA * confidence if confidence > 0.0 else 0.0
        alpha = max(0.0, min(1.0, alpha))
        beta = 1.0 - alpha

        if color == 'Red':
            self.model_red_ema = beta * self.model_red_ema + alpha * 1.0
            self.model_green_ema = beta * self.model_green_ema
        elif color == 'Green':
            self.model_green_ema = beta * self.model_green_ema + alpha * 1.0
            self.model_red_ema = beta * self.model_red_ema
        else:
            self.model_red_ema = beta * self.model_red_ema
            self.model_green_ema = beta * self.model_green_ema

        self.model_red_ema = max(0.0, min(1.0, self.model_red_ema))
        self.model_green_ema = max(0.0, min(1.0, self.model_green_ema))
        ema_color = self._model_ema_to_color()
        self._refresh_model_ema_summary()
        return ema_color

    def _decay_model_color_ema(self):
        decay = max(0.0, min(1.0, self.MODEL_EMA_DECAY))
        factor = 1.0 - decay
        self.model_red_ema *= factor
        self.model_green_ema *= factor
        self._refresh_model_ema_summary()

    def _model_ema_to_color(self):
        margin = self.MODEL_EMA_MARGIN
        threshold = self.MODEL_EMA_THRESHOLD
        red = self.model_red_ema
        green = self.model_green_ema
        if (red - green) >= margin and red >= threshold:
            return 'Red'
        if (green - red) >= margin and green >= threshold:
            return 'Green'
        return 'Unknown'

    def _refresh_model_ema_summary(self):
        self.last_model_ema_summary = f"ModelEMA:R{self.model_red_ema:.2f}/G{self.model_green_ema:.2f}"

    def _integrate_detection_with_ema(self, color, confidence):
        if color in {'Red', 'Green'} and confidence is not None:
            return self._update_model_color_ema(color, confidence)
        self._decay_model_color_ema()
        return self._model_ema_to_color()

    def update_gui(self):
        if not rclpy.ok(): return
        if not self.gui_is_first_print: sys.stdout.write(f"\x1b[{self.GUI_HEIGHT}A"); sys.stdout.write("\x1b[J")
        self.gui_is_first_print = False
        print((f"--- Traffic Light Detector ({self.gui_mode_label}) ---\n"
               f"  Detection   : {self.gui_detection_status}\n"
               f"  State       : {self.gui_state}\n"
               f"  Confidence  : {self.gui_confidence}\n"
               f"  Service     : {self.gui_service_status}\n"
               f"  Last Error  : {self.gui_last_error}\n"
               f"-------------------------------------"), flush=True)

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
        if self.show_camera_windows:
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
