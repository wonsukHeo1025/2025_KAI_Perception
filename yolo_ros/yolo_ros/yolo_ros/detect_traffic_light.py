#! /usr/bin/env python
# -*- coding:utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import numpy as np
from yolo_msgs.msg import DetectionArray
import message_filters
import sys
import json
import os
from rcl_interfaces.msg import ParameterDescriptor

class TrafficLightDetector(Node):
    """
    YOLO 또는 수동으로 정의된 ROI를 기반으로 신호등 색상을 감지하는 통합 노드.
    HSV 색상 범위를 트랙바로 실시간 조절하고, GUI 창 표시 여부를 파라미터로 제어하는 기능이 통합되었습니다.
    """
    GUI_HEIGHT = 6

    def __init__(self):
        super().__init__('traffic_light_detector')
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.ROI_FILE = os.path.join(script_dir, "rois.json")

        # --- 파라미터 선언 ---
        self.declare_parameter('roi_mode', 'manual', ParameterDescriptor(description='ROI 감지 모드: \'yolo\' 또는 \'manual\''))
        self.declare_parameter('pixel_threshold', 200, ParameterDescriptor(description='(Manual 모드용) 색상 검출을 위한 최소 픽셀 수 임계값'))
        self.declare_parameter('show_camera_windows', True, ParameterDescriptor(description='메인 카메라 창 표시 여부'))
        self.declare_parameter('show_control_windows', True, ParameterDescriptor(description='색상 제어 창 표시 여부'))
        self.declare_parameter('show_mask_windows', True, ParameterDescriptor(description='마스크 시각화 창 표시 여부'))
        
        # --- 파라미터 값 읽어오기 ---
        self.roi_mode = self.get_parameter('roi_mode').get_parameter_value().string_value
        self.threshold_pixels = self.get_parameter('pixel_threshold').get_parameter_value().integer_value
        self.show_camera_windows = self.get_parameter('show_camera_windows').get_parameter_value().bool_value
        self.show_control_windows = self.get_parameter('show_control_windows').get_parameter_value().bool_value
        self.show_mask_windows = self.get_parameter('show_mask_windows').get_parameter_value().bool_value
        
        self.hsv_ranges = {
            'red1': {'h_min': 0, 'h_max': 31, 's_min': 150, 's_max': 255, 'v_min': 173, 'v_max': 255},
            'red2': {'h_min': 170, 'h_max': 179, 's_min': 150, 's_max': 255, 'v_min': 173, 'v_max': 255},
            'green': {'h_min': 40, 'h_max': 80, 's_min': 150, 's_max': 255, 'v_min': 173, 'v_max': 255}
        }

        self.bridge = CvBridge()
        self.green_client = self.create_client(Trigger, '/green')
        self.virtual_green_service = self.create_service(Trigger, '/virtual_green', self.virtual_green_callback)
        
        self.mission_triggered = False; self.is_calling_green_service = False
        self.retry_green_service_timer = None
        self.gui_is_first_print = True; self.gui_state = "Waiting..."
        self.gui_service_status = "Idle"; self.gui_last_error = "None"
        self.gui_confidence = "N/A"
        
        if self.show_camera_windows or self.show_control_windows or self.show_mask_windows:
            self.setup_control_windows()

        if self.roi_mode == 'yolo': self.init_yolo_mode()
        elif self.roi_mode == 'manual': self.init_manual_mode()
        else:
            self.get_logger().fatal(f"Invalid roi_mode: '{self.roi_mode}'. Shutting down.")
            self.create_timer(0.1, self.destroy_node); return

        self.get_logger().info(f'TrafficLightDetector node started in \'{self.roi_mode}\' mode.')

    def create_hsv_gradient(self, h_range, s_range, v_value, width, height):
        h_min, h_max = h_range; s_min, s_max = s_range
        if h_min > h_max: h_min, h_max = h_max, h_min
        h = np.linspace(h_min, h_max, width); s = np.linspace(s_min, s_max, height)
        H, S = np.meshgrid(h, s)
        V = np.full((height, width), v_value, dtype=np.float64)
        hsv_gradient = np.stack([H, S, V], axis=-1).astype(np.uint8)
        return cv2.cvtColor(hsv_gradient, cv2.COLOR_HSV2BGR)

    def setup_control_windows(self):
        """OpenCV 창과 트랙바를 설정합니다. (Threshold 트랙바 제외)"""
        if self.show_control_windows:
            red_controls_window, green_controls_window = "Red Controls", "Green Controls"
            cv2.namedWindow(red_controls_window); cv2.namedWindow(green_controls_window)
            for key in ['h_min', 'h_max', 's_min', 's_max', 'v_min', 'v_max']:
                max_val = 179 if 'h' in key else 255
                cv2.createTrackbar(f"G_{key}", green_controls_window, self.hsv_ranges['green'][key], max_val, lambda v, k=key: self.hsv_ranges['green'].__setitem__(k, v))
            for key in ['h_min', 'h_max']: cv2.createTrackbar(f"R1_{key}", red_controls_window, self.hsv_ranges['red1'][key], 179, lambda v, k=key: self.hsv_ranges['red1'].__setitem__(k, v))
            for key in ['h_min', 'h_max']: cv2.createTrackbar(f"R2_{key}", red_controls_window, self.hsv_ranges['red2'][key], 179, lambda v, k=key: self.hsv_ranges['red2'].__setitem__(k, v))
            for key in ['s_min', 's_max', 'v_min', 'v_max']:
                cv2.createTrackbar(f"R_{key}", red_controls_window, self.hsv_ranges['red1'][key], 255, lambda v, k=key: (self.hsv_ranges['red1'].__setitem__(k, v), self.hsv_ranges['red2'].__setitem__(k, v)))
        
        if self.show_mask_windows:
            cv2.namedWindow("Mask 1"); cv2.namedWindow("Mask 2")

    def init_yolo_mode(self):
        self.gui_detection_status = "YOLO-Based"
        if self.show_camera_windows: 
            cv2.namedWindow("Camera 1"); cv2.namedWindow("Camera 2")
        
        image_sub1 = message_filters.Subscriber(self, CompressedImage, '/usb_cam_1/image_raw/compressed')
        yolo_sub1 = message_filters.Subscriber(self, DetectionArray, '/camera_1/detections')
        image_sub2 = message_filters.Subscriber(self, CompressedImage, '/usb_cam_2/image_raw/compressed')
        yolo_sub2 = message_filters.Subscriber(self, DetectionArray, '/camera_2/detections')
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer([image_sub1, yolo_sub1, image_sub2, yolo_sub2], queue_size=10, slop=0.2)
        self.time_synchronizer.registerCallback(self.yolo_synchronized_callback)

    def init_manual_mode(self):
        self.gui_detection_status = "Manual"
        self.rois = {1: None, 2: None}; self.roi_points = {1: [], 2: []}
        self.temp_roi_end_point = {1: None, 2: None}; self.edit_mode = False
        self.load_rois_from_file()
        image_sub1 = message_filters.Subscriber(self, CompressedImage, '/usb_cam_1/image_raw/compressed')
        image_sub2 = message_filters.Subscriber(self, CompressedImage, '/usb_cam_2/image_raw/compressed')
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer([image_sub1, image_sub2], queue_size=10, slop=0.2)
        self.time_synchronizer.registerCallback(self.manual_synchronized_callback)
        if self.show_camera_windows:
            cv2.namedWindow("Camera 1"); cv2.namedWindow("Camera 2")
            # Camera 1 창 생성 후 Threshold 트랙바 추가
            cv2.createTrackbar("Threshold (Pixels)", "Camera 1", self.threshold_pixels, 1000, lambda v: setattr(self, 'threshold_pixels', v))
            cv2.setMouseCallback("Camera 1", self.mouse_callback, 1)
            cv2.setMouseCallback("Camera 2", self.mouse_callback, 2)

    def yolo_synchronized_callback(self, img1_msg, yolo1_msg, img2_msg, yolo2_msg):
        if self.mission_triggered or not rclpy.ok(): return

        frame1 = self.bridge.compressed_imgmsg_to_cv2(img1_msg, 'bgr8')
        frame2 = self.bridge.compressed_imgmsg_to_cv2(img2_msg, 'bgr8')

        # 각 카메라 스트림에서 최상의 탐지 결과 찾기
        best_detection1 = self.find_best_traffic_light_detection(yolo1_msg)
        best_detection2 = self.find_best_traffic_light_detection(yolo2_msg)

        # 두 카메라를 통틀어 가장 신뢰도 높은 탐지 결과 선택
        overall_best_detection = None
        if best_detection1 and best_detection2:
            overall_best_detection = best_detection1 if best_detection1.score > best_detection2.score else best_detection2
        elif best_detection1:
            overall_best_detection = best_detection1
        elif best_detection2:
            overall_best_detection = best_detection2
        
        final_color = "Unknown"
        
        if overall_best_detection:
            # 탐지된 신호등 색상 로그 출력
            self.get_logger().info(f"YOLO detected: '{overall_best_detection.class_name}' with confidence {overall_best_detection.score:.2f}")
            
            # 탐지된 클래스 이름을 상태로 매핑
            if 'green light' in overall_best_detection.class_name:
                final_color = "Green"
            elif 'red light' in overall_best_detection.class_name:
                final_color = "Red"
            
            self.gui_confidence = f"{overall_best_detection.score:.2f}"

            # 녹색 신호등이 탐지되면 미션 트리거
            if final_color == "Green":
                self.trigger_green_mission()
        else:
            self.gui_confidence = "N/A"

        # GUI 상태 업데이트
        self.gui_state = final_color.capitalize()
        self.gui_detection_status = f"YOLO-Based ({'Detected' if final_color != 'Unknown' else 'Not Detected'})"
        self.update_gui()

        # 시각화
        if rclpy.ok() and (self.show_camera_windows or self.show_control_windows or self.show_mask_windows):
            if self.show_camera_windows:
                self.draw_yolo_boxes(frame1, yolo1_msg)
                self.draw_yolo_boxes(frame2, yolo2_msg)
                cv2.imshow("Camera 1", frame1)
                cv2.imshow("Camera 2", frame2)
            key = cv2.waitKey(1) & 0xFF


    def manual_synchronized_callback(self, img1_msg, img2_msg):
        frame1 = self.bridge.compressed_imgmsg_to_cv2(img1_msg, 'bgr8')
        frame2 = self.bridge.compressed_imgmsg_to_cv2(img2_msg, 'bgr8')
        self.process_and_update(frame1, self.rois.get(1), frame2, self.rois.get(2))

    def process_and_update(self, frame1, roi1_tuple, frame2, roi2_tuple):
        if self.mission_triggered or not rclpy.ok(): return

        if self.show_camera_windows: self.threshold_pixels = cv2.getTrackbarPos("Threshold (Pixels)", "Camera 1")
        if self.show_control_windows:
            grad_w, grad_h = 400, 75
            g = self.hsv_ranges['green']
            green_grad = self.create_hsv_gradient((g['h_min'], g['h_max']), (g['s_min'], g['s_max']), g['v_max'], grad_w, grad_h)
            cv2.imshow("Green Controls", green_grad)
            r1, r2 = self.hsv_ranges['red1'], self.hsv_ranges['red2']
            red_grad1 = self.create_hsv_gradient((r1['h_min'], r1['h_max']), (r1['s_min'], r1['s_max']), r1['v_max'], grad_w // 2, grad_h)
            red_grad2 = self.create_hsv_gradient((r2['h_min'], r2['h_max']), (r2['s_min'], r2['s_max']), r2['v_max'], grad_w // 2, grad_h)
            cv2.imshow("Red Controls", np.hstack([red_grad1, red_grad2]))
            
        res1 = self.process_single_stream(frame1, roi1_tuple, 1)
        res2 = self.process_single_stream(frame2, roi2_tuple, 2)

        if rclpy.ok() and (self.show_camera_windows or self.show_control_windows or self.show_mask_windows):
            if self.show_camera_windows:
                self.visualizing(res1['result_img'], res1['roi'], 1, res1['result'])
                self.visualizing(res2['result_img'], res2['roi'], 2, res2['result'])
            key = cv2.waitKey(1) & 0xFF
            if self.roi_mode == 'manual' and key == ord('q'):
                self.edit_mode = not self.edit_mode
                self.get_logger().info(f'Mode changed to: {"ROI Edit" if self.edit_mode else "Color Detection"}')

        res1_pixels = max(res1['result']['red_pixels'], res1['result']['green_pixels'])
        res2_pixels = max(res2['result']['red_pixels'], res2['result']['green_pixels'])
        best_res = res1 if res1_pixels > res2_pixels else res2
        final_color = best_res['result']['color']
        self.gui_state = final_color.capitalize()
        pixel_str = f"R:{best_res['result']['red_pixels']}, G:{best_res['result']['green_pixels']} / Thr:{self.threshold_pixels}"
        self.gui_detection_status = f"{self.roi_mode.capitalize()}-Based ({'Detected' if final_color != 'Unknown' else 'Not Detected'}) | {pixel_str}"
        self.gui_confidence = "N/A (HSV Range-based)"

        if final_color.lower() == 'green': self.trigger_green_mission()
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
        detection_result = {'color': 'Unknown', 'red_pixels': 0, 'green_pixels': 0, 'color_mask': None}

        if box_found:
            x, y, w, h = roi_tuple
            roi_img = frame[y:y+h, x:x+w]
            if roi_img.size > 0: detection_result = self.detect_color_with_hsv_range(roi_img)
        
        if self.show_mask_windows:
            mask_img = detection_result.get('color_mask')
            if mask_img is not None: cv2.imshow(f"Mask {stream_id}", mask_img)
            else: cv2.imshow(f"Mask {stream_id}", np.zeros((100, 100, 3), dtype=np.uint8))

        return {'result': detection_result, 'result_img': result_img, 'roi': roi_tuple, 'box_found': box_found}

    def find_best_traffic_light_detection(self, yolo_msg):
        """YOLO 탐지 결과에서 가장 신뢰도 높은 신호등 객체를 찾습니다."""
        best_detection = None
        highest_score = -1.0
        target_classes = ['green light', 'red light', 'unknown light']
        for detection in yolo_msg.detections:
            if detection.class_name in target_classes:
                if detection.score > highest_score:
                    highest_score = detection.score
                    best_detection = detection
        return best_detection

    def draw_yolo_boxes(self, frame, yolo_msg):
        """프레임에 YOLO 탐지 결과를 시각화합니다."""
        target_classes = ['green light', 'red light', 'unknown light']
        for detection in yolo_msg.detections:
            if detection.class_name in target_classes:
                cx = detection.bbox.center.position.x
                cy = detection.bbox.center.position.y
                w = detection.bbox.size.x
                h = detection.bbox.size.y
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                x2 = int(cx + w / 2)
                y2 = int(cy + h / 2)

                color_map = {
                    'green light': (0, 255, 0),
                    'red light': (0, 0, 255),
                    'unknown light': (255, 255, 0)
                }
                color = color_map.get(detection.class_name, (255, 0, 255))
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{detection.class_name}: {detection.score:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def detect_color_with_hsv_range(self, roi_img):
        hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        r1,r2,g = self.hsv_ranges['red1'],self.hsv_ranges['red2'],self.hsv_ranges['green']
        lower_red1 = np.array([r1['h_min'],r1['s_min'],r1['v_min']]); upper_red1 = np.array([r1['h_max'],r1['s_max'],r1['v_max']])
        lower_red2 = np.array([r2['h_min'],r2['s_min'],r2['v_min']]); upper_red2 = np.array([r2['h_max'],r2['s_max'],r2['v_max']])
        lower_green= np.array([g['h_min'],g['s_min'],g['v_min']]); upper_green= np.array([g['h_max'],g['s_max'],g['v_max']])
        mask_red1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        red_pixel_count = cv2.countNonZero(mask_red)
        mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)
        green_pixel_count = cv2.countNonZero(mask_green)
        detected_color = "Unknown"
        if red_pixel_count > self.threshold_pixels and red_pixel_count > green_pixel_count: detected_color = "Red"
        elif green_pixel_count > self.threshold_pixels and green_pixel_count > red_pixel_count: detected_color = "Green"
        color_mask_display = np.zeros_like(roi_img)
        color_mask_display[mask_red > 0] = (0, 0, 255)
        color_mask_display[mask_green > 0] = (0, 255, 0)
        return { 'color': detected_color, 'red_pixels': red_pixel_count, 'green_pixels': green_pixel_count, 'color_mask': color_mask_display }

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
        if os.path.exists(self.ROI_FILE):
            try:
                with open(self.ROI_FILE, 'r') as f: self.rois = {int(k): tuple(v) for k, v in json.load(f).items()}
                self.get_logger().info(f'Successfully loaded ROIs from {self.ROI_FILE}')
            except Exception as e: self.get_logger().error(f'Failed to load ROIs: {e}')
    def save_rois_to_file(self):
        try:
            with open(self.ROI_FILE, 'w') as f: json.dump(self.rois, f, indent=4)
            self.get_logger().info(f'Successfully saved ROIs to {self.ROI_FILE}')
        except Exception as e: self.get_logger().error(f'Failed to save ROIs: {e}')
    def virtual_green_callback(self, request, response):
        if self.mission_triggered: response.success=False; response.message='Mission already triggered.'; return response
        self.get_logger().info('<<<<< VIRTUAL GREEN LIGHT triggered by service call! >>>>>')
        self.trigger_green_mission(); response.success=True; response.message='Triggering green mission.'; return response
    def trigger_green_mission(self):
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
        if self.show_camera_windows or self.show_control_windows or self.show_mask_windows:
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