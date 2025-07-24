#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from custom_interface.msg import TrackedConeArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import numpy as np

# 클래스 이름에서 색상으로의 매핑
COLOR_MAP = {
    "red cone": ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
    "yellow cone":ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
    "blue cone":ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
    "unknown":  ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), # Unknown은 녹색으로 표시
}
DEFAULT_COLOR = ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0) # 매핑되지 않은 클래스는 회색으로 표시

class FusedConeColorVisualizer(Node):
    """
    클래스 이름이 포함된 융합된 콘 데이터를 구독하고
    RViz2를 위한 색상 코딩된 시각화 마커를 발행합니다.
    3D 콘 데이터(X, Y, Z 좌표)를 지원합니다.
    칼만 필터의 속도 예측을 화살표로 시각화합니다.
    """
    def __init__(self, node_name='fused_cone_color_visualizer'):
        super().__init__(node_name)

        # 파라미터 선언
        self.declare_parameter('input_topic', '/fused_sorted_cones_ukf')
        self.declare_parameter('marker_topic', '/vis/cone/fused/ukf') # 퓨전된 UKF 콘
        self.declare_parameter('arrow_marker_topic', '/vis/cone/velocity') # 속도 화살표
        self.declare_parameter('text_marker_topic', '/vis/cone/text') # 트랙 ID 텍스트
        self.declare_parameter('marker_namespace', 'fused_cones_colored')
        self.declare_parameter('velocity_marker_namespace', 'velocity_arrows')
        self.declare_parameter('text_marker_namespace', 'track_id_text')
        self.declare_parameter('marker_scale', [0.35, 0.35, 0.35]) # x, y, z 스케일

        # 파라미터 가져오기
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        marker_topic = self.get_parameter('marker_topic').get_parameter_value().string_value
        arrow_marker_topic = self.get_parameter('arrow_marker_topic').get_parameter_value().string_value
        text_marker_topic = self.get_parameter('text_marker_topic').get_parameter_value().string_value
        self._marker_ns = self.get_parameter('marker_namespace').get_parameter_value().string_value
        self._velocity_marker_ns = self.get_parameter('velocity_marker_namespace').get_parameter_value().string_value
        self._text_marker_ns = self.get_parameter('text_marker_namespace').get_parameter_value().string_value
        marker_scale_list = self.get_parameter('marker_scale').get_parameter_value().double_array_value

        if len(marker_scale_list) != 3:
            self.get_logger().error("'marker_scale' 파라미터는 3개의 값(x, y, z)을 가져야 합니다. 기본값 사용.")
            marker_scale_list = [0.35, 0.35, 0.35]
        self._marker_scale_x = marker_scale_list[0]
        self._marker_scale_y = marker_scale_list[1]
        self._marker_scale_z = marker_scale_list[2]

        # QoS 설정
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 구독자
        self.subscription = self.create_subscription(
            TrackedConeArray,
            input_topic,
            self._cone_data_callback,
            qos_profile
        )

        # 발행자
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, qos_profile)
        self.arrow_marker_pub = self.create_publisher(MarkerArray, arrow_marker_topic, qos_profile)
        self.text_marker_pub = self.create_publisher(MarkerArray, text_marker_topic, qos_profile)

        # 상태
        self._previous_marker_count = 0 # 이전 마커 수 추적용
        self._previous_velocity_marker_count = 0 # 속도 화살표 수 추적
        self._previous_text_marker_count = 0 # 텍스트 마커 수 추적
        
        # 속도 추정을 위한 이전 위치 저장
        self._previous_positions = {}  # track_id: (timestamp, position)
        self._velocity_estimates = {}  # track_id: velocity_vector

        self.get_logger().info(f"'{node_name}' 시작됨.")
        self.get_logger().info(f"구독 토픽: '{input_topic}'")
        self.get_logger().info(f"퓨전 UKF 콘 발행 토픽: '{marker_topic}'")
        self.get_logger().info(f"속도 화살표 발행 토픽: '{arrow_marker_topic}'")
        self.get_logger().info(f"트랙 ID 텍스트 발행 토픽: '{text_marker_topic}'")

    def _get_color_for_class(self, class_name: str) -> ColorRGBA:
        """주어진 클래스 이름에 대한 해당 색상을 반환합니다."""
        return COLOR_MAP.get(class_name.lower(), DEFAULT_COLOR) # 소문자로 매칭

    def _cone_data_callback(self, msg: TrackedConeArray):
        """
        들어오는 콘 데이터를 처리하고 색상 코딩된 마커를 발행합니다.
        이제 TrackedConeArray에서 3D 콘 데이터(X, Y, Z 좌표)를 지원합니다.
        """
        marker_array = MarkerArray()
        arrow_marker_array = MarkerArray()  # 화살표용 별도 MarkerArray
        now = self.get_clock().now().to_msg() # 마커용 현재 시간 사용

        cones_data = [] # (x, y, z, class_name, track_id) 튜플을 저장
        current_time_sec = self.get_clock().now().nanoseconds / 1e9

        # 1. TrackedConeArray 메시지 파싱
        try:
            for tracked_cone in msg.cones:
                x = tracked_cone.position.x
                y = tracked_cone.position.y
                z = tracked_cone.position.z
                class_name = tracked_cone.color # TrackedCone의 'color' 필드에 클래스 이름이 저장된다고 가정
                track_id = tracked_cone.track_id
                cones_data.append((x, y, z, class_name, track_id))
                
                # 속도 추정을 위한 위치 업데이트 (XY 평면만 사용)
                current_pos = np.array([x, y, z])
                if track_id in self._previous_positions:
                    prev_time, prev_pos = self._previous_positions[track_id]
                    dt = current_time_sec - prev_time
                    if dt > 0.01:  # 최소 시간 간격
                        velocity = (current_pos - prev_pos) / dt
                        # Z축 속도는 0으로 설정 (2D 추적)
                        velocity[2] = 0.0
                        self._velocity_estimates[track_id] = velocity
                else:
                    self._velocity_estimates[track_id] = np.array([0.0, 0.0, 0.0])
                
                self._previous_positions[track_id] = (current_time_sec, current_pos)
        except AttributeError as e:
            self.get_logger().error(f"TrackedConeArray 파싱 오류: {e}. 'cones', 'position', 'color' 속성이 존재하는지 확인하세요.")
            self._publish_delete_markers(msg.header.frame_id or "map", now)
            return
        except Exception as e:
            self.get_logger().error(f"TrackedConeArray 파싱 중 예상치 못한 오류: {e}")
            self._publish_delete_markers(msg.header.frame_id or "map", now)
            return

        if not cones_data and msg.cones:
            self.get_logger().warn("메시지에 콘 데이터가 있었지만, cones_data 리스트로 파싱하지 못했습니다.")

        # 2. 이전에 발행된 마커에 대한 DELETE 마커 생성
        # 콘 수가 감소하거나 사라지면 이전 마커가 RViz에서 제거되도록 합니다.
        frame_id = msg.header.frame_id if msg.header.frame_id else "map" # 비어있으면 기본값
        for i in range(self._previous_marker_count):
            delete_marker = Marker()
            delete_marker.header.frame_id = frame_id
            delete_marker.header.stamp = now
            delete_marker.ns = self._marker_ns
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)
            
        # 속도 화살표 삭제 (별도 MarkerArray에 추가)
        for i in range(self._previous_velocity_marker_count):
            delete_marker = Marker()
            delete_marker.header.frame_id = frame_id
            delete_marker.header.stamp = now
            delete_marker.ns = self._velocity_marker_ns
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            arrow_marker_array.markers.append(delete_marker)

        # 텍스트 마커 삭제 (기존 MarkerArray에 추가)
        for i in range(self._previous_text_marker_count):
            delete_marker = Marker()
            delete_marker.header.frame_id = frame_id
            delete_marker.header.stamp = now
            delete_marker.ns = self._text_marker_ns
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        # 3. 현재 콘에 대한 ADD 마커 생성
        current_marker_count = 0
        current_velocity_count = 0
        current_text_count = 0
        for i, (x, y, z, class_name, track_id) in enumerate(cones_data):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = now
            marker.ns = self._marker_ns
            marker.id = i # 인덱스를 ID로 사용
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # 위치: 실제 z 좌표 사용
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = float(z)
            marker.pose.orientation.w = 1.0 # 구에는 회전이 필요 없음

            # 파라미터에서 스케일
            marker.scale.x = self._marker_scale_x
            marker.scale.y = self._marker_scale_y
            marker.scale.z = self._marker_scale_z

            # 클래스 이름 기반 색상
            marker.color = self._get_color_for_class(class_name)
            if marker.color == DEFAULT_COLOR:
                self.get_logger().debug(f"콘 {i}, 클래스 '{class_name}'이 기본 색상으로 매핑됨.")

            marker_array.markers.append(marker)
            current_marker_count += 1
            
            # 속도 화살표 추가
            if track_id in self._velocity_estimates:
                velocity = self._velocity_estimates[track_id]
                speed = np.linalg.norm(velocity)
                
                # 속도가 충분히 클 때만 화살표 표시 (노이즈 필터링)
                if speed > 0.1:  # 0.1 m/s 이상
                    arrow_marker = Marker()
                    arrow_marker.header.frame_id = frame_id
                    arrow_marker.header.stamp = now
                    arrow_marker.ns = self._velocity_marker_ns
                    arrow_marker.id = current_velocity_count
                    arrow_marker.type = Marker.ARROW
                    arrow_marker.action = Marker.ADD
                    
                    # 화살표 시작점과 끝점
                    start_point = Point()
                    start_point.x = float(x)
                    start_point.y = float(y)
                    start_point.z = float(z)
                    
                    # 0.5초 후 예측 위치 (기존 1초에서 절반으로)
                    prediction_time = 0.5
                    end_point = Point()
                    end_point.x = float(x + velocity[0] * prediction_time)
                    end_point.y = float(y + velocity[1] * prediction_time)
                    end_point.z = float(z + velocity[2] * prediction_time)
                    
                    arrow_marker.points = [start_point, end_point]
                    
                    # 화살표 크기 (속도에 비례, 변화폭 1/3로 감소)
                    arrow_marker.scale.x = 0.05 * (1.0 + speed * 0.167)  # 샤프트 직경 (0.5 -> 0.167)
                    arrow_marker.scale.y = 0.08 * (1.0 + speed * 0.167)  # 헤드 직경 (0.5 -> 0.167)
                    arrow_marker.scale.z = 0.1  # 헤드 길이
                    
                    # 색상 (속도에 따라)
                    arrow_marker.color.a = 0.8
                    if speed < 1.0:
                        # 느림: 녹색
                        arrow_marker.color.r = 0.0
                        arrow_marker.color.g = 1.0
                        arrow_marker.color.b = 0.0
                    elif speed < 3.0:
                        # 중간: 노란색  
                        arrow_marker.color.r = 1.0
                        arrow_marker.color.g = 1.0
                        arrow_marker.color.b = 0.0
                    else:
                        # 빠름: 빨간색
                        arrow_marker.color.r = 1.0
                        arrow_marker.color.g = 0.0
                        arrow_marker.color.b = 0.0
                    
                    arrow_marker_array.markers.append(arrow_marker)
                    current_velocity_count += 1

            # 텍스트 마커 추가 (기존 MarkerArray에 추가)
            text_marker = Marker()
            text_marker.header.frame_id = frame_id
            text_marker.header.stamp = now
            text_marker.ns = self._text_marker_ns
            text_marker.id = current_text_count
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            # 위치: 콘 위에 위치
            text_marker.pose.position.x = float(x)
            text_marker.pose.position.y = float(y)
            text_marker.pose.position.z = float(z + 0.3)  # 콘 위 30cm에 텍스트 표시
            text_marker.pose.orientation.w = 1.0

            # 텍스트 내용
            text_marker.text = str(track_id)

            # 텍스트 크기
            text_marker.scale.z = 0.2  # 텍스트 높이 (미터 단위)

            # 텍스트 색상 (흰색, 투명하지 않음)
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0

            text_marker_array.markers.append(text_marker)
            current_text_count += 1

        # 4. 다음 콜백을 위해 마커 수 업데이트
        self._previous_marker_count = current_marker_count
        self._previous_velocity_marker_count = current_velocity_count
        self._previous_text_marker_count = current_text_count

        # 5. MarkerArray 발행
        if marker_array.markers: # 추가하거나 삭제할 항목이 있는 경우에만 발행
            self.marker_pub.publish(marker_array)
        
        # 6. 화살표 MarkerArray 발행
        if arrow_marker_array.markers: # 화살표 추가하거나 삭제할 항목이 있는 경우에만 발행
            self.arrow_marker_pub.publish(arrow_marker_array)
            
        # 7. 텍스트 MarkerArray 발행
        if text_marker_array.markers: # 텍스트 추가하거나 삭제할 항목이 있는 경우에만 발행
            self.text_marker_pub.publish(text_marker_array)

    def _publish_delete_markers(self, frame_id: str, timestamp):
        """삭제 마커만 발행하는 헬퍼 함수."""
        marker_array = MarkerArray()
        arrow_marker_array = MarkerArray()
        
        for i in range(self._previous_marker_count):
            delete_marker = Marker()
            delete_marker.header.frame_id = frame_id
            delete_marker.header.stamp = timestamp
            delete_marker.ns = self._marker_ns
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)
        
        for i in range(self._previous_velocity_marker_count):
            delete_marker = Marker()
            delete_marker.header.frame_id = frame_id
            delete_marker.header.stamp = timestamp
            delete_marker.ns = self._velocity_marker_ns
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            arrow_marker_array.markers.append(delete_marker)
        
        for i in range(self._previous_text_marker_count):
            delete_marker = Marker()
            delete_marker.header.frame_id = frame_id
            delete_marker.header.stamp = timestamp
            delete_marker.ns = self._text_marker_ns
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            text_marker_array.markers.append(delete_marker)
            
        if marker_array.markers:
            self.marker_pub.publish(marker_array)
        if arrow_marker_array.markers:
            self.arrow_marker_pub.publish(arrow_marker_array)
        if text_marker_array.markers:
            self.text_marker_pub.publish(text_marker_array)
            
        self._previous_marker_count = 0 # 삭제 후 카운트 재설정
        self._previous_velocity_marker_count = 0
        self._previous_text_marker_count = 0

    def destroy_node(self):
        """종료 전 정리."""
        self.get_logger().info("마커 정리 중...")
        now = self.get_clock().now().to_msg()
        # 마지막으로 알려진 모든 ID에 대해 삭제 마커만 발행
        last_frame_id = "map" # 기본 폴백 프레임

        self._publish_delete_markers(last_frame_id, now)
        # 발행자가 보내는 짧은 시간 여유
        if self.context and self.context.ok():
             self.context.sleep_for(0.1)
        super().destroy_node()
        self.get_logger().info("Fused Cone Color Visualizer 종료됨.")

# 메인 실행
def main(args=None):
    rclpy.init(args=args)
    visualizer_node = None # None으로 초기화
    try:
        visualizer_node = FusedConeColorVisualizer()
        rclpy.spin(visualizer_node)
    except KeyboardInterrupt:
        if visualizer_node:
             visualizer_node.get_logger().info('키보드 인터럽트, 종료 중.')
        else:
             print('노드 초기화 전 키보드 인터럽트.')
    except ImportError as e:
         # 노드 생성 전에 발생했다면 다시 import 오류 포착
         print(f"실행 중 Import 오류: {e}")
    except Exception as e:
        # spin 중 다른 예상치 못한 오류 포착
        if visualizer_node:
            visualizer_node.get_logger().fatal(f"처리되지 않은 예외: {e}", include_traceback=True)
        else:
            print(f"노드 초기화 전 처리되지 않은 예외: {e}")
    finally:
        # spin 루프가 예상치 못하게 종료되더라도 정리 작업이 이루어지도록 보장
        if visualizer_node and rclpy.ok():
             visualizer_node.destroy_node()
        if rclpy.ok():
             rclpy.shutdown()

if __name__ == '__main__':
    main()