#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from custom_interface.msg import TrackedConeArray
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

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
    """
    def __init__(self, node_name='fused_cone_color_visualizer'):
        super().__init__(node_name)

        # 파라미터 선언
        self.declare_parameter('input_topic', '/fused_sorted_cones_ukf')
        self.declare_parameter('marker_topic', '/visualization_marker_fused_colored') # 고유한 토픽
        self.declare_parameter('marker_namespace', 'fused_cones_colored')
        self.declare_parameter('marker_scale', [0.35, 0.35, 0.35]) # x, y, z 스케일

        # 파라미터 가져오기
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        marker_topic = self.get_parameter('marker_topic').get_parameter_value().string_value
        self._marker_ns = self.get_parameter('marker_namespace').get_parameter_value().string_value
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

        # 상태
        self._previous_marker_count = 0 # 이전 마커 수 추적용

        self.get_logger().info(f"'{node_name}' 시작됨.")
        self.get_logger().info(f"구독 토픽: '{input_topic}'")
        self.get_logger().info(f"마커 발행 토픽: '{marker_topic}'")

    def _get_color_for_class(self, class_name: str) -> ColorRGBA:
        """주어진 클래스 이름에 대한 해당 색상을 반환합니다."""
        return COLOR_MAP.get(class_name.lower(), DEFAULT_COLOR) # 소문자로 매칭

    def _cone_data_callback(self, msg: TrackedConeArray):
        """
        들어오는 콘 데이터를 처리하고 색상 코딩된 마커를 발행합니다.
        이제 TrackedConeArray에서 3D 콘 데이터(X, Y, Z 좌표)를 지원합니다.
        """
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg() # 마커용 현재 시간 사용

        cones_data = [] # (x, y, z, class_name) 튜플을 저장

        # 1. TrackedConeArray 메시지 파싱
        try:
            for tracked_cone in msg.cones:
                x = tracked_cone.position.x
                y = tracked_cone.position.y
                z = tracked_cone.position.z
                class_name = tracked_cone.color # TrackedCone의 'color' 필드에 클래스 이름이 저장된다고 가정
                cones_data.append((x, y, z, class_name))
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

        # 3. 현재 콘에 대한 ADD 마커 생성
        current_marker_count = 0
        for i, (x, y, z, class_name) in enumerate(cones_data):
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

        # 4. 다음 콜백을 위해 마커 수 업데이트
        self._previous_marker_count = current_marker_count

        # 5. MarkerArray 발행
        if marker_array.markers: # 추가하거나 삭제할 항목이 있는 경우에만 발행
            self.marker_pub.publish(marker_array)

    def _publish_delete_markers(self, frame_id: str, timestamp):
        """삭제 마커만 발행하는 헬퍼 함수."""
        marker_array = MarkerArray()
        for i in range(self._previous_marker_count):
            delete_marker = Marker()
            delete_marker.header.frame_id = frame_id
            delete_marker.header.stamp = timestamp
            delete_marker.ns = self._marker_ns
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)
        if marker_array.markers:
            self.marker_pub.publish(marker_array)
        self._previous_marker_count = 0 # 삭제 후 카운트 재설정

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