#! /usr/bin/env python
# -*- coding:utf-8 -*-

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

class TestMissionNode(Node):
    """
    '/green' 서비스 호출을 받아 내부 플래그를 변경하고,
    플래그 상태에 따라 주기적으로 다른 작업을 수행하는 ROS2 노드.
    """
    def __init__(self):
        # 'mission_node' 라는 이름으로 노드 초기화
        super().__init__('test_mission_node')

        # 1. 미션 플래그를 False로 초기화
        self._green_flag = False

        # 2. '/green' 이름의 Trigger 타입 서비스 서버 생성
        #    서비스 요청이 오면 self.green_service_callback 메서드가 실행됨
        self.green_service = self.create_service(
            Trigger,
            '/green',
            self.green_service_callback)

        # 3. 1초마다 self.mission_timer_callback 메서드를 실행하는 타이머 생성
        self.timer = self.create_timer(1.0, self.mission_timer_callback)

        self.get_logger().info('미션 노드가 시작되었습니다. /green 서비스 호출을 대기합니다.')

    def green_service_callback(self, request, response):
        """
        /green 서비스가 호출되었을 때 실행되는 콜백 함수 (신호 수신부)
        """
        if not self._green_flag:
            self.get_logger().info('✅ /green 신호를 수신했습니다! 플래그를 True로 변경합니다.')
            # 플래그를 True로 설정
            self._green_flag = True
        else:
            self.get_logger().warn('이미 /green 신호를 받은 상태입니다.')

        # 서비스 호출에 대한 응답 설정
        response.success = True
        response.message = 'Green flag is now set to True.'
        
        return response

    def mission_timer_callback(self):
        """
        1초마다 주기적으로 실행되는 함수 (메인 로직)
        """
        # 플래그 상태를 확인하고 그에 맞는 동작 수행
        if self._green_flag:
            self.get_logger().info('🚀 초록불 확인! 미션을 수행합니다. (예: 모터 구동)')
            # 여기에 미션 수행과 관련된 실제 코드를 추가할 수 있습니다.
            # 예시: 미션이 한 번만 수행되어야 한다면 타이머를 멈추거나 노드를 종료
            # self.timer.cancel()
            # self.get_logger().info('미션이 완료되어 타이머를 정지합니다.')
        else:
            self.get_logger().info('⏳ 초록불 신호를 대기 중입니다...')

def main(args=None):
    rclpy.init(args=args)
    mission_node = TestMissionNode()
    try:
        rclpy.spin(mission_node)
    except KeyboardInterrupt:
        mission_node.get_logger().info('키보드 인터럽트로 노드를 종료합니다.')
    finally:
        mission_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()