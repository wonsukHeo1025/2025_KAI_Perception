#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
import numpy as np
from scipy.spatial.transform import Rotation

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Point, Quaternion, Vector3
import tf2_ros
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class ImuOdometryNode(Node):
    """
    Estimates robot odometry (odom -> os_sensor) based on IMU data
    and publishes the TF transform and Odometry message.
    Also publishes a static transform from map -> odom.
    """
    def __init__(self):
        super().__init__('imu_odometry_node')

        # --- Parameters ---
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'os_sensor')  # base_link 대신 os_sensor 사용
        # Assume IMU frame is relative to os_sensor
        self.declare_parameter('imu_frame', 'os_imu')
        self.declare_parameter('imu_topic', '/ouster/imu')
        self.declare_parameter('gravity', 9.80665) # Standard gravity
        self.declare_parameter('use_orientation_for_gravity', False) # 기본값은 False로 설정
        self.declare_parameter('apply_simple_gravity_compensation', True) # 기본적으로 활성화

        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.imu_frame = self.get_parameter('imu_frame').value
        imu_topic = self.get_parameter('imu_topic').value
        self.gravity_z = self.get_parameter('gravity').value
        self.use_orientation_for_gravity = self.get_parameter('use_orientation_for_gravity').value
        self.apply_simple_gravity_compensation = self.get_parameter('apply_simple_gravity_compensation').value

        # --- State Variables (Pose of os_sensor relative to odom_frame) ---
        self.last_time = None
        self.position = np.zeros(3, dtype=np.float64) # [x, y, z] in odom frame
        self.orientation = Rotation.identity()      # Rotation object (odom -> os_sensor)
        self.velocity = np.zeros(3, dtype=np.float64) # [vx, vy, vz] in odom frame

        # --- TF Broadcasters ---
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # --- QoS 프로필 설정 ---
        imu_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # --- Publishers ---
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        # --- Subscriber ---
        self.imu_sub = self.create_subscription(
            Imu,
            imu_topic,
            self._imu_callback,
            qos_profile=imu_qos
        )

        # --- Publish static map -> odom transform (identity) ---
        self._publish_static_transform()

        # --- imu_frame -> base_frame 변환 게시 ---
        self._publish_imu_sensor_transform()

        self.get_logger().info(f"IMU Odometry Node started. Publishing {self.odom_frame} -> {self.base_frame}")
        self.get_logger().info(f"Publishing static {self.map_frame} -> {self.odom_frame}")
        self.get_logger().info(f"Subscribing to IMU topic: {imu_topic}")
        self.get_logger().info(f"Using os_sensor as base frame (instead of base_link)")

    def _publish_static_transform(self):
        """Publishes the static identity transform from map to odom."""
        static_transform = TransformStamped()
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = self.map_frame
        static_transform.child_frame_id = self.odom_frame
        static_transform.transform.translation.x = 0.0
        static_transform.transform.translation.y = 0.0
        static_transform.transform.translation.z = 0.0
        static_transform.transform.rotation.x = 0.0
        static_transform.transform.rotation.y = 0.0
        static_transform.transform.rotation.z = 0.0
        static_transform.transform.rotation.w = 1.0
        self.tf_static_broadcaster.sendTransform(static_transform)
        self.get_logger().info(f"Published static transform {self.map_frame} -> {self.odom_frame}")
        self.get_logger().info(f"System initialized with map and odom at os_sensor origin")

    def _publish_imu_sensor_transform(self):
        """Publishes the static transform from os_sensor to os_imu."""
        imu_transform = TransformStamped()
        imu_transform.header.stamp = self.get_clock().now().to_msg()
        imu_transform.header.frame_id = self.base_frame  # os_sensor
        imu_transform.child_frame_id = self.imu_frame    # os_imu
        
        # 변환 행렬 적용 (mm를 m로 변환하여 위치 오프셋 적용)
        imu_transform.transform.translation.x = -0.006253   # -6.253mm
        imu_transform.transform.translation.y = 0.011775    # 11.775mm
        imu_transform.transform.translation.z = -0.007645   # -7.645mm
        
        # 회전은 없음 (단위 회전)
        imu_transform.transform.rotation.x = 0.0
        imu_transform.transform.rotation.y = 0.0
        imu_transform.transform.rotation.z = 0.0
        imu_transform.transform.rotation.w = 1.0
        
        self.tf_static_broadcaster.sendTransform(imu_transform)
        self.get_logger().info(f"Published static transform {self.base_frame} -> {self.imu_frame}")

    def _imu_callback(self, msg: Imu):
        """Processes IMU message to update odometry."""
        current_time = Time.from_msg(msg.header.stamp)

        if self.last_time is None:
            self.last_time = current_time
            return # Need dt for integration

        dt = (current_time - self.last_time).nanoseconds / 1e9

        if dt <= 0:
            self.get_logger().warn(f"Non-positive dt calculated ({dt:.4f}). Skipping IMU update.")
            # Don't update last_time if dt is invalid, wait for next valid message
            return
        elif dt > 0.5: # Check for large gaps
            self.get_logger().warn(f"Large dt ({dt:.3f}s). Odometry might jump.")
            # Reset velocity if dt is too large? Optional.
            # self.velocity = np.zeros(3)

        self.last_time = current_time

        # --- Extract IMU data ---
        # If IMU frame is not coincident with os_sensor, transform may be needed
        # For this example, we'll assume IMU data is already in os_sensor frame
        omega_base = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ], dtype=np.float64)

        accel_imu = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ], dtype=np.float64)

        # --- Update Orientation ---
        # Rotate using angular velocity in the base_frame
        delta_rotation = Rotation.from_rotvec(omega_base * dt)
        self.orientation = self.orientation * delta_rotation
        
        # 수정된 부분: normalize 대신 단위 쿼터니언으로 변환
        quat = self.orientation.as_quat()
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 0:
            quat = quat / quat_norm
            self.orientation = Rotation.from_quat(quat)

        # --- Update Position (with Gravity Compensation) ---
        accel_corrected_imu = accel_imu.copy() # Start with measured acceleration
        gravity_compensated = False

        # 1. Orientation 기반 중력 보정 (설정된 경우)
        if self.use_orientation_for_gravity:
            use_orientation = msg.orientation_covariance[0] != -1.0
            if use_orientation:
                q_world_imu = np.array([
                    msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
                ], dtype=np.float64)

                # Check if quaternion is valid (close to unit norm)
                if abs(np.linalg.norm(q_world_imu) - 1.0) < 0.1:
                    try:
                        # rotation_world_to_imu: Rotation that transforms vectors from world to IMU frame
                        rotation_world_to_imu = Rotation.from_quat(q_world_imu)

                        # Define gravity in the world (odom/map) frame (ENU assumed: Z is up)
                        gravity_world = np.array([0.0, 0.0, self.gravity_z], dtype=np.float64) # Z-up world

                        # Transform world gravity vector into the IMU frame
                        gravity_imu = rotation_world_to_imu.apply(gravity_world)

                        # Subtract gravity component from measured acceleration
                        accel_corrected_imu = accel_imu - gravity_imu
                        gravity_compensated = True
                        self.get_logger().debug("Gravity compensation applied using IMU orientation.")

                    except Exception as e:
                        self.get_logger().warn(f"Error during orientation-based gravity compensation: {e}")
                else:
                    self.get_logger().warn("IMU orientation quaternion norm deviates significantly from 1. Skipping orientation-based gravity.")
            else:
                self.get_logger().debug("IMU orientation covariance indicates invalid data. Skipping orientation-based gravity.")

        # 2. 단순 Z축 중력 보정 (설정된 경우 그리고 orientation 기반 보정이 실패했을 때)
        if not gravity_compensated and self.apply_simple_gravity_compensation:
            # 간단히 Z축 방향으로 중력 가속도 빼기
            # IMU 프레임에서 Z축이 어느 방향인지에 따라 부호 조정 필요
            accel_corrected_imu[2] += self.gravity_z  # 중력 방향이 음수 Z라면 더해줌
            self.get_logger().debug("Simple gravity compensation applied (fixed Z-axis correction).")
            gravity_compensated = True

        if not gravity_compensated:
            self.get_logger().warn("No gravity compensation applied - using raw acceleration!")

        # Rotate corrected acceleration from base/IMU frame to odom frame
        # self.orientation represents rotation from odom to base
        # So, apply() rotates from base to odom
        accel_odom = self.orientation.apply(accel_corrected_imu)

        # Integrate acceleration in the odom frame
        # Simple Euler integration:
        self.position = self.position + self.velocity * dt + 0.5 * accel_odom * dt**2
        self.velocity = self.velocity + accel_odom * dt

        # --- Publish TF (odom -> os_sensor) ---
        tfs = TransformStamped()
        tfs.header.stamp = current_time.to_msg()
        tfs.header.frame_id = self.odom_frame
        tfs.child_frame_id = self.base_frame  # Now os_sensor

        tfs.transform.translation.x = self.position[0]
        tfs.transform.translation.y = self.position[1]
        tfs.transform.translation.z = self.position[2]

        quat = self.orientation.as_quat() # [x, y, z, w]
        tfs.transform.rotation.x = quat[0]
        tfs.transform.rotation.y = quat[1]
        tfs.transform.rotation.z = quat[2]
        tfs.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(tfs)

        # --- Publish Odometry Message ---
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame  # Now os_sensor

        # Pose in odom frame
        odom_msg.pose.pose.position = Point(x=self.position[0], y=self.position[1], z=self.position[2])
        odom_msg.pose.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

        # Twist in os_sensor frame
        # Velocity is currently in odom frame, rotate it back to base_frame
        vel_base = self.orientation.apply(self.velocity, inverse=True)
        odom_msg.twist.twist.linear = Vector3(x=vel_base[0], y=vel_base[1], z=vel_base[2])
        odom_msg.twist.twist.angular = Vector3(x=omega_base[0], y=omega_base[1], z=omega_base[2]) # From IMU

        # --- Covariance (Placeholders - Fill with realistic values if known) ---
        # Example: High uncertainty, especially for position/velocity from pure IMU
        P_pos = 1.0; P_orient = 0.1; P_vel = 0.5; P_ang_vel = 0.05
        odom_msg.pose.covariance = [
            P_pos, 0.0,   0.0,   0.0,     0.0,     0.0,      # xx, xy, xz, xa, xy, xz
            0.0,   P_pos, 0.0,   0.0,     0.0,     0.0,      # yx, yy, yz, ya, yb, yc
            0.0,   0.0,   P_pos, 0.0,     0.0,     0.0,      # zx, zy, zz, za, zb, zc
            0.0,   0.0,   0.0,   P_orient,0.0,     0.0,      # ax, ay, az, aa, ab, ac
            0.0,   0.0,   0.0,   0.0,     P_orient,0.0,      # bx, by, bz, ba, bb, bc
            0.0,   0.0,   0.0,   0.0,     0.0,     P_orient  # cx, cy, cz, ca, cb, cc
        ]
        odom_msg.twist.covariance = [
            P_vel, 0.0,   0.0,      0.0,       0.0,       0.0,
            0.0,   P_vel, 0.0,      0.0,       0.0,       0.0,
            0.0,   0.0,   P_vel,    0.0,       0.0,       0.0,
            0.0,   0.0,   0.0,      P_ang_vel, 0.0,       0.0,
            0.0,   0.0,   0.0,      0.0,       P_ang_vel, 0.0,
            0.0,   0.0,   0.0,      0.0,       0.0,       P_ang_vel
        ]

        self.odom_pub.publish(odom_msg)

        # 주기적으로 현재 상태 로깅 (디버깅 용이)
        if self.last_time is not None and (current_time - self.last_time).nanoseconds % int(1e9) < 100000000:
            self.get_logger().debug(f"Position: {self.position}, Orientation: {quat}")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ImuOdometryNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node: 
            # 수정된 부분: include_traceback 옵션 제거하고 스택 트레이스 별도 출력
            node.get_logger().error(f"Unhandled exception: {e}")
            import traceback
            node.get_logger().error(traceback.format_exc())
        else: 
            print(f"Exception before node init: {e}")
    finally:
        if node: node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()