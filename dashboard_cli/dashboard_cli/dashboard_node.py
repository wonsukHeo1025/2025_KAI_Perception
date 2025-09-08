#!/usr/bin/env python3
import time
from typing import Dict

import rclpy
from rclpy.node import Node

from .topics import TopicsManager
from .nodes import NodeWatcher
from .ui import render_dashboard, clear_screen
from .spec import TOPICS, EXPECTED_NODES, UI_REFRESH_HZ, SHOW_NODES


class DashboardNode(Node):
    def __init__(self):
        super().__init__('dashboard_cli')
        # Topics
        self.tm = TopicsManager(self)
        for t in TOPICS:
            self.tm.add_topic(t.name, t.topic, t.type, t.min_hz, t.checks)

        # Nodes
        self.node_watcher = NodeWatcher(self, EXPECTED_NODES)

        # Timers
        ui_period = 1.0 / max(1e-3, UI_REFRESH_HZ)
        self.create_timer(0.5, self._scan_nodes)
        self.create_timer(ui_period, self._update_ui)

        self.last_render = 0.0
        self.get_logger().info('Dashboard CLI started')

    def _scan_nodes(self):
        try:
            self.node_watcher.scan()
        except Exception as e:
            self.get_logger().warn(f'Node scan failed: {e}')

    def _update_ui(self):
        # split by module for UI convenience
        status = self.tm.get_status()
        perception_keys = ['cam1', 'cam2', 'lidar', 'cones_lidar', 'cones_fused', 'cones_ukf', 'odom']
        planning_keys = ['local_path', 'speed_profile']
        control_keys = ['steer_cmd', 'steer_fb', 'rpm_cmd', 'rpm_target', 'speed_fb', 'throttle']
        safety_keys = ['aeb']

        def pick(keys):
            return {k: status[k] for k in keys if k in status}

        perception = pick(perception_keys)
        planning = pick(planning_keys)
        control = pick(control_keys)
        safety = pick(safety_keys)
        nodes = self.node_watcher.status() if SHOW_NODES else {}

        clear_screen()
        print(render_dashboard(perception, planning, control, safety, nodes))


def main():
    rclpy.init()
    node = DashboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
