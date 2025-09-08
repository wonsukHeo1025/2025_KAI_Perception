from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass
import time

import rclpy
from rclpy.node import Node

from .metrics import TopicStats

# Message imports (guarded where not installed)
from std_msgs.msg import Float32, Float32MultiArray, Int32, UInt8
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import Marker, MarkerArray

try:
    from yolo_msgs.msg import DetectionArray  # type: ignore
except Exception:  # pragma: no cover
    DetectionArray = None  # type: ignore
try:
    from custom_interface.msg import TrackedConeArray  # type: ignore
except Exception:  # pragma: no cover
    TrackedConeArray = None  # type: ignore
try:
    from throttle_msgs.msg import ThrottleData  # type: ignore
except Exception:  # pragma: no cover
    ThrottleData = None  # type: ignore


_TYPE_MAP = {
    'std_msgs/msg/Float32': Float32,
    'std_msgs/msg/Float32MultiArray': Float32MultiArray,
    'std_msgs/msg/Int32': Int32,
    'std_msgs/msg/UInt8': UInt8,
    'sensor_msgs/msg/Image': Image,
    'sensor_msgs/msg/PointCloud2': PointCloud2,
    'nav_msgs/msg/Path': Path,
    'nav_msgs/msg/Odometry': Odometry,
    'visualization_msgs/msg/Marker': Marker,
    'visualization_msgs/msg/MarkerArray': MarkerArray,
    'yolo_msgs/msg/DetectionArray': DetectionArray,
    'custom_interface/msg/TrackedConeArray': TrackedConeArray,
    'throttle_msgs/msg/ThrottleData': ThrottleData,
}


@dataclass
class TopicMonitor:
    name: str
    topic: str
    type_str: str
    min_hz: float
    checks: Dict[str, Any]
    stats: TopicStats

    # computed values (for UI)
    extra_value: Optional[str] = None


class TopicsManager:
    def __init__(self, node: Node):
        self.node = node
        self.monitors: Dict[str, TopicMonitor] = {}
        self.subscriptions = []

    def _resolve_type(self, type_str: str):
        t = _TYPE_MAP.get(type_str)
        if t is None:
            self.node.get_logger().warn(f"Message type '{type_str}' not available; skipping {type_str}")
        return t

    def add_topic(self, name: str, topic: str, type_str: str, min_hz: float, checks: Optional[Dict[str, Any]] = None):
        checks = checks or {}
        msg_type = self._resolve_type(type_str)
        if msg_type is None:
            return
        mon = TopicMonitor(name=name, topic=topic, type_str=type_str, min_hz=float(min_hz), checks=checks, stats=TopicStats())
        self.monitors[name] = mon

        def callback(msg, mon=mon):
            # Update stats and derive extra metrics per message type
            mon.stats.add_event(data=msg)
            mon.extra_value = self._extract_value(mon, msg)

        sub = self.node.create_subscription(msg_type, topic, callback, 10)
        self.subscriptions.append(sub)

    def _extract_value(self, mon: TopicMonitor, msg: Any) -> Optional[str]:
        # Provide compact summary per topic for UI
        try:
            if isinstance(msg, PointCloud2):
                pts = int(msg.width) * int(msg.height)
                return f"{pts} pts"
            if TrackedConeArray is not None and isinstance(msg, TrackedConeArray):
                return f"{len(msg.cones)} cones"
            if isinstance(msg, Float32):
                return f"{msg.data:.2f}"
            if isinstance(msg, Int32):
                return str(msg.data)
            if isinstance(msg, Float32MultiArray):
                return f"{len(msg.data)} vals"
            if DetectionArray is not None and isinstance(msg, DetectionArray):
                return f"{len(msg.detections)} dets"
            if isinstance(msg, Path):
                return f"{len(msg.poses)} pts"
        except Exception:
            return None
        return None

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        now = time.time()
        for name, mon in self.monitors.items():
            hz = mon.stats.hz
            age = mon.stats.age
            ok = True
            if mon.min_hz > 0:
                ok = ok and (hz >= mon.min_hz * 0.8)  # allow some slack
            # extra checks
            if 'min_points' in mon.checks and isinstance(mon.stats.last_data, PointCloud2):
                pts = int(mon.stats.last_data.width) * int(mon.stats.last_data.height)
                ok = ok and (pts >= int(mon.checks['min_points']))

            out[name] = {
                'topic': mon.topic,
                'type': mon.type_str,
                'hz': hz,
                'age': age,
                'ok': ok,
                'value': mon.extra_value,
            }
        return out

