from dataclasses import dataclass
from typing import List, Tuple
import rclpy
from rclpy.node import Node


@dataclass
class NodeStatus:
    present: bool


class NodeWatcher:
    def __init__(self, node: Node, expected: List[str]):
        self.node = node
        self.expected = expected
        self.present_set = set()  # cache from last scan

    def scan(self) -> None:
        names: List[Tuple[str, str]] = self.node.get_node_names_and_namespaces()
        # names is list of (name, ns) or list of names depending on rclpy version
        found = set()
        try:
            for item in names:
                if isinstance(item, tuple):
                    found.add(item[0])
                else:
                    found.add(item)
        except Exception:
            # Fallback for mixed versions
            found = set([str(x) for x in names])
        self.present_set = found

    def status(self) -> dict:
        return {name: NodeStatus(present=(name in self.present_set)) for name in self.expected}

