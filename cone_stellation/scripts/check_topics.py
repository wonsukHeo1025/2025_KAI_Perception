#!/usr/bin/env python3
"""
Quick script to check topic connections
"""
import rclpy
from rclpy.node import Node


class TopicChecker(Node):
    def __init__(self):
        super().__init__('topic_checker')
        
        self.timer = self.create_timer(2.0, self.check_topics)
        
    def check_topics(self):
        topics = self.get_topic_names_and_types()
        
        print("\n=== Active Topics ===")
        for topic, types in topics:
            print(f"{topic}: {types}")
            
        # Check specific topics
        cone_topics = [t for t, _ in topics if 'cone' in t.lower()]
        print(f"\n=== Cone-related topics: {cone_topics}")
        
        # Check publishers and subscribers
        print("\n=== Topic Info ===")
        for topic in ['/fused_sorted_cones_ukf_sim', '/lidar/cone_detection_cones']:
            pubs = self.get_publishers_info_by_topic(topic)
            subs = self.get_subscriptions_info_by_topic(topic)
            print(f"\n{topic}:")
            print(f"  Publishers: {len(pubs)}")
            print(f"  Subscribers: {len(subs)}")
            for sub in subs:
                print(f"    - {sub.node_name}/{sub.node_namespace}")


def main():
    rclpy.init()
    node = TopicChecker()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()