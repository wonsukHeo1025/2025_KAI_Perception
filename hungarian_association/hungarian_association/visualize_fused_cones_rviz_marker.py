#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from custom_interface.msg import ModifiedFloat32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA # For defining marker colors

# Define a mapping from class names (lowercase) to colors
# Adding flexibility for common variations like "red" vs "red cone"
COLOR_MAP = {
    "red":      ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
    "red cone": ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
    "yellow":   ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
    "yellow cone":ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
    "blue":     ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
    "blue cone":ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
    "unknown":  ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), # Green for Unknown
}
DEFAULT_COLOR = ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0) # Gray for unmapped classes

class FusedConeColorVisualizer(Node):
    """
    Subscribes to fused cone data with class names and publishes
    color-coded visualization markers for RViz2.
    Now supports 3D cone data (X, Y, Z coordinates).
    """
    def __init__(self, node_name='fused_cone_color_visualizer'):
        super().__init__(node_name)

        # --- Parameters ---
        self.declare_parameter('input_topic', '/fused_sorted_cones_ukf')
        self.declare_parameter('marker_topic', '/visualization_marker_fused_colored') # Distinct topic
        self.declare_parameter('marker_namespace', 'fused_cones_colored')
        self.declare_parameter('marker_scale', [0.35, 0.35, 0.35]) # x, y, z scale
        self.declare_parameter('marker_fixed_z', 0.3) # Fallback height for visualization if Z data is missing

        # --- Get Parameters ---
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        marker_topic = self.get_parameter('marker_topic').get_parameter_value().string_value
        self._marker_ns = self.get_parameter('marker_namespace').get_parameter_value().string_value
        marker_scale_list = self.get_parameter('marker_scale').get_parameter_value().double_array_value
        self._fixed_z = self.get_parameter('marker_fixed_z').get_parameter_value().double_value

        if len(marker_scale_list) != 3:
            self.get_logger().error("Param 'marker_scale' must have 3 values (x, y, z). Using default.")
            marker_scale_list = [0.35, 0.35, 0.35]
        self._marker_scale_x = marker_scale_list[0]
        self._marker_scale_y = marker_scale_list[1]
        self._marker_scale_z = marker_scale_list[2]

        # QoS 설정 추가
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # --- Subscriber ---
        self.subscription = self.create_subscription(
            ModifiedFloat32MultiArray,
            input_topic,
            self._cone_data_callback,
            qos_profile  # QoS 프로파일 적용
        )

        # --- Publisher ---
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, qos_profile)

        # --- State ---
        self._previous_marker_count = 0 # To keep track for deleting old markers

        self.get_logger().info(f"'{node_name}' started.")
        self.get_logger().info(f"Subscribing to: '{input_topic}'")
        self.get_logger().info(f"Publishing markers to: '{marker_topic}'")

    def _get_color_for_class(self, class_name: str) -> ColorRGBA:
        """Returns the corresponding color for a given class name."""
        return COLOR_MAP.get(class_name.lower(), DEFAULT_COLOR) # Use lower case for matching

    def _cone_data_callback(self, msg: ModifiedFloat32MultiArray):
        """
        Processes incoming cone data and publishes color-coded markers.
        Now supports 3D cone data (X, Y, Z).
        """
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg() # Use current time for markers

        cones_data = [] # Store tuples of (x, y, z, class_name)

        # --- 1. Parse the incoming message ---
        num_cones = 0
        stride = 0
        
        # --- MODIFIED: Check for 3D format (stride=3) ---
        if len(msg.layout.dim) == 2 and msg.layout.dim[1].size == 3:
            # New 3D format with [x, y, z] values per cone
            num_cones = msg.layout.dim[0].size
            stride = msg.layout.dim[1].stride # Should be 3 for [x, y, z]

            if num_cones != len(msg.class_names):
                self.get_logger().warn(
                    f"Mismatch between number of cones in layout ({num_cones}) "
                    f"and number of class names ({len(msg.class_names)}). "
                    f"Will use minimum count for visualization."
                )
                # Adjust num_cones to the minimum to avoid index errors
                num_cones = min(num_cones, len(msg.class_names))

            # Check if data array size is sufficient
            expected_data_len = num_cones * stride
            if expected_data_len > len(msg.data):
                 self.get_logger().error(
                     f"Data array size ({len(msg.data)}) is smaller than expected "
                     f"based on layout ({expected_data_len}). Cannot process message."
                 )
                 # Publish delete markers for safety and return
                 self._publish_delete_markers(msg.header.frame_id or "map", now)
                 return

            try:
                for i in range(num_cones):
                    idx_x = i * stride + 0
                    idx_y = i * stride + 1
                    idx_z = i * stride + 2  # New: Z coordinate index
                    x = msg.data[idx_x]
                    y = msg.data[idx_y]
                    z = msg.data[idx_z]  # Get actual Z coordinate
                    class_name = msg.class_names[i]
                    cones_data.append((x, y, z, class_name))  # Now includes Z

            except IndexError as e:
                 self.get_logger().error(f"Index Error while parsing 3D cone data: {e}. Check message layout/data consistency.")
                 self._publish_delete_markers(msg.header.frame_id or "map", now)
                 return
            except Exception as e:
                self.get_logger().error(f"Unexpected error parsing 3D cone data: {e}")
                self._publish_delete_markers(msg.header.frame_id or "map", now)
                return
                
        # --- Legacy 2D format support (stride=2) ---
        elif len(msg.layout.dim) == 2 and msg.layout.dim[1].size == 2:
            self.get_logger().debug("Received old 2D format. Using fixed_z for visualization.")
            num_cones = msg.layout.dim[0].size
            stride = msg.layout.dim[1].stride # Should be 2 for [x, y]
            
            if num_cones != len(msg.class_names):
                self.get_logger().warn(
                    f"Mismatch between number of cones in layout ({num_cones}) "
                    f"and number of class names ({len(msg.class_names)}). "
                    f"Will use minimum count for visualization."
                )
                # Adjust num_cones to the minimum to avoid index errors
                num_cones = min(num_cones, len(msg.class_names))

            # Check if data array size is sufficient
            expected_data_len = num_cones * stride
            if expected_data_len > len(msg.data):
                 self.get_logger().error(
                     f"Data array size ({len(msg.data)}) is smaller than expected "
                     f"based on layout ({expected_data_len}). Cannot process message."
                 )
                 # Publish delete markers for safety and return
                 self._publish_delete_markers(msg.header.frame_id or "map", now)
                 return

            try:
                for i in range(num_cones):
                    idx_x = i * stride + 0
                    idx_y = i * stride + 1
                    x = msg.data[idx_x]
                    y = msg.data[idx_y]
                    z = self._fixed_z  # Use fixed Z value for legacy format
                    class_name = msg.class_names[i]
                    cones_data.append((x, y, z, class_name))  # Still store as 3D with fixed Z

            except IndexError as e:
                 self.get_logger().error(f"Index Error while parsing 2D cone data: {e}.")
                 self._publish_delete_markers(msg.header.frame_id or "map", now)
                 return

        # --- Fallback: Assume flat list if layout is missing/empty ---
        elif len(msg.data) > 0 and len(msg.layout.dim) == 0:
             # Fallback 1: First try to interpret as flat [x,y,z,x,y,z,...] triplets
             if len(msg.data) % 3 == 0:
                 self.get_logger().debug("Message layout missing. Trying to interpret as [x,y,z] triplets.")
                 num_cones_from_data = len(msg.data) // 3
                 
                 if num_cones_from_data != len(msg.class_names):
                     self.get_logger().warn(
                        f"Fallback: Mismatch between cones in data ({num_cones_from_data}) "
                        f"and class names ({len(msg.class_names)}). Using minimum."
                     )
                     num_cones_to_process = min(num_cones_from_data, len(msg.class_names))
                 else:
                     num_cones_to_process = num_cones_from_data

                 for i in range(num_cones_to_process):
                      x = msg.data[i*3]
                      y = msg.data[i*3 + 1]
                      z = msg.data[i*3 + 2]
                      class_name = msg.class_names[i]
                      cones_data.append((x, y, z, class_name))
                      
             # Fallback 2: Try as flat [x,y,x,y,...] pairs with fixed Z
             elif len(msg.data) % 2 == 0:
                 self.get_logger().debug("Message layout missing. Interpreting as [x,y] pairs with fixed Z.")
                 num_cones_from_data = len(msg.data) // 2
                 
                 if num_cones_from_data != len(msg.class_names):
                     self.get_logger().warn(
                        f"Fallback: Mismatch between cones in data ({num_cones_from_data}) "
                        f"and class names ({len(msg.class_names)}). Using minimum."
                     )
                     num_cones_to_process = min(num_cones_from_data, len(msg.class_names))
                 else:
                     num_cones_to_process = num_cones_from_data

                 for i in range(num_cones_to_process):
                      x = msg.data[i*2]
                      y = msg.data[i*2 + 1]
                      z = self._fixed_z  # Use fixed Z for 2D data
                      class_name = msg.class_names[i]
                      cones_data.append((x, y, z, class_name))
             else:
                 self.get_logger().warn("Data length is not divisible by 2 or 3. Cannot form valid coordinates.")

        elif num_cones == 0 and len(msg.class_names) == 0:
             self.get_logger().debug("Received message with 0 cones.")
             # Proceed to delete old markers
        else:
            self.get_logger().warn(f"Received message with unexpected layout/data: {msg.layout}. Cannot parse.")
            # Proceed to delete old markers

        # --- 2. Create DELETE markers for previously published markers ---
        # This ensures that if the number of cones decreases or disappears,
        # the old ones are removed from RViz.
        frame_id = msg.header.frame_id if msg.header.frame_id else "map" # Default if empty
        for i in range(self._previous_marker_count):
            delete_marker = Marker()
            delete_marker.header.frame_id = frame_id
            delete_marker.header.stamp = now
            delete_marker.ns = self._marker_ns
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        # --- 3. Create ADD markers for current cones ---
        current_marker_count = 0
        for i, (x, y, z, class_name) in enumerate(cones_data):  # Now unpacks 4 values
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = now
            marker.ns = self._marker_ns
            marker.id = i # Use index as ID
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Position: Use x, y, z from data (not using fixed_z parameter anymore)
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = float(z)  # Use actual Z coordinate 
            marker.pose.orientation.w = 1.0 # No rotation needed for a sphere

            # Scale from parameters
            marker.scale.x = self._marker_scale_x
            marker.scale.y = self._marker_scale_y
            marker.scale.z = self._marker_scale_z

            # Color based on class name
            marker.color = self._get_color_for_class(class_name)
            if marker.color == DEFAULT_COLOR:
                self.get_logger().debug(f"Cone {i} with class '{class_name}' mapped to default color.")


            # Lifetime (optional, but DELETE action is generally more robust)
            # marker.lifetime = Duration(seconds=1.0).to_msg()

            marker_array.markers.append(marker)
            current_marker_count += 1

        # --- 4. Update the count of markers for the next callback ---
        self._previous_marker_count = current_marker_count

        # --- 5. Publish the MarkerArray ---
        if marker_array.markers: # Only publish if there's something to add or delete
            self.marker_pub.publish(marker_array)

    def _publish_delete_markers(self, frame_id: str, timestamp):
        """Helper function to publish only delete markers."""
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
        self._previous_marker_count = 0 # Reset count after deleting

    def destroy_node(self):
        """Cleanup before shutdown."""
        self.get_logger().info("Cleaning up markers...")
        now = self.get_clock().now().to_msg()
        # Publish one last time with only delete markers for all previously known IDs
        # Need a frame_id, try to get last known or default
        # A more robust way would be storing the last valid frame_id
        last_frame_id = "map" # Default fallback frame
        try:
            # This might fail if no message was ever received
            # In a real scenario, store the frame_id from the last valid message
            pass # Placeholder - ideally store last known frame_id
        except AttributeError:
             pass # Use default frame

        self._publish_delete_markers(last_frame_id, now)
        # Give a short moment for the publisher to send
        if self.context and self.context.ok():
             self.context.sleep_for(0.1)
        super().destroy_node()
        self.get_logger().info("Fused Cone Color Visualizer shut down.")

# --- Main Execution ---
def main(args=None):
    rclpy.init(args=args)
    visualizer_node = None # Initialize to None
    try:
        visualizer_node = FusedConeColorVisualizer()
        rclpy.spin(visualizer_node)
    except KeyboardInterrupt:
        if visualizer_node:
             visualizer_node.get_logger().info('Keyboard interrupt, shutting down.')
        else:
             print('Keyboard interrupt before node initialization.')
    except ImportError as e:
         # Catch import error again in main if it happened before node creation
         # (Though the initial check should prevent this)
         print(f"Import Error during execution: {e}")
    except Exception as e:
        # Catch any other unexpected errors during spin
        if visualizer_node:
            visualizer_node.get_logger().fatal(f"Unhandled exception: {e}", include_traceback=True)
        else:
            print(f"Unhandled exception before node initialization: {e}")
    finally:
        # Ensure cleanup happens even if spin loop is exited unexpectedly
        if visualizer_node and rclpy.ok():
             visualizer_node.destroy_node()
        if rclpy.ok():
             rclpy.shutdown()

if __name__ == '__main__':
    main()