# /sorted_cones_time Migration Plan: ModifiedFloat32MultiArray → TrackedConeArray

## Overview
This document outlines the plan to migrate `/sorted_cones_time` topic from `ModifiedFloat32MultiArray` to `TrackedConeArray` for better consistency with the rest of the system.

## Current Implementation
- **Topic**: `/sorted_cones_time`
- **Message Type**: `custom_interface/msg/ModifiedFloat32MultiArray`
- **Publisher**: `OutlierFilter::publishArrayMsg()`
- **Data Format**: 2D array where each row contains [x, y, z, ...] for each cone

## Target Implementation
- **Topic**: `/sorted_cones_time` (same name for backward compatibility)
- **Message Type**: `custom_interface/msg/TrackedConeArray`
- **Publisher**: New function `OutlierFilter::publishTrackedConeArray()`
- **Benefits**:
  - Consistent with `/cones/lidar/ukf` 
  - Easier frame transformations
  - Better type safety
  - Supports color and track_id from the start

## Migration Steps

### Step 1: Add New Publisher Function
Create a new function in `cone_detection_node.cpp`:

```cpp
void OutlierFilter::publishTrackedConeArray(
    const rclcpp::Publisher<custom_interface::msg::TrackedConeArray>::SharedPtr &publisher,
    const std::vector<ConeDescriptor> &cones,
    const rclcpp::Time &timestamp,
    const std::string &frame_id) {
    
    if (!publisher || cones.empty()) {
        return;
    }
    
    try {
        if (publisher->get_subscription_count() > 0) {
            custom_interface::msg::TrackedConeArray msg;
            msg.header.stamp = timestamp;
            msg.header.frame_id = frame_id;
            
            // Convert each ConeDescriptor to TrackedCone
            for (size_t i = 0; i < cones.size(); ++i) {
                custom_interface::msg::TrackedCone cone;
                
                // Position
                cone.position.x = cones[i].x;
                cone.position.y = cones[i].y;
                cone.position.z = cones[i].z;
                
                // Color (default "unknown" - no color info at this stage)
                cone.color = "unknown";
                
                // Track ID (use index for now, no tracking at this stage)
                cone.track_id = static_cast<int32_t>(i);
                
                // Velocity (zero - no tracking at this stage)
                cone.velocity.x = 0.0;
                cone.velocity.y = 0.0;
                cone.velocity.z = 0.0;
                
                // Confidence (1.0 - all cones passed validation)
                cone.confidence = 1.0;
                
                msg.cones.push_back(cone);
            }
            
            publisher->publish(msg);
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in publishTrackedConeArray: %s", e.what());
    }
}
```

### Step 2: Update Publisher Declaration
In `cone_detection_node.h`:
```cpp
// Change from:
rclcpp::Publisher<custom_interface::msg::ModifiedFloat32MultiArray>::SharedPtr cones_time_pub;

// To:
rclcpp::Publisher<custom_interface::msg::TrackedConeArray>::SharedPtr cones_time_pub;
```

### Step 3: Update Publisher Creation
In constructor:
```cpp
// Change from:
cones_time_pub = this->create_publisher<custom_interface::msg::ModifiedFloat32MultiArray>("/sorted_cones_time", 10);

// To:
cones_time_pub = this->create_publisher<custom_interface::msg::TrackedConeArray>("/sorted_cones_time", 10);
```

### Step 4: Update Publishing Calls
Replace all calls to `publishArrayMsg` for `cones_time_pub` with `publishTrackedConeArray`:
```cpp
// Change from:
publishArrayMsg(cones_time_pub, output_array, msg->header.stamp, "os_sensor");

// To:
publishTrackedConeArray(cones_time_pub, validated_cones, msg->header.stamp, "os_sensor");
```

### Step 5: Update Documentation
- Update README.md to reflect new message type
- Update any dependent packages' documentation

## Testing Plan
1. Build the package with changes
2. Run with existing bag files to verify output
3. Check that downstream consumers can handle the new format
4. Verify frame_transformer.py works with both old and new formats

## Rollback Plan
If issues arise:
1. The frame_transformer.py already supports both formats
2. Can quickly revert changes by switching back to ModifiedFloat32MultiArray
3. Downstream consumers should be updated to handle both formats during transition

## Future Enhancements
Once migration is complete:
1. Remove ModifiedFloat32MultiArray support from frame_transformer.py
2. Consider adding actual color information if available from intensity
3. Consider adding preliminary tracking IDs based on spatial proximity