# YOLO Model Configuration Guide

## Model Path Configuration

The YOLO ROS package now supports flexible model path configuration through multiple methods:

### 1. Environment Variable (Recommended for Production)

Set the `YOLO_MODEL_PATH` environment variable to specify the model location:

```bash
export YOLO_MODEL_PATH=/path/to/your/model.pt
```

You can add this to your `.bashrc` or `.env` file for persistence.

### 2. Package-Relative Path (Default)

If no environment variable is set, the package will look for models in the following order:
1. `<package_root>/best.pt` - Main package directory
2. `<package_root>/yolo_ros/best.pt` - Alternative location

### 3. Launch File Parameter

When using launch files, you can specify the model path as a launch argument:

```bash
ros2 launch yolo_bringup yolo.launch.py model:=/path/to/your/model.pt
```

## Available Models

Currently, the package includes:
- `best.pt` - Pre-trained YOLO model for cone detection (Blue, Yellow, Crimson cones)

## Python Node Configuration

The following Python nodes have been updated to use portable path handling:
- `detect_publish.py` - Main detection node
- `detect_publish_hsv.py` - Detection with HSV color validation
- `detect_publish_ycbcr.py` - Detection with YCbCr color validation

Each node will:
1. First check for the `YOLO_MODEL_PATH` environment variable
2. If not found, use the package-relative `best.pt` file
3. Log the model path being used for transparency

## Example Usage

### Using Environment Variable
```bash
export YOLO_MODEL_PATH=/home/user/models/yolov8_custom.pt
ros2 run yolo_ros detect_publish
```

### Using Launch File
```bash
ros2 launch yolo_bringup yolo.launch.py model:=/home/user/models/yolov8_custom.pt
```

### Using Default Package Model
```bash
# No configuration needed - will use best.pt from package
ros2 run yolo_ros detect_publish
```