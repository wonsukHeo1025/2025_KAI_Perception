from setuptools import setup

package_name = "yolo_ros"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Miguel Ángel González Santamarta",
    maintainer_email="mgons@unileon.es",
    description="YOLO for ROS 2",
    license="GPL-3",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "yolo_node = yolo_ros.yolo_node:main",
            "debug_node = yolo_ros.debug_node:main",
            "tracking_node = yolo_ros.tracking_node:main",
            "detect_3d_node = yolo_ros.detect_3d_node:main",
            "detect_publish = yolo_ros.detect_publish:main",
            "detect_publish_ycbcr = yolo_ros.detect_publish_ycbcr:main",
            "detect_publish_hsv = yolo_ros.detect_publish_hsv:main",
            "yolo_debug_node = yolo_ros.yolo_debug_node:main",
        ],
    },
)
