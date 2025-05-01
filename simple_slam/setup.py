from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'simple_slam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'numpy',
        'scipy',
        'sensor_msgs',
        'nav_msgs',
        'geometry_msgs',
        'visualization_msgs',
        'std_msgs',
        'tf2_ros_py',
        'tf2_geometry_msgs',
        'tf2_msgs',
        'custom_interface',
        'message_filters',
    ],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Basic IMU Odometry and Cone Mapping',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'imu_odometry = simple_slam.imu_odometry_node:main',
            'cone_mapper = simple_slam.cone_mapper_node:main',
            'lio_node = simple_slam.lio_node:main',
        ],
    },
)