from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'hungarian_association'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user1',
    maintainer_email='kikiws70@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'hungarian_association_node = hungarian_association.yolo_lidar_fusion:main',
            'kalman_filtering_node = hungarian_association.kalman_filtering:main',
            'visualize_fused_cones_rviz_marker_node = hungarian_association.visualize_fused_cones_rviz_marker:main',
            'yolo_lidar_multicam_fusion_node = hungarian_association.yolo_lidar_multicam_fusion:main'
        ],
    },
)
