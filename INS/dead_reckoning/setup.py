from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dead_reckoning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user1',
    maintainer_email='kikiws70@gmail.com',
    description='Dead reckoning using IMU data with tf and path visualization',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'dead_reckoning_node = dead_reckoning.dead_reckoning_node:main',
            'imu_calibration = dead_reckoning.imu_calibration:main',
            'advanced_imu_calibration = dead_reckoning.improved_calibration:main',
        ],
    },
)
