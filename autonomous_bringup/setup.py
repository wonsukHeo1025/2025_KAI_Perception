from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'autonomous_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch 디렉토리 안의 모든 .launch.py 파일을 설치
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        # config 디렉토리 안의 모든 .yaml 파일을 설치 (config 디렉토리를 사용한다면)
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Launch package for the autonomous vehicle system',
    license='Apache License 2.0', # 혹은 원하는 라이선스
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 만약 이 패키지에 실행할 스크립트가 있다면 여기에 추가
            # 예: 'my_script = autonomous_bringup.my_script:main'
        ],
    },
)