from setuptools import find_packages
from setuptools import setup

setup(
    name='cone_detection',
    version='0.1.0',
    packages=find_packages(
        include=('cone_detection', 'cone_detection.*')),
)
