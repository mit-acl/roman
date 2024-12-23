from setuptools import setup, find_packages

setup(
    name='roman',
    version='0.1.1',    
    description='Package for tracking FastSAM segments',
    url='url',
    author='Mason Peterson',
    author_email='masonbp@mit.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib', 
        'gtsam',
        'scikit-image',
        'open3d==0.18.0',
        'yolov7-package',
        'shapely',
        'opencv-python',
        'pyyaml',
        'fastsam @ git+ssh://git@github.com/CASIA-IVA-Lab/FastSAM@4d153e9',
        'robotdatapy>=1.0.0'
    ],
)

