from setuptools import setup, find_packages

project_description = """
ROMAN is a view-invariant global localization method that maps open-set objects and uses the 
geometry, shape, and semantics of objects to find the transformation between a current pose and 
previously created object map. This enables loop closure between robots even when a scene is 
observed from opposite views.
"""

setup(
    name='roman',
    version='0.1.2',    
    description=project_description,
    url='url',
    author='Mason Peterson, Lucas Jia, and Yulun Tian',
    author_email='masonbp@mit.edu, yixuany@mit.edu, yut034@ucsd.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy<2',
        'scipy',
        'matplotlib', 
        'gtsam',
        'scikit-image',
        'open3d>=0.18.0',
        'yolov7-package',
        'shapely',
        'opencv-python>=4.6.0',
        'pyyaml',
        'torch==2.4.0',
        'torchvision==0.19.0',
        'fastsam @ git+https://github.com/CASIA-IVA-Lab/FastSAM.git@4d153e9',
        'robotdatapy>=1.1.0',
        'gdown'
    ],
)

