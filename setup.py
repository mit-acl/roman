from setuptools import setup, find_packages

# TODO: Don't forget to add an __init__.py in directory!

setup(
    name='segment_track',
    version='0.1.0',    
    description='Package for tracking FastSAM segments',
    url='url',
    author='Mason Peterson',
    author_email='masonbp@mit.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ],
)
