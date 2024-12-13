# ROMAN

<img src="./media/opposite_view_loop_closure.jpg" alt="Opposite view loop closure" width="500"/>


Welcome to ROMAN(<ins>R</ins>obust <ins>O</ins>bject <ins>M</ins>ap <ins>A</ins>lignment A<ins>n</ins>ywhere).
ROMAN is a view-invariant global localization method that maps open-set objects and uses the geometry, shape, and semantics of objects to find the transformation between a current pose and previously created object map.
This enables loop closure between robots even when a scene is observed from *opposite views.*

Included in this repository is code for open-set object mapping and object map registration using our robust data association algorithm.
For ROS1/2 integration, please see the [roman_ros](https://github.com/mit-acl/roman_ros) repo. Additionally, the ROMAN project website can be viewed [here](https://acl.mit.edu/ROMAN-project/).

## Citation

If you find ROMAN useful in your work, please cite our paper:

M.B. Peterson, Y.X. Jia, Y. Tian and J.P. How, "ROMAN: Open-Set Object Map Alignment for Robust View-Invariant Global Localization,"
*arXiv preprint arXiv:2410.08262*, 2024.

```
@article{peterson2024roman,
  title={ROMAN: Open-Set Object Map Alignment for Robust View-Invariant Global Localization},
  author={Peterson, Mason B and Jia, Yi Xuan and Tian, Yulun and Thomas, Annika and How, Jonathan P},
  journal={arXiv preprint arXiv:2410.08262},
  year={2024}
}
```

## System Overview

<img src="./media/system_diagram.png" alt="System diagram" width="500"/>

ROMAN has three modules: mapping, data association, and
pose graph optimization. The front-end mapping pipeline tracks
segments across RGB-D images to generate segment maps. The data
association module incorporates semantics and shape geometry attributes from submaps along with gravity as a prior into the ROMAN
alignment module to align maps and detect loop closures. These loop
closures and VIO are then used for pose graph optimization.

The `roman` package has a Python submodule corresponding to each pipeline module. Code for creating open-set object maps can be found in `roman.map`. Code for finding loop closures via data association of object maps can be found in `roman.align`. Finally, code for interfacing ROMAN with Kimera-RPGO for pose graph optimization can be found in `roman.offline_rpgo`.

## Dependencies

Direct dependencies, [CLIPPER](https://github.com/mit-acl/CLIPPER) and [Kimera-RPGO](https://github.com/MIT-SPARK/Kimera-RPGO) are installed with the install script. 
If you would like to use Kimera-RPGO with ROMAN (required for full demo), please also follow the [Kimera-RPGO dependency instructions](https://github.com/MIT-SPARK/Kimera-RPGO#Dependencies).

## Install

First, **activate any virtual environment you would like to use with ROMAN**.

Then, clone and install with:

```
git clone git@github.com:mit-acl/roman.git
./roman/install.sh
```

While mapping and loop closure can be performed standalone, we use Kimera-RPGO for pose graph optimization. 
We are working to make that publicly runnable in the future.

## Demo

A short demo is available to run ROMAN on small subset of the [Kimera Multi Data](https://github.com/MIT-SPARK/Kimera-Multi-Data).
The subset includes two robots (`sparkal1` and `sparkal2`) traveling along a path in opposite directions. 
We demonstrate ROMAN's open-set object mapping and object-based loop closure to estimate the two robots' cumulative 200 m long trajectories with 1.2 m RMSE absolute trajectory error using our vision-only pipeline.

Instructions for running the demo:

1. Download a small portion of the [Kimera Multi Data](https://github.com/MIT-SPARK/Kimera-Multi-Data) that is used for the ROMAN SLAM demo. The data subset is available for download [here](https://drive.google.com/drive/folders/1ANdi4IyroWzJmd85ap1V-IMF8-I9haUB?usp=sharing).

2. Download FastSAM model weights [here](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view).

3. In your `.bashrc` or in the terminal where you will run the ROMAN demo export the following environment variables: 

```
export ROMAN_DEMO_DATA=<path to the demo data>
export FASTSAM_WEIGHTS_PATH=<path to weights downloaded in step 2>
```

4. `cd` into this repo and run the following to start the demo

```
mkdir demo_output
python3 demo/demo.py \
    -r sparkal1 sparkal2 -e ROBOT \
    -p demo/params/demo \
    -o demo_output
```

Optionally, the mapping process can be visualized with the `-m` argument to show the map projected on the camera image as it is created or `-3` command to show a 3D visualization of the map.
However, these will cause the demo to run slower. 

The output includes map visualization, loop closure accuracy results, and pose graph optimization results including root mean squared absolute trajectory error. 

<!-- ![demo](./media/demo.mp4) -->

## Running on Custom Data

ROMAN requires RGB-D images and odometry information. ROMAN should be runnable on any data with this information, using [robotdatapy](https://github.com/mbpeterson70/robotdatapy) to interface [pose data](https://github.com/mbpeterson70/robotdatapy/blob/main/robotdatapy/data/pose_data.py) and [image data](https://github.com/mbpeterson70/robotdatapy/blob/main/robotdatapy/data/img_data.py). Currently supported data types include ROS1/2 bags, zip files of images, and csv files for poses, with additional data sources in development. 
Click [here](./demo/README.md/#custom-data) for more information on running on custom data.

## Acknowledgements

This research is supported by Ford Motor Company, DSTA, ONR, and
ARL DCIST under Cooperative Agreement Number W911NF-17-2-0181.

Additional thanks to Yun Chang for his help in interfacing with Kimera-RPGO.
