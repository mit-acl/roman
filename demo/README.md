# ROMAN Demo

Using `demo.py`, the full ROMAN pipeline can be run on a multi-robot dataset, or mapping, loop closures, or pose graph optimization can be separately run.

After following the [install instructions](../README.md/#install) and the [demo setup instructions](../README.md/#demo), the demo can be run with the following command:

```
cd roman
mkdir demo_output
python3 demo/demo.py \
    -r sparkal1 sparkal2 -e ROBOT \
    -p demo/params/demo \
    -o demo_output
```

## Custom Data

Up to this point, ROMAN has mostly been used to process ROS 1/2 data (agnostic of ROS version).
This documentation gives instructions for how to image data from a ROS bag and how to load pose data from a ROS bag or from a CSV file.
To run ROMAN on played back data or in real-time, use the `roman_ros` repository. 
If you are interested in loading in another type of data, the [ROMANMapRunner class](../roman/map/run.py) can be edited to load other data types using the `robotdatapy` [PoseData](https://github.com/mbpeterson70/robotdatapy/blob/main/robotdatapy/data/pose_data.py) and [ImgData](https://github.com/mbpeterson70/robotdatapy/blob/main/robotdatapy/data/img_data.py) interfaces.

To run ROMAN on your data, start by copying the [demo params directory](./params/demo).
Within [mapping.yaml](./params/demo/mapping.yaml), the `img_data` and `pose_data` fields will need to be edited to load your data.

### Loading ROS Image Data

Update the following subfields within the `img_data` field in your new `mapping.yaml` file.

```
path: <bag file path>
img_topic: <topic name>
cam_info_topic: <topic name>
depth_img_topic: <topic name>
depth_cam_info_topic: <topic name>
depth_compressed: <boolean>
```

### Loading ROS Pose Data

Update the following subfields within the `pose_data` field in your new `mapping.yaml` file.

```
path: <bag file path>
topic: <topic name of type Pose or Odometry>
time_tol: <time in seconds> # Pose data will not be interpolated if there is not a recorded pose within time_tol of a desired time
```

If the pose is recorded using `/tf` topic rather tha a Pose or Odometry message, then use the following:

```
path: <bag file path>
file_type: bag_tf
parent: <parent frame_id>
child: <child frame_id>
inv: <Whether the transformation from /tf should be inverted or not>
```

Additionally the following subfields should be provided if they are not identity: `T_camera_flu` and `T_odombase_camera`. These are used for determining (1) the pose of the a forward-left-up coordinate frame with respect to the camera frame and (2) the pose of the camera expressed in the odometry base frame. `T_camera_flu` is used to inform ROMAN roughly along which axis gravity will point, since ROMAN uses gravity direction to help guide loop closures. `T_odombase_camera` is necessary if the camera frame is not the child of the pose topic.

These two transformation subfields can be described with the following formats:

```
input_type: "string"
string: <"T_RDFFLU" or "T_FLURDF">
```

or 

```
input_type: "tf"
parent: <parent frame_id>
child: <child frame_id>
inv: <Whether the transformation from /tf_static should be inverted or not>
```