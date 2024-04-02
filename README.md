# `segment_track`

![demo](./media/tracking.png)

Object detection based segment tracking and 3D reconstruction using triangulation of image detections.

Frame to frame segment data association is computed using mask object detection mask similarity.

# Install

TODO: setup dependencies for easy install

For now, activate your favorite virtual environment and `cd` into this repo and run:

```
mkdir dependencies
cd dependencies
git clone git@github.com:mbpeterson70/plot_utils.git
git clone git@github.com:mbpeterson70/robot_utils.git
pip install ./plot_utils/
pip install ./robot_utils/
```

# Demo

1. Set the following environment variables (consider putting in `.bashrc` for running repeatedly):

```
export WEST_POINT_2023_PATH=<path>
export FASTSAM_WEIGHTS_PATH=<path>
```

2. `cd` into `./demo` and run `python3 ./demo.py -p <params file>`

