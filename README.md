# ROMAN

![demo](./media/tracking.png)

Open-set object mapping pipeline. 
Objects are detected using [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM).
3D point cloud observations are created by using FastSAM masks on depth images, which are then merged into object tracks in real-time, using voxel-based data association.

# Install

`cd` into this directory and `pip install .`

# Demo

1. Set the following environment variables (consider putting in `.bashrc` for running repeatedly):

```
export BAG_PATH=<path to Kimera Multi bag file>
export GT_PATH=<path to Kimera Multi ground truth csv file>
export FASTSAM_WEIGHTS_PATH=<path to FastSAM weights>
export ROBOT=<acl_jackal, acl_jackal2, sparkal1, sparkal2, hathor, thoth, apis, OR sobek>
```

2. `cd` into `./demo` and run `python3 ./demo.py -p ./params/kmd_gt.yaml -o <output prefix (without file extensions)`

---

This research is supported by Ford Motor Company, DSTA, ONR, and
ARL DCIST under Cooperative Agreement Number W911NF-17-2-0181.

