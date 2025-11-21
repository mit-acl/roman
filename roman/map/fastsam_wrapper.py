#########################################
# 
# fastsam_wrapper.py
#
# A Python wrapper for sending RGBD images to FastSAM and using segmentation 
# masks to create object observations.
# 
# Authors: Jouko Kinnari, Mason Peterson, Lucas Jia, Annika Thomas, Qingyuan Li
# 
# Dec. 21, 2024
#
#########################################


import cv2 as cv
import numpy as np
from numpy.typing import ArrayLike
import open3d as o3d
import copy
import torch
from yolov7_package import Yolov7Detector
import math
import time
from PIL import Image
from fastsam import FastSAMPrompt
from fastsam import FastSAM
import clip
import logging
from transformers import AutoImageProcessor, AutoModel

from robotdatapy.camera import CameraParams

from roman.map.observation import Observation
from roman.params.fastsam_params import FastSAMParams
from roman.utils import expandvars_recursive
from roman.viz import viz_pointcloud_on_img

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)

class FastSAMWrapper():

    def __init__(self, 
        weights, 
        conf=.5, 
        iou=.9,
        imgsz=(1024, 1024),
        device='cuda',
        mask_downsample_factor=1,
        rotate_img=None,
        use_pointcloud=False
    ):
        """Wrapper for running FastSAM on images (RGB/depth data)

        Args:
            weights (str): Path to FastSAM weights.
            conf (float, optional): FastSAM confidence threshold. Defaults to .5.
            iou (float, optional): FastSAM IOU threshold. Defaults to .9.
            imgsz (tuple, optional): Image size to feed into FastSAM. Defaults to (1024, 1024).
            device (str, optional): 'cuda' or 'cpu. Defaults to 'cuda'.
            mask_downsample_factor (int, optional): For creating smaller data observations. 
                Defaults to 1.
            rotate_img (_type_, optional): 'CW', 'CCW', or '180' for rotating image before 
                feeding into FastSAM. Defaults to None.
            use_pointcloud (bool, optional): True if depth data source is pointcloud
        """
        # parameters
        self.weights = weights
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self.mask_downsample_factor = mask_downsample_factor
        self.rotate_img = rotate_img
        self.use_pointcloud = use_pointcloud

        # member variables
        self.observations = []
        self.model = FastSAM(weights)
        # setup default filtering
        self.setup_filtering()

        assert self.device == 'cuda' or self.device == 'cpu', "Device should be 'cuda' or 'cpu'."
        assert self.rotate_img is None or self.rotate_img == 'CW' or self.rotate_img == 'CCW' \
            or self.rotate_img == '180', "Invalid rotate_img option."
            
    @classmethod
    def from_params(cls, params: FastSAMParams, depth_cam_params: CameraParams):
        fastsam = cls(
            weights=expandvars_recursive(params.weights_path),
            imgsz=params.imgsz,
            device=params.device,
            mask_downsample_factor=params.mask_downsample_factor,
            rotate_img=params.rotate_img,
            use_pointcloud=params.use_pointcloud,
            conf=params.conf,
            iou=params.iou
        )
        fastsam.setup_rgbd_params(
            depth_cam_params=depth_cam_params, 
            max_depth=params.max_depth,
            depth_scale=params.depth_scale,
            voxel_size=params.voxel_size,
            erosion_size=params.erosion_size,
            plane_filter_params=params.plane_filter_params
        )

        img_area = depth_cam_params.width * depth_cam_params.height
        fastsam.setup_filtering(
            ignore_labels=params.ignore_labels,
            use_keep_labels=params.use_keep_labels,
            keep_labels=params.keep_labels,
            keep_labels_option=params.keep_labels_option,
            yolo_weights=expandvars_recursive(params.yolo_weights_path),
            yolo_det_img_size=params.yolo_imgsz,
            allow_tblr_edges=[True, True, True, True],
            area_bounds=[img_area / (params.min_mask_len_div**2), img_area / (params.max_mask_len_div**2)],
            semantics=params.semantics,
            frame_descriptor=params.frame_descriptor,
            triangle_ignore_masks=params.triangle_ignore_masks
        )

        return fastsam
            
    def setup_filtering(self,
        ignore_labels = [],
        use_keep_labels=False,
        keep_labels = [],
        keep_labels_option='intersect',          
        yolo_weights=None,
        yolo_det_img_size=None,
        area_bounds=np.array([0, np.inf]),
        allow_tblr_edges = [True, True, True, True],
        keep_mask_minimal_intersection=0.3,
        semantics: str = None,
        frame_descriptor: str = None,
        triangle_ignore_masks=None
    ):
        """
        Filtering setup function

        Args:
            ignore_labels (list, optional): List of yolo labels to ignore masks. Defaults to [].
            use_keep_labels (bool, optional): Use list of labels to only keep masks within keep mask. Defaults to False.
            keep_labels (list, optional): List of yolo labels to keep masks. Defaults to [].
            keep_labels_option (str, optional): 'intersect' or 'contain'. Defaults to 'intersect'.
            yolo_det_img_size (List[int], optional): Two-item list denoting yolo image size. Defaults to None.
            area_bounds (np.array, shape=(2,), optional): Two element array indicating min and max number of pixels. Defaults to np.array([0, np.inf]).
            allow_tblr_edges (list, optional): Allow masks touching top, bottom, left, and right edge. Defaults to [True, True, True, True].
            keep_mask_minimal_intersection (float, optional): Minimal intersection of mask within keep mask to be kept. Defaults to 0.3.
        """
        assert not use_keep_labels or keep_labels_option == 'intersect' or keep_labels_option == 'contain', "Keep labels option should be one of: intersect, contain"
        self.ignore_labels = ignore_labels
        self.use_keep_labels = use_keep_labels
        self.keep_labels = keep_labels
        self.keep_labels_option=keep_labels_option
        if len(ignore_labels) > 0 or use_keep_labels:
            if yolo_det_img_size is None:
                yolo_det_img_size=self.imgsz
            self.yolov7_det = Yolov7Detector(traced=False, img_size=yolo_det_img_size, weights=yolo_weights)
        
        self.area_bounds = area_bounds
        self.allow_tblr_edges= allow_tblr_edges
        self.keep_mask_minimal_intersection = keep_mask_minimal_intersection
        self.run_yolo = len(ignore_labels) > 0 or use_keep_labels
        self.semantics = semantics
        if semantics is None or semantics.lower() == 'none':
            self.semantics_model = None
            self.semantics_preprocess = None
        elif semantics.lower() == 'clip':
            clip_model = 'ViT-L/14'
            self.semantics_model, self.semantics_preprocess = clip.load(clip_model, device=self.device)
        elif semantics.lower() == 'dino':
            self.semantics_preprocess = AutoImageProcessor.from_pretrained('facebook/dinov2-base', do_center_crop=False)
            self.semantics_model = AutoModel.from_pretrained('facebook/dinov2-base')
            self.semantics_model.eval()
            self.semantics_model.to(self.device)
        else:
            raise ValueError(f"Invalid semantics option: {semantics}. Choose from 'clip', 'dino', or 'none'.")
        self.semantic_patches_shape = None
        self.frame_descriptor_type = frame_descriptor
        if frame_descriptor is not None:
            assert self.semantics.lower() == 'dino', "Frame descriptor only supported with DINO semantics."
        
        if triangle_ignore_masks is not None:
            self.constant_ignore_mask = np.zeros((self.depth_cam_params.height, self.depth_cam_params.width), dtype=np.uint8)
            for triangle in triangle_ignore_masks:
                assert len(triangle) == 3, "Triangle must have 3 points."
                for pt in triangle:
                    assert len(pt) == 2, "Each point must have 2 coordinates."
                    assert all([isinstance(x, int) for x in pt]), "Coordinates must be integers."
                cv.fillPoly(self.constant_ignore_mask, [np.array(triangle)], 1)
            self.constant_ignore_mask = self.apply_rotation(self.constant_ignore_mask)
        else:
            self.constant_ignore_mask = None
            
    def setup_rgbd_params(
        self, 
        depth_cam_params, 
        max_depth, 
        depth_scale=1e3,
        voxel_size=0.05, 
        within_depth_frac=0.25, 
        pcd_stride=4,
        erosion_size=0,
        plane_filter_params=None,
    ):
        """Setup params for processing RGB-D depth measurements

        Args:
            depth_cam_params (CameraParams): parameters of depth camera
            max_depth (float): maximum depth to be included in point cloud
            depth_scale (float, optional): scale of depth image. Defaults to 1e3.
            voxel_size (float, optional): Voxel size when downsampling point cloud. Defaults to 0.05.
            within_depth_frac(float, optional): Fraction of points that must be within max_depth. Defaults to 0.5.
            pcd_stride (int, optional): Stride for downsampling point cloud. Defaults to 4.
            plane_filter_params (List[float], optional): If an object's oriented bounding box's extent from max to min is > > <, mask is rejected. Defaults to None.
        """
        self.depth_cam_params = depth_cam_params
        self.max_depth = max_depth
        self.within_depth_frac = within_depth_frac
        self.depth_scale = depth_scale
        if not self.use_pointcloud:
            self.depth_cam_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                width=int(depth_cam_params.width),
                height=int(depth_cam_params.height),
                fx=depth_cam_params.fx,
                fy=depth_cam_params.fy,
                cx=depth_cam_params.cx,
                cy=depth_cam_params.cy,
            )
        self.voxel_size = voxel_size
        self.pcd_stride = pcd_stride
        if erosion_size > 0:
            # see: https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
            erosion_shape = cv.MORPH_ELLIPSE
            self.erosion_element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                (erosion_size, erosion_size))
        else:
            self.erosion_element = None
        self.plane_filter_params = plane_filter_params

    def run(self, t, pose, img, depth_data=None):
        """
        Takes and image and returns filtered FastSAM masks as Observations.

        Args:
            img (cv image): camera image

        Returns:
            self.observations (list): list of Observations
            frame_descriptor (np.ndarray): semantic descriptor of the frame if frame_descriptor is not None, else None
        """
        self.observations = []
        
        # rotate image
        img_orig = img
        img = self.apply_rotation(img)

        if self.use_pointcloud:
            pcl, pcl_proj = depth_data

        if self.run_yolo:
            ignore_mask, keep_mask = self._create_mask(img)
        else:
            ignore_mask = None
            keep_mask = None

        if self.constant_ignore_mask is not None:
            ignore_mask = np.bitwise_or(ignore_mask, self.constant_ignore_mask) \
                if ignore_mask is not None else self.constant_ignore_mask  
        
        # run fastsam
        masks = self._process_img(img, ignore_mask=ignore_mask, keep_mask=keep_mask)
                
        if self.semantics == 'dino':
            # Process the image for DINO
            dino_shape = 768
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            preprocessed = self.semantics_preprocess(images=img_rgb, return_tensors="pt").to(self.device)
            dino_output = self.semantics_model(**preprocessed)
            dino_output_patches = self.get_output_patches(
                model_output=dino_output.last_hidden_state, 
                img_shape=img.shape, 
                feature_dim=dino_shape
            )
            dino_features = self.get_per_pixel_features(
                model_output_patches=dino_output_patches,
                img_shape=img.shape
            )
            
        frame_descriptor = None
        if self.frame_descriptor_type is not None:
            frame_descriptor = self.get_frame_descriptor(dino_output_patches)
        
        for mask in masks:
            
            mask = self.unapply_rotation(mask)

            # Extract point cloud of object from RGBD
            ptcld = None
            if depth_data is not None:
                if self.use_pointcloud:

                    # get 3D points that project within the mask
                    inside_mask = mask[pcl_proj[:, 1], pcl_proj[:, 0]] == 1
                    inside_mask_points = pcl[inside_mask]
                    pre_truncate_len = len(inside_mask_points)
                    ptcld_test = inside_mask_points[inside_mask_points[:, 2] < self.max_depth]

                    if len(ptcld_test) < self.within_depth_frac*pre_truncate_len:
                        continue

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(inside_mask_points)
                    
                else:
                    depth_obj = copy.deepcopy(depth_data)
                    if self.erosion_element is not None:
                        eroded_mask = cv.erode(mask, self.erosion_element)
                        depth_obj[eroded_mask==0] = 0
                    else:
                        depth_obj[mask==0] = 0
                    logger.debug(f"img_depth type {depth_data.dtype}, shape={depth_data.shape}")

                    # Extract point cloud without truncation to heuristically check if enough of the object
                    # is within the max depth
                    pcd_test = o3d.geometry.PointCloud.create_from_depth_image(
                        o3d.geometry.Image(np.ascontiguousarray(depth_obj).astype(np.dtype(depth_obj.dtype).type)),
                        self.depth_cam_intrinsics,
                        depth_scale=self.depth_scale,
                        # depth_trunc=self.max_depth,
                        stride=self.pcd_stride,
                        project_valid_depth_only=True
                    )
                    ptcld_test = np.asarray(pcd_test.points)
                    pre_truncate_len = len(ptcld_test)
                    ptcld_test = ptcld_test[ptcld_test[:,2] < self.max_depth]
                    # require some fraction of the points to be within the max depth
                    if len(ptcld_test) < self.within_depth_frac*pre_truncate_len:
                        continue
                    
                    pcd = o3d.geometry.PointCloud.create_from_depth_image(
                        o3d.geometry.Image(np.ascontiguousarray(depth_obj).astype(np.dtype(depth_obj.dtype).type)),
                        self.depth_cam_intrinsics,
                        depth_scale=self.depth_scale,
                        depth_trunc=self.max_depth,
                        stride=self.pcd_stride,
                        project_valid_depth_only=True
                    )

                # shared for depth & rangesens, once PointCloud object is created

                pcd.remove_non_finite_points()
                pcd_sampled = pcd.voxel_down_sample(voxel_size=self.voxel_size)
                if not pcd_sampled.is_empty():
                    ptcld = np.asarray(pcd_sampled.points)
                if ptcld is None:
                    continue
                
                if self.plane_filter_params is not None:
                    # Create oriented bounding box
                    try:
                        obb = o3d.geometry.OrientedBoundingBox.create_from_points(
                                o3d.utility.Vector3dVector(ptcld))
                        extent = np.sort(obb.extent)[::-1] # in descending order
                        if  extent[0] > self.plane_filter_params[0] and \
                            extent[1] > self.plane_filter_params[1] and \
                            extent[2] < self.plane_filter_params[2]:
                                continue
                    except:
                        continue

            # Generate downsampled mask
            mask_downsampled = np.array(cv.resize(
                mask,
                (mask.shape[1]//self.mask_downsample_factor, mask.shape[0]//self.mask_downsample_factor), 
                interpolation=cv.INTER_NEAREST
            )).astype('uint8')

            if self.semantics == 'clip':
                ### Use bounding box
                bbox = self.mask_bounding_box(mask.astype('uint8'))
                if bbox is None:
                    assert False, "Bounding box is None."
                    self.observations.append(Observation(t, pose, mask, mask_downsampled, ptcld))
                else:
                    min_col, min_row, max_col, max_row = bbox
                    img_bbox = self.apply_rotation(img_orig[min_row:max_row, min_col:max_col])
                    img_bbox = cv.cvtColor(img_bbox, cv.COLOR_BGR2RGB)
                    processed_img = self.semantics_preprocess(Image.fromarray(img_bbox, mode='RGB')).to(self.device)
                    clip_embedding = self.semantics_model.encode_image(processed_img.unsqueeze(dim=0))
                    clip_embedding = clip_embedding.squeeze().cpu().detach().numpy()
                    self.observations.append(Observation(t, pose, mask, mask_downsampled, ptcld, semantic_descriptor=clip_embedding))
            elif self.semantics == 'dino':
                assert mask.shape[0] == dino_features.shape[0] and mask.shape[1] == dino_features.shape[1], \
                    "Mask and DINO features must have the same shape."
                dino_mask = dino_features[mask.astype(bool)] # num-pixels x dino_shape
                dino_mask = dino_mask.cpu().detach().numpy()
                mean_dino = np.mean(dino_mask, axis=0) # dino_shape
                mean_dino = mean_dino / np.linalg.norm(mean_dino) # normalize
                self.observations.append(Observation(t, pose, mask, mask_downsampled, ptcld, semantic_descriptor=mean_dino))
            else:
                self.observations.append(Observation(t, pose, mask, mask_downsampled, ptcld))
                
        return self.observations, frame_descriptor
    
    def apply_rotation(self, img, unrotate=False):
        if self.rotate_img is None:
            return img
        elif self.rotate_img == 'CW':
            k = 3 if not unrotate else 1
        elif self.rotate_img == 'CCW':
            k = 1 if not unrotate else 3
        elif self.rotate_img == '180':
            k = 2
        else:
            raise Exception("Invalid rotate_img option.")
        if type(img) == np.ndarray:
            result = np.rot90(img, k)
        else:
            result = torch.rot90(img, k)
        return result
        
    def unapply_rotation(self, img):
        return self.apply_rotation(img, unrotate=True)

    def _create_mask(self, img):
        
        if len(img.shape) == 2: # image is mono
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        classes, boxes, scores = self.yolov7_det.detect(img)
        ignore_boxes = []
        keep_boxes = []
        for i, cl in enumerate(classes[0]):
            if self.yolov7_det.names[cl] in self.ignore_labels:
                ignore_boxes.append(boxes[0][i])

            if self.yolov7_det.names[cl] in self.keep_labels:
                keep_boxes.append(boxes[0][i])

        ignore_mask = np.zeros(img.shape[:2]).astype(np.int8)
        for box in ignore_boxes:
            x0, y0, x1, y1 = np.array(box).astype(np.int64).reshape(-1).tolist()
            box_before_truncation = np.array([x0, y0, x1, y1])
            x0 = max(x0, 0)
            y0 = max(y0, 0)
            x1 = min(x1, ignore_mask.shape[1])
            y1 = min(y1, ignore_mask.shape[0])

            try:
                ignore_mask[y0:y1,x0:x1] = np.ones((y1-y0, x1-x0)).astype(np.int8)
            except:
                print("Ignore box: ", box_before_truncation)
                print("Ignore box after truncating: ", x0, y0, x1, y1)
                print("Ignore mask shape: ", ignore_mask.shape)
                raise Exception("Invalid ignore box.") 
    

        if self.use_keep_labels:
            keep_mask = np.zeros(img.shape[:2]).astype(np.int8)
            for box in keep_boxes:
                x0, y0, x1, y1 = np.array(box).astype(np.int64).reshape(-1).tolist()
                x0 = max(x0, 0)
                y0 = max(y0, 0)
                x1 = min(x1, keep_mask.shape[1])
                y1 = min(y1, keep_mask.shape[0])
                keep_mask[y0:y1,x0:x1] = np.ones((y1-y0, x1-x0)).astype(np.int8)
        else:
            keep_mask = None

        return ignore_mask, keep_mask

    def _delete_edge_masks(self, segmask):
        [numMasks, h, w] = segmask.shape
        contains_edge = np.zeros(numMasks).astype(np.bool_)
        for i in range(numMasks):
            mask = segmask[i,:,:]
            edge_width = 5
            # TODO: should be a parameter
            contains_edge[i] = (np.sum(mask[:,:edge_width]) > 0 and not self.allow_tblr_edges[2]) or (np.sum(mask[:,-edge_width:]) > 0 and not self.allow_tblr_edges[3]) or \
                            (np.sum(mask[:edge_width,:]) > 0 and not self.allow_tblr_edges[0]) or (np.sum(mask[-edge_width:, :]) > 0 and not self.allow_tblr_edges[1])
        return np.delete(segmask, contains_edge, axis=0)

    def _process_img(self, image_bgr, ignore_mask=None, keep_mask=None):
        """Process FastSAM on image, returns segment masks and center points from results

        Args:
            image_bgr ((h,w,3) np.array): color image
            fastSamModel (FastSAM): FastSAM object
            device (str, optional): 'cuda' or 'cpu'. Defaults to 'cuda'.
            plot (bool, optional): Plots (slow) for visualization. Defaults to False.
            ignore_edges (bool, optional): Filters out edge-touching segments. Defaults to False.

        Returns:
            segmask ((n,h,w) np.array): n segmented masks (binary mask over image)
            blob_means ((n, 2) list): pixel means of segmasks
            blob_covs ((n, (2, 2) np.array) list): list of covariances (ellipses describing segmasks)
            (fig, ax) (Matplotlib fig, ax): fig and ax with visualization
        """

        # OpenCV uses BGR images, but FastSAM and Matplotlib require an RGB image, so convert.
        image = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)

        # Run FastSAM
        everything_results = self.model(image, 
                                        retina_masks=True, 
                                        device=self.device, 
                                        imgsz=self.imgsz, 
                                        conf=self.conf, 
                                        iou=self.iou)
        prompt_process = FastSAMPrompt(image, everything_results, device=self.device)
        segmask = prompt_process.everything_prompt()

        # If there were segmentations detected by FastSAM, transfer them from GPU to CPU and convert to Numpy arrays
        if (len(segmask) > 0):
            segmask = segmask.cpu().numpy()
        else:
            segmask = None

        if (segmask is not None):
            # FastSAM provides a numMask-channel image in shape C, H, W where each channel in the image is a binary mask
            # of the detected segment
            [numMasks, h, w] = segmask.shape

            # filter out edge-touching segments
            # could do with multi-dimensional summing faster instead of looping over masks
            if not np.all(self.allow_tblr_edges):
                segmask = self._delete_edge_masks(segmask)
                [numMasks, h, w] = segmask.shape

            to_delete = []
            for maskId in range(numMasks):
                # Extract the single binary mask for this mask id
                mask_this_id = segmask[maskId,:,:]

                # filter out ignore mask
                if ignore_mask is not None and np.any(np.bitwise_and(mask_this_id.astype(np.int8), ignore_mask)):
                    to_delete.append(maskId)
                    continue

                # Only keep masks that are within keep_mask
                # if keep_mask is not None and not np.any(np.bitwise_and(mask_this_id.astype(np.int8), keep_mask)):
                #     print("Delete maskID: ", maskId)
                #     to_delete.append(maskId)
                #     continue
                # if keep_mask is not None and self.keep_labels_option == 'intersect' and (not np.any(np.bitwise_and(mask_this_id.astype(np.int8), keep_mask))):
                if keep_mask is not None and self.keep_labels_option == 'intersect' and (np.bitwise_and(mask_this_id.astype(np.int8), keep_mask).sum() < self.keep_mask_minimal_intersection*mask_this_id.astype(np.int8).sum()):
                    to_delete.append(maskId)
                    continue

                if self.area_bounds is not None:
                    area = np.sum(mask_this_id)
                    if area < self.area_bounds[0] or area > self.area_bounds[1]:
                        to_delete.append(maskId)
                        continue

            segmask = np.delete(segmask, to_delete, axis=0)

        else: 
            return []

        return segmask
    
    def mask_bounding_box(self, mask):
        # Find the indices of the True values
        true_indices = np.argwhere(mask)

        if len(true_indices) == 0:
            # No True values found, return None or an appropriate response
            return None

        # Calculate the mean of the indices
        mean_coords = np.mean(true_indices, axis=0)

        # Calculate the width and height based on the min and max indices in each dimension
        min_row, min_col = np.min(true_indices, axis=0)
        max_row, max_col = np.max(true_indices, axis=0)
        width = max_col - min_col + 1
        height = max_row - min_row + 1

        # Define a bounding box around the mean coordinates with the calculated width and height
        min_row = int(max(mean_coords[0] - height // 2, 0))
        max_row = int(min(mean_coords[0] + height // 2, mask.shape[0] - 1))
        min_col = int(max(mean_coords[1] - width // 2, 0))
        max_col = int(min(mean_coords[1] + width // 2, mask.shape[1] - 1))

        return (min_col, min_row, max_col, max_row,)

    def get_output_patches(self, model_output: ArrayLike, img_shape: ArrayLike, feature_dim: int) -> ArrayLike:
        """
        Extract (Dino) output patches

        Args:
            model_output (ArrayLike): Last hidden state of (Dino) model
            img_shape (ArrayLike): Original image shape
            feature_dim (int): Expected (Dino) feature dimension

        Returns:
            ArrayLike: Reshaped (Dino) output
        """
        model_output_flat_patches = model_output[:,1:, :]
        if self.semantic_patches_shape is None:
            ratio = img_shape[1] / img_shape[0] # width / height
            num_patches = model_output_flat_patches.shape[1]
            h = np.round(np.sqrt(num_patches / ratio)).astype(int) # number of patches along y-axis
            w = np.round(np.sqrt(num_patches * ratio)).astype(int) # number of patches along x-axis

            self.semantic_patches_shape = (1, h, w, feature_dim)
            
        model_output_patches = model_output_flat_patches.reshape(self.semantic_patches_shape)

        return model_output_patches # 1 x h x w x feature_dim

    def get_per_pixel_features(self, model_output_patches: ArrayLike, img_shape: ArrayLike) -> ArrayLike:
        """
        Extract (Dino) per-pixel features

        Args:
            model_output_patches (ArrayLike): Reshaped (Dino) output patches
            img_shape (ArrayLike): Original image shape

        Returns:
            ArrayLike: Reshaped (Dino) output
        """
        # interpolate the feature map to match the size of the original image
        per_pixel_features = torch.nn.functional.interpolate(
            model_output_patches.permute(0, 3, 1, 2), # permute to be batch, channels, height, width
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
        ) # 1 x dino_shape x h x w

        # reshape
        per_pixel_features = per_pixel_features[0].permute(1, 2, 0) # h x w x feature_dim

        return per_pixel_features # h x w x feature_dim
        
    def get_frame_descriptor(self, dino_features: torch.Tensor) -> np.ndarray:   
        with torch.no_grad(): # prevent memory leak
            dino_features_flat = dino_features.view(-1, dino_features.shape[-1])
            if self.frame_descriptor_type == 'dino-gap':
                frame_descriptor = torch.sum(dino_features_flat, dim=0)
            elif self.frame_descriptor_type == 'dino-gmp':
                frame_descriptor = torch.max(dino_features_flat, dim=0).values
            elif self.frame_descriptor_type == 'dino-gem':
                cubed_descriptor = torch.mean(dino_features_flat ** 3, dim=0)
                frame_descriptor = torch.sign(cubed_descriptor) * \
                                   (torch.abs(cubed_descriptor).clamp(min=1e-12) ** (1.0 / 3)) # avoid NaN from negative or zero root
            else:
                raise ValueError(f"frame descriptor must be one of 'dino-gap', 'dino-gmp', or 'dino-gem'.")
                
            frame_descriptor /= torch.norm(frame_descriptor)
                                            
        return frame_descriptor.cpu().detach().numpy()
            