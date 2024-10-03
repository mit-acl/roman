#########################################
# Authors: Jouko Kinnari, Mason Peterson, Lucas Jia, Annika Thomas


import cv2 as cv
import numpy as np
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
from segment_track.observation import Observation
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)

def ssc(keypoints, num_ret_points, tolerance, cols, rows):
    exp1 = rows + cols + 2 * num_ret_points
    exp2 = (
        4 * cols
        + 4 * num_ret_points
        + 4 * rows * num_ret_points
        + rows * rows
        + cols * cols
        - 2 * rows * cols
        + 4 * rows * cols * num_ret_points
    )
    exp3 = math.sqrt(exp2)
    exp4 = num_ret_points - 1

    sol1 = -round(float(exp1 + exp3) / exp4)  # first solution
    sol2 = -round(float(exp1 - exp3) / exp4)  # second solution

    high = (
        sol1 if (sol1 > sol2) else sol2
    )  # binary search range initialization with positive solution
    low = math.floor(math.sqrt(len(keypoints) / num_ret_points))

    prev_width = -1
    selected_keypoints = []
    result_list = []
    result = []
    complete = False
    k = num_ret_points
    k_min = round(k - (k * tolerance))
    k_max = round(k + (k * tolerance))

    while not complete:
        width = low + (high - low) / 2
        if (
            width == prev_width or low > high
        ):  # needed to reassure the same radius is not repeated again
            result_list = result  # return the keypoints from the previous iteration
            break

        c = width / 2  # initializing Grid
        num_cell_cols = int(math.floor(cols / c))
        num_cell_rows = int(math.floor(rows / c))
        covered_vec = [
            [False for _ in range(num_cell_cols + 1)] for _ in range(num_cell_rows + 1)
        ]
        result = []

        for i in range(len(keypoints)):
            row = int(
                math.floor(keypoints[i].pt[1] / c)
            )  # get position of the cell current point is located at
            col = int(math.floor(keypoints[i].pt[0] / c))
            if not covered_vec[row][col]:  # if the cell is not covered
                result.append(i)
                # get range which current radius is covering
                row_min = int(
                    (row - math.floor(width / c))
                    if ((row - math.floor(width / c)) >= 0)
                    else 0
                )
                row_max = int(
                    (row + math.floor(width / c))
                    if ((row + math.floor(width / c)) <= num_cell_rows)
                    else num_cell_rows
                )
                col_min = int(
                    (col - math.floor(width / c))
                    if ((col - math.floor(width / c)) >= 0)
                    else 0
                )
                col_max = int(
                    (col + math.floor(width / c))
                    if ((col + math.floor(width / c)) <= num_cell_cols)
                    else num_cell_cols
                )
                for row_to_cover in range(row_min, row_max + 1):
                    for col_to_cover in range(col_min, col_max + 1):
                        if not covered_vec[row_to_cover][col_to_cover]:
                            # cover cells within the square bounding box with width w
                            covered_vec[row_to_cover][col_to_cover] = True

        if k_min <= len(result) <= k_max:  # solution found
            result_list = result
            complete = True
        elif len(result) < k_min:
            high = width - 1  # update binary search range
        else:
            low = width + 1
        prev_width = width

    for i in range(len(result_list)):
        selected_keypoints.append(keypoints[result_list[i]])

    return selected_keypoints
'''
Helper function from Annika
'''
# my function to calculate a boounding box around a mask
def mask_bounding_box(mask):
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

class FastSAMWrapper():

    def __init__(self, 
        weights, 
        conf=.5, 
        iou=.9,
        imgsz=(1024, 1024),
        device='cuda',
        mask_downsample_factor=1,
        rotate_img=None,
    ):
        """Wrapper for running FastSAM on images (especially RGB/depth pairs)

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
        """
        # parameters
        self.weights = weights
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self.mask_downsample_factor = mask_downsample_factor
        self.rotate_img = rotate_img

        # member variables
        self.observations = []
        self.model = FastSAM(weights)
        # setup default filtering
        self.setup_filtering()

        self.orb_num_initial = 100
        self.orb_num_final = 10
        
        assert self.device == 'cuda' or self.device == 'cpu', "Device should be 'cuda' or 'cpu'."
        assert self.rotate_img is None or self.rotate_img == 'CW' or self.rotate_img == 'CCW' \
            or self.rotate_img == '180', "Invalid rotate_img option."
            
    def setup_filtering(self,
        ignore_labels = [],
        use_keep_labels=False,
        keep_labels = [],
        keep_labels_option='intersect',          
        yolo_det_img_size=None,
        area_bounds=np.array([0, np.inf]),
        allow_tblr_edges = [True, True, True, True],
        keep_mask_minimal_intersection=0.3,
        clip_embedding=False,
        clip_model='ViT-L/14',
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
            self.yolov7_det = Yolov7Detector(traced=False, img_size=yolo_det_img_size)
        
        self.area_bounds = area_bounds
        self.allow_tblr_edges= allow_tblr_edges
        self.keep_mask_minimal_intersection = keep_mask_minimal_intersection
        self.run_yolo = len(ignore_labels) > 0 or use_keep_labels
        self.clip_embedding = clip_embedding
        if clip_embedding:
            self.clip_model, self.clip_preprocess = clip.load(clip_model, device=self.device)
            
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

    def run(self, t, pose, img, img_depth=None, plot=False):
        """
        Takes and image and returns filtered FastSAM masks as Observations.

        Args:
            img (cv image): camera image
            plot (bool, optional): Returns plots if true. Defaults to False.

        Returns:
            self.observations (list): list of Observations
        """
        self.observations = []
        
        # rotate image
        img_orig = img
        img = self.apply_rotation(img)

        if self.run_yolo:
            ignore_mask, keep_mask = self._create_mask(img)
        else:
            ignore_mask = None
            keep_mask = None
        
        # run fastsam
        masks = self._process_img(img, ignore_mask=ignore_mask, keep_mask=keep_mask)
        
        for mask in masks:
            
            mask = self.unapply_rotation(mask)
            # kp, des = self.compute_orb(img[y0:y1, x0:x1], 
            #     num_final=self.orb_num_final, num_initial=self.orb_num_initial)
            # Extract point cloud of object from RGBD
            ptcld = None
            if img_depth is not None:
                depth_obj = copy.deepcopy(img_depth)
                if self.erosion_element is not None:
                    eroded_mask = cv.erode(mask, self.erosion_element)
                    depth_obj[eroded_mask==0] = 0
                else:
                    depth_obj[mask==0] = 0
                logger.debug(f"img_depth type {img_depth.dtype}, shape={img_depth.shape}")

                # Extract point cloud without truncation to heuristically check if enough of the object
                # is within the max depth
                pcd_test = o3d.geometry.PointCloud.create_from_depth_image(
                    o3d.geometry.Image(np.ascontiguousarray(depth_obj).astype(np.uint16)),
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
                    o3d.geometry.Image(np.ascontiguousarray(depth_obj).astype(np.uint16)),
                    self.depth_cam_intrinsics,
                    depth_scale=self.depth_scale,
                    depth_trunc=self.max_depth,
                    stride=self.pcd_stride,
                    project_valid_depth_only=True
                )
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

            if self.clip_embedding:
                ### Use masked image
                # print("Img size: ", img.shape)
                # print("Mask shape: ", mask.shape)
                # # print("Masked image: ", cv.bitwise_and(img, img, mask = mask.astype('uint8')).shape)
                # pre_processed_img = self.clip_preprocess(Image.fromarray(cv.bitwise_and(img, img, mask = mask.astype('uint8')))).to(self.device)
                # clip_embedding = self.clip_model.encode_image(pre_processed_img.unsqueeze(dim=0))
                # print("Clip embedding shape: ", clip_embedding.shape)
                # clip_embedding = clip_embedding.squeeze().cpu().detach().numpy()

                ### Use bounding box
                bbox = mask_bounding_box(mask.astype('uint8'))
                if bbox is None:
                    assert False, "Bounding box is None."
                    self.observations.append(Observation(t, pose, mask, mask_downsampled, ptcld))
                else:
                    min_col, min_row, max_col, max_row = bbox
                    img_bbox = self.apply_rotation(img_orig[min_row:max_row, min_col:max_col])
                    img_bbox = cv.cvtColor(img_bbox, cv.COLOR_BGR2RGB)
                    processed_img = self.clip_preprocess(Image.fromarray(img_bbox, mode='RGB')).to(self.device)
                    clip_embedding = self.clip_model.encode_image(processed_img.unsqueeze(dim=0))
                    clip_embedding = clip_embedding.squeeze().cpu().detach().numpy()
                    self.observations.append(Observation(t, pose, mask, mask_downsampled, ptcld, clip_embedding=clip_embedding))
                
            else:
                self.observations.append(Observation(t, pose, mask, mask_downsampled, ptcld))
                
        return self.observations
    
    def apply_rotation(self, img, unrotate=False):
        if self.rotate_img is None:
            result = img
        elif self.rotate_img == 'CW':
            result = cv.rotate(img, cv.ROTATE_90_CLOCKWISE 
                               if not unrotate else cv.ROTATE_90_COUNTERCLOCKWISE)
        elif self.rotate_img == 'CCW':
            result = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE 
                               if not unrotate else cv.ROTATE_90_CLOCKWISE)
        elif self.rotate_img == '180':
            result = cv.rotate(img, cv.ROTATE_180)
        else:
            raise Exception("Invalid rotate_img option.")
        return result
        
    def unapply_rotation(self, img):
        return self.apply_rotation(img, unrotate=True)
    
    
    def compute_orb(self, img, num_final=10, num_initial=100):
        # partially borrowed from: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
        # Initiate ORB detector
        orb = cv.ORB_create(nfeatures=num_initial)
        
        # find the keypoints with ORB
        kp = orb.detect(img, None)

        # keypoints should be sorted by strength in descending order
        # before feeding to SSC to work correctly
        if len(kp) < self.orb_num_final:
            kp_sel = kp
        else:
            kp = sorted(kp, key=lambda x: x.response, reverse=True)
            kp_sel = ssc(
                kp, self.orb_num_final, .1, img.shape[1], img.shape[0]
            )

        if len(kp_sel) > 0:
            kp_sel, des = orb.compute(img, kp_sel)
        else:
            des = []
        
        return kp_sel, des

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