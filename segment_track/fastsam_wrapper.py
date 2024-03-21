#########################################
# Authors: Jouko Kinnari, Mason Peterson


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import skimage
from yolov7_package import Yolov7Detector
import math

from FastSAM.fastsam import FastSAMPrompt
from FastSAM.fastsam import FastSAM
from fastsam3D.utils import compute_blob_mean_and_covariance, plotErrorEllipse
import fastsam3D.segment_qualification as seg_qual

from segment_track.observation import Observation

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

class FastSAMWrapper():
    
    def __init__(self, 
        weights, 
        conf=.5, 
        iou=.9,
        imgsz=1024,
        device='cuda'
    ):
        # parameters
        self.weights = weights
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        
        # member variables
        self.observations = []
        self.model = FastSAM(weights)
        # setup default filtering
        self.setup_filtering()

        self.orb_num_initial = 100
        self.orb_num_final = 10
            
    def setup_filtering(self,
        ignore_people=False,
        yolo_det_img_size=None,
        max_cov_axis_ratio=np.inf,
        area_bounds=np.array([0, np.inf]),
        allow_tblr_edges = [True, True, True, True],
    ):
        self.ignore_people = ignore_people
        if self.ignore_people:
            if yolo_det_img_size is None:
                yolo_det_img_size=(int(self.imgsz), int(self.imgsz))
            self.yolov7_det = Yolov7Detector(traced=False, img_size=yolo_det_img_size)
        self.max_cov_axis_ratio = max_cov_axis_ratio
        self.area_bounds = area_bounds
        self.allow_tblr_edges= allow_tblr_edges
        
    def run(self, t, pose, img, plot=False):
        """
        Takes and image and returns filtered FastSAM masks as Observations.

        Args:
            img (cv image): camera image
            plot (bool, optional): Returns plots if true. Defaults to False.

        Returns:
            self.observations (list): list of Observations
        """
        self.observations = []

        if self.ignore_people:
            ignore_mask = self._create_people_mask(img)
        else:
            ignore_mask = None
            
        # run fastsam
        masks, means, covs, (fig, ax) = self._process_img(
            img, plot=plot, ignore_mask=ignore_mask)

        widths = []
        heights = []
        bboxes = []
        for _, mask in enumerate(masks):
            nonzero_indices = np.transpose(np.nonzero(mask))
            x0 = np.min(nonzero_indices[:,1])
            x1 = np.max(nonzero_indices[:,1])
            y0 = np.min(nonzero_indices[:,0])
            y1 = np.max(nonzero_indices[:,0])
            bboxes.append([x0, y0, x1, y1])
            widths.append(x1-x0)
            heights.append(y1-y0)
        
        for mean, w, h, mask, bbox in zip(means, widths, heights, masks, bboxes):
            x0, y0, x1, y1 = bbox
            # kp, des = self.compute_orb(img[y0:y1, x0:x1], 
            #     num_final=self.orb_num_final, num_initial=self.orb_num_initial)
            self.observations.append(Observation(t, pose, np.array(mean), w, h, mask))
        
        # TODO: fix plotting
        # if plot_dir is not None:
        #     self._plot_3d_pos(centroids, means, plot_dir, i, fig, ax)
            
        return self.observations
    
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

    def _create_people_mask(self, img):
        classes, boxes, scores = self.yolov7_det.detect(img)
        person_boxes = []
        for i, cl in enumerate(classes[0]):
            if self.yolov7_det.names[cl] == 'person':
                person_boxes.append(boxes[0][i])

        mask = np.zeros(img.shape[:2]).astype(np.int8)
        for box in person_boxes:
            x0, y0, x1, y1 = np.array(box).astype(np.int64).reshape(-1).tolist()
            x0 = max(x0, 0)
            y0 = max(y0, 0)
            x1 = min(x1, mask.shape[1])
            y1 = min(y1, mask.shape[0])
            mask[y0:y1,x0:x1] = np.ones((y1-y0, x1-x0)).astype(np.int8)
        return mask

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

    def _process_img(self, image_bgr, plot=False, ignore_mask=None):
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
        
        # Create a matplotlib figure to plot image and ellipsoids on.
        if plot:
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

        # OpenCV uses BGR images, but FastSAM and Matplotlib require an RGB image, so convert.
        image = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)

        # Let's also make a 1-channel grayscale image
        image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        if plot:
            # ...and also a 3-channel RGB image, which we will eventually use for showing FastSAM output on
            image_gray_rgb = np.stack((image_gray,)*3, axis=-1)

        # Run FastSAM
        everything_results = self.model(image, 
                                        retina_masks=True, 
                                        device=self.device, 
                                        imgsz=self.imgsz, 
                                        conf=self.conf, 
                                        iou=self.iou)
        prompt_process = FastSAMPrompt(image, everything_results, device=self.device)
        segmask = prompt_process.everything_prompt()

        blob_means = []
        blob_covs = []

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
                
            # Prepare a mask of IDs where each pixel value corresponds to the mask ID
            if plot:
                segmasks_flat = np.zeros((h,w),dtype=int)

            to_delete = []
            for maskId in range(numMasks):
                # Extract the single binary mask for this mask id
                mask_this_id = segmask[maskId,:,:]

                # From the pixel coordinates of the mask, compute centroid and a covariance matrix
                blob_mean, blob_cov = compute_blob_mean_and_covariance(mask_this_id)
                
                # filter out ignore mask
                if ignore_mask is not None and \
                    np.any(np.bitwise_and(segmask[maskId,:,:].astype(np.int8), ignore_mask)):
                    to_delete.append(maskId)
                    continue
                
                # qualify covariance
                # filter out segments based on covariance size and shape
                if self.max_cov_axis_ratio is not None:
                    if seg_qual.is_elongated([blob_cov], max_axis_ratio=self.max_cov_axis_ratio):
                        to_delete.append(maskId)
                        continue
                # if self.area_bounds is not None:
                #     if seg_qual.is_out_of_bounds_area([blob_cov], bounds_area=area_bounds):
                #         to_delete.append(maskId)
                #         continue
                    
                if self.area_bounds is not None:
                    area = np.sum(segmask[maskId,:,:])
                    if area < self.area_bounds[0] or area > self.area_bounds[1]:
                        to_delete.append(maskId)
                        continue
                
                # Store centroids and covariances in lists
                blob_means.append(blob_mean)
                blob_covs.append(blob_cov)

                if plot:
                    # Replace the 1s corresponding with masked areas with maskId (i.e. number of mask)
                    segmasks_flat = np.where(mask_this_id < 1, segmasks_flat, maskId)

            segmask = np.delete(segmask, to_delete, axis=0)

            if plot:
                # Using skimage, overlay masked images with colors
                image_gray_rgb = skimage.color.label2rgb(segmasks_flat, image_gray_rgb)

        else: 
            return [], [], [], (None, None)

        if plot:
            # For each centroid and covariance, plot an ellipse
            for m, c in zip(blob_means, blob_covs):
                plotErrorEllipse(ax, m[0], m[1], c, "r",stdMultiplier=2.0)

            # Show image using Matplotlib, set axis limits, save and close the output figure.
            ax.imshow(image_gray_rgb)
            ax.set_xlim([0,w])
            ax.set_ylim([h,0])        
            return segmask, blob_means, blob_covs, (fig, ax)
            
        return segmask, blob_means, blob_covs, (None, None)
    
    def _plot_3d_pos(self, centroids, means, plot_dir, idx, fig, ax):
        for j in range(len(centroids)):
            centroids[j] = self.T_FLURDF @ centroids[j]
        
        for j, mn in enumerate(means):
            ax.text(mn[0]+5, mn[1]+5, f"{np.round(centroids[j].reshape(-1), 1).tolist()}", fontsize=8)
        
        fig.savefig(f'{plot_dir}/image_{idx}.png')
        plt.close(fig)
        
        fig, ax = plt.subplots()
        for c in centroids:
            ax.plot(c.item(0), c.item(1), 'o', color='green')
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        plt.savefig(f'{plot_dir}/measurements_{idx}.png')
        