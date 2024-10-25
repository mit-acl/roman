import numpy as np
import cv2 as cv

from roman.object.segment import Segment

def draw(t, img, pose, tracker, fastsam, observations, reprojected_bboxs, viz_bbox=False, viz_mask=False):

    if viz_mask:
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_fastsam = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            img_fastsam = img.copy()
        img_fastsam = np.concatenate([img_fastsam[...,None]]*3, axis=2)
    
    if len(img.shape) == 2:
        img = np.concatenate([img[...,None]]*3, axis=2)
        
    if viz_bbox:
        segment: Segment
        for i, segment in enumerate(tracker.segments + tracker.inactive_segments + tracker.segment_graveyard):
            # only draw segments seen in the last however many seconds
            if segment.last_seen < t - tracker.segment_graveyard_time - 10:
                continue
            outline = segment.outline_2d(pose)
            if outline is None:
                continue
            # if i < len(tracker.segments):
            #     color = (0, 255, 0)
            # elif i < len(tracker.segments) + len(tracker.inactive_segments):
            #     color = (255, 0, 0)
            # else:
            #     color = (180, 0, 180)
            color = segment.viz_color
            for i in range(len(outline) - 1):
                start_point = tuple(outline[i].astype(np.int32))
                end_point = tuple(outline[i+1].astype(np.int32))
                img = cv.line(img, start_point, end_point, color, thickness=2)

            img = cv.putText(img, str(segment.id), (np.array(outline[0]) + np.array([10., 10.])).astype(np.int32), 
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if viz_mask:
        matched_masks = []
        for segment in tracker.segments:
            if segment.last_seen == t:
                colored_mask = np.zeros_like(img)
                rand_color = segment.viz_color
                matched_masks.append(segment.last_mask)
                try:
                    colored_mask = segment.last_mask.astype(np.int32)[..., np.newaxis]*rand_color
                except:
                    import ipdb; ipdb.set_trace()
                colored_mask = colored_mask.astype(np.uint8)
                img_fastsam = cv.addWeighted(img_fastsam, 1.0, colored_mask, 0.5, 0)
                mass_x, mass_y = np.where(segment.last_mask >= 1)
                img_fastsam = cv.putText(img_fastsam, str(segment.id), (int(np.mean(mass_y)), int(np.mean(mass_x))), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, rand_color.tolist(), 2)
        
        for obs in observations:
            alread_shown = False
            for mask in matched_masks:
                if np.all(mask == obs.mask):
                    alread_shown = True
                    break
            if alread_shown:
                continue
            white_mask = obs.mask.astype(np.int32)[..., np.newaxis]*np.ones(3)*255
            white_mask = white_mask.astype(np.uint8)
            img_fastsam = cv.addWeighted(img_fastsam, 1.0, white_mask, 0.5, 0)

        for seg_id, bbox in reprojected_bboxs:
            np.random.seed(seg_id)
            rand_color = np.random.randint(0, 255, 3)
            cv.rectangle(img_fastsam, np.array([bbox[0][0], bbox[0][1]]).astype(np.int32), 
                        np.array([bbox[1][0], bbox[1][1]]).astype(np.int32), color=rand_color.tolist(), thickness=2)
            
    # rotate images
    img = fastsam.apply_rotation(img)
    if viz_mask:
        img_fastsam = fastsam.apply_rotation(img_fastsam)
            
    # concatenate images
    if viz_bbox and viz_mask:
        img_ret = np.concatenate([img, img_fastsam], axis=1)
    elif viz_bbox:
        img_ret = img
    elif viz_mask:
        img_ret = img_fastsam
    return img_ret