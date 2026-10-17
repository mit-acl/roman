"""Per-segment color-name histograms, accumulated alongside ROMAN's DINOv2 descriptors.

DINOv2 is trained to ignore color: every crop gets ColorJitter(hue=0.1) with p=0.8, i.e. a
hue shift of up to +-36 degrees, and RandomGrayscale with p=0.2 (dinov2/data/augmentations.py).
Red, orange and yellow are about 30 degrees apart, so the features treat them as the same.

This keeps a separate 11-d color descriptor per segment instead: for every FastSAM mask, the
mean probability of the 11 basic color names (black, blue, brown, grey, green, orange, pink,
purple, red, white, yellow) over the mask's pixels, from the pixel-to-color-name table of
van de Weijer et al., "Learning Color Names for Real-World Applications", IEEE TIP 2009
(data/w2c.txt.gz, from https://lear.inrialpes.fr/people/vandeweijer/color_names.html). A segment
averages its observations with equal weight, as ROMAN does for the DINOv2 descriptor, and
merged segments combine by observation count.

For a text query, color words are pulled out of it (e.g. 'yellow' from 'yellow airplane
wing') and each segment gets the fraction of its pixels with that color name
(color_fraction). With several color words ('red and white target bag') the smallest
fraction counts, so every named color must be there.

install_hooks() wraps FastSAMWrapper.run and Segment.__init__/update/update_from_segment, so
ROMAN itself is unchanged and the map is built exactly as before.
"""
import os

import cv2 as cv
import numpy as np

COLOR_NAMES = ['black', 'blue', 'brown', 'grey', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']
# query word -> the basic color names it covers
COLOR_WORDS = {c: [c] for c in COLOR_NAMES}
COLOR_WORDS.update({'gray': ['grey'], 'maroon': ['red', 'brown'], 'magenta': ['pink', 'purple'],
                    'violet': ['purple'], 'navy': ['blue'], 'silver': ['grey'], 'tan': ['brown']})
W2C_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'w2c.txt.gz')


def load_w2c(path=W2C_PATH):
    """32768 x 11 table: row r + 32 g + 1024 b (8-level sRGB bins) -> P(color name)."""
    return np.loadtxt(path)[:, 3:].astype(np.float32)


def mask_color_names(img_bgr, mask, w2c, erode_px=2):
    """Mean color-name probabilities over the mask's pixels. The mask is eroded slightly so
    pixels on the boundary, which often belong to the background, are left out."""
    m = mask != 0
    if erode_px:
        eroded = cv.erode(m.astype(np.uint8), np.ones((2 * erode_px + 1,) * 2, np.uint8)) != 0
        if eroded.any():
            m = eroded
    px = img_bgr[m].astype(np.int32) // 8
    return w2c[px[:, 2] + 32 * px[:, 1] + 1024 * px[:, 0]].mean(axis=0)


def install_hooks(w2c):
    from roman.map.fastsam_wrapper import FastSAMWrapper
    from roman.object.segment import Segment

    run, init, update, update_from_segment = (FastSAMWrapper.run, Segment.__init__, Segment.update,
                                              Segment.update_from_segment)

    def run_with_color(self, t, pose, img, depth_data=None):
        out = run(self, t, pose, img, depth_data=depth_data)
        for obs in self.observations:
            obs.color_names = mask_color_names(img, obs.mask, w2c)
        return out

    def add(seg, color_sum, cnt):
        seg.color_names_sum = seg.color_names_sum + color_sum
        seg.color_names_cnt += cnt

    def init_with_color(self, observation, *args, **kwargs):
        init(self, observation, *args, **kwargs)
        self.color_names_sum, self.color_names_cnt = np.zeros(len(COLOR_NAMES)), 0
        if getattr(observation, 'color_names', None) is not None:
            add(self, observation.color_names, 1)

    def update_with_color(self, observation, integrate_points=True):
        update(self, observation, integrate_points=integrate_points)
        # same condition as the DINOv2 descriptor; update_from_segment passes stored copies
        # with integrate_points=False and adds the other segment's totals itself
        if integrate_points and getattr(observation, 'color_names', None) is not None:
            add(self, observation.color_names, 1)

    def update_from_segment_with_color(self, segment):
        update_from_segment(self, segment)
        add(self, segment.color_names_sum, segment.color_names_cnt)

    FastSAMWrapper.run = run_with_color
    Segment.__init__ = init_with_color
    Segment.update = update_with_color
    Segment.update_from_segment = update_from_segment_with_color


def segment_colors(segments):
    """(ids, n x 11 mean color-name probabilities, observation counts) for segments built
    with the hooks installed."""
    ids = np.array([s.id for s in segments])
    cnt = np.array([s.color_names_cnt for s in segments])
    colors = np.stack([s.color_names_sum / max(s.color_names_cnt, 1) for s in segments])
    return ids, colors, cnt


def query_colors(query):
    return [w for w in query.lower().replace(',', ' ').split() if w in COLOR_WORDS]


def color_fraction(colors, query):
    """n_segments fraction of pixels with every color word in the query (min over words);
    1 if the query names no color."""
    words = query_colors(query)
    if not words:
        return np.ones(len(colors))
    idx = {c: i for i, c in enumerate(COLOR_NAMES)}
    return np.min([colors[:, [idx[c] for c in COLOR_WORDS[w]]].sum(axis=1) for w in words], axis=0)
