
import cv2
import matplotlib
from matplotlib.patches import Arc
from pathlib import Path
from utils.kitti_utils import KITTI_post_process, depth_completion

import copy
from utils.common_utils import load_flist, img2label

# ====================================Images Suffixe====================================================
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
# class_mapping = {'Car': 1, 'Van': 2, 'Truck': 3, 'Pedestrian': 4, 'Person_sitting': 5, 'Cyclist': 6, 'Tram': 7,
#                 'Misc': 8}
class_mapping = {'Vehicle': 1, 'Pedestrian': 2, 'Cyclist': 3}
categories = ['N/A', 'Vehicle', 'Pedestrian', 'Cyclist']


# ==================================kitti sonar=============================================
class kitti_under(object):
    def __init__(self, calib_paths, image_2_paths, label_2_paths, velo_paths):
        self.image_2_paths = load_flist(image_2_paths)
        self.calib_paths = img2label(self.image_2_paths, calib_paths, suffix='.txt')
        self.label_2_paths = img2label(self.image_2_paths, label_2_paths, suffix='.txt')
        self.velo_paths = img2label(self.image_2_paths, velo_paths, suffix='.bin')
        self.Kitti_post_process = KITTI_post_process(self.calib_paths, self.image_2_paths, self.label_2_paths, self.velo_paths)


    def kitti_data(self, frame):
        # ================================prepare img, depth=====================================================
        if frame == None:
            frame = 0
        img_ori, lidar_depth2img, depth, _ = self.Kitti_post_process.velo2pixel_and_velo2depth(frame, clip_distance=2.0,
                                                                                               plt_img=False)

        depth = depth_completion(depth, plt_img=False)
        # ====================================================================

        return img_ori, depth
