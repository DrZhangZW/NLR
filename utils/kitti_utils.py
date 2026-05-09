#====================kitti data process: velo_2_detph========================
import matplotlib.pyplot as plt
import glob
import collections
import cv2
import numpy as np
kitti_classes = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,  # 坐姿行人
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7,  # 其他杂项
    'DontCare': -1  # 忽略区域
}
#=============================depth map utils==============================
# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)


def fill_in_fast(depth_map, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    """Fast, in-place depth completion.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_map: dense depth map
    """

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                top_pixel_values[pixel_col_idx]

        # Large Fill
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = cv2.medianBlur(depth_map, 5)

    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map


def fill_in_multiscale(depth_map, max_depth=100.0,
                       dilation_kernel_far=CROSS_KERNEL_3,
                       dilation_kernel_med=CROSS_KERNEL_5,
                       dilation_kernel_near=CROSS_KERNEL_7,
                       extrapolate=False,
                       blur_type='bilateral',
                       show_process=False):
    """Slower, multi-scale dilation version with additional noise removal that
    provides better qualitative results.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        dilation_kernel_far: dilation kernel to use for 30.0 < depths < 80.0 m
        dilation_kernel_med: dilation kernel to use for 15.0 < depths < 30.0 m
        dilation_kernel_near: dilation kernel to use for 0.1 < depths < 15.0 m
        extrapolate:whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'gaussian' - provides lower RMSE
            'bilateral' - preserves local structure (recommended)
        show_process: saves process images into an OrderedDict

    Returns:
        depth_map: dense depth map
        process_dict: OrderedDict of process images
    """

    # Convert to float32
    depths_in = np.float32(depth_map)

    # Calculate bin masks before inversion
    valid_pixels_near = (depths_in > 0.1) & (depths_in <= 15.0)
    valid_pixels_med = (depths_in > 15.0) & (depths_in <= 30.0)
    valid_pixels_far = (depths_in > 30.0)

    # Invert (and offset)
    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = (s1_inverted_depths > 0.1)
    s1_inverted_depths[valid_pixels] = \
        max_depth - s1_inverted_depths[valid_pixels]

    # Multi-scale dilation
    dilated_far = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_far),
        dilation_kernel_far)
    dilated_med = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_med),
        dilation_kernel_med)
    dilated_near = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_near),
        dilation_kernel_near)

    # Find valid pixels for each binned dilation
    valid_pixels_near = (dilated_near > 0.1)
    valid_pixels_med = (dilated_med > 0.1)
    valid_pixels_far = (dilated_far > 0.1)

    # Combine dilated versions, starting farthest to nearest
    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    # Small hole closure
    s3_closed_depths = cv2.morphologyEx(
        s2_dilated_depths, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Median blur to remove outliers
    s4_blurred_depths = np.copy(s3_closed_depths)
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = (s3_closed_depths > 0.1)
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Calculate a top mask
    top_mask = np.ones(depths_in.shape, dtype=bool)
    for pixel_col_idx in range(s4_blurred_depths.shape[1]):
        pixel_col = s4_blurred_depths[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.1)
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    # Get empty mask
    valid_pixels = (s4_blurred_depths > 0.1)
    empty_pixels = ~valid_pixels & top_mask

    # Hole fill
    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_9)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image or create top mask
    s6_extended_depths = np.copy(s5_dilated_depths)
    top_mask = np.ones(s5_dilated_depths.shape, dtype=bool)

    top_row_pixels = np.argmax(s5_dilated_depths > 0.1, axis=0)
    top_pixel_values = s5_dilated_depths[top_row_pixels,
                                         range(s5_dilated_depths.shape[1])]

    for pixel_col_idx in range(s5_dilated_depths.shape[1]):
        if extrapolate:
            s6_extended_depths[0:top_row_pixels[pixel_col_idx],
                               pixel_col_idx] = top_pixel_values[pixel_col_idx]
        else:
            # Create top mask
            top_mask[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = False

    # Fill large holes with masked dilations
    s7_blurred_depths = np.copy(s6_extended_depths)
    for i in range(6):
        empty_pixels = (s7_blurred_depths < 0.1) & top_mask
        dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_5)
        s7_blurred_depths[empty_pixels] = dilated[empty_pixels]

    # Median blur
    blurred = cv2.medianBlur(s7_blurred_depths, 5)
    valid_pixels = (s7_blurred_depths > 0.1) & top_mask
    s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    if blur_type == 'gaussian':
        # Gaussian blur
        blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)
        valid_pixels = (s7_blurred_depths > 0.1) & top_mask
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    elif blur_type == 'bilateral':
        # Bilateral blur
        blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 2.0)
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Invert (and offset)
    s8_inverted_depths = np.copy(s7_blurred_depths)
    valid_pixels = np.where(s8_inverted_depths > 0.1)
    s8_inverted_depths[valid_pixels] = \
        max_depth - s8_inverted_depths[valid_pixels]

    depths_out = s8_inverted_depths

    process_dict = None
    if show_process:
        process_dict = collections.OrderedDict()

        process_dict['s0_depths_in'] = depths_in

        process_dict['s1_inverted_depths'] = s1_inverted_depths
        process_dict['s2_dilated_depths'] = s2_dilated_depths
        process_dict['s3_closed_depths'] = s3_closed_depths
        process_dict['s4_blurred_depths'] = s4_blurred_depths
        process_dict['s5_combined_depths'] = s5_dilated_depths
        process_dict['s6_extended_depths'] = s6_extended_depths
        process_dict['s7_blurred_depths'] = s7_blurred_depths
        process_dict['s8_inverted_depths'] = s8_inverted_depths

        process_dict['s9_depths_out'] = depths_out

    return depths_out, process_dict


class KITTI_post_process(object):


    def __init__(self,calib_paths,image_2_paths,label_2_paths,velodyne_paths):
        self.velo_paths = velodyne_paths
        self.calib_paths = calib_paths
        self.image_2_paths = image_2_paths
        self.label_2_paths = label_2_paths

    #========================video_velo2pixel_and_velo2depth=============================
    def velo2pixel_and_velo2depth_video(self):
        image = cv2.imread(self.image_2_paths[0])
        width, height = image.shape[1], image.shape[0]
        #====================================================
        # cv2.putText
        text = "Hello, World!"
        font = cv2.FONT_HERSHEY_SIMPLEX  # 选择字体
        font_scale = 1  # 字体大小
        color = (0, 0, 0)  # 字体颜色（黑色）
        thickness = 2  # 字体粗细
        org = (50, 50)  # 文字起始位置
        # ====================================================
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vid = cv2.VideoWriter('velo2pixel_and_velo2depth.avi', fourcc, 3, (width, int(3*height)))
        for frame, point in enumerate(self.velo_paths):
            img_ori,lidar_depth2img, depth_image,combine = self.velo2pixel_and_velo2depth(frame)
            frame = cv2.putText(combine, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
            vid.write(frame)
        print('video saved')
        vid.release()
    def velo2pixel_and_velo2depth(self,frame=None,clip_distance=2.0,velo=None,plt_img=False):
        if frame == None:
            frame=0
        #==================load velo, calib, image==============================
        if velo is not None:
            velo=velo
        else:
            velo = (np.fromfile(self.velo_paths[frame], dtype=np.float32)).reshape((-1, 4))
        img_ori = cv2.cvtColor(cv2.imread(self.image_2_paths[frame]), cv2.COLOR_BGR2RGB)
        calib = load_calibration(self.calib_paths[frame])  # 处理标定文件
        # =======================Project LiDAR points to image===========================
        # ===================pts2d is the coordinate of the radar point cloud on the image
        pts_2d, point_depth = self.compute_velo_coor_2_pixel_coor(velo[:, :3], calib)
        img = np.copy(img_ori)
        img_height, img_width, _ = img.shape
        xmin, ymin, xmax, ymax = 0, 0, img_width, img_height
        valid_indx = ((pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin)
                      & (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin))
        valid_indx = valid_indx & (velo[:, 0] > clip_distance) if clip_distance is not None else valid_indx
        valid_pixel_2d = pts_2d[valid_indx, :]
        valid_pixel_depth = point_depth[valid_indx]
        # ===============================================================================
        lidar_depth2img = self.draw_lidar_depth2image(img, valid_pixel_2d, valid_pixel_depth)
        depth_image = np.zeros(img.shape[0:2])
        depth_image[valid_pixel_2d[:, 1].astype(int), valid_pixel_2d[:, 0].astype(int)] = valid_pixel_depth
        #=========================visual result======================================
        combine=None
        if plt_img:
            depth_image_copy = np.copy(depth_image)
            depth_image_copy = self.normalize_val(depth_image,depth_image_copy.min(),depth_image_copy.max())
            depth_image_copy = cv2.cvtColor(depth_image_copy, cv2.COLOR_GRAY2BGR)
            combine = np.vstack([img_ori, lidar_depth2img, depth_image_copy])
            plt.subplots(1, 1, figsize=(6, 18))
            plt.title("Velodyne points to camera image Result")
            plt.imshow(combine)
            plt.show(block=True)

        """
        plt.subplots(1, 1, figsize=(6, 18))
        plt.title("Velodyne points to camera image Result")
        plt.imshow(lidar_depth2img)
        plt.show(block=True)
        plt.subplots(1, 1, figsize=(6, 18))
        plt.title("Velodyne points to depth Result")
        plt.imshow(self.normalize_val(depth_image, min_v=60, max_v=70))
        plt.show(block=True)
        """
        return img_ori,lidar_depth2img, depth_image,combine
    #=========================depth2velo===============
    def depth2velo(self, frame, depth):
        #load frame and depth
        calib = load_calibration(self.calib_paths[frame])  # 处理标定文件
        lidar =  self.compute_depth_2_velo(depth, calib, max_high=1)
        lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
        lidar = lidar.astype(np.float32)
        #lidar.tofile('{}/{}.bin'.format( , predix))
        #print('Finish Depth {}'.format(predix))
        #==================load velo, calib, image==============================
        return lidar
    #========================labels_boxes2pixel_in_image=============================
    def labels_boxes2pixel_in_image_video(self):
        image = cv2.imread(self.image_2_paths[0])
        width, height = image.shape[1], image.shape[0]
        #====================================================
        # cv2.putText
        text = "Hello, World!"
        font = cv2.FONT_HERSHEY_SIMPLEX  # 选择字体
        font_scale = 1  # 字体大小
        color = (0, 0, 0)  # 字体颜色（黑色）
        thickness = 2  # 字体粗细
        org = (50, 50)  # 文字起始位置
        # ====================================================
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vid = cv2.VideoWriter('labels_boxes2pixel_in_image.avi', fourcc, 3, (width, int(3*height)))
        for frame, point in enumerate(self.velo_paths):
            img_bbox2d, img_bbox3d, box3d_to_pixel2d,combine = self.labels_boxes2pixel_in_image(frame)
            frame = cv2.putText(combine, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
            vid.write(frame)
        print('video saved')
        vid.release()
    def labels_boxes2pixel_in_image(self,frame=None,plt_img=False):
        if frame == None:
            frame=0
        #==================load velo, calib, image==============================
        img = cv2.cvtColor(cv2.imread(self.image_2_paths[frame]), cv2.COLOR_BGR2RGB)
        calib = load_calibration(self.calib_paths[frame])  # 处理标定文件
        labels = [line.rstrip() for line in open(self.label_2_paths[frame])]

        objects = [Object3d(label) for label in labels]
        # =========================post process============================
        img1 = np.copy(img)  # for 2d bbox
        img2 = np.copy(img)  # for 3d bbox
        box3d_to_pixel2d = {"corners_3d": [], "box3d_pts_2d": []}
        for obj in objects:
            # 2d coordinate
            if obj.type == "DontCare":
                continue
            else:
                cv2.rectangle(img1, (int(obj.xmin), int(obj.ymin)), (int(obj.xmax), int(obj.ymax)), (0, 255, 0), 2, )
                cv2.putText(img1, str(obj.type), (int(obj.xmin), int(obj.ymin)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0),
                            2)

            box3d_pts_2d, corners_3d = self.compute_box_3d(obj, calib.P2)
            box3d_to_pixel2d['corners_3d'].append(corners_3d)

            box3d_to_pixel2d['box3d_pts_2d'].append(box3d_pts_2d)
            if box3d_pts_2d is None:
                print("something wrong in the 3D box.")
                continue
            img2 = self.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 0))
        #==========================plot img========================
        combine=None
        if plt_img:
            combine = np.vstack([img, img1, img2])
            plt.subplots(1, 1, figsize=(6, 18))
            plt.title("Velodyne points to camera image Result")
            plt.imshow(combine)
            plt.show(block=True)
        return img1, img2, box3d_to_pixel2d,combine,objects
    #========================velo_points_2_top_view=============================
    def velo_points_2_top_view_video(self):
        #====================================================
        # cv2.putText
        text = "Hello, World!"
        font = cv2.FONT_HERSHEY_SIMPLEX  # 选择字体
        font_scale = 1  # 字体大小
        color = (0, 0, 0)  # 字体颜色（黑色）
        thickness = 2  # 字体粗细
        org = (50, 50)  # 文字起始位置
        # ====================================================
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vid = cv2.VideoWriter('velo_points_2_top_view.avi', fourcc, 3, (401,401))
        for frame, point in enumerate(self.velo_paths):
            top_view = self.velo_points_2_top_view(frame)
            frame = cv2.putText(top_view, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
            vid.write(frame)
        print('video saved')
        vid.release()
    def velo_points_2_top_view(self,frame=None,x_range=(-20, 20), y_range=(-20, 20), z_range=(-2, 2), scale=10,velo=None):
        # x_range= -20m-->20m
        if frame == None:
            frame=0
        #==================load velo, calib, image==============================
        if velo is not None:
            velo=velo
        else:
            velo = (np.fromfile(self.velo_paths[frame], dtype=np.float32)).reshape((-1, 4))
        x = velo[:, 0]
        y = velo[:, 1]
        z = velo[:, 2]
        dist = np.sqrt(x ** 2 + y ** 2)

        fig = plt.figure(figsize=(13, 7))
        ax = fig.add_subplot(111, projection='3d')
        plt.title("3D Tracklet display")
        ax.scatter(x,y,z, s=0.1, c='k', marker='.', alpha=0.5)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim3d(-10, 30)
        ax.set_ylim3d(-20, 20)
        ax.set_zlim3d(-2, 15)
        plt.show(block=True)

        #===================extract in-range points=========================
        x_lim = self.in_range_points(x, x, y, z, x_range, y_range, z_range)
        y_lim = self.in_range_points(y, x, y, z, x_range, y_range, z_range)
        dist_lim = self.in_range_points(dist, x, y, z, x_range, y_range, z_range)
        # * x,y,z range are based on lidar coordinates
        x_size = int((y_range[1] - y_range[0]))
        y_size = int((x_range[1] - x_range[0]))
        # convert 3D lidar coordinates(vehicle coordinates) to 2D image coordinates
        # scale - for high resolution
        x_img = -(y_lim * scale).astype(np.int32)
        y_img = -(x_lim * scale).astype(np.int32)
        # shift negative points to positive points (shift minimum value to 0)
        x_img = x_img + int(np.trunc(y_range[1] * scale))
        y_img = y_img + int(np.trunc(x_range[1] * scale))
        # normalize distance value & convert to depth map
        max_dist = np.sqrt((max(x_range) ** 2) + (max(y_range) ** 2))
        dist_lim_ori = np.copy(dist_lim)
        dist_lim = self.normalize_depth(dist_lim, min_v=0, max_v=max_dist)
        # array to img
        img = np.zeros([y_size * scale + 1, x_size * scale + 1], dtype=np.uint8)
        img[y_img, x_img] = dist_lim
        #========================plt results=================
        plt.subplots(1, 1, figsize=(5, 5))
        plt.title("Velodyne points on Top view")
        plt.imshow(img)
        plt.axis('off')
        plt.show(block=True)
        #================================================
        return img
    #==================================================================================
    def normalize_depth(self,val, min_v, max_v):
        """
        print 'nomalized depth value'
        nomalize values to 0-255 & close distance value has high value. (similar to stereo vision's disparity map)
        """
        return (((max_v - val) / (max_v - min_v)) * 255).astype(np.uint8)
    def normalize_val(sefl,val, min_v, max_v):
        """
        print 'nomalized depth value'
        nomalize values to 0-255 & close distance value has low value.
        """
        val = (val - min_v) / (max_v - min_v)
        val = (val * 255).astype(np.uint8)
        return val
    def compute_velo_coor_2_pixel_coor(self,velo,calib):
        #  Y = P2_(3x4) * R0_rect_(3x3) * Tr_velo_to_cam_(3x4) * X
        # velo is a 3d point [x,y,z]
        # step1 x=[x_v,y_v,z_v]--->X=[x_v,y_v,z_v,1]
        # step2  cam_coor: (Tr_velo_to_cam)_3x4 * X_4x1
        # step3 rect_cam_coor: R0_rect * cam_coor
        # step4 proj_2d_point: P2 * rect_cam_coor
        # =======================================================================
        n = velo.shape[0]
        pts_3d_point = np.hstack((velo, np.ones((n, 1))))  # return [x_v,y_v,z_v,1]
        # ===========[x_c,y_c,z_c,1]=(Tr_velo_to_cam)_3x4 *[x_v,y_v,z_v,1]=======
        pts_3d_2_cam = np.dot(calib.Tr_velo_to_cam, pts_3d_point.T).T
        # ===================[x_c_rect,y_c_rect,z_c_rect,1]=R0_rect *[x_c,y_c,z_c,1]================
        pts_3d_2_cam_rect = np.dot(calib.R0_rect, pts_3d_2_cam.T).T
        depth = pts_3d_2_cam_rect[:, 2]
        # =====================s[u,v,1]=P2*[x_c_rect,y_c_rect,z_c_rect,1]============================
        pts_3d_2_cam_rect = np.hstack((pts_3d_2_cam_rect, np.ones((n, 1))))
        pts_3d_2_2d = np.dot(calib.P2, pts_3d_2_cam_rect.T).T

        # ===================s[x,y,1]=s[u,v,1]===================================
        # ====================pts_3d_2_2d=[sx,sy,s]==========================
        pts_3d_2_2d[:, 0] /= pts_3d_2_2d[:, 2]
        pts_3d_2_2d[:, 1] /= pts_3d_2_2d[:, 2]
        return pts_3d_2_2d[:, 0:2], depth
    #===============================================================
    def compute_depth_2_velo(self,depth,calib,max_high=None):
        # depth: contain: (u,v)--z_c,
        #  (u,v)-->(x_c,y_c,z_c)-->(x_v,y_v,z_v)
        if max_high==None:
            max_high=1
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        points = np.stack([c, r, depth])
        points = points.reshape((3, -1))
        points = points.T
        #=====================(u,v,z_c)-->(x_c,y_c,z_c)======================
        n = points.shape[0]
        x = ((points[:, 0] - calib.c_u) * points[:, 2]) / calib.f_u + calib.b_x
        y = ((points[:, 1] - calib.c_v) * points[:, 2]) / calib.f_v + calib.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = points[:, 2]
        #====================(x_c,y_c,z_c)--->R^-1(x_c,y_c,z_c)=======================
        pts_3d_ref = np.transpose(np.dot(np.linalg.inv(calib.R0_rect), np.transpose(pts_3d_rect)))
        n = pts_3d_ref.shape[0]
        pts_3d_ref = np.hstack((pts_3d_ref, np.ones((n, 1))))
        cloud = np.dot(pts_3d_ref, np.transpose(calib.Tr_cam_to_velo))
        #===========================================
        valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
        cloud = cloud[valid]
        return cloud
    #=========================================================================
    def draw_lidar_depth2image(self,img, imgfov_pts_2d, imgfov_depth):
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("hsv", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        for i in range(imgfov_pts_2d.shape[0]):
            depth = imgfov_depth[i]
            x = int(255) if int(640.0 / depth)>255 else int(640.0 / depth)
            #x = int(0) if int(640.0 / depth)< 0 else int(640.0 / depth)
            color = cmap[x, :]
            cv2.circle(img, (int(np.round(imgfov_pts_2d[i, 0])), int(np.round(imgfov_pts_2d[i, 1]))),
                       2, color=tuple(color), thickness=-1, )
        return img

    def compute_box_3d(self,obj, P):
        """ Takes an object and a projection matrix (P) and projects the 3d
            bounding box into the image plane.
            Returns:
                corners_2d: (8,2) array in left image coord.
                corners_3d: (8,3) array in in rect camera coord.
        """

        def roty(t):
            """ Rotation about the y-axis. """
            c = np.cos(t)
            s = np.sin(t)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        def project_to_image(pts_3d, P):
            """ Project 3d points to image plane.

            Usage: pts_2d = projectToImage(pts_3d, P)
              input: pts_3d: nx3 matrix
                     P:      3x4 projection matrix
              output: pts_2d: nx2 matrix

              P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
              => normalize projected_pts_2d(2xn)

              <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
                  => normalize projected_pts_2d(nx2)
            """
            n = pts_3d.shape[0]
            pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
            # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
            # pts_2d = np.dot(P, pts_3d_extend.T).T # 这一句与下面一句是等价的
            pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
            pts_2d[:, 0] /= pts_2d[:, 2]
            pts_2d[:, 1] /= pts_2d[:, 2]
            return pts_2d[:, 0:2]

        # compute rotational matrix around yaw axis
        R = roty(obj.ry)

        # 3d bounding box dimensions
        l = obj.l
        w = obj.w
        h = obj.h

        # 3d bounding box corners
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        # rotate and translate 3d bounding box
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        # print corners_3d.shape
        corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
        corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
        corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
        # print 'cornsers_3d: ', corners_3d
        # only draw 3d bounding box for objs in front of the camera
        if np.any(corners_3d[2, :] < 0.1):
            corners_2d = None
            return corners_2d, np.transpose(corners_3d)

        # project the 3d bounding box into the image plane，相机坐标转像素坐标
        corners_2d = project_to_image(np.transpose(corners_3d), P)
        # print 'corners_2d: ', corners_2d
        return corners_2d, np.transpose(corners_3d)
    def draw_projected_box3d(self,image, qs, color=(0, 255, 0), thickness=2):
        """ Draw 3d bounding box in image
                qs: (8,3) array of vertices for the 3d box in following order:
                    1 -------- 0
                   /|         /|
                  2 -------- 3 .
                  | |        | |
                  . 5 -------- 4
                  |/         |/
                  6 -------- 7
        """
        qs = qs.astype(np.int32)
        for k in range(0, 4):
            # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            # use LINE_AA for opencv3
            # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

            i, j = k, k + 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        return image

    def in_range_points(self,points, x, y, z, x_range, y_range, z_range):
        """ extract in-range points """
        return points[np.logical_and.reduce((x > x_range[0], x < x_range[1], y > y_range[0],
                                             y < y_range[1], z > z_range[0], z < z_range[1]))]



class load_calibration(object):
    def __init__(self,path):
        """
        Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref
        y_image2 = P^2_rect * x_rect

        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z
        """
        #=================load txt data========================
        calibs = {}
        with open(path, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    calibs[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        #=======================read P2, R_0_rect, Tr_velo_cam==================================
        self.P2 = np.reshape(calibs['P2'],[3,4])
        self.R0_rect =  np.reshape(calibs["R0_rect"], [3, 3])
        self.Tr_velo_to_cam = np.reshape(calibs["Tr_velo_to_cam"], [3, 4])
        #======Inverse a rigid body transform matrix (3x4 as [R|t])===[R'|-R't; 0|1]============
        self.Tr_cam_to_velo = np.zeros_like(self.Tr_velo_to_cam)  # 3x4
        self.Tr_cam_to_velo[0:3, 0:3] = np.transpose(self.Tr_velo_to_cam[0:3, 0:3])
        self.Tr_cam_to_velo[0:3, 3] = np.dot(-np.transpose(self.Tr_velo_to_cam[0:3, 0:3]), self.Tr_velo_to_cam[0:3, 3])
        #======================Camera intrinsics and extrinsics=================================
        self.c_u,self.c_v = self.P2[0, 2],self.P2[1, 2]
        self.f_u, self.f_v = self.P2[0, 0],self.P2[1, 1]
        self.b_x = self.P2[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P2[1, 3] / (-self.f_v)

class Object3d(object):
    """ 3d object label """
    def __init__(self, label_file_line):
        data = label_file_line.split(" ")
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        #=========================================================
        # 类别合并逻辑
        if self.type in ['Car', 'Van', 'Truck', 'Tram','Misc']:
            self.type = 'Vehicle'
        elif self.type in ['Pedestrian', 'Person_sitting']:
            self.type = 'Pedestrian'
        elif self.type == 'Cyclist':
            self.type = 'Cyclist'





    def is_valid_distance(self, max_distance=20):
        """判断目标是否在有效距离内"""
        if self.h <0 or self.w <0 or self.l <0:
            return False  # 跳过无效标注
        distance = np.sqrt(self.t[0]**2 + self.t[1]**2 + self.t[2]**2)
        return distance <= max_distance
def depth_completion(projected_depths,fill_type='multiscale',extrapolate = True,blur_type = 'gaussian',plt_img=False):


    if fill_type == 'fast':
        final_depths = fill_in_fast(
            projected_depths, extrapolate=extrapolate, blur_type=blur_type) #depth_map_utils.
    elif fill_type == 'multiscale':
        final_depths, process_dict = fill_in_multiscale(
            projected_depths, extrapolate=extrapolate, blur_type=blur_type,
            show_process=True)
    else:
        raise ValueError('Invalid fill_type {}'.format(fill_type))
    if plt_img:
        plt.subplots(1, 1, figsize=(6, 18))
        plt.title("Velodyne points to camera image Result")
        plt.imshow(final_depths)
        plt.show(block=True)

    """
    # Display images from process_dict
    if fill_type == 'multiscale':
        img_size = (570, 165)
        x_start = 80
        y_start = 50
        x_offset = img_size[0]
        y_offset = img_size[1]
        x_padding = 0
        y_padding = 28

        img_x = x_start
        img_y = y_start
        max_x = 1900

        row_idx = 0
        for key, value in process_dict.items():

            image_jet = cv2.applyColorMap(
                np.uint8(value / np.amax(value) * 255),
                cv2.COLORMAP_JET)
            vis_utils.cv2_show_image(
                key, image_jet,
                img_size, (img_x, img_y))

            img_x += x_offset + x_padding
            if (img_x + x_offset + x_padding) > max_x:
                img_x = x_start
                row_idx += 1
            img_y = y_start + row_idx * (y_offset + y_padding)

            # Save process images
            cv2.imwrite('process/' + key + '.png', final_depths)
        cv2.imshow('depth',final_depths)
        cv2.waitKey()
    """

    return final_depths


if __name__ == '__main__':
    #==================================calib,image,label,velodyne path===========================
    path_calib = r'E:\Python-master\dataset\KITTI\object\calib'
    path_image_2 = r'E:\Python-master\dataset\KITTI\object\image2'
    path_label_2 = r'E:\Python-master\dataset\KITTI\object\label2'
    path_velodyne = r'E:\Python-master\dataset\KITTI\object\velodyne'
    kitti=KITTI_post_process(path_calib,path_image_2,path_label_2,path_velodyne)
    kitti.velo2pixel_and_velo2depth_video()
    img_ori,lidar_depth2img, depth_image,combine = kitti.velo2pixel_and_velo2depth(frame=1)
    depth_image = np.float32(depth_image)
    velo_valid = kitti.depth2velo(frame=1,depth=depth_image)
    kitti.velo_points_2_top_view(frame=1,velo=velo_valid)

    depth_com=depth_completion(depth_image)
    velo_com = kitti.depth2velo(frame=1,depth=depth_com)
    kitti.velo_points_2_top_view(frame=1,velo=velo_com)

    #kitti.labels_boxes2pixel_in_image_video()
    #kitti.velo_points_2_top_view_video()







