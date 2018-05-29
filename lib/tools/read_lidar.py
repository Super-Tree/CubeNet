import numpy as np
import os
import matplotlib.pyplot as plt

def show_bv(i):
    root_dir = "/home/hexindong/DATASET/kittidataset/KITTI/object/train"
    velodyne = os.path.join(root_dir, "velodyne/")
    bird = os.path.join(root_dir, "lidar_bv_10x0.24/")

    side_range = (-30., 30.)
    fwd_range = (0., 60)
    height_range = (-2, 0.4)

    filename = velodyne + str(i).zfill(6) + ".bin"
    print("Processing: ", filename)
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    bird_view = point_cloud_2_top(scan, res=0.1, zres=0.3,
                                  side_range=side_range,  # left-most to right-most
                                  fwd_range=fwd_range,  # back-most to forward-most
                                  height_range=height_range)
    bv = bird_view
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.axis('off')
        plt.title(i)
        plt.imshow(bv[:, :, i])
    plt.show()
    # fig.show()
    # save
    plt.imshow(bv[:, :, 8])
    plt.show()
    np.save(bird + str(i).zfill(6) + ".npy", bird_view)

    # test


def show_front():
    root_dir = "/home/hexindong/DATASET/kittidataset/KITTI/object/train"
    velodyne = os.path.join(root_dir, "velodyne/")
    bird = os.path.join(root_dir, "lidar_bv_10x0.24/")

    for i in range(200):
        filename = velodyne + str(i).zfill(6) + ".bin"
        print("Processing:0.0", filename)
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        img = point_cloud_to_panorama(scan)
        plt.imshow(img[:, :, 0])
        plt.show()
    #####################################
    side_range = (-20., 20.)
    fwd_range = (0., 40)
    height_range = (-2, 0.4)  #
    plt.rcParams['figure.figsize'] = (10, 10)

def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)

def point_cloud_2_top(points,res=0.1,zres=0.3,side_range=(-20., 20.),fwd_range=(0., 40.), height_range=(-2., 0.4),):
    """ Creates an birds eye view representation of the point cloud data for MV3D.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        zres:        (float)
                    Desired resolution on Z-axis in metres to use.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        numpy array encoding height features , density and intensity.
    """
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    reflectance = points[:, 3]

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    z_max = int((height_range[1] - height_range[0]) / zres)
    # z_max =
    top = np.zeros([y_max + 1, x_max + 1, z_max + 1], dtype=np.float32)

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and(
        (x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and(
        (y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)

    # # ASSIGN EACH POINT TO A HEIGHT SLICE
    # # n_slices-1 is used because values above max_height get assigned to an
    # # extra index when we call np.digitize().
    # bins = np.linspace(height_range[0], height_range[1], num=n_slices-1)
    # slice_indices = np.digitize(z_points, bins=bins, right=False)
    # # RESCALE THE REFLECTANCE VALUES - to be between the range 0-255
    # pixel_values = scale_to_255(r_points, min=0.0, max=1.0)
    # FILL PIXEL VALUES IN IMAGE ARRAY
    # -y is used because images start from top left
    # x_max = int((side_range[1] - side_range[0]) / res)
    # y_max = int((fwd_range[1] - fwd_range[0]) / res)
    # im = np.zeros([y_max, x_max, n_slices], dtype=np.uint8)
    # im[-y_img, x_img, slice_indices] = pixel_values

    for i, height in enumerate(np.arange(height_range[0], height_range[1], zres)):
        z_filt = np.logical_and((z_points >= height),
                                (z_points < height + zres))
        zfilter = np.logical_and(filter, z_filt)
        indices = np.argwhere(zfilter).flatten()

        # KEEPERS
        xi_points = x_points[indices]
        yi_points = y_points[indices]
        zi_points = z_points[indices]
        ref_i = reflectance[indices]

        # print(f_filt.shape)

        # CONVERT TO PIXEL POSITION VALUES - Based on resolution
        x_img = (-yi_points / res).astype(np.int32)  # x axis is -y in LIDAR
        y_img = (-xi_points / res).astype(np.int32)  # y axis is -x in LIDAR

        # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
        # floor & ceil used to prevent anything being rounded to below 0 after
        # shift
        x_img -= int(np.floor(side_range[0] / res))
        y_img += int(np.floor(fwd_range[1] / res))

        # CLIP HEIGHT VALUES - to between min and max heights
        pixel_values = zi_points - height_range[0]
        # pixel_values = zi_points

        # FILL PIXEL VALUES IN IMAGE ARRAY
        top[y_img, x_img, i] = pixel_values

        # max_intensity = np.max(prs[idx])
        top[y_img, x_img, z_max] = ref_i

    return top

def point_cloud_to_panorama(points, v_res=0.42, h_res=0.35, v_fov=(-24.9, 2.0),
                            d_range=(0, 100), y_fudge=3, side_range=(-20., 20.),
                            fwd_range=(0.,40), height_range=(-2, 0.4)):
    """ Takes point cloud data as input and creates a 360 degree panoramic
        image, returned as a numpy array.

    Args:
        points: (np array)
            The numpy array containing the point cloud. .
            The shape should be at least Nx3 (allowing for more columns)
            - Where N is the number of points, and
            - each point is specified by at least 3 values (x, y, z)
        v_res: (float)
            vertical angular resolution in degrees. This will influence the
            height of the output image.
        h_res: (float)
            horizontal angular resolution in degrees. This will influence
            the width of the output image.
        v_fov: (tuple of two floats)
            Field of view in degrees (-min_negative_angle, max_positive_angle)
        d_range: (tuple of two floats) (default = (0,100))
            Used for clipping distance values to be within a min and max range.
        y_fudge: (float)
            A hacky fudge factor to use if the theoretical calculations of
            vertical image height do not match the actual data.
    Returns:
        A numpy array representing a 360 degree panoramic image of the point
        cloud.
    """
    # side_range = (-30., 30.)
    # fwd_range = (0., 60)
    # height_range = (-2, 0.4)  #
    xi_points = points[:, 0]
    yi_points = points[:, 1]
    zi_points = points[:, 2]
    reflectance = points[:, 3]

    f_filt = np.logical_and(
        (xi_points > fwd_range[0]), (xi_points < fwd_range[1]))
    s_filt = np.logical_and(
        (yi_points > -side_range[1]), (yi_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    z_filt = np.logical_and((zi_points >= height_range[0]),
                            (zi_points < height_range[1]))
    zfilter = np.logical_and(filter, z_filt)
    indices = np.argwhere(zfilter).flatten()
    print 'indice size'
    print indices.size

    x_points = xi_points[indices]
    print 'xi_points'
    print x_points
    y_points = yi_points[indices]
    z_points = zi_points[indices]
    r_points = reflectance[indices]
    r_max = max(r_points)
    z_max = max(z_points)
    r_min = min(r_points)
    z_min = min(z_points)

    # Projecting to 2D
    # x_points = points[:, 0]
    # y_points = points[:, 1]
    # z_points = points[:, 2]
    # r_points = points[:, 3]

    # d_points = np.sqrt(x_points ** 2 + y_points ** 2)  # map distance relative to origin
    # print 'd_points size', len(d_points)
    d_points = np.sqrt(x_points ** 2 + y_points ** 2 + z_points ** 2)  # abs distance
    # d_points = r_points
    # d_points = z_points

    # d_points = np.zeros(indices.size)
    # for i in range(indices.size):
    #     d_points[i] = z_points[i]

    # We use map distance, because otherwise it would not project onto a cylinder,
    # instead, it would map onto a segment of slice of a sphere.

    # RESOLUTION AND FIELD OF VIEW SETTINGS
    v_fov_total = -v_fov[0] + v_fov[1]

    # CONVERT TO RADIANS
    v_res_rad = v_res * (np.pi / 180)
    h_res_rad = h_res * (np.pi / 180)

    # MAPPING TO CYLINDER
    de_points = np.sqrt(x_points ** 2 + y_points ** 2)
    x_img = np.arctan2(y_points, x_points) / h_res_rad
    y_img = -(np.arctan2(z_points, de_points) / v_res_rad)

    # THEORETICAL MAX HEIGHT FOR IMAGE
    d_plane = (v_fov_total / v_res) / (v_fov_total * (np.pi / 180))
    h_below = d_plane * np.tan(-v_fov[0] * (np.pi / 180))
    h_above = d_plane * np.tan(v_fov[1] * (np.pi / 180))
    y_max = int(np.ceil(h_below + h_above + y_fudge))

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -180.0 / h_res / 2
    x_img = np.trunc(-x_img - x_min).astype(np.int32)
    x_max = int(np.ceil(180.0 / h_res))

    y_min = -((v_fov[1] / v_res) + y_fudge)
    y_img = np.trunc(y_img - y_min).astype(np.int32)

    # CLIP DISTANCES
    d_points = np.clip(d_points, a_min=d_range[0], a_max=d_range[1])

    # CONVERT TO IMAGE ARRAY
    img = np.ones([y_max + 1, x_max + 1, 3], dtype=np.uint8)*255
    distance = np.sqrt(x_points ** 2 + y_points ** 2 + z_points ** 2)
    dis_max = max(distance)
    dis_min = min(distance)
    img[y_img, x_img, 0] = scale_to_255(distance, min=dis_min, max=dis_max)
    img[y_img, x_img, 1] = scale_to_255(z_points, min=z_min, max=z_max)
    img[y_img, x_img, 2] = scale_to_255(r_points, min=r_min, max=r_max)
    return img


if __name__ == '__main__':
    while True:
        idx = input('Type a new index: ')
        show_bv(idx)


