import numpy as np


def voxel_coord_2_world(voxel_coord, origin, spacing, directions=None):
    """
    This function converts voxel coordinates to world coordinates using origin, spacing, and directions.

    Args:
      voxel_coord: The voxel coordinates of a point in a 3D image volume. These are typically integers
    representing the location of the point in the volume along the x, y, and z axes.
      origin: The origin is a list or array of length N, where N is the number of dimensions in the
    image. It represents the physical coordinates of the center of the voxel at index (0, 0, ..., 0) in
    the image.
      spacing: The spacing parameter is a list or array containing the size of each voxel in
    millimeters. It specifies the physical distance between adjacent voxels in the three dimensions of
    the image volume.
      directions: The directions parameter is an optional input that specifies the orientation of the
    image axes in the world coordinate system. It is a numpy array of length equal to the number of
    dimensions in the image. By default, it is set to an array of ones, which assumes that the image
    axes are aligned with the

    Returns:
      the world coordinates of a voxel given its voxel coordinates, origin, spacing, and directions.
    """
    if directions is None:
        directions = np.array([1] * len(voxel_coord))
    stretched_voxel_coord = np.array(voxel_coord) * np.array(spacing).astype(float)
    world_coord = np.array(origin) + stretched_voxel_coord * np.array(directions)

    return world_coord


def world_2_voxel_coord(world_coord, origin, spacing):
    """
    The function converts world coordinates to voxel coordinates using origin and spacing information.

    Args:
      world_coord: A list or array of 3D coordinates in the world coordinate system.
      origin: The origin is a list or array of three values representing the x, y, and z coordinates of
    the origin point in the world coordinate system. This is the point from which the voxel coordinates
    will be calculated.
      spacing: Spacing is a list or array of three values representing the distance between adjacent
    voxels in the x, y, and z directions, respectively. For example, if the spacing is [1.0, 1.0, 1.0],
    it means that adjacent voxels are 1 unit

    Returns:
      the voxel coordinates of a point in the world coordinate system, given the origin and spacing of
    the voxel grid.
    """
    stretched_voxel_coord = np.array(world_coord) - np.array(origin)
    voxel_coord = np.absolute(stretched_voxel_coord / np.array(spacing))

    return voxel_coord


def change_axes_of_image(npy_image, orientation):
    """
    The function changes the orientation of a numpy image array based on the given orientation
    parameter.

    Args:
      npy_image: The input image in numpy array format.
      orientation: The orientation parameter is a list of three values that determine the direction of
    each axis of the image. The default value is [1, -1, -1], which means that the first axis (x) is
    oriented from left to right, the second axis (y) is oriented from top to

    Returns:
      the numpy array image with its axes changed based on the given orientation.
    """
    if orientation[0] < 0:
        npy_image = np.flip(npy_image, axis=0)
    if orientation[1] > 0:
        npy_image = np.flip(npy_image, axis=1)
    if orientation[2] > 0:
        npy_image = np.flip(npy_image, axis=2)
    return npy_image


def clip_image(image, min_window=-1200, max_window=600):
    """
    The function clips the pixel values of an image to a specified range.

    Args:
      image: The input image that needs to be clipped.
      min_window: The minimum pixel value that the image can have after clipping. Any pixel value below
    this value will be set to this minimum value.
      max_window: The maximum pixel value that the image can have after clipping. Any pixel value above
    this value will be set to this maximum value. Defaults to 600
    Returns:
      The function `clip_image` is returning the clipped image, where the pixel values are limited to
    the range between `min_window` and `max_window`. The returned value is a numpy array.
    """
    return np.clip(image, min_window, max_window)


def normalize_min_max_and_clip(image, min_window=-1200.0, max_window=600.0):
    """
    The function normalizes the HU values of an image to a range of [-1, 1] using a specified window and
    clips the values outside this range.

    Args:
    image: The input image to be normalized.
    min_window: The minimum Hounsfield unit (HU) value for the windowing range. This is typically the
    lower end of the range of values that are considered to be within the range of interest for a
    particular imaging study or application.
    max_window: The maximum value of the window used to normalize the Hounsfield Units (HU) of the
    medical image. This value is typically chosen based on the tissue of interest and the imaging
    modality used. In this case, the value is set to 600.0.

    Returns:
    the normalized image with HU values in the range of [-1, 1] using the specified window of
    [min_window, max_window].
    """
    image = (image - min_window) / (max_window - min_window)
    image = image * 2 - 1.0
    image = image.clip(-1, 1)
    return image


def normalize_mean_std(image, global_mean=None, global_std=None):
    """
    The function normalizes an image by subtracting its mean and dividing by its standard deviation,
    with the option to use a global mean and standard deviation.

    Args:
      image: The input image that needs to be normalized.
      global_mean: The global mean is the mean value of the entire dataset or population. It is used to
    normalize the image by subtracting the global mean from each pixel value. If the global mean is not
    provided, the function calculates it from the input image.
      global_std: The global standard deviation is a statistical measure that indicates how much the
    values in a dataset deviate from the mean value. In the context of this function, the global_std
    parameter is used to normalize an image by subtracting the global mean and dividing by the global
    standard deviation. If the global_mean and

    Returns:
      the normalized image.
    """
    if not global_mean or not global_std:
        mean = np.mean(image)
        std = np.std(image)
    else:
        mean, std = global_mean, global_std

    image = (image - mean) / (std + 1e-5)
    return image


def clip_and_normalize_mean_std(image, min_window=-1200, max_window=600):
    """
    The function clips an input image to a specified range and normalizes it based on its mean and
    standard deviation.

    Args:
      image: a 2D or 3D numpy array representing a medical image
      min_window: The minimum value to which the pixel values in the image will be clipped. Any pixel
    value below this minimum value will be set to the minimum value.
      max_window: The maximum value of the window used to clip the image pixel values. Any pixel value
    above this value will be set to this value. Defaults to 600

    Returns:
      the normalized image after clipping the pixel values between the minimum and maximum window
    values, and then normalizing the pixel values by subtracting the mean and dividing by the standard
    deviation.
    """
    image = np.clip(image, min_window, max_window)
    mean = np.mean(image)
    std = np.std(image)

    image = (image - mean) / (std + 1e-5)
    return image
