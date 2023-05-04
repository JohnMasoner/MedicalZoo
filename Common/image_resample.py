import numpy as np
from scipy.ndimage import zoom


class MedicalImageResampler:
    """
    The MedicalImageResampler class provides static methods for resampling medical images and masks to a
    target spacing or size while preserving label information and accounting for overlap.
    """

    @staticmethod
    def resample_to_spacing(npy_image, source_spacing, target_spacing, order=1):
        """
        resample_to_spacing.

        Args:
            npy_image: numpy.ndarray, the input image
            source_spacing: source_spacing to get the pixel distance
            target_spacing: target spacing, source_spacing/target_spacing get zoom factor
            order:
        """
        scale = np.array(source_spacing[::-1]) / np.array(target_spacing[::-1])
        zoom_factor = np.array(target_spacing[::-1]) / np.array(source_spacing[::-1])
        target_npy_image = zoom(npy_image, scale, order=order)
        return target_npy_image, zoom_factor, scale

    @staticmethod
    def resample_to_size(npy_image, target_size, order=1):
        """
        The function resamples a numpy image to a target size using zoom function and returns the resampled
        image and zoom factor.

        Args:
          npy_image: A numpy array representing an image.
          target_size: The desired size of the resampled image. It is a tuple of (height, width) in pixels.
          order: The order parameter specifies the order of the spline interpolation used for resampling the
        image. It can take values from 0 to 5, with 0 representing nearest-neighbor interpolation and higher
        values representing higher-order spline interpolation. The default value is 1, which corresponds to
        bilinear interpolation. Defaults to 1

        Returns:
          two values: the resampled image (target_npy_image) and the zoom factor used to resample the image
        (zoom_factor).
        """
        source_size = npy_image.shape
        scale = np.array(target_size) / source_size
        zoom_factor = source_size / np.array(target_size)
        target_npy_image = zoom(npy_image, scale, order=order)
        return target_npy_image, zoom_factor

    @staticmethod
    def resample_mask_to_spacing(
        npy_mask, source_spacing, target_spacing, num_label, order=1
    ):
        """
        This function resamples a given mask to a target spacing while preserving the label information.

        Args:
          npy_mask: A numpy array representing a mask image.
          source_spacing: The spacing (distance between adjacent pixels/voxels) of the input mask in the
        original coordinate system. It is a tuple of three values representing the spacing along the x, y,
        and z axes respectively.
          target_spacing: The desired spacing of the resampled mask. It is a list or array of three values
        representing the spacing in x, y, and z directions.
          num_label: The number of labels or classes in the mask. For example, if the mask has two classes
        (foreground and background), then num_label would be 2.
          order: The order parameter specifies the order of the interpolation used during the resampling
        process. It can take integer values from 0 to 5, with 0 being nearest-neighbor interpolation and 5
        being a high-order spline interpolation. The default value is 1, which corresponds to bilinear
        interpolation. Defaults to 1

        Returns:
          three values: target_npy_mask, zoom_factor, and scale.
        """
        scale = np.array(source_spacing) / np.array(target_spacing)
        zoom_factor = np.array(target_spacing) / np.array(source_spacing)
        target_npy_mask = np.zeros_like(npy_mask)
        target_npy_mask = zoom(target_npy_mask, scale, order=order)
        for i in range(1, num_label + 1):
            current_mask = npy_mask.copy()

            current_mask[current_mask != i] = 0
            current_mask[current_mask == i] = 1

            current_mask = zoom(current_mask, scale, order=order)
            current_mask = (current_mask > 0.5).astype(np.uint8)
            target_npy_mask[current_mask != 0] = i
        return target_npy_mask, zoom_factor, scale

    @staticmethod
    def resample_mask_to_size(npy_mask, target_size, num_label, order=1):
        """
        The function resamples a numpy mask to a target size while preserving the labels and returns the
        resampled mask and the zoom factor.

        Args:
          npy_mask: A numpy array representing a mask image.
          target_size: The desired size of the resampled mask. It is a tuple of (height, width, depth) or
        (height, width) depending on the dimensionality of the mask.
          num_label: num_label is the number of labels or classes in the mask. For example, if the mask has
        two classes - foreground and background, then num_label would be 2.
          order: The order parameter specifies the order of the interpolation used during resampling. It can
        take values from 0 to 5, with 0 being nearest-neighbor interpolation and 5 being a high-order spline
        interpolation. The default value is 1, which corresponds to bilinear interpolation. Defaults to 1

        Returns:
          two values: `target_npy_mask` and `zoom_factor`.
        """
        source_size = npy_mask.shape
        scale = np.array(target_size) / source_size
        zoom_factor = source_size / np.array(target_size)
        target_npy_mask = np.zeros_like(npy_mask)
        target_npy_mask = zoom(target_npy_mask, scale, order=order)
        for i in range(1, num_label + 1):
            current_mask = npy_mask.copy()

            current_mask[current_mask != i] = 0
            current_mask[current_mask == i] = 1

            current_mask = zoom(current_mask, scale, order=order)
            current_mask = (current_mask > 0.5).astype(np.uint8)
            target_npy_mask[current_mask != 0] = i
        return target_npy_mask, zoom_factor

    @staticmethod
    def resample_overlap_mask_to_spacing(
        npy_mask, source_spacing, target_spacing, order=1
    ):
        """
        This function resamples a given mask to a target spacing while accounting for overlap.

        Args:
          npy_mask: A numpy array representing a binary mask.
          source_spacing: The spacing (distance between adjacent pixels/voxels) of the input numpy mask
        array in the format (z, y, x) or (y, x) depending on the dimensionality of the mask.
          target_spacing: The desired spacing of the resampled mask. It is a tuple or list of three values
        representing the spacing in x, y, and z dimensions, respectively.
          order: The order parameter specifies the order of the interpolation used during resampling. It can
        take integer values from 0 to 5, with 0 being nearest-neighbor interpolation and 5 being a
        high-order spline interpolation. The default value is 1, which corresponds to bilinear
        interpolation. Defaults to 1

        Returns:
          three values: target_npy_mask, zoom_factor, and scale.
        """
        scale = np.array(source_spacing[::-1]) / np.array(target_spacing[::-1])
        scale = np.concatenate((scale, [1]))
        zoom_factor = np.array(target_spacing[::-1]) / np.array(source_spacing[::-1])
        zoom_factor = np.concatenate((zoom_factor, [1]))
        target_npy_mask = np.zeros_like(npy_mask)
        target_npy_mask = zoom(target_npy_mask, scale, order=order)
        current_mask = npy_mask.copy()

        current_mask[current_mask != 1] = 0
        current_mask[current_mask == 1] = 1

        current_mask = zoom(current_mask, scale, order=order)
        current_mask = (current_mask > 0.5).astype(np.uint8)
        target_npy_mask[current_mask != 0] = 1
        return target_npy_mask, zoom_factor, scale

    @staticmethod
    def resample_overlap_mask_to_size(npy_mask, target_size, order=1):
        """
        The function resamples and overlaps a given numpy mask to a target size using zoom function and
        returns the resampled mask and zoom factor.

        Args:
          npy_mask: A numpy array representing a binary mask.
          target_size: The desired size of the resampled mask. It should be a tuple of integers representing
        the target size in each dimension (e.g. (256, 256)).
          order: The order parameter specifies the order of the interpolation used during resampling. It can
        take values from 0 to 5, with 0 being nearest-neighbor interpolation and 5 being a high-order spline
        interpolation. The default value is 1, which corresponds to bilinear interpolation. Defaults to 1

        Returns:
          a resampled mask array of size `target_size` with overlapping regions from the input `npy_mask`
        array. The function also returns the zoom factor used to resample the mask.
        """
        source_size = npy_mask.shape
        scale = np.array(target_size) / source_size
        zoom_factor = source_size / np.array(target_size)
        target_npy_mask = np.zeros_like(npy_mask)
        target_npy_mask = zoom(target_npy_mask, scale, order=order)
        current_mask = npy_mask.copy()

        current_mask[current_mask != 1] = 0
        current_mask[current_mask == 1] = 1

        current_mask = zoom(current_mask, scale, order=order)
        current_mask = (current_mask > 0.5).astype(np.uint8)
        target_npy_mask[current_mask != 0] = 1
        return target_npy_mask, zoom_factor
