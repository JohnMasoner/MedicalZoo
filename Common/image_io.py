import os

import SimpleITK as sitk


def load_nii_file(data_path):
    """
    This function loads a NIfTI file and returns its metadata and image data as a dictionary.

    Args:
    data_path: The path to the NIfTI file that needs to be loaded.

    Returns:
    a dictionary containing the following keys:
    - "sitk_image": a SimpleITK image object
    - "npy_image": a numpy array representing the image data
    - "origin": a tuple representing the origin of the image
    - "spacing": a tuple representing the spacing of the image
    - "direction": a tuple representing the direction of the image.
    """
    assert os.path.exists(
        data_path
    ), f"Could not find {data_path}, Please check your configuration"
    try:
        sitk_image = sitk.ReadImage(data_path)

        assert sitk_image is not None, f"Could not load {data_path}"

        # origin = list(reversed(sitk_image.GetOrigin()))
        # spacing = list(reversed(sitk_image.GetSpacing()))
        spacing = sitk_image.GetSpacing()
        origin = sitk_image.GetOrigin()
        direction = sitk_image.GetDirection()
        # direction = [direction[8], direction[4], direction[0]]
        res = {
            "sitk_image": sitk_image,
            "npy_image": sitk.GetArrayFromImage(sitk_image),
            "origin": origin,
            "spacing": spacing,
            "direction": direction,
        }
        return res
    except Exception as err:
        print(f"Exiting some errors in {data_path} \nThe error message: {err}")
    return None


def save_ct_from_sitk(sitk_image, save_path, sitk_type=None, use_compression=False):
    """
    This function saves a SimpleITK image to a specified path with optional type casting and
    compression.

    Args:
      sitk_image: This is a SimpleITK image object that contains the image data to be saved.
      save_path: The file path where the SimpleITK image will be saved.
      sitk_type: The data type to which the SimpleITK image should be cast before saving. If it is not
    specified, the image will be saved in its original data type.
      use_compression: use_compression is a boolean parameter that determines whether or not to use
    compression when saving the image. If set to True, the image will be compressed to reduce its file
    size. If set to False, the image will be saved without compression. Defaults to False
    """
    if sitk_type is not None:
        sitk_image = sitk.Cast(sitk_image, sitk_type)
    sitk.WriteImage(sitk_image, save_path, use_compression)


def save_ct_from_npy(
    npy_image,
    save_path,
    origin=None,
    spacing=None,
    direction=None,
    sitk_type=None,
    use_compression=False,
):
    """
    This function saves a CT image from a numpy array using SimpleITK, with options to set origin,
    spacing, direction, image type, and compression.

    Args:
      npy_image: A numpy array representing a CT image.
      save_path: The file path where the CT image will be saved.
      origin: The origin is a tuple of three floats that represents the physical coordinates of the
    pixel with index (0,0,0) in the image. It defines the position of the image in physical space.
      spacing: Spacing refers to the physical distance between adjacent pixels in the image. It is
    usually represented as a tuple of three values (x, y, z) where x, y, and z are the distances between
    adjacent pixels in the x, y, and z directions respectively.
      direction: The direction parameter is used to set the direction cosines of the image. It is a
    9-element list or tuple representing the 3x3 direction matrix. The direction matrix describes the
    orientation of the image in physical space.
      sitk_type: sitk_type is a parameter that specifies the SimpleITK image type to use when saving the
    image. It can be one of the following values: sitk.sitkUInt8, sitk.sitkInt16, sitk.sitkUInt16,
    sitk.sitkInt
      use_compression: use_compression is a boolean parameter that determines whether or not to use
    compression when saving the image. If set to True, the image will be saved with compression, which
    can reduce the file size but may also increase the time required to save the image. If set to False,
    the image will be. Defaults to False
    """
    sitk_image = sitk.GetImageFromArray(npy_image)
    if origin is not None:
        sitk_image.SetOrigin(origin)
    if spacing is not None:
        sitk_image.SetSpacing(spacing)
    if direction is not None:
        sitk_image.SetDirection(direction)
    save_ct_from_sitk(sitk_image, save_path, sitk_type, use_compression)


def convert_npy_image_2_sitk(
    npy_image, origin=None, spacing=None, direction=None, sitk_type=sitk.sitkFloat32
):
    """
    This function converts a numpy array image to a SimpleITK image with specified origin, spacing,
    direction, and data type.

    Args:
      npy_image: A numpy array representing an image.
      origin: The origin parameter is a tuple of three floats representing the physical coordinates of
    the origin of the image in millimeters. It specifies the position of the first voxel in the image
    with respect to the physical world coordinate system.
      spacing: Spacing refers to the physical distance between adjacent pixels or voxels in an image. It
    is usually expressed in millimeters or micrometers. In medical imaging, spacing is an important
    parameter as it affects the accuracy of measurements and the quality of image registration.
      direction: The direction parameter is a 9-element list or tuple that specifies the orientation of
    the image axes in 3D space. It is used to define the relationship between the image voxel
    coordinates and the physical world coordinates. The first three elements of the direction vector
    represent the direction of the first image axis in
      sitk_type: The data type of the SimpleITK image. It is set to sitk.sitkFloat32 by default, but can
    be changed to other data types such as sitk.sitkUInt8 or sitk.sitkInt16.

    Returns:
      a SimpleITK image object created from a NumPy array, with optional specifications for origin,
    spacing, direction, and SimpleITK image type.
    """
    sitk_image = sitk.GetImageFromArray(npy_image)
    sitk_image = sitk.Cast(sitk_image, sitk_type)
    if origin is not None:
        sitk_image.SetOrigin(origin)
    if spacing is not None:
        sitk_image.SetSpacing(spacing)
    if direction is not None:
        sitk_image.SetDirection(direction)
    return sitk_image


def convert_npy_mask_2_sitk(
    npy_mask,
    label=None,
    origin=None,
    spacing=None,
    direction=None,
    sitk_type=sitk.sitkUInt8,
):
    """
    This function converts a numpy array mask to a SimpleITK image with specified label, origin,
    spacing, direction, and data type.

    Args:
      npy_mask: A numpy array representing a binary or labeled mask.
      label: The label parameter is an optional integer value that can be used to replace all non-zero
    values in the input numpy array with the specified label value. This can be useful when converting
    segmentation masks where each label corresponds to a different object or region of interest.
      origin: The origin parameter is a tuple of three floats that represents the physical coordinates
    of the pixel with index (0,0,0) in the image. It defines the position of the image in physical
    space.
      spacing: Spacing refers to the physical distance between adjacent pixels or voxels in an image. It
    is usually expressed in millimeters or micrometers. In medical imaging, spacing is an important
    parameter as it affects the accuracy of measurements and the quality of image registration.
      direction: The direction parameter is used to set the direction cosines of the image. It is a
    9-element list or tuple representing the 3x3 direction matrix. The direction matrix describes the
    orientation of the image in physical space.
      sitk_type: sitk_type is a parameter that specifies the SimpleITK image type to which the numpy
    array mask will be converted. It has a default value of sitk.sitkUInt8, which means that the
    resulting SimpleITK image will have pixel values in the range [0, 255].

    Returns:
      a SimpleITK image object (sitk_mask) after converting a numpy array (npy_mask) to a SimpleITK
    image. The sitk_mask image object has optional modifications to its metadata (label, origin,
    spacing, and direction) based on the input arguments.
    """
    if label is not None:
        npy_mask[npy_mask != 0] = label
    sitk_mask = sitk.GetImageFromArray(npy_mask)
    sitk_mask = sitk.Cast(sitk_mask, sitk_type)
    if origin is not None:
        sitk_mask.SetOrigin(origin)
    if spacing is not None:
        sitk_mask.SetSpacing(spacing)
    if direction is not None:
        sitk_mask.SetDirection(direction)
    return sitk_mask
