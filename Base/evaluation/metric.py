import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from surface_dice import compute_dice_coefficient
from surface_dice import compute_surface_dice_at_tolerance
from surface_dice import compute_surface_distances
from torch import nn


def compute_flare_metric(
    gt_data, seg_data, case_spacing
):  # pylint: disable=missing-docstring
    """
    This function computes the Dice similarity coefficient and surface Dice at tolerance metric for
    ground truth and segmented data.

    Args:
      gt_data: a numpy array containing the ground truth segmentation data for a medical image, with
    shape (4, height, width, depth)
      seg_data: This parameter is a numpy array containing the predicted segmentation mask for a medical
    image. It has shape (4, H, W, D), where H, W, and D are the height, width, and depth of the image,
    respectively. The first dimension corresponds to the four different classes that can
      case_spacing: case_spacing is the spacing between slices of a medical image volume. It is used in
    the computation of surface distances between the ground truth and predicted masks in the
    compute_flare_metric function.

    Returns:
      two lists: `all_dice` and `all_surface_dice`.
    """
    all_dice = []
    all_surface_dice = []
    gt_data = gt_data.astype(np.uint8)
    seg_data = seg_data.astype(np.uint8)
    for i in range(4):
        gt_mask, pred_mask = gt_data[i], seg_data[i]
        if np.sum(gt_mask) == 0 and np.sum(pred_mask) == 0:
            DSC_i = 1
            NSD_i = 1
        elif np.sum(gt_mask) == 0 and np.sum(pred_mask) > 0:
            DSC_i = 0
            NSD_i = 0
        else:
            surface_distances = compute_surface_distances(
                gt_mask, pred_mask, case_spacing
            )
            DSC_i = compute_dice_coefficient(gt_mask, pred_mask)
            NSD_i = compute_surface_dice_at_tolerance(surface_distances, 1)
        all_dice.append(DSC_i)
        all_surface_dice.append(NSD_i)

    return all_dice, all_surface_dice


# This is a PyTorch module that calculates the Dice coefficient for binary or multi-class segmentation
# tasks.
class DiceMetric(nn.Module):  # pylint: disable=missing-docstring
    def __init__(self, dims=(2, 3, 4)):
        super().__init__()
        self.dims = dims

    def forward(
        self, predict, gt, activation="sigmoid", is_average=True
    ):  # pylint: disable=missing-docstring
        predict = predict.float()
        gt = gt.float()

        if activation == "sigmoid":
            pred = F.sigmoid(predict)
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1
        elif activation == "softmax":
            pred = F.softmax(predict, dim=1)

        intersection = torch.sum(pred * gt, dim=self.dims)
        union = torch.sum(pred, dim=self.dims) + torch.sum(gt, dim=self.dims)
        dice = (2.0 * intersection + 1e-8) / (union + 1e-8)
        dice = dice.mean(0) if is_average else dice.sum(0)

        return dice


def compute_dice(predict, gt):  # pylint: disable=missing-docstring
    """
    This function computes the Dice coefficient between two arrays.

    Args:
      predict: A numpy array representing the predicted segmentation mask.
      gt: The ground truth segmentation mask, which is a binary array where each pixel is either 0 or 1
    indicating whether that pixel belongs to the object of interest or not.

    Returns:
      the Dice coefficient, which is a measure of similarity between two sets of data. Specifically, it
    is computing the Dice coefficient between two binary arrays, `predict` and `gt`, which represent
    predicted and ground truth segmentation masks, respectively. The Dice coefficient is a value between
    0 and 1, where 1 indicates perfect overlap between the two masks and 0 indicates no overlap
    """
    predict = predict.astype(np.float)
    gt = gt.astype(np.float)
    intersection = np.sum(predict * gt)
    union = np.sum(predict + gt)
    dice = (2.0 * intersection + 1e-8) / (union + 1e-8)

    return dice


def compute_precision_recall_f1(
    predict, gt, num_class
):  # pylint: disable=missing-docstring
    """
    This function computes precision, recall, and F1 score for multiple classes based on predicted and
    ground truth labels.

    Args:
      predict: An array of predicted labels for each sample in the dataset.
      gt: gt stands for "ground truth" and refers to the true labels or classes of the data. It is a
    numpy array of shape (n_samples,) where n_samples is the number of samples in the dataset.
      num_class: The number of classes in the classification problem.

    Returns:
      three lists: precision, recall, and f1. Each list contains num_class elements, where each element
    corresponds to a different class label.
    """
    tp, tp_fp, tp_fn = [0.0] * num_class, [0.0] * num_class, [0.0] * num_class
    precision, recall, f1 = [0.0] * num_class, [0.0] * num_class, [0.0] * num_class
    for label in range(num_class):
        t_labels = gt == label
        p_labels = predict == label
        tp[label] += np.sum(t_labels == (p_labels * 2 - 1))
        tp_fp[label] += np.sum(p_labels)
        tp_fn[label] += np.sum(t_labels)
        precision[label] = tp[label] / (tp_fp[label] + 1e-8)
        recall[label] = tp[label] / (tp_fn[label] + 1e-8)
        f1[label] = (
            2
            * precision[label]
            * recall[label]
            / (precision[label] + recall[label] + 1e-8)
        )

    return precision, recall, f1


def compute_overlap_metrics(pred_npy, target_npy, metrics=None):
    """
    The function computes overlap metrics between two input images using SimpleITK library.

    Args:
      pred_npy: A numpy array representing the predicted segmentation mask.
      target_npy: A numpy array representing the ground truth segmentation mask.
      metrics: A list of strings representing the metrics to be computed. If None, the default metric
    "dice" will be computed. The available metrics are "jaccard", "dice", "volume_similarity",
    "false_negative", and "false_positive".

    Returns:
      The function `compute_overlap_metrics` returns a dictionary containing the overlap measures
    computed between two input images (pred and target) based on the specified metrics. The keys of the
    dictionary correspond to the names of the metrics, and the values are the computed values of the
    metrics.
    """
    if metrics is None:
        metrics = ["dice"]

    for metric in metrics:
        if metric not in {
            "jaccard",
            "dice",
            "volume_similarity",
            "false_negative",
            "false_positive",
        }:
            raise ValueError(f"Does not exist the {metric} metric")

    pred = sitk.GetImageFromArray(pred_npy)
    target = sitk.GetImageFromArray(target_npy)
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(target, pred)

    overlap_results = {}
    for metric in metrics:
        if metric == "jaccard":
            overlap_results["jaccard"] = overlap_measures_filter.GetJaccardCoefficient()
        elif metric == "dice":
            overlap_results["dice"] = overlap_measures_filter.GetDiceCoefficient()
        elif metric == "volume_similarity":
            overlap_results[
                "volume_similarity"
            ] = overlap_measures_filter.GetVolumeSimilarity()
        elif metric == "false_negative":
            overlap_results[
                "false_negative"
            ] = overlap_measures_filter.GetFalseNegativeError()
        elif metric == "false_positive":
            overlap_results[
                "false_positive"
            ] = overlap_measures_filter.GetFalsePositiveError()

    return overlap_results


def compute_hausdorff_distance(pred_npy, target_npy):
    """
    This function computes the Hausdorff distance between two 3D images using SimpleITK library in
    Python.

    Args:
      pred_npy: A NumPy array representing the predicted segmentation mask.
      target_npy: It is a numpy array representing the ground truth segmentation mask or image that we
    want to compare with the predicted segmentation mask or image.

    Returns:
      the Hausdorff distance between two input images, which is a measure of the maximum distance
    between the closest points of the two images.
    """
    pred = sitk.GetImageFromArray(pred_npy)
    target = sitk.GetImageFromArray(target_npy)
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter.Execute(target, pred)

    return hausdorff_distance_filter.GetHausdorffDistance()


def compute_surface_distance_statistics(pred_npy, target_npy):
    """
    This function computes surface distance statistics between two binary segmentations using SimpleITK
    library in Python.

    Args:
      pred_npy: The predicted segmentation as a numpy array.
      target_npy: `target_npy` is a numpy array representing the ground truth segmentation. It is used
    to compute the surface distance measures.

    Returns:
      a dictionary containing the following surface distance statistics: mean_surface_distance,
    median_surface_distance, std_surface_distance, and max_surface_distance.
    """
    pred = sitk.GetImageFromArray(pred_npy)
    target = sitk.GetImageFromArray(target_npy)

    # Symmetric surface distance measures
    # Use the absolute values of the distance map to compute the surface distances (distance map sign, outside or
    # inside relationship, is irrelevant)
    segmented_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(pred, squaredDistance=False, useImageSpacing=True)
    )
    segmented_surface = sitk.LabelContour(pred)
    reference_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(target, squaredDistance=False)
    )
    reference_surface = sitk.LabelContour(target)

    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum())

    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    seg2ref_distance_map = reference_distance_map * sitk.Cast(
        segmented_surface, sitk.sitkFloat32
    )
    ref2seg_distance_map = segmented_distance_map * sitk.Cast(
        reference_surface, sitk.sitkFloat32
    )

    # Get all non-zero distances and then add zero distances if required.
    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
    seg2ref_distances = seg2ref_distances + list(
        np.zeros(num_segmented_surface_pixels - len(seg2ref_distances))
    )
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
    ref2seg_distances = ref2seg_distances + list(
        np.zeros(num_reference_surface_pixels - len(ref2seg_distances))
    )
    all_surface_distances = seg2ref_distances + ref2seg_distances

    # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In
    # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two
    # segmentations, though in our case it is. More on this below.
    surface_distance_results = {}
    surface_distance_results["mean_surface_distance"] = np.mean(all_surface_distances)
    surface_distance_results["median_surface_distance"] = np.median(
        all_surface_distances
    )
    surface_distance_results["std_surface_distance"] = np.std(all_surface_distances)
    surface_distance_results["max_surface_distance"] = np.max(all_surface_distances)

    return surface_distance_results
