import numpy as np
from skimage import morphology
from skimage import filters


def apply_postprocessing(volume, method="opening"):
    match method:
        case "opening":
            return post_process_morphology(volume, morphology.opening)
        case "closing":
            return post_process_morphology(volume, morphology.closing)
        case "gaussian_smoothing":
            return gaussian_blur_thresholding(volume)
        case _:
            return volume


def post_process_morphology(volume, post_function):
    classes = np.unique(volume)
    classes = classes[classes != 0]
    footprint = morphology.ball(radius=2)
    processed_volume = np.zeros_like(volume)

    for class_label in classes:
        mask = volume == class_label
        processed_mask = post_function(mask, footprint=footprint)
        processed_volume[processed_mask] = class_label

    return processed_volume


def gaussian_blur_thresholding(volume):
    classes = np.unique(volume)
    classes = classes[classes != 0]
    processed_volume = np.zeros_like(volume)

    for class_label in classes:
        mask = volume == class_label
        mask_smoothed = filters.gaussian(mask, sigma=1, preserve_range=True)
        mask = mask_smoothed > 0.5
        processed_volume[mask] = class_label

    return processed_volume