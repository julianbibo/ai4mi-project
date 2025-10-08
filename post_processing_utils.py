import numpy as np
from skimage import morphology
from skimage import filters


def apply_postprocessing(volume, method="opening", radius=2, sigma=1):
    match method:
        case "opening":
            return post_process_morphology(volume, morphology.opening, radius=radius)
        case "closing":
            return post_process_morphology(volume, morphology.closing, radius=radius)
        case "gaussian_smoothing":
            return gaussian_blur_thresholding(volume, sigma=sigma)
        case _:
            return volume


def post_process_morphology(volume, post_function, radius=2):
    classes = np.unique(volume)
    classes = classes[classes != 0]
    footprint = morphology.ball(radius=radius)
    processed_volume = np.zeros_like(volume)

    for class_label in classes:
        mask = volume == class_label
        processed_mask = post_function(mask, footprint=footprint)

        # if a class prediction is completely removed after 
        # applying the filter, leave it as it was.
        if processed_mask.sum() == 0:
            processed_mask = mask

        processed_volume[processed_mask] = class_label

    return processed_volume


def gaussian_blur_thresholding(volume, sigma=1):
    classes = np.unique(volume)
    classes = classes[classes != 0]
    processed_volume = np.zeros_like(volume)

    for class_label in classes:
        mask = volume == class_label
        mask_smoothed = filters.gaussian(mask, sigma=sigma, preserve_range=True)
        mask_smoothed = mask_smoothed > 0.5

        if mask_smoothed.sum() == 0:
            mask_smoothed = mask

        processed_volume[mask_smoothed] = class_label

    return processed_volume