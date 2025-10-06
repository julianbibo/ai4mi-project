import pickle
import random
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable
import os, re
from PIL import Image
import skimage

import numpy as np
import nibabel as nib

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Stitching params")

    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--dest_folder', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--grp_regex', type=str, default="(Patient_\d\d)_\d\d\d\d")
    parser.add_argument(
        "--source_scan_pattern",
        type=str,
        default="data/train/train/{id_}/GT.nii.gz"
    )
    return parser.parse_args()

def stitch(
    data_folder: Path,
    dest_folder: Path,
    num_classes: int,
    grp_regex: str,
    source_scan_pattern: str,
) -> None:     
    # gather patient IDs
    files = os.listdir(data_folder)

    ids = []
    for file in data_folder.glob("*.png"):
        res = re.search(rf"{grp_regex}", str(file))
        if res is not None:
            ids.append(res.group(1))
    ids = set(ids)

    print(ids)

    for id in ids:
        # load original scan
        original_scan = nib.load(source_scan_pattern.format(id_=id))
        original_scan_shape = original_scan.shape

        patient_files = sorted(list(filter(lambda file: id in file, files)))
        assert len(patient_files) == original_scan_shape[2], f"Too few slices for patient {id}!"
        
        out = np.empty(original_scan_shape, dtype=np.uint8)

        for i, file in enumerate(patient_files):
            # load slice
            slice = np.asarray(Image.open(data_folder / file))
            
            # resize to original scan's size
            slice = skimage.transform.resize(
                slice,
                original_scan_shape[:2],
                order=0,
                preserve_range=True,
            )

            # store slice (note that files are already sorted by slice index)
            out[:, :, i] = slice

        # scale by 1/63 if necessary
        if np.max(out) >= num_classes:
            out = out // 63

        # make sure output directory exists
        out_path = dest_folder / f"{id}.nii.gz"
        dest_folder.mkdir(parents=True, exist_ok=True)

        nib.save(
            nib.nifti1.Nifti1Image(out, original_scan.affine),
            out_path,
        )


if __name__ == "__main__":
    args = parse_args()

    stitch(
        Path(args.data_folder),
        Path(args.dest_folder),
        args.num_classes,
        args.grp_regex,
        args.source_scan_pattern,
    )