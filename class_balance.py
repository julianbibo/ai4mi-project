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

    return parser.parse_args()

def save_class_balance(
    data_folder: Path,
    dest_folder: Path,
    num_classes: int,
    grp_regex: str,
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

    # class balances
    class_balance = np.zeros(num_classes)

    for id in ids:
        patient_files = sorted(list(filter(lambda file: id in file, files)))
        
        for file in patient_files:
            # load slice
            slice = np.asarray(Image.open(data_folder / file))

            # scale by 1/63 if necessary
            if np.max(slice) >= num_classes:
                slice = slice // 63

            inds, counts = np.unique(slice, return_counts=True)

            for idx, count in zip(inds, counts):
                class_balance[idx] += count

    # make sure output directory exists
    out_path = dest_folder / f"class_balance.txt"
    dest_folder.mkdir(parents=True, exist_ok=True)

    print(f"Computed class balance: {class_balance}")

    with open(out_path, "w") as f:
        f.write(str(list(class_balance)))

        print(f"Saved class balance to {f}")


if __name__ == "__main__":
    args = parse_args()

    save_class_balance(
        Path(args.data_folder),
        Path(args.dest_folder),
        args.num_classes,
        args.grp_regex,
    )