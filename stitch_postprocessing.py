# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import re
import skimage
import numpy as np
import nibabel as nib
from pathlib import Path
from functools import partial
from utils import tqdm_
from PIL import Image
from post_processing_utils import apply_postprocessing


def main(args: argparse.Namespace) -> None:
    file_paths = sorted(args.data_folder.glob("*"))

    m = re.compile(args.grp_regex)
    patient_dict = {}
    # get the paths to slices of each patient.
    for fp in file_paths:
        stem = fp.stem
        match = re.match(m, stem)
        id = match.group(1)

        if id not in patient_dict:
            patient_dict[id] = []
        patient_dict[id].append(fp)

    for id, paths in patient_dict.items():
        # Load all the slices and stak them.
        slices = []
        for path in sorted(paths):
            image = Image.open(path)
            arr = np.array(image).astype(np.uint8)
            slices.append(arr)

        vol = np.stack(slices, axis=2) / 63
        vol = vol.astype(np.uint8)

        # Resize and save the predictions.
        source_pattern = args.source_scan_pattern.replace("{id_}", id)
        gt_img = nib.load(source_pattern)
        gt = gt_img.get_fdata().astype(np.uint8)
        vol = skimage.transform.resize(
            vol,
            gt.shape,
            anti_aliasing=False,
            order=0,
            mode="constant",
            preserve_range=True,
        ).astype(np.uint8)

        vol = apply_postprocessing(vol, args.post_processing, args.radius, args.sigma)

        new_img = nib.Nifti1Image(vol, affine=gt_img.affine, header=gt_img.header)
        args.dest_folder.mkdir(exist_ok=True, parents=True)
        nib.save(new_img, args.dest_folder / (id + ".nii.gz"))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitching arguments")

    parser.add_argument("--data_folder", type=Path, required=True)
    parser.add_argument("--dest_folder", type=Path, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--grp_regex", type=str, required=True)
    parser.add_argument("--source_scan_pattern", type=str, required=True)

    # Post processing arguments:
    parser.add_argument(
        "--post_processing",
        default="closing",
        choices=["none", "opening", "closing", "gaussian_smoothing"],
    )
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=1)

    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
