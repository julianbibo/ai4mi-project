from collections import defaultdict
import numpy as np
import nibabel as nib
from PIL import Image
from skimage.transform import resize

import argparse
import re
from pathlib import Path

# should be callable with:
# python stitch.py --data_folder FOLDER_SLICED_DATA \
# --dest_folder FOLDER_STITCHED_DATA --num_classes 5 \
# --grp_regex "(Patient_\d\d)_\d\d\d\d" \
# --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"
CLASS_MAPPING = {0: 0, 63: 1, 126: 2, 189: 3, 252: 4}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--dest_folder", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--grp_regex", type=str, default=r"(Patient_\d\d)_\d\d\d\d")
    parser.add_argument(
        "--source_scan_pattern",
        type=str,
        default="data/segthor_train/train/{id_}/GT.nii.gz",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    data_folder = Path(args.data_folder)
    dest_folder = Path(args.dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)

    # collect all patient slices
    print("Collecting PNG filenames")
    pattern = re.compile(args.grp_regex)
    patient_slice_filenames = defaultdict(list)
    for file in data_folder.glob("*.png"):
        match = pattern.search(file.name)
        if match:
            # print("matched", file.name, match.group(1))
            patient_id = match.group(1).strip()
            patient_slice_filenames[patient_id].append(file)

    # for each patient, sort slices and stitch them
    for patient_id, patient_pngs in patient_slice_filenames.items():
        print(f"Stitching {len(patient_pngs)} slices for patient {patient_id}")
        patient_pngs.sort()

        slice_arrays = []
        for filename in patient_pngs:
            arr = np.array(Image.open(filename)).astype(np.uint8)
            remapped = np.zeros_like(arr, dtype=np.uint8)
            for k, v in CLASS_MAPPING.items():
                remapped[arr == k] = v
            slice_arrays.append(remapped)

        volume = np.stack(slice_arrays, axis=-1)  # (H, W, D)

        orig_path = args.source_scan_pattern.format(id_=patient_id)
        orig_img_nib: nib.nifti1.Nifti1Image = nib.load(orig_path)  # type: ignore
        orig_img_np: np.ndarray = np.asarray(orig_img_nib.dataobj)
        # resize to orig size
        volume: np.ndarray = resize(
            volume, orig_img_np.shape, order=0, preserve_range=True, anti_aliasing=False
        )  # type: ignore
        volume = volume.astype(np.uint8)
        assert orig_img_np.shape == volume.shape, (
            f"Shape mismatch: {orig_img_np.shape} vs {volume.shape}"
        )

        # Save stitched volume
        out_path = dest_folder / f"{patient_id}.nii.gz"
        nib.save(nib.Nifti1Image(volume, affine=orig_img_nib.affine), str(out_path))  # type: ignore
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
