import numpy as np
from pathlib import Path


def main():
    base_path = Path("results")
    results = []

    # Find all folders with the required npy files
    for x_path in base_path.iterdir():
        metrics_path = x_path / "metrics"
        dice_path = metrics_path / "3d_dice.npy"
        hd95_path = metrics_path / "3d_hd95.npy"

        if dice_path.is_file() and hd95_path.is_file():
            dice: np.ndarray = np.load(dice_path)
            # print(f"{dice.shape=}")  # shape (10, 5)
            hd95: np.ndarray = np.load(hd95_path)

            # Mean including class 0
            mean_dice_all = dice.mean()
            mean_hd95_all = hd95.mean()

            # Mean excluding class 0 (assume class 0 is index 0)
            if dice.shape[0] > 1:
                mean_dice_no0 = dice[:, 1:].mean()
                mean_hd95_no0 = hd95[:, 1:].mean()
            else:
                mean_dice_no0 = float("nan")
                mean_hd95_no0 = float("nan")

            results.append(
                {
                    "folder": x_path.name,
                    "mean_dice_all": mean_dice_all,
                    "mean_dice_no0": mean_dice_no0,
                    "mean_hd95_all": mean_hd95_all,
                    "mean_hd95_no0": mean_hd95_no0,
                }
            )

    # Sort by mean_dice_all from high to low
    results.sort(key=lambda r: r["mean_dice_all"], reverse=True)

    # Print results
    print("Best runs, sorted by mean dice (incl background):")
    for r in results:
        print(
            f"{r['folder']}: mean dice (incl 0): {r['mean_dice_all']:.4f}, "
            f"mean dice (excl 0): {r['mean_dice_no0']:.4f} "
            # f"mean hd95 (incl 0): {r['mean_hd95_all']:.4f}, "
            # f"mean hd95 (excl 0): {r['mean_hd95_no0']:.4f}"
        )


if __name__ == "__main__":
    main()
