"""Run the project's interpolation comparison and write reproducible outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from rasterizer import make_checker_texture, rasterize, save_result


ANGLES = [0, 15, 30, 45, 60, 70]


def measure_uv_and_image_error(affine, perspective, texture_size: int) -> tuple[float, float, float]:
    mask = affine.mask & perspective.mask
    duv = affine.uv[mask] - perspective.uv[mask]
    pixel_error = np.sqrt((duv[:, 0] * texture_size) ** 2 + (duv[:, 1] * texture_size) ** 2)
    rgb_error = np.abs(affine.image.astype(np.int16) - perspective.image.astype(np.int16))[mask]
    uv_mean = float(pixel_error.mean()) if pixel_error.size else 0.0
    uv_p95 = float(np.percentile(pixel_error, 95)) if pixel_error.size else 0.0
    rgb_mae = float(rgb_error.mean()) if rgb_error.size else 0.0
    return uv_mean, uv_p95, rgb_mae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    texture = make_checker_texture()
    rows = []
    for angle in ANGLES:
        affine = rasterize(angle, "affine", texture)
        perspective = rasterize(angle, "perspective", texture)
        uv_mean, uv_p95, rgb_mae = measure_uv_and_image_error(affine, perspective, texture.shape[0])

        save_result(affine, str(args.output / f"angle_{angle:02d}_affine.png"))
        save_result(perspective, str(args.output / f"angle_{angle:02d}_perspective.png"))
        rows.append(
            {
                "angle_degrees": angle,
                "covered_pixels": int(affine.mask.sum()),
                "mean_uv_error_texture_px": f"{uv_mean:.4f}",
                "p95_uv_error_texture_px": f"{uv_p95:.4f}",
                "rgb_mae_0_255": f"{rgb_mae:.4f}",
            }
        )
        print(
            f"angle={angle:2d} deg | mean UV error={uv_mean:7.3f} texel | "
            f"P95={uv_p95:7.3f} texel | RGB MAE={rgb_mae:6.2f}"
        )

    csv_path = args.output / "experiment_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

