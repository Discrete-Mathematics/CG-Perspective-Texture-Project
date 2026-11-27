"""Budgeted collaborative Catmull-Rom filtering under perspective-correct UVs.

Usage, from this project directory:
    python src/run_sota_experiments.py

The reference for filter error is exact 16-tap Catmull-Rom at the same UV.
An analytic signal is reported separately and is not an area-integrated
antialiasing reference. Stochastic numbers are means over several seeds.
"""
from __future__ import annotations

import csv
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collaborative_filter import (  # noqa: E402
    magnification_factor,
    mask_sampling_catmull_rom,
    two_tap_stochastic,
    verification,
)
from rasterizer import HEIGHT, WIDTH, rasterize  # noqa: E402
from texture_reconstruction import (  # noqa: E402
    analytic_texture,
    reconstruct_bicubic_direct,
    sampled_texture,
)


ANGLES = [0, 45, 70]
SIZES = [32, 128, 512]
TEXTURES = ["smooth_low", "smooth_high", "checker"]
SEEDS = 4


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def error_metrics(predicted: np.ndarray, reference: np.ndarray) -> dict:
    delta = predicted - reference
    mse = float(np.mean(delta * delta))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "psnr_db": float("inf") if mse < 1e-30 else float(-10.0 * np.log10(mse)),
    }


def display_image(colors: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:] = (17, 25, 38)
    image[mask] = np.round(np.clip(colors, 0, 1) * 255).astype(np.uint8)
    return image


def mean_metric(rows: list[dict], key: str) -> float:
    values = [row[key] for row in rows]
    if any(np.isinf(value) for value in values):
        return float("inf")
    return float(np.mean(values))


def comparison_figure(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = [
        ("smooth_high", 32, 0, "High magnification"),
        ("smooth_high", 512, 70, "Low magnification"),
    ]
    methods = ["bicubic_exact", "two_tap_stf", "mask_cplus"]
    titles = ["Exact Catmull-Rom", "Two-tap STF", "Mask Sampling + C+"]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    for row, (texture, size, angle, label) in enumerate(cases):
        for column, (method, title) in enumerate(zip(methods, titles)):
            image = np.asarray(Image.open(output / f"{texture}_n{size}_a{angle}_{method}.png"))
            axes[row, column].imshow(image[140:390, 150:430], interpolation="nearest")
            axes[row, column].set_title(title if row == 0 else "", fontsize=11)
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(f"{label}\n{texture}, {size}², {angle}°", fontsize=10)
    figure.suptitle("Same perspective UVs. Crops are 280 by 240 pixels from the rendered frame.", fontsize=12)
    figure.tight_layout()
    figure.savefig(output / "comparison_sota.png", dpi=160)
    plt.close(figure)


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "results" / "sota"
    output.mkdir(parents=True, exist_ok=True)
    checks = verification()
    rows: list[dict] = []
    for angle in ANGLES:
        geometry = rasterize(angle, "perspective")
        covered = geometry.mask
        uv = geometry.uv[covered]
        ys, xs = np.where(covered)
        pixel_xy = np.column_stack((xs, ys))
        for size in SIZES:
            magnification = magnification_factor(geometry.uv, covered, size)
            interior = np.all((uv > 2.0 / size) & (uv < 1.0 - 2.0 / size), axis=1)
            for texture_name in TEXTURES:
                texture = sampled_texture(size, texture_name)
                analytic = analytic_texture(uv[:, 0], uv[:, 1], texture_name)
                exact = reconstruct_bicubic_direct(texture, uv)
                collaborative_errors = []
                stochastic_errors = []
                collaborative_analytic = []
                stochastic_analytic = []
                collaborative_evals = []
                stochastic_evals = []
                perfect_percent = None
                full_percent = None
                preview = {}
                for seed in range(SEEDS):
                    rng = np.random.default_rng(20261124 + 1000 * seed + 17 * size + angle)
                    collaborative, stats, _ = mask_sampling_catmull_rom(
                        texture, uv, pixel_xy, WIDTH, rng)
                    stochastic, evals = two_tap_stochastic(texture, uv, rng)
                    collaborative_errors.append(error_metrics(collaborative, exact))
                    stochastic_errors.append(error_metrics(stochastic, exact))
                    collaborative_analytic.append(error_metrics(collaborative[interior], analytic[interior]))
                    stochastic_analytic.append(error_metrics(stochastic[interior], analytic[interior]))
                    collaborative_evals.append(stats.evaluations_per_pixel)
                    stochastic_evals.append(float(np.mean(evals)))
                    perfect_percent = stats.perfect_pixel_percent
                    full_percent = stats.perfect_full_tile_percent
                    if seed == 0 and texture_name == "smooth_high" and (size, angle) in {(32, 0), (512, 70)}:
                        preview = {"mask": collaborative, "stf": stochastic, "exact": exact}
                if preview:
                    Image.fromarray(display_image(preview["exact"], covered)).save(
                        output / f"{texture_name}_n{size}_a{angle}_bicubic_exact.png")
                    Image.fromarray(display_image(preview["stf"], covered)).save(
                        output / f"{texture_name}_n{size}_a{angle}_two_tap_stf.png")
                    Image.fromarray(display_image(preview["mask"], covered)).save(
                        output / f"{texture_name}_n{size}_a{angle}_mask_cplus.png")
                exact_analytic = error_metrics(exact[interior], analytic[interior])
                row = {
                    "texture": texture_name,
                    "texture_size": size,
                    "angle_degrees": angle,
                    "covered_pixels": int(len(uv)),
                    "median_magnification": magnification,
                    "perfect_pixel_percent": perfect_percent,
                    "perfect_full_tile_percent": full_percent,
                    "exact_analytic_psnr_db": exact_analytic["psnr_db"],
                    "mask_cplus_evals_per_pixel": float(np.mean(collaborative_evals)),
                    "mask_cplus_psnr_vs_bicubic_db": mean_metric(collaborative_errors, "psnr_db"),
                    "mask_cplus_rmse_vs_bicubic": mean_metric(collaborative_errors, "rmse"),
                    "mask_cplus_analytic_psnr_db": mean_metric(collaborative_analytic, "psnr_db"),
                    "two_tap_evals_per_pixel": float(np.mean(stochastic_evals)),
                    "two_tap_psnr_vs_bicubic_db": mean_metric(stochastic_errors, "psnr_db"),
                    "two_tap_rmse_vs_bicubic": mean_metric(stochastic_errors, "rmse"),
                    "two_tap_analytic_psnr_db": mean_metric(stochastic_analytic, "psnr_db"),
                    "seeds": SEEDS,
                }
                rows.append(row)
                print(
                    f"{texture_name:12} N={size:<3} a={angle:<2} mag={magnification:6.2f} "
                    f"perfect={perfect_percent:5.1f}% full={full_percent:5.1f}% "
                    f"C+ {row['mask_cplus_psnr_vs_bicubic_db']:.2f} dB  "
                    f"STF {row['two_tap_psnr_vs_bicubic_db']:.2f} dB",
                    flush=True,
                )
    save_csv(output / "sota_results.csv", rows)
    summary_groups = []
    for size in SIZES:
        for angle in ANGLES:
            subset = [row for row in rows if row["texture_size"] == size and row["angle_degrees"] == angle]
            summary_groups.append({
                "texture_size": size,
                "angle_degrees": angle,
                "median_magnification": subset[0]["median_magnification"],
                "perfect_pixel_percent": float(np.mean([row["perfect_pixel_percent"] for row in subset])),
                "perfect_full_tile_percent": float(np.mean([row["perfect_full_tile_percent"] for row in subset])),
                "mask_cplus_evals_per_pixel": float(np.mean([row["mask_cplus_evals_per_pixel"] for row in subset])),
                "mask_cplus_psnr_vs_bicubic_db": mean_metric(subset, "mask_cplus_psnr_vs_bicubic_db"),
                "two_tap_psnr_vs_bicubic_db": mean_metric(subset, "two_tap_psnr_vs_bicubic_db"),
                "mask_cplus_analytic_psnr_db": mean_metric(subset, "mask_cplus_analytic_psnr_db"),
                "two_tap_analytic_psnr_db": mean_metric(subset, "two_tap_analytic_psnr_db"),
                "exact_analytic_psnr_db": mean_metric(subset, "exact_analytic_psnr_db"),
            })
    save_csv(output / "sota_summary.csv", summary_groups)
    summary = {
        "experiment": "Mask Sampling collaborative Catmull-Rom with C+ fallback, versus exact Catmull-Rom and two-tap stochastic texture filtering",
        "source": "Akenine-Moller, Ebelin, Pharr, and Wronski, Collaborative Texture Filtering, HPG 2025, sections 3.3, 3.4, and 4.2",
        "source_url": "https://arxiv.org/abs/2506.17770",
        "scope": "CPU tile model of an 8 by 4 wave. No GPU intrinsics, neural decoder, DLSS, or hardware timing.",
        "settings": {
            "angles": ANGLES, "texture_sizes": SIZES, "textures": TEXTURES,
            "seeds": SEEDS, "resolution": [WIDTH, HEIGHT], "tile": [8, 4],
        },
        "checks": checks,
        "groups": summary_groups,
        "timing": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__},
    }
    def jsonable(value):
        if isinstance(value, float):
            if math.isinf(value):
                return "inf"
            if math.isnan(value):
                return None
        if isinstance(value, dict):
            return {key: jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [jsonable(item) for item in value]
        return value

    (output / "summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    comparison_figure(output)
    print(json.dumps({"checks": checks, "groups": summary_groups}, indent=2))


if __name__ == "__main__":
    main()
