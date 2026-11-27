# Perspective-Correct Texture Mapping

This project implements a small CPU triangle rasterizer and compares screen-space affine UV interpolation with perspective-correct UV interpolation.

## Reproduce the experiment

Requirements: Python 3.10 or later, NumPy, and Pillow.

From this folder, run:

```powershell
python -m pip install -r requirements.txt
python src/run_experiments.py
```

The script writes six pairs of 512 by 512 renders and `results/experiment_results.csv`. Each pair uses the same generated checkerboard texture, camera, projected geometry, and nearest-neighbor sampling. Only the UV interpolation formula changes.

## Experiment

The quad is rotated about its vertical axis at 0, 15, 30, 45, 60, and 70 degrees. The baseline interpolates UV linearly in screen space. The variant uses clip-space reciprocal-w weights. The rasterizer uses edge-function coverage, barycentric coordinates, a depth buffer, and nearest-neighbor texture sampling.

The CSV reports mean and 95th-percentile UV error in texture pixels, plus RGB mean absolute error between the two rendered outputs. RGB MAE is averaged over covered pixels and channels on the 0 to 255 scale. The error values are deterministic for the supplied software and parameters.

## Scope and limitations

The experiment uses one planar quad, no lighting, no multisampling, no mipmapping, and no near-plane clipping. It isolates perspective distortion in UV interpolation and is not a production renderer.

## Files

- `src/rasterizer.py`: geometry transform, triangle coverage, depth test, interpolation, and texture lookup.
- `src/run_experiments.py`: sweep the viewing angle, save comparisons, and write the CSV.
- `results/`: generated render pairs and measurements.

