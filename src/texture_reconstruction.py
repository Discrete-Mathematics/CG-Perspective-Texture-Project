"""Texture reconstruction and an exact CPU texel-sharing reference.

The sharing experiment is inspired by the List Merge idea in Akenine-Moller
et al., Collaborative Texture Filtering (HPG 2025), sections 3 and 3.1.
It is NOT their full GPU algorithm: every required texel is fetched even if a
tile needs more than one texel per active pixel. No stochastic fallback,
wave intrinsics, neural decoder, or GPU bandwidth model is implemented.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def analytic_texture(u, v, kind: str) -> np.ndarray:
    """Known continuous linear-RGB signal, or discontinuous checker control."""
    tau = 2.0 * np.pi
    if kind == "smooth_low":
        return np.stack((0.5 + 0.45 * np.sin(tau * (3*u + 0.4*v)),
                         0.5 + 0.45 * np.cos(tau * (2*v - 0.3*u)),
                         0.5 + 0.45 * np.sin(tau * (2*u + 2*v))), axis=-1)
    if kind == "smooth_high":
        return np.stack((0.5 + 0.45 * np.sin(tau * (11*u + 3*v)),
                         0.5 + 0.45 * np.cos(tau * (2*u + 9*v)),
                         0.5 + 0.45 * np.sin(tau * (10*u - 8*v))), axis=-1)
    if kind == "checker":
        odd = ((np.floor(12*u) + np.floor(12*v)).astype(np.int64) % 2).astype(bool)
        return np.where(odd[..., None], np.array([.95, .12, .08]), np.array([.05, .88, .92]))
    raise ValueError(kind)


def sampled_texture(size: int, kind: str) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    return analytic_texture((xx+.5)/size, 1-(yy+.5)/size, kind)


def catmull_rom_weights(t: np.ndarray) -> np.ndarray:
    """Keys cubic convolution at a=-1/2, offsets (-1,0,1,2)."""
    t2, t3 = t*t, t*t*t
    return np.stack((-.5*t+t2-.5*t3, 1-2.5*t2+1.5*t3,
                     .5*t+2*t2-1.5*t3, -.5*t2+.5*t3), axis=-1)


def texel_coordinates(uv: np.ndarray, size: int):
    x = uv[:, 0] * size - .5
    y = (1.0-uv[:, 1]) * size - .5
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    return ix, iy, x-ix, y-iy


def reconstruct_nearest(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    size = texture.shape[0]
    x = np.clip(np.floor(uv[:, 0]*size).astype(np.int64), 0, size-1)
    y = np.clip(np.floor((1.0-uv[:, 1])*size).astype(np.int64), 0, size-1)
    return texture[y, x]


def reconstruct_bilinear(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    size = texture.shape[0]
    ix, iy, tx, ty = texel_coordinates(uv, size)
    output = np.zeros((len(uv), 3), dtype=np.float64)
    for dy in range(2):
        wy = ty if dy else 1-ty
        for dx in range(2):
            wx = tx if dx else 1-tx
            output += (wx*wy)[:, None] * texture[np.clip(iy+dy, 0, size-1), np.clip(ix+dx, 0, size-1)]
    return output


def reconstruct_bicubic_direct(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Independent 16-term implementation used as an equality oracle."""
    size = texture.shape[0]
    ix, iy, tx, ty = texel_coordinates(uv, size)
    wx, wy = catmull_rom_weights(tx), catmull_rom_weights(ty)
    output = np.zeros((len(uv), 3), dtype=np.float64)
    for dy in range(4):
        for dx in range(4):
            color = texture[np.clip(iy+dy-1, 0, size-1), np.clip(ix+dx-1, 0, size-1)]
            output += (wx[:, dx]*wy[:, dy])[:, None]*color
    return output


@dataclass
class SharingStats:
    pixels: int
    tiles: int
    direct_texel_evaluations: int
    shared_unique_texel_evaluations: int
    unique_texels_per_pixel: float
    logical_evaluation_reduction_percent: float
    tiles_within_one_evaluation_per_active_pixel_percent: float
    max_unique_texels_in_tile: int


def reconstruct_bicubic_shared(texture: np.ndarray, uv: np.ndarray,
                               pixel_xy: np.ndarray, image_width: int):
    """Merge each 8x4 screen tile's 4x4 footprints before evaluating texels.

    np.unique substitutes for the paper's parallel list merge. Its keys include
    the tile ID, so a texel requested by different tiles is evaluated repeatedly.
    The count measures logical expensive texel evaluations, NOT cache misses,
    memory transactions, or the number of gathers from the shared color array.
    All work, including merging, is included in timings of this function.
    """
    size = texture.shape[0]
    ix, iy, tx, ty = texel_coordinates(uv, size)
    wx, wy = catmull_rom_weights(tx), catmull_rom_weights(ty)
    offsets = np.arange(-1, 3)
    xs = np.clip(ix[:, None]+offsets, 0, size-1)
    ys = np.clip(iy[:, None]+offsets, 0, size-1)
    indices = (ys[:, :, None]*size + xs[:, None, :]).reshape(-1, 16)
    weights = (wy[:, :, None]*wx[:, None, :]).reshape(-1, 16)
    tiles_wide = (image_width+7)//8
    tile_ids = (pixel_xy[:, 1]//4)*tiles_wide + pixel_xy[:, 0]//8
    keys = tile_ids[:, None]*(size*size)+indices
    unique_keys, inverse = np.unique(keys.ravel(), return_inverse=True)
    # This is the only source-texture evaluation in the shared path.
    unique_colors = texture.reshape(-1, 3)[unique_keys % (size*size)]
    gathered = unique_colors[inverse].reshape(-1, 16, 3)
    output = np.sum(weights[:, :, None]*gathered, axis=1)
    tile_count = int(tile_ids.max())+1
    active = np.bincount(tile_ids, minlength=tile_count)
    unique_per_tile = np.bincount(unique_keys//(size*size), minlength=tile_count)
    occupied = active > 0
    stats = SharingStats(
        pixels=len(uv), tiles=int(occupied.sum()),
        direct_texel_evaluations=16*len(uv), shared_unique_texel_evaluations=len(unique_keys),
        unique_texels_per_pixel=len(unique_keys)/len(uv),
        logical_evaluation_reduction_percent=100*(1-len(unique_keys)/(16*len(uv))),
        tiles_within_one_evaluation_per_active_pixel_percent=100*float(np.mean(unique_per_tile[occupied] <= active[occupied])),
        max_unique_texels_in_tile=int(unique_per_tile.max()),
    )
    return output, stats


def verification() -> dict:
    """Nontrivial invariants: weight partition, polynomial reproduction, sharing."""
    rng = np.random.default_rng(20261124)
    t = rng.random(5000)
    weights = catmull_rom_weights(t)
    partition_error = float(np.abs(weights.sum(axis=-1)-1).max())
    size = 32
    uv = rng.uniform(.12, .88, (4000, 2))
    yy, xx = np.mgrid[:size, :size]
    u, v = (xx+.5)/size, 1-(yy+.5)/size
    linear_texture = np.stack((u, v, .2*u+.3*v+.1), axis=-1)
    linear_reference = np.stack((uv[:, 0], uv[:, 1], .2*uv[:, 0]+.3*uv[:, 1]+.1), axis=-1)
    linear_error = float(np.abs(reconstruct_bicubic_direct(linear_texture, uv)-linear_reference).max())
    constant_error = float(np.abs(reconstruct_bicubic_direct(np.full((size, size, 3), .37), uv)-.37).max())
    pixel_xy = np.column_stack((np.arange(len(uv)) % 100, np.arange(len(uv)) // 100))
    # Boundary cases exercise clamping and duplicate stencil entries.
    uv[:8] = [[0,0], [0,1], [1,0], [1,1], [.001,.4], [.999,.5], [.2,.001], [.3,.999]]
    texture = rng.random((size, size, 3))
    direct = reconstruct_bicubic_direct(texture, uv)
    shared, _ = reconstruct_bicubic_shared(texture, uv, pixel_xy, 100)
    sharing_error = float(np.abs(direct-shared).max())
    assert max(partition_error, linear_error, constant_error, sharing_error) < 1e-12
    return dict(weight_sum_max_abs_error=partition_error,
                linear_reproduction_max_abs_error=linear_error,
                constant_reproduction_max_abs_error=constant_error,
                direct_shared_max_abs_error=sharing_error)
