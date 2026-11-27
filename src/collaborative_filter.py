"""Budgeted collaborative Catmull-Rom filtering.

CPU reference for Mask Sampling with the C+ fallback from Akenine-Moller,
Ebelin, Pharr, and Wronski, Collaborative Texture Filtering, HPG 2025
(sections 3.3 and 3.4). This is not a GPU wave-intrinsic implementation:
there is no shared memory, ballot, or neural texel decoder. Each aligned
8 by 4 screen tile stands in for one 32-lane wave.

Perfect filtering is used only when the clamped 4 by 4 footprints fit in a
16 by 16 texel window and the number of unique texels does not exceed the
number of active pixels in the tile. Otherwise each active pixel evaluates
one texel. C+ spends duplicate proposals on footprint texels that are still
missing, then fills unknown taps with equation (1) of the paper. Catmull-Rom
weights are signed, so proposals are drawn with probability proportional to
the absolute weight. A one-texel known set returns that texel, matching the
paper's one-tap special case.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from texture_reconstruction import (
    catmull_rom_weights,
    reconstruct_bicubic_direct,
    texel_coordinates,
)


OFFSETS = np.array([-1, 0, 1, 2], dtype=np.int64)


@dataclass
class FilterStats:
    pixels: int
    tiles: int
    full_tiles: int
    perfect_pixels: int
    perfect_full_tile_pixels: int
    full_tile_pixels: int
    texel_evaluations: int

    @property
    def perfect_pixel_percent(self) -> float:
        return 100.0 * self.perfect_pixels / self.pixels

    @property
    def perfect_full_tile_percent(self) -> float:
        if self.full_tile_pixels == 0:
            return float("nan")
        return 100.0 * self.perfect_full_tile_pixels / self.full_tile_pixels

    @property
    def evaluations_per_pixel(self) -> float:
        return self.texel_evaluations / self.pixels


def _stencil(uv: np.ndarray, size: int):
    ix, iy, tx, ty = texel_coordinates(uv, size)
    wx = catmull_rom_weights(tx)
    wy = catmull_rom_weights(ty)
    xs = ix[:, None] + OFFSETS
    ys = iy[:, None] + OFFSETS
    clamped_x = np.clip(xs, 0, size - 1)
    clamped_y = np.clip(ys, 0, size - 1)
    texel_ids = (clamped_y[:, :, None] * size + clamped_x[:, None, :]).reshape(-1, 16)
    weights = (wy[:, :, None] * wx[:, None, :]).reshape(-1, 16)
    return xs, ys, texel_ids.astype(np.int64), weights


def _eq1(texel_ids: np.ndarray, weights: np.ndarray, produced: np.ndarray, texflat: np.ndarray) -> np.ndarray:
    """Paper equation (1), with the single-known-texel special case."""
    known = np.isin(texel_ids, produced)
    colors = texflat[texel_ids]
    kept = np.where(known, weights, 0.0)
    weighted = np.sum(kept[:, :, None] * colors, axis=1)
    order = np.argsort(texel_ids, axis=1)
    sorted_ids = np.take_along_axis(texel_ids, order, axis=1)
    distinct = np.all(sorted_ids[:, 1:] != sorted_ids[:, :-1], axis=1)
    count = known.sum(axis=1)
    output = np.zeros((len(texel_ids), 3), dtype=np.float64)
    one = distinct & (count == 1)
    if np.any(one):
        output[one] = colors[one, np.argmax(known[one], axis=1)]
    many = distinct & (count > 1)
    if np.any(many):
        weight_sum = kept[many].sum(axis=1)
        mean = np.sum(colors[many] * known[many, :, None], axis=1) / count[many, None]
        output[many] = weighted[many] + (1.0 - weight_sum[:, None]) * mean
    for index in np.flatnonzero(~distinct):
        chosen = known[index]
        ids = texel_ids[index, chosen]
        if ids.size == 0:
            continue
        coeff = weights[index, chosen]
        unique_ids, inverse = np.unique(ids, return_inverse=True)
        summed = np.zeros(len(unique_ids), dtype=np.float64)
        np.add.at(summed, inverse, coeff)
        palette = texflat[unique_ids]
        if len(unique_ids) == 1:
            output[index] = palette[0]
        else:
            output[index] = summed @ palette + (1.0 - summed.sum()) * palette.mean(axis=0)
    return output


def _sample_ids(texel_ids: np.ndarray, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    magnitude = np.abs(weights)
    mass = magnitude.sum(axis=1)
    probability = np.divide(magnitude, mass[:, None], out=np.full_like(magnitude, 1.0 / 16.0), where=mass[:, None] > 0)
    cdf = np.cumsum(probability, axis=1)
    choice = np.clip(np.sum(cdf < rng.random(len(texel_ids))[:, None], axis=1), 0, 15)
    return texel_ids[np.arange(len(texel_ids)), choice]


def _cplus_produced(texel_ids: np.ndarray, weights: np.ndarray, rng: np.random.Generator) -> list[int]:
    """One evaluation per active pixel: unique proposals, then missing taps."""
    sampled = _sample_ids(texel_ids, weights, rng)
    produced: list[int] = []
    seen: set[int] = set()
    for texel in sampled.tolist():
        if texel not in seen:
            seen.add(int(texel))
            produced.append(int(texel))
    active = len(texel_ids)
    initial = len(produced)
    extras = active - initial
    for slot in range(extras):
        # Equation (2), adapted from a 32-lane wave to the active pixels in this tile.
        if extras == 1:
            source = 0
        else:
            source = int(np.clip(np.rint((active - 1) * slot / (extras - 1)), 0, active - 1))
        # The paper's lane map spreads extra work; the sample itself must be a
        # footprint texel that no lane has produced yet.
        missing = [k for k in range(16) if int(texel_ids[source, k]) not in seen]
        if not missing:
            for alternative in range(active):
                missing = [k for k in range(16) if int(texel_ids[alternative, k]) not in seen]
                if missing:
                    source = alternative
                    break
        if not missing:
            break
        rank = np.abs(weights[source, missing])
        if float(rank.sum()) <= 0:
            rank = np.ones(len(missing))
        rank = rank / rank.sum()
        pick = int(rng.choice(np.asarray(missing), p=rank))
        texel = int(texel_ids[source, pick])
        if texel not in seen:
            seen.add(texel)
            produced.append(texel)
    return produced


def mask_sampling_catmull_rom(texture: np.ndarray, uv: np.ndarray, pixel_xy: np.ndarray,
                              image_width: int, rng: np.random.Generator):
    """Mask Sampling, falling back to C+ when the 8 by 4 tile cannot be exact."""
    size = texture.shape[0]
    texflat = np.ascontiguousarray(texture.reshape(-1, 3))
    xs, ys, texel_ids, weights = _stencil(uv, size)
    tiles_wide = (image_width + 7) // 8
    tile_ids = (pixel_xy[:, 1] // 4) * tiles_wide + pixel_xy[:, 0] // 8
    order = np.argsort(tile_ids, kind="mergesort")
    sorted_tiles = tile_ids[order]
    cuts = np.flatnonzero(np.diff(sorted_tiles)) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(order)]

    output = np.zeros((len(uv), 3), dtype=np.float64)
    perfect = np.zeros(len(uv), dtype=bool)
    evaluations = 0
    full_tiles = 0
    perfect_full = 0
    full_pixels = 0
    for start, end in zip(starts, ends):
        members = order[start:end]
        active = len(members)
        full = active == 32
        if full:
            full_tiles += 1
            full_pixels += active
        span_x = int(xs[members].max() - xs[members].min() + 1)
        span_y = int(ys[members].max() - ys[members].min() + 1)
        ids = texel_ids[members]
        unique_ids, inverse = np.unique(ids.ravel(), return_inverse=True)
        if len(unique_ids) <= active and span_x <= 16 and span_y <= 16:
            gathered = texflat[unique_ids][inverse].reshape(active, 16, 3)
            output[members] = np.sum(weights[members, :, None] * gathered, axis=1)
            perfect[members] = True
            evaluations += int(len(unique_ids))
            if full:
                perfect_full += active
            continue
        produced = np.asarray(_cplus_produced(ids, weights[members], rng), dtype=np.int64)
        output[members] = _eq1(ids, weights[members], produced, texflat)
        evaluations += int(len(produced))
    stats = FilterStats(
        pixels=len(uv), tiles=len(starts), full_tiles=full_tiles,
        perfect_pixels=int(perfect.sum()), perfect_full_tile_pixels=perfect_full,
        full_tile_pixels=full_pixels, texel_evaluations=evaluations,
    )
    return output, stats, perfect


def two_tap_stochastic(texture: np.ndarray, uv: np.ndarray, rng: np.random.Generator):
    """Unbiased Catmull-Rom estimator with one positive-lobe and one negative-lobe sample.

    The HPG 2025 Catmull-Rom comparison uses two evaluations for one-tap STF
    because the kernel is signed. Sampling each lobe proportional to its mass
    and returning mass * color is unbiased for that lobe. A lobe with no mass
    is skipped.
    """
    size = texture.shape[0]
    texflat = texture.reshape(-1, 3)
    _, _, texel_ids, weights = _stencil(uv, size)
    output = np.zeros((len(uv), 3), dtype=np.float64)
    evaluations = np.zeros(len(uv), dtype=np.int64)
    for sign, lobe in ((1.0, np.clip(weights, 0, None)), (-1.0, np.clip(-weights, 0, None))):
        mass = lobe.sum(axis=1)
        active = mass > 1e-15
        if not np.any(active):
            continue
        probability = np.zeros_like(lobe)
        probability[active] = lobe[active] / mass[active, None]
        cdf = np.cumsum(probability, axis=1)
        choice = np.clip(np.sum(cdf < rng.random(len(uv))[:, None], axis=1), 0, 15)
        missed = probability[np.arange(len(uv)), choice] <= 0
        if np.any(missed):
            choice[missed] = np.argmax(probability[missed], axis=1)
        chosen = texel_ids[np.arange(len(uv)), choice]
        output[active] += sign * mass[active, None] * texflat[chosen[active]]
        evaluations[active] += 1
    return output, evaluations


def magnification_factor(uv_image: np.ndarray, mask: np.ndarray, size: int) -> float:
    """Median screen texels-per-pixel reciprocal. Values above 1 are magnification."""
    texel_x = uv_image[..., 0] * size
    texel_y = (1.0 - uv_image[..., 1]) * size
    horizontal = mask[:, :-1] & mask[:, 1:]
    vertical = mask[:-1, :] & mask[1:, :]
    dx = np.maximum(np.abs(np.diff(texel_x, axis=1)), np.abs(np.diff(texel_y, axis=1)))
    dy = np.maximum(np.abs(np.diff(texel_x, axis=0)), np.abs(np.diff(texel_y, axis=0)))
    derivatives = np.concatenate((dx[horizontal], dy[vertical]))
    derivatives = derivatives[derivatives > 1e-8]
    if derivatives.size == 0:
        return float("nan")
    return float(np.median(1.0 / derivatives))


def verification() -> dict:
    """Exact-tile agreement, constant reproduction, and two-tap unbiasedness."""
    rng = np.random.default_rng(20261124)
    size = 32
    texture = rng.random((size, size, 3))
    uv = np.tile(np.array([[0.37, 0.62]]), (32, 1))
    pixel_xy = np.column_stack((np.arange(32) % 8, np.arange(32) // 8))
    exact = reconstruct_bicubic_direct(texture, uv)
    shared, stats, perfect = mask_sampling_catmull_rom(texture, uv, pixel_xy, 32, rng)
    exact_error = float(np.max(np.abs(shared - exact)))
    assert exact_error < 1e-12 and perfect.all() and stats.texel_evaluations <= 32

    constant = np.full((size, size, 3), 0.41)
    scattered = rng.uniform(0.08, 0.92, (180, 2))
    scattered_xy = np.column_stack((np.arange(180) % 40, np.arange(180) // 40))
    constant_out, _, _ = mask_sampling_catmull_rom(constant, scattered, scattered_xy, 48, rng)
    constant_error = float(np.max(np.abs(constant_out - 0.41)))
    assert constant_error < 1e-10

    probe = rng.uniform(0.2, 0.8, (1, 2))
    reference = reconstruct_bicubic_direct(texture, probe)[0]
    draws = np.vstack([two_tap_stochastic(texture, probe, rng)[0][0] for _ in range(4000)])
    unbiased_error = float(np.max(np.abs(draws.mean(axis=0) - reference)))
    assert unbiased_error < 0.03

    border = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.001, 0.5], [0.999, 0.2]])
    border_xy = np.column_stack((np.arange(len(border)) % 8, np.zeros(len(border), dtype=int)))
    border_out, _, _ = mask_sampling_catmull_rom(texture, border, border_xy, 16, rng)
    assert np.isfinite(border_out).all()
    return {
        "exact_tile_max_abs_error": exact_error,
        "constant_max_abs_error": constant_error,
        "two_tap_mean_max_abs_error": unbiased_error,
    }
