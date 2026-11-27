"""Run controlled texture reconstruction and exact texel-sharing experiments.

Usage: python src/run_advanced_experiments.py
Primary results require only NumPy and Pillow. Matplotlib is optional for plots.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import platform
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image
from rasterizer import rasterize, WIDTH, HEIGHT, BACKGROUND
from texture_reconstruction import (analytic_texture, sampled_texture,
    reconstruct_nearest, reconstruct_bilinear, reconstruct_bicubic_direct,
    reconstruct_bicubic_shared, verification)

ANGLES = [0, 30, 60, 70]
SIZES = [32, 64, 128]
TEXTURES = ['smooth_low', 'smooth_high', 'checker']
METHODS = ['nearest', 'bilinear', 'bicubic_direct', 'bicubic_shared']


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(predicted, reference):
    diff = predicted-reference
    mse = float(np.mean(diff*diff))
    clipped_diff = np.clip(predicted, 0, 1)-reference
    return {'mse_raw': mse, 'rmse_raw': np.sqrt(mse),
            'psnr_db_raw': -10*np.log10(mse) if mse else float('inf'),
            'mean_signed_bias_raw': float(diff.mean()),
            'rmse_after_display_clamp': float(np.sqrt(np.mean(clipped_diff**2))),
            'out_of_range_channel_percent': float(100*np.mean((predicted<0)|(predicted>1))),
            'min_raw': float(predicted.min()), 'max_raw': float(predicted.max())}


def image_array(colors, mask):
    image = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:] = BACKGROUND
    image[mask] = np.round(np.clip(colors, 0, 1)*255).astype(np.uint8)
    return image


def timed(fn, repeats):
    result = fn()  # Warmup, then median of complete filtering calls.
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append(1000*(time.perf_counter()-start))
    return result, statistics.median(samples), samples


def plot_results(output, quality, costs):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10})
    labels = {'smooth_low':'Smooth low frequency', 'smooth_high':'Smooth high frequency', 'checker':'Hard checker control'}
    colors = ['#59686F', '#DB704E', '#147A83']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, texture in zip(axes, TEXTURES):
        subset = [r for r in quality if r['texture']==texture]
        for method, color in zip(METHODS[:3], colors):
            values = [next(r['aggregate_psnr_db'] for r in subset if r['method']==method and r['texture_size']==s) for s in SIZES]
            ax.plot(SIZES, values, marker='o', label=method.replace('_direct','').replace('_',' '), color=color)
        ax.set_title(labels[texture]); ax.set_xlabel('Texture resolution'); ax.set_ylabel('PSNR (dB)')
        ax.set_xticks(SIZES); ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/'quality_psnr.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for size, color in zip(SIZES, colors):
        values = [next(r['unique_texels_per_pixel'] for r in costs if r['texture_size']==size and r['angle_degrees']==a) for a in ANGLES]
        ax.plot(ANGLES, values, marker='o', label=f'{size} x {size}', color=color)
    ax.axhline(16, color='#A9B9BD', linestyle='--', label='Direct bicubic: 16')
    ax.set(xlabel='Plane rotation (degrees)', ylabel='Logical texel evaluations / pixel', ylim=(0,17))
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(output/'logical_cost.png', dpi=180); plt.close(fig)
    # Plots of measured outputs, not fabricated illustration: selected same-scene crops.
    for kind, target in [('smooth_high','comparison_smooth.png'), ('checker','comparison_checker.png')]:
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
        for ax, method in zip(axes, ['reference','nearest','bilinear','bicubic_shared']):
            arr = np.asarray(Image.open(output/f'{kind}_n32_a30_{method}.png'))
            ax.imshow(arr[150:350, 150:350], interpolation='nearest')
            ax.set_title(method.replace('_shared','').replace('_',' ')); ax.axis('off')
        fig.suptitle(f'{labels[kind]}, 32 x 32 texture, 30 degrees (same 200 x 200 crop)')
        fig.tight_layout(); fig.savefig(output/target, dpi=180); plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'results'/'advanced')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    assert args.repeats >= 1
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    checks = verification()
    quality_rows, cost_rows = [], []
    max_match = 0.0
    for angle in ANGLES:
        geometry = rasterize(angle, 'perspective')
        full_uv = geometry.uv[geometry.mask]
        ys, xs = np.where(geometry.mask)
        pixel_xy = np.column_stack((xs, ys))
        for size in SIZES:
            # Same evaluation mask for every method; keep all rendering pixels for
            # the sharing-cost count and output PNG, excluding borders only in quality.
            interior = np.all((full_uv > 2.0/size) & (full_uv < 1.0-2.0/size), axis=1)
            for texture_name in TEXTURES:
                texture = sampled_texture(size, texture_name)
                reference = analytic_texture(full_uv[:,0], full_uv[:,1], texture_name)
                calls = [lambda: reconstruct_nearest(texture,full_uv),
                         lambda: reconstruct_bilinear(texture,full_uv),
                         lambda: reconstruct_bicubic_direct(texture,full_uv),
                         lambda: reconstruct_bicubic_shared(texture,full_uv,pixel_xy,WIDTH)]
                outputs = {}
                for method, call in zip(METHODS, calls):
                    result, median_ms, samples_ms = timed(call, args.repeats)
                    if method=='bicubic_shared':
                        result, stats = result
                    outputs[method] = result
                    row = {'texture':texture_name, 'texture_size':size, 'angle_degrees':angle,
                           'method':method, 'covered_pixels':len(full_uv), 'evaluation_pixels':int(interior.sum()),
                           **metrics(result[interior], reference[interior]),
                           'filter_median_ms':median_ms,
                           'timing_repeats':args.repeats,
                           'timing_samples_ms':';'.join(f'{v:.6f}' for v in samples_ms)}
                    quality_rows.append(row)
                    if angle==30 and size==32:
                        Image.fromarray(image_array(result, geometry.mask)).save(output/f'{texture_name}_n32_a30_{method}.png')
                if angle==30 and size==32:
                    Image.fromarray(image_array(reference, geometry.mask)).save(output/f'{texture_name}_n32_a30_reference.png')
                match = float(np.max(np.abs(outputs['bicubic_direct']-outputs['bicubic_shared'])))
                max_match = max(max_match, match)
                assert match < 1e-12, match
                if texture_name == TEXTURES[0]:
                    cost_rows.append({'texture_size':size, 'angle_degrees':angle, **asdict(stats)})
                print(f'{texture_name:12} N={size:3} angle={angle:2} bicubic equality={match:.2e}', flush=True)

    save_csv(output/'quality_results.csv', quality_rows)
    save_csv(output/'cost_results.csv', cost_rows)
    grouped = []
    for texture in TEXTURES:
        for size in SIZES:
            for method in METHODS:
                rows = [r for r in quality_rows if r['texture']==texture and r['texture_size']==size and r['method']==method]
                mean_mse = float(np.mean([r['mse_raw'] for r in rows]))
                grouped.append({'texture':texture,'texture_size':size,'method':method,
                                'aggregate_rmse':float(np.sqrt(mean_mse)),
                                'aggregate_psnr_db':float(-10*np.log10(mean_mse)),
                                'mean_out_of_range_channel_percent':float(np.mean([r['out_of_range_channel_percent'] for r in rows])),
                                'mean_filter_median_ms':float(np.mean([r['filter_median_ms'] for r in rows]))})
    save_csv(output/'quality_by_texture.csv', grouped)
    cost_summary = []
    for size in SIZES:
        rows = [r for r in cost_rows if r['texture_size']==size]
        pixels = sum(r['pixels'] for r in rows); unique = sum(r['shared_unique_texel_evaluations'] for r in rows)
        cost_summary.append({'texture_size':size,'total_pixels':pixels,
                             'unique_texels_per_pixel':unique/pixels,
                             'direct_texels_per_pixel':16,
                             'logical_evaluation_reduction_percent':100*(1-unique/(16*pixels))})
    save_csv(output/'cost_summary.csv', cost_summary)
    per_texture = []
    for texture in TEXTURES:
        item = {'texture':texture}
        for method in METHODS[:3]:
            rows = [r for r in quality_rows if r['texture']==texture and r['method']==method]
            mse = float(np.mean([r['mse_raw'] for r in rows]))
            item[method+'_rmse'] = float(np.sqrt(mse))
            item[method+'_psnr_db'] = float(-10*np.log10(mse))
        item['bicubic_gain_over_bilinear_db'] = item['bicubic_direct_psnr_db']-item['bilinear_psnr_db']
        per_texture.append(item)
    summary = {
        'experiment':'Perspective-correct UV plus Catmull-Rom bicubic reconstruction and exact CPU texel sharing',
        'source_url':'https://arxiv.org/html/2506.17770v1',
        'source_scope':'Inspired by sections 3 and 3.1 (List Merge). Not a complete GPU replication or SOTA performance benchmark.',
        'settings':{'angles':ANGLES,'texture_sizes':SIZES,'texture_types':TEXTURES,'resolution':[WIDTH,HEIGHT],
                    'sharing_tile_pixels':[8,4], 'quality_cases':36,'method_rows':144},
        'reference':'Known analytic signal evaluated at perspective-correct UV; pointwise reconstruction reference, not pixel-footprint-integrated antialiasing ground truth.',
        'color_space':'Synthetic values are treated as linear RGB; metrics use raw float64 outputs before clipping. PNGs clamp to [0,1] and quantize without a display transfer curve.',
        'aggregation':'Per-case MSE averages pixels/channels. Aggregate MSE is the unweighted mean of case MSE values; aggregate PSNR is -10log10(aggregate MSE). Cost summary pools valid rendering pixels over angles. Border margin 2 texels excluded from all quality metrics.',
        'limit':'Full stencil union with no 32-texel cap; tiles needing more texels are still evaluated exactly. Logical counts do not estimate GPU cache misses or bandwidth. No GPU wave intrinsics, neural texture decompression, or paper stochastic fallbacks.',
        'checks':{**checks,'experiment_direct_shared_max_abs_error':max_match},
        'quality_summary':per_texture,'cost_summary':cost_summary,
        'timing':{'platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,
                  'repeats':args.repeats,'scope':'Warmup then median full filtering-call time; shared path includes merge/gather and stats. Excludes geometry and reference computation. Python/NumPy CPU timing is not GPU speed.'},
        'plots_created':plot_results(output, grouped, cost_rows),
    }
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'quality_summary':per_texture,'cost_summary':cost_summary,'checks':summary['checks']},indent=2))


if __name__=='__main__':
    main()
