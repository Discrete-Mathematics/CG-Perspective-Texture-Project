"""A small CPU triangle rasterizer for a perspective-texture interpolation study."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from PIL import Image


WIDTH = 512
HEIGHT = 512
TEXTURE_SIZE = 256
BACKGROUND = np.array([17, 25, 38], dtype=np.uint8)


@dataclass
class RenderResult:
    image: np.ndarray
    uv: np.ndarray
    mask: np.ndarray


def make_checker_texture(size: int = TEXTURE_SIZE, cells: int = 12) -> np.ndarray:
    """Create a deterministic, high-contrast texture without external assets."""
    yy, xx = np.indices((size, size))
    tile = size / cells
    checks = ((xx // tile).astype(int) + (yy // tile).astype(int)) % 2
    light = np.array([238, 228, 202], dtype=np.uint8)
    dark = np.array([28, 112, 142], dtype=np.uint8)
    tex = np.where(checks[..., None] == 0, light, dark)

    # A thin orange border makes the texture orientation easier to see.
    border = max(1, size // 96)
    tex[:border] = (235, 116, 74)
    tex[-border:] = (235, 116, 74)
    tex[:, :border] = (235, 116, 74)
    tex[:, -border:] = (235, 116, 74)
    return tex


def perspective_matrix(fov_degrees: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.deg2rad(fov_degrees) * 0.5)
    return np.array(
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )


def scene_vertices(angle_degrees: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return clip-space corners, UVs, and triangle indices for a tilted quad."""
    positions = np.array(
        [
            [-1.35, -0.95, 0.0, 1.0],
            [1.35, -0.95, 0.0, 1.0],
            [1.35, 0.95, 0.0, 1.0],
            [-1.35, 0.95, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)

    angle = np.deg2rad(angle_degrees)
    rotate_y = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    translate = np.eye(4, dtype=np.float64)
    translate[2, 3] = -4.2
    projection = perspective_matrix(52.0, WIDTH / HEIGHT, 0.1, 50.0)
    clip = (projection @ translate @ rotate_y @ positions.T).T
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return clip, uvs, triangles


def _edge(a: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])


def _sample_nearest(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    tex_h, tex_w = texture.shape[:2]
    u = np.clip(uv[..., 0], 0.0, 1.0 - np.finfo(np.float64).eps)
    v = np.clip(uv[..., 1], 0.0, 1.0 - np.finfo(np.float64).eps)
    tx = np.minimum((u * tex_w).astype(np.int32), tex_w - 1)
    ty = np.minimum(((1.0 - v) * tex_h).astype(np.int32), tex_h - 1)
    return texture[ty, tx]


def rasterize(angle_degrees: float, mode: str, texture: np.ndarray | None = None) -> RenderResult:
    """Rasterize a textured perspective-projected quad.

    mode='affine' linearly interpolates UV in screen space.
    mode='perspective' divides the interpolated UV/w by interpolated 1/w.
    """
    if mode not in {"affine", "perspective"}:
        raise ValueError("mode must be 'affine' or 'perspective'")
    if texture is None:
        texture = make_checker_texture()

    clip, vertex_uv, triangles = scene_vertices(angle_degrees)
    inv_w = 1.0 / clip[:, 3]
    ndc = clip[:, :3] * inv_w[:, None]
    screen = np.empty((len(clip), 2), dtype=np.float64)
    screen[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * (WIDTH - 1)
    screen[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (HEIGHT - 1)

    image = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:] = BACKGROUND
    z_buffer = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float64)
    uv_buffer = np.full((HEIGHT, WIDTH, 2), np.nan, dtype=np.float64)
    covered = np.zeros((HEIGHT, WIDTH), dtype=bool)

    for tri in triangles:
        points = screen[tri]
        area = _edge(points[0], points[1], np.array(points[2, 0]), np.array(points[2, 1])).item()
        if abs(area) < 1e-12:
            continue

        x_min = max(0, int(np.floor(points[:, 0].min())))
        x_max = min(WIDTH - 1, int(np.ceil(points[:, 0].max())))
        y_min = max(0, int(np.floor(points[:, 1].min())))
        y_max = min(HEIGHT - 1, int(np.ceil(points[:, 1].max())))
        if x_min > x_max or y_min > y_max:
            continue

        yy, xx = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]
        px = xx.astype(np.float64) + 0.5
        py = yy.astype(np.float64) + 0.5
        l0 = _edge(points[1], points[2], px, py) / area
        l1 = _edge(points[2], points[0], px, py) / area
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
        if not inside.any():
            continue

        lambdas = np.stack((l0, l1, l2), axis=-1)
        tri_inv_w = inv_w[tri]
        tri_uv = vertex_uv[tri]
        if mode == "affine":
            uv = lambdas @ tri_uv
        else:
            weighted = lambdas * tri_inv_w[None, None, :]
            denominator = weighted.sum(axis=-1, keepdims=True)
            numerator = weighted @ tri_uv
            uv = numerator / np.maximum(denominator, 1e-15)

        depth = lambdas @ ndc[tri, 2]
        z_view = z_buffer[y_min : y_max + 1, x_min : x_max + 1]
        visible = inside & (depth < z_view)
        if not visible.any():
            continue

        sampled = _sample_nearest(texture, uv)
        image_region = image[y_min : y_max + 1, x_min : x_max + 1]
        uv_region = uv_buffer[y_min : y_max + 1, x_min : x_max + 1]
        mask_region = covered[y_min : y_max + 1, x_min : x_max + 1]
        image_region[visible] = sampled[visible]
        z_view[visible] = depth[visible]
        uv_region[visible] = uv[visible]
        mask_region[visible] = True

    return RenderResult(image=image, uv=uv_buffer, mask=covered)


def save_result(result: RenderResult, path: str) -> None:
    Image.fromarray(result.image, mode="RGB").save(path)

