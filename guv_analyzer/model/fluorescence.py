"""Fluorescence measurement for detected GUVs — lumen (area) and membrane (donut)."""

from dataclasses import dataclass

import numpy as np

from .guv_detector import DetectedGUV


@dataclass
class FluorescenceResult:
    """Fluorescence measurements for one GUV on one channel."""
    channel_index: int
    area_mean: float  # Mean intensity inside the lumen
    area_std: float
    membrane_mean: float  # Mean intensity on the membrane ring
    membrane_std: float
    area_pixel_count: int
    membrane_pixel_count: int


def measure_fluorescence(
    channel_data: np.ndarray,
    guv: DetectedGUV,
    channel_index: int,
    membrane_width_px: float = 4.0,
) -> FluorescenceResult:
    """Measure lumen and membrane fluorescence for a single GUV on a single channel.

    The lumen is the interior circle excluding the membrane ring.
    The membrane is an annular ring centered on the detected radius.

    Args:
        channel_data: 2D array for one channel.
        guv: Detected GUV with center and radius.
        channel_index: Which channel this measurement is for.
        membrane_width_px: Width of the membrane annulus in pixels.
    """
    h, w = channel_data.shape
    cx, cy, r = guv.center_x, guv.center_y, guv.radius
    half_w = membrane_width_px / 2.0

    # Create coordinate grids (only in the bounding box for efficiency)
    y_min = max(0, int(cy - r - half_w - 1))
    y_max = min(h, int(cy + r + half_w + 2))
    x_min = max(0, int(cx - r - half_w - 1))
    x_max = min(w, int(cx + r + half_w + 2))

    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2

    # Lumen mask: inside the circle but outside the membrane inner edge
    inner_r = r - half_w
    lumen_mask = dist_sq < (inner_r ** 2)

    # Membrane mask: annular ring
    outer_r = r + half_w
    membrane_mask = (dist_sq >= (inner_r ** 2)) & (dist_sq <= (outer_r ** 2))

    # Extract pixels
    roi = channel_data[y_min:y_max, x_min:x_max].astype(np.float64)
    lumen_pixels = roi[lumen_mask]
    membrane_pixels = roi[membrane_mask]

    return FluorescenceResult(
        channel_index=channel_index,
        area_mean=float(np.mean(lumen_pixels)) if len(lumen_pixels) > 0 else 0.0,
        area_std=float(np.std(lumen_pixels)) if len(lumen_pixels) > 0 else 0.0,
        membrane_mean=float(np.mean(membrane_pixels)) if len(membrane_pixels) > 0 else 0.0,
        membrane_std=float(np.std(membrane_pixels)) if len(membrane_pixels) > 0 else 0.0,
        area_pixel_count=int(lumen_pixels.size),
        membrane_pixel_count=int(membrane_pixels.size),
    )


def measure_all_channels(
    channel_arrays: np.ndarray,
    guv: DetectedGUV,
    membrane_width_px: float = 4.0,
) -> list[FluorescenceResult]:
    """Measure fluorescence across all channels for one GUV.

    Args:
        channel_arrays: 3D array (C, H, W).
        guv: Detected GUV.
        membrane_width_px: Width of membrane annulus.

    Returns:
        List of FluorescenceResult, one per channel.
    """
    results = []
    for ch_idx in range(channel_arrays.shape[0]):
        result = measure_fluorescence(
            channel_arrays[ch_idx], guv, ch_idx, membrane_width_px
        )
        results.append(result)
    return results
