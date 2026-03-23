"""Load TIFF/JPG/PNG/LIF images and extract metadata."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ImageMetadata:
    """Metadata extracted from a microscopy image."""
    filepath: Path = field(default_factory=lambda: Path())
    num_channels: int = 1
    width: int = 0
    height: int = 0
    dtype: str = "uint8"
    # Scale in µm per pixel (None if unknown)
    scale_um_per_px: float | None = None
    unit: str = ""
    # Per-channel display ranges [(min, max), ...]
    display_ranges: list[tuple[float, float]] = field(default_factory=list)
    # Per-channel LUT color names
    channel_colors: list[str] = field(default_factory=list)
    # Whether quantitative fluorescence is valid
    quantitative: bool = False
    # Optional manual laser power
    laser_power: str = ""


def _classify_lut_color(lut: np.ndarray) -> str:
    """Determine the dominant color of a 3x256 LUT array."""
    # Sum each row (R, G, B channel) to see which is dominant
    sums = lut.sum(axis=1).astype(float)
    r, g, b = sums[0], sums[1], sums[2]
    total = r + g + b
    if total == 0:
        return "gray"
    if r > 0 and g > 0 and b > 0:
        # Check if roughly equal → gray/white
        ratios = np.array([r, g, b]) / total
        if ratios.min() > 0.25:
            return "gray"
    # R+B with no G → magenta
    if r > 0 and b > 0 and g == 0:
        return "magenta"
    if r > g and r > b:
        if b > g * 0.5:
            return "magenta"
        return "red"
    if g > r and g > b:
        return "green"
    if b > r and b > g:
        if r > g * 0.5:
            return "magenta"
        return "blue"
    if r > 0 and g > 0:
        return "yellow"
    return "gray"


def get_lif_series_list(filepath: str | Path) -> list[tuple[str, tuple[int, ...], int]]:
    """Return list of (name, dimensions, num_channels) for each series in a .lif file."""
    from readlif.reader import LifFile

    lif = LifFile(str(filepath))
    series_info = []
    for img in lif.get_iter_image():
        dims = img.dims
        # dims is a named tuple with x, y, z, t, m attributes
        n_channels = img.channels
        size = (dims.x, dims.y)
        if dims.z > 1:
            size = (*size, dims.z)
        series_info.append((img.name, size, n_channels))
    return series_info


def load_image(
    filepath: str | Path, series_index: int = 0
) -> tuple[np.ndarray, ImageMetadata]:
    """Load an image and return (channel_data, metadata).

    Args:
        filepath: Path to image file.
        series_index: For .lif files with multiple series, which series to load.

    Returns:
        channel_data: numpy array with shape (C, H, W) — always 3D even for single channel.
        metadata: ImageMetadata with extracted info.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix in (".tif", ".tiff"):
        return _load_tiff(filepath)
    elif suffix in (".lif",):
        return _load_lif(filepath, series_index)
    elif suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        return _load_standard(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _load_tiff(filepath: Path) -> tuple[np.ndarray, ImageMetadata]:
    """Load a TIFF file, handling ImageJ multi-channel composites."""
    import tifffile

    tif = tifffile.TiffFile(str(filepath))
    meta = ImageMetadata(filepath=filepath, quantitative=True)

    # Read image data via series (handles ImageJ composites correctly)
    if tif.series:
        data = tif.series[0].asarray()
        axes = tif.series[0].axes
    else:
        data = tif.asarray()
        axes = "YX"

    # Normalize to (C, H, W)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]  # (1, H, W)
    elif data.ndim == 3:
        if axes and axes[0] in ("C", "S", "I"):
            pass  # Already (C, H, W)
        else:
            # Assume (H, W, C) — e.g., RGB
            data = np.moveaxis(data, -1, 0)

    meta.num_channels = data.shape[0]
    meta.height = data.shape[1]
    meta.width = data.shape[2]
    meta.dtype = str(data.dtype)

    # Extract scale from TIFF tags
    page = tif.pages[0]
    x_res_tag = page.tags.get(282)  # XResolution
    y_res_tag = page.tags.get(283)  # YResolution
    if x_res_tag is not None:
        num, denom = x_res_tag.value
        if num > 0:
            px_per_unit = num / denom  # pixels per unit
            meta.scale_um_per_px = 1.0 / px_per_unit

    # Extract unit and channel info from ImageJ metadata
    ij = tif.imagej_metadata
    if ij:
        meta.unit = ij.get("unit", "")
        channels = ij.get("channels", 1)
        if meta.num_channels == 1 and channels > 1:
            meta.num_channels = channels

        # Display ranges
        ranges = ij.get("Ranges")
        if ranges:
            for i in range(0, len(ranges), 2):
                if i + 1 < len(ranges):
                    meta.display_ranges.append((ranges[i], ranges[i + 1]))

        # LUT colors
        luts = ij.get("LUTs")
        if luts:
            for lut in luts:
                meta.channel_colors.append(_classify_lut_color(lut))

    # Fill default display ranges if missing
    while len(meta.display_ranges) < meta.num_channels:
        info = np.iinfo(data.dtype) if np.issubdtype(data.dtype, np.integer) else None
        lo = 0.0
        hi = float(info.max) if info else 1.0
        meta.display_ranges.append((lo, hi))

    # Fill default colors if missing
    default_colors = ["green", "red", "blue", "cyan", "magenta", "yellow"]
    while len(meta.channel_colors) < meta.num_channels:
        idx = len(meta.channel_colors)
        meta.channel_colors.append(default_colors[idx % len(default_colors)])

    tif.close()
    return data, meta


def _load_lif(filepath: Path, series_index: int = 0) -> tuple[np.ndarray, ImageMetadata]:
    """Load a Leica .lif file series as (C, H, W) array with metadata."""
    from readlif.reader import LifFile

    lif = LifFile(str(filepath))
    images = list(lif.get_iter_image())
    if series_index >= len(images):
        raise ValueError(
            f"Series index {series_index} out of range (file has {len(images)} series)"
        )
    img = images[series_index]

    meta = ImageMetadata(filepath=filepath, quantitative=True)
    n_channels = img.channels
    meta.num_channels = n_channels

    # Build (C, H, W) array by reading each channel at z=0, t=0
    channel_arrays = []
    for c in range(n_channels):
        frame = img.get_frame(z=0, t=0, c=c)
        arr = np.array(frame)
        channel_arrays.append(arr)

    data = np.stack(channel_arrays)  # (C, H, W)
    meta.height = data.shape[1]
    meta.width = data.shape[2]
    meta.dtype = str(data.dtype)

    # Extract scale from image metadata
    # readlif scale is in pixels per micrometer: (n_pixels - 1) / (length_m * 1e6)
    # So µm/px = 1.0 / scale
    scale = img.scale
    if scale and len(scale) >= 1 and scale[0] is not None and scale[0] > 0:
        meta.scale_um_per_px = 1.0 / scale[0]
        meta.unit = "µm"

    # Auto-range display from actual data for each channel
    for c in range(n_channels):
        ch = data[c]
        lo = float(ch.min())
        hi = float(ch.max())
        if hi <= lo:
            hi = lo + 1.0
        meta.display_ranges.append((lo, hi))

    # Default channel colors
    default_colors = ["green", "red", "blue", "cyan", "magenta", "yellow"]
    for i in range(n_channels):
        meta.channel_colors.append(default_colors[i % len(default_colors)])

    return data, meta


def _load_standard(filepath: Path) -> tuple[np.ndarray, ImageMetadata]:
    """Load JPG/PNG using OpenCV — detection only, no quantitative fluorescence."""
    import cv2

    img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Failed to load image: {filepath}")

    meta = ImageMetadata(filepath=filepath, quantitative=False)

    if img.ndim == 2:
        data = img[np.newaxis, :, :]
        meta.num_channels = 1
        meta.channel_colors = ["gray"]
    elif img.ndim == 3:
        # OpenCV loads as BGR
        if img.shape[2] == 3:
            # Convert to RGB channel order
            data = np.stack([img[:, :, 2], img[:, :, 1], img[:, :, 0]])
            meta.num_channels = 3
            meta.channel_colors = ["red", "green", "blue"]
        elif img.shape[2] == 4:
            data = np.stack([img[:, :, 2], img[:, :, 1], img[:, :, 0], img[:, :, 3]])
            meta.num_channels = 4
            meta.channel_colors = ["red", "green", "blue", "alpha"]
        else:
            data = np.moveaxis(img, -1, 0)
            meta.num_channels = data.shape[0]
            meta.channel_colors = [f"ch{i}" for i in range(meta.num_channels)]
    else:
        raise ValueError(f"Unexpected image dimensions: {img.shape}")

    meta.height = data.shape[1]
    meta.width = data.shape[2]
    meta.dtype = str(data.dtype)

    # No scale info from JPG/PNG
    meta.display_ranges = [(0.0, 255.0)] * meta.num_channels

    return data, meta
