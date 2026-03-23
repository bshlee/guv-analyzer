"""GUV circle detection using HoughCircles + contour-based hybrid approach."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DetectionParams:
    """Parameters for GUV detection."""
    min_radius_px: int = 8
    max_radius_px: int = 500
    blur_sigma: float = 2.0
    sensitivity: float = 0.85  # 0-1, maps to HoughCircles param2
    min_distance_px: int = 15
    use_clahe: bool = True
    border_margin_px: int = 5
    # Contour-based detection params
    circularity_min: float = 0.65  # How circular a contour must be (0-1)


@dataclass
class DetectedGUV:
    """A single detected GUV."""
    id: int
    center_x: float
    center_y: float
    radius: float  # in pixels
    excluded: bool = False

    @property
    def diameter_px(self) -> float:
        return self.radius * 2


@dataclass
class OverlapGroup:
    """A group of mutually overlapping GUVs."""
    group_id: int
    guv_ids: list[int]
    bbox: tuple[int, int, int, int]  # x_min, y_min, x_max, y_max


def _circle_overlap_fraction(r1: float, r2: float, dist: float) -> float:
    """Compute the overlap area as a fraction of the smaller circle's area.

    Returns a value in [0, 1]:
    - 0 means no overlap (or just touching)
    - 1 means the smaller circle is fully contained
    """
    r_small = min(r1, r2)
    r_large = max(r1, r2)
    small_area = np.pi * r_small ** 2

    if dist >= r1 + r2:
        return 0.0
    if dist <= abs(r1 - r2):
        return 1.0  # smaller fully inside larger

    # Standard circle–circle intersection area formula
    d = dist
    part1 = r1 ** 2 * np.arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
    part2 = r2 ** 2 * np.arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
    part3 = 0.5 * np.sqrt(
        (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
    )
    intersection = part1 + part2 - part3
    return float(intersection / small_area)


def find_overlaps(guvs: list['DetectedGUV'], padding_px: int = 20,
                  img_width: int = 0, img_height: int = 0,
                  overlap_threshold: float = 0.30) -> list[OverlapGroup]:
    """Find groups of overlapping GUVs using union-find.

    Two GUVs are considered overlapping when the intersection area exceeds
    *overlap_threshold* (fraction of the smaller circle's area).
    Default 0.20 = 20%.

    Returns only groups with 2+ members.
    """
    if len(guvs) < 2:
        return []

    # Union-find
    parent: dict[int, int] = {g.id: g.id for g in guvs}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Check all pairs for overlap exceeding the threshold
    for i, g1 in enumerate(guvs):
        for g2 in guvs[i + 1:]:
            dist = np.sqrt((g1.center_x - g2.center_x) ** 2 +
                           (g1.center_y - g2.center_y) ** 2)
            if dist < g1.radius + g2.radius:
                frac = _circle_overlap_fraction(g1.radius, g2.radius, dist)
                if frac >= overlap_threshold:
                    union(g1.id, g2.id)

    # Group by root
    from collections import defaultdict
    groups_map: dict[int, list[int]] = defaultdict(list)
    for g in guvs:
        groups_map[find(g.id)].append(g.id)

    # Build OverlapGroup objects (only groups with 2+ members)
    guv_map = {g.id: g for g in guvs}
    result = []
    for group_idx, (_, member_ids) in enumerate(sorted(groups_map.items())):
        if len(member_ids) < 2:
            continue
        # Compute bounding box
        x_min = int(min(guv_map[gid].center_x - guv_map[gid].radius for gid in member_ids) - padding_px)
        y_min = int(min(guv_map[gid].center_y - guv_map[gid].radius for gid in member_ids) - padding_px)
        x_max = int(max(guv_map[gid].center_x + guv_map[gid].radius for gid in member_ids) + padding_px)
        y_max = int(max(guv_map[gid].center_y + guv_map[gid].radius for gid in member_ids) + padding_px)
        # Clamp to image bounds
        if img_width > 0 and img_height > 0:
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width, x_max)
            y_max = min(img_height, y_max)
        result.append(OverlapGroup(
            group_id=len(result) + 1,
            guv_ids=sorted(member_ids),
            bbox=(x_min, y_min, x_max, y_max),
        ))

    return result


def _preprocess(img: np.ndarray, params: DetectionParams) -> np.ndarray:
    """Common preprocessing: normalize to uint8, optional CLAHE, blur."""
    if img.dtype != np.uint8:
        if img.max() > 0:
            img = ((img.astype(np.float64) / img.max()) * 255).astype(np.uint8)
        else:
            return img

    if params.use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

    ksize = max(3, int(params.blur_sigma * 4) | 1)
    img = cv2.GaussianBlur(img, (ksize, ksize), params.blur_sigma)
    return img


def _detect_hough(blurred: np.ndarray, params: DetectionParams) -> list[tuple[float, float, float]]:
    """Detect circles using HoughCircles with multi-pass param1 sweep.

    A single high param1 (Sobel edge threshold) can miss GUVs in
    compressed images (JPG/PNG) or faint membranes.  Running multiple
    passes at decreasing thresholds catches both strong and weak edges.
    """
    param2 = max(0.1, 1.0 - params.sensitivity)
    all_circles: list[tuple[float, float, float]] = []

    for p1 in (300, 150, 50):
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT_ALT,
            dp=1.5,
            minDist=params.min_distance_px,
            param1=p1,
            param2=param2,
            minRadius=params.min_radius_px,
            maxRadius=params.max_radius_px,
        )
        if circles is not None:
            all_circles.extend(
                (float(cx), float(cy), float(r)) for cx, cy, r in circles[0]
            )

    return all_circles


def _detect_contours(blurred: np.ndarray, original: np.ndarray,
                     params: DetectionParams) -> list[tuple[float, float, float]]:
    """Detect circular objects via adaptive thresholding + contour analysis."""
    results = []

    # Multi-level thresholding to catch GUVs at different intensities
    # Adaptive threshold works well for varying local brightness
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=51, C=-3
    )

    # Also try Otsu on CLAHE-enhanced image for global threshold
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for binary in [adaptive, otsu]:
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < np.pi * params.min_radius_px ** 2 * 0.5:
                continue
            if area > np.pi * params.max_radius_px ** 2 * 1.5:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            # Circularity: 4π·area / perimeter²  (1.0 = perfect circle)
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < params.circularity_min:
                continue

            # Fit minimum enclosing circle
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)

            # Check that the enclosing circle area is close to contour area
            # (rejects elongated shapes that pass circularity check)
            circle_area = np.pi * radius ** 2
            fill_ratio = area / circle_area
            if fill_ratio < 0.3:
                continue

            if params.min_radius_px <= radius <= params.max_radius_px:
                results.append((float(cx), float(cy), float(radius)))

    # --- Ring detection via Canny edges ---
    # GUV membranes appear as bright rings; thresholding may only capture the
    # membrane band, not a filled disc.  Canny + contour fitting handles this.
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in edge_contours:
        if len(cnt) < 20:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if not (params.min_radius_px <= radius <= params.max_radius_px):
            continue

        # For edge contours, use perimeter-based circularity:
        # ideal circle perimeter = 2πr; ratio near 1.0 = circular
        ideal_perimeter = 2 * np.pi * radius
        perimeter_ratio = perimeter / ideal_perimeter if ideal_perimeter > 0 else 0
        # Accept contours whose perimeter is 0.7–1.5× ideal (allows partial rings)
        if perimeter_ratio < 0.7 or perimeter_ratio > 1.5:
            continue

        results.append((float(cx), float(cy), float(radius)))

    return results


def _merge_detections(
    all_detections: list[tuple[float, float, float]],
    min_distance: int,
) -> list[tuple[float, float, float]]:
    """Merge overlapping detections, keeping the one with larger radius."""
    if not all_detections:
        return []

    # Sort by radius descending — prefer larger detections
    sorted_dets = sorted(all_detections, key=lambda d: d[2], reverse=True)
    merged = []

    for cx, cy, r in sorted_dets:
        is_duplicate = False
        for mx, my, mr in merged:
            dist = np.sqrt((cx - mx) ** 2 + (cy - my) ** 2)
            # If centers are within the smaller radius, it's a duplicate
            if dist < min(r, mr) * 0.8:
                is_duplicate = True
                break
        if not is_duplicate:
            merged.append((cx, cy, r))

    return merged


def detect_guvs(
    channel: np.ndarray,
    params: DetectionParams,
) -> list[DetectedGUV]:
    """Detect circular GUVs using a hybrid Hough + contour approach.

    Args:
        channel: 2D array (single channel).
        params: Detection parameters.

    Returns:
        List of DetectedGUV objects.
    """
    img = channel.copy()

    # Ensure uint8
    if img.dtype != np.uint8:
        if img.max() > 0:
            img = ((img.astype(np.float64) / img.max()) * 255).astype(np.uint8)
        else:
            return []

    blurred = _preprocess(img, params)

    # Run both detection methods
    hough_results = _detect_hough(blurred, params)
    contour_results = _detect_contours(blurred, img, params)

    # Merge all detections
    all_detections = hough_results + contour_results
    merged = _merge_detections(all_detections, params.min_distance_px)

    # Filter by border margin
    h, w = channel.shape
    margin = params.border_margin_px
    results = []

    for cx, cy, r in merged:
        if (cx - r < margin or cx + r > w - margin or
                cy - r < margin or cy + r > h - margin):
            continue

        results.append(DetectedGUV(
            id=len(results) + 1,
            center_x=cx,
            center_y=cy,
            radius=r,
        ))

    return results
