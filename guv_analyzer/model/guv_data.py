"""Data classes and export utilities for GUV analysis results."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .guv_detector import DetectedGUV
from .fluorescence import FluorescenceResult


@dataclass
class GUVMeasurement:
    """Complete measurement for one GUV across all channels."""
    guv: DetectedGUV
    diameter_um: float | None  # None if no scale info
    fluorescence: list[FluorescenceResult] = field(default_factory=list)


def build_dataframe(
    measurements: list[GUVMeasurement],
    scale_um_per_px: float | None = None,
    channel_colors: list[str] | None = None,
) -> pd.DataFrame:
    """Convert measurements to a pandas DataFrame for display/export."""
    rows = []
    for m in measurements:
        row = {
            "ID": m.guv.id,
            "Center X (px)": round(m.guv.center_x, 1),
            "Center Y (px)": round(m.guv.center_y, 1),
            "Radius (px)": round(m.guv.radius, 1),
            "Diameter (px)": round(m.guv.diameter_px, 1),
        }
        if m.diameter_um is not None:
            row["Diameter (µm)"] = round(m.diameter_um, 2)

        for fl in m.fluorescence:
            ch_label = f"Ch{fl.channel_index + 1}"
            if channel_colors and fl.channel_index < len(channel_colors):
                ch_label += f" ({channel_colors[fl.channel_index]})"
            row[f"{ch_label} Area Mean"] = round(fl.area_mean, 2)
            row[f"{ch_label} Area Std"] = round(fl.area_std, 2)
            row[f"{ch_label} Membrane Mean"] = round(fl.membrane_mean, 2)
            row[f"{ch_label} Membrane Std"] = round(fl.membrane_std, 2)

        rows.append(row)

    return pd.DataFrame(rows)


def filter_active_measurements(measurements: list[GUVMeasurement]) -> list[GUVMeasurement]:
    """Return only measurements whose GUV is not excluded."""
    return [m for m in measurements if not m.guv.excluded]


def export_csv(df: pd.DataFrame, filepath: str | Path) -> None:
    """Export DataFrame to CSV."""
    df.to_csv(str(filepath), index=False)


def export_excel(df: pd.DataFrame, filepath: str | Path) -> None:
    """Export DataFrame to Excel."""
    df.to_excel(str(filepath), index=False, engine="openpyxl")
