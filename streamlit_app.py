"""GUV Analyzer — Streamlit Web Interface.

Reuses the model layer from guv_analyzer (no PyQt6 dependency).
Run with: streamlit run streamlit_app.py
"""

import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from guv_analyzer.model.image_loader import load_image, get_lif_series_list, ImageMetadata
from guv_analyzer.model.guv_detector import detect_guvs, DetectionParams, DetectedGUV, find_overlaps
from guv_analyzer.model.fluorescence import measure_all_channels
from guv_analyzer.model.guv_data import GUVMeasurement, build_dataframe, filter_active_measurements
from guv_analyzer.feedback import FEEDBACK_CONFIG as _FB_CONFIG

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="GUV Analyzer", layout="wide")

# ---------------------------------------------------------------------------
# Color map (shared with ImageCanvas)
# ---------------------------------------------------------------------------
COLOR_MAP = {
    "gray": (1, 1, 1),
    "red": (1, 0, 0),
    "green": (0, 1, 0),
    "blue": (0, 0, 1),
    "cyan": (0, 1, 1),
    "magenta": (1, 0, 1),
    "yellow": (1, 1, 0),
}

# RGB colors for cv2 overlay drawing
OVERLAY_COLOR_RGB = (0, 255, 255)    # cyan
HIGHLIGHT_COLOR_RGB = (255, 255, 0)  # yellow
EXCLUDED_COLOR_RGB = (255, 80, 80)   # red


# ---------------------------------------------------------------------------
# Rendering helpers (pure numpy — ported from image_canvas.py)
# ---------------------------------------------------------------------------

def render_channel(channel: np.ndarray, color: str, lo: float, hi: float) -> np.ndarray:
    """Render a single channel as (H, W, 3) uint8 RGB."""
    img = channel.astype(np.float64)
    img = np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    h, w = img.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    r_f, g_f, b_f = COLOR_MAP.get(color, (1, 1, 1))
    rgb[:, :, 0] = (img * r_f).astype(np.uint8)
    rgb[:, :, 1] = (img * g_f).astype(np.uint8)
    rgb[:, :, 2] = (img * b_f).astype(np.uint8)
    return rgb


def render_composite(channels: np.ndarray, colors: list[str],
                     display_ranges: list[tuple[float, float]]) -> np.ndarray:
    """Render multi-channel composite as (H, W, 3) uint8 RGB."""
    c, h, w = channels.shape
    composite = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(c):
        ch = channels[i].astype(np.float64)
        lo, hi = display_ranges[i] if i < len(display_ranges) else (0, 255)
        ch = np.clip((ch - lo) / max(hi - lo, 1) * 255, 0, 255)
        col = colors[i] if i < len(colors) else "gray"
        r_f, g_f, b_f = COLOR_MAP.get(col, (1, 1, 1))
        composite[:, :, 0] += ch * r_f
        composite[:, :, 1] += ch * g_f
        composite[:, :, 2] += ch * b_f
    return np.clip(composite, 0, 255).astype(np.uint8)


def draw_guv_overlay(rgb: np.ndarray, guvs: list[DetectedGUV],
                     highlight_id: int | None = None,
                     excluded_ids: set[int] | None = None) -> np.ndarray:
    """Draw GUV circles and ID labels on an RGB image. Returns a copy."""
    if excluded_ids is None:
        excluded_ids = set()
    img = rgb.copy()
    for guv in guvs:
        is_hl = (guv.id == highlight_id)
        is_excluded = (guv.id in excluded_ids)
        if is_excluded:
            color = EXCLUDED_COLOR_RGB
            thickness = 2
        elif is_hl:
            color = HIGHLIGHT_COLOR_RGB
            thickness = 3
        else:
            color = OVERLAY_COLOR_RGB
            thickness = 2
        center = (int(round(guv.center_x)), int(round(guv.center_y)))
        radius = int(round(guv.radius))
        cv2.circle(img, center, radius, color, thickness)
        # ID label
        label_pos = (center[0] + radius + 4, center[1] + 4)
        cv2.putText(img, str(guv.id), label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Cached loader — avoids re-reading on every Streamlit rerun
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading image...")
def cached_load_image(file_bytes: bytes, filename: str, series_index: int = 0):
    """Write uploaded bytes to a temp file and call the model loader."""
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    channel_data, metadata = load_image(tmp_path, series_index=series_index)
    # Convert metadata.filepath to string so it's cacheable
    metadata.filepath = Path(filename)
    return channel_data, metadata


@st.cache_data(show_spinner="Reading LIF series list...")
def cached_lif_series(file_bytes: bytes, filename: str):
    """Get series list from a .lif file."""
    with tempfile.NamedTemporaryFile(suffix=".lif", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    return get_lif_series_list(tmp_path)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "channel_data": None,
        "metadata": None,
        "guvs": [],
        "measurements": [],
        "results_df": None,
        "highlight_id": None,
        "uploaded_file_id": None,
        "series_index": 0,
        "overlap_groups": [],
        "excluded_ids": set(),
        "selected_group_idx": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    _init_state()

    # ── CSS: sticky top bar ──────────────────────────────────────────────
    st.markdown("""
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"]:has(
        > div > div > div > div[data-testid="stMarkdown"] > div > p > span.top-bar-marker
    ) {
        position: sticky;
        top: 0;
        z-index: 999;
        background: var(--background-color, #0e1117);
        padding: 0.25rem 0;
        border-bottom: 1px solid rgba(250,250,250,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    # ── TOP BAR: file upload + image info (sticky) ───────────────────────
    with st.container(border=True):
        # Hidden marker for CSS sticky targeting
        st.markdown('<span class="top-bar-marker"></span>', unsafe_allow_html=True)

        top_left, top_right = st.columns([1, 2])

        with top_left:
            uploaded = st.file_uploader(
                "Upload image",
                type=["tif", "tiff", "lif", "jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )

            series_index = 0
            if uploaded is not None:
                file_bytes = uploaded.getvalue()
                file_id = f"{uploaded.name}_{len(file_bytes)}"

                # LIF series selector
                if uploaded.name.lower().endswith(".lif"):
                    try:
                        series_list = cached_lif_series(file_bytes, uploaded.name)
                    except Exception as e:
                        st.error(f"Failed to read .lif file: {e}")
                        return
                    if len(series_list) > 1:
                        series_names = [
                            f"{name} ({w}x{h}, {nc}ch)"
                            for name, (w, h, *_), nc in series_list
                        ]
                        series_index = st.selectbox(
                            "Select series", range(len(series_names)),
                            format_func=lambda i: series_names[i],
                        )

                # Load image if new file or different series
                needs_load = (
                    file_id != st.session_state.uploaded_file_id
                    or series_index != st.session_state.series_index
                )
                if needs_load:
                    try:
                        channel_data, metadata = cached_load_image(
                            file_bytes, uploaded.name, series_index
                        )
                    except Exception as e:
                        st.error(f"Failed to load image: {e}")
                        return
                    st.session_state.channel_data = channel_data
                    st.session_state.metadata = metadata
                    st.session_state.uploaded_file_id = file_id
                    st.session_state.series_index = series_index
                    st.session_state.guvs = []
                    st.session_state.measurements = []
                    st.session_state.results_df = None
                    st.session_state.highlight_id = None
                    st.session_state.overlap_groups = []
                    st.session_state.excluded_ids = set()
                    st.session_state.selected_group_idx = 0

        with top_right:
            meta: ImageMetadata | None = st.session_state.metadata
            if meta is not None:
                scale_str = f"{meta.scale_um_per_px:.4f} µm/px" if meta.scale_um_per_px else "N/A"
                quant_str = "Yes" if meta.quantitative else "No"
                st.markdown(
                    f"**{meta.filepath.name}** · "
                    f"{meta.width} x {meta.height} · "
                    f"{meta.num_channels} ch · "
                    f"Scale: {scale_str} · "
                    f"Quantitative: {quant_str}",
                )

    # ── Bail early if no image loaded ────────────────────────────────────
    channel_data = st.session_state.channel_data
    meta = st.session_state.metadata

    if channel_data is None or meta is None:
        st.info("Upload a microscopy image (.tif, .tiff, .lif, .jpg, .png) to get started.")
        return

    # ── Build channel name list (needed by both panels) ──────────────────
    ch_names = []
    for i in range(meta.num_channels):
        color = meta.channel_colors[i] if i < len(meta.channel_colors) else "?"
        ch_names.append(f"Ch{i+1} ({color})")

    # ── TWO-COLUMN LAYOUT: [main content (3) | detection panel (1)] ──────
    main_col, right_col = st.columns([3, 1], gap="medium")

    # ── RIGHT COLUMN: Detection parameters ───────────────────────────────
    with right_col:
        st.subheader("Detection")

        det_channel = 0
        if meta.num_channels > 1:
            det_channel = st.selectbox(
                "Detection channel", range(len(ch_names)),
                format_func=lambda i: ch_names[i],
                key="det_ch",
            )

        # Scale-aware sliders: show µm when scale is known, px otherwise
        sc = meta.scale_um_per_px
        has_scale = sc is not None

        if has_scale:
            min_radius_um = st.slider("Min radius (µm)", 0.5, 200.0, 2.5, 0.5)
            max_radius_um = st.slider("Max radius (µm)", 1.0, 500.0, 25.0, 0.5)
            min_distance_um = st.slider("Min distance (µm)", 0.5, 100.0, 5.0, 0.5)
            membrane_width_um = st.slider("Membrane width (µm)", 0.1, 10.0, 1.5, 0.1)
            # Convert µm → px
            min_radius = max(3, int(round(min_radius_um / sc)))
            max_radius = max(min_radius + 1, int(round(max_radius_um / sc)))
            min_distance = max(5, int(round(min_distance_um / sc)))
            membrane_width = max(1.0, membrane_width_um / sc)
        else:
            min_radius = st.slider("Min radius (px)", 3, 2000, 8)
            max_radius = st.slider("Max radius (px)", 10, 5000, 500)
            min_distance = st.slider("Min distance (px)", 5, 1000, 15)
            membrane_width = st.slider("Membrane width (px)", 1.0, 30.0, 4.0, 1.0)

        sensitivity = st.slider("Sensitivity", 0.1, 1.0, 0.85, 0.05)
        blur_sigma = st.slider("Blur sigma", 0.5, 20.0, 2.0, 0.5)
        circularity = st.slider("Min circularity", 0.3, 1.0, 0.65, 0.05)
        use_clahe = st.checkbox("Use CLAHE", value=True)
        overlap_threshold = st.slider(
            "Overlap threshold", 0.0, 1.0, 0.30, 0.05,
            help="Min overlap area (fraction of smaller GUV) to flag a pair. "
                 "0 = any touching, 1 = fully contained.",
        )

        btn1, btn2 = st.columns(2)
        with btn1:
            detect_clicked = st.button("Detect", type="primary", use_container_width=True)
        with btn2:
            clear_clicked = st.button("Clear", use_container_width=True)

        if detect_clicked:
            if channel_data is not None:
                params = DetectionParams(
                    min_radius_px=min_radius,
                    max_radius_px=max_radius,
                    blur_sigma=blur_sigma,
                    sensitivity=sensitivity,
                    min_distance_px=min_distance,
                    use_clahe=use_clahe,
                    circularity_min=circularity,
                )
                with st.spinner("Detecting GUVs..."):
                    guvs = detect_guvs(channel_data[det_channel], params)
                    measurements = []
                    for guv in guvs:
                        fl = measure_all_channels(channel_data, guv, membrane_width)
                        diameter_um = None
                        if sc is not None:
                            diameter_um = guv.diameter_px * sc
                        measurements.append(GUVMeasurement(
                            guv=guv, diameter_um=diameter_um, fluorescence=fl,
                        ))
                    colors = meta.channel_colors
                    df = build_dataframe(measurements, sc, colors)

                h_img, w_img = channel_data.shape[1], channel_data.shape[2]
                overlap_groups = find_overlaps(guvs, padding_px=20,
                                               img_width=w_img, img_height=h_img,
                                               overlap_threshold=overlap_threshold)

                st.session_state.guvs = guvs
                st.session_state.measurements = measurements
                st.session_state.results_df = df
                st.session_state.highlight_id = None
                st.session_state.overlap_groups = overlap_groups
                st.session_state.excluded_ids = set()
                st.session_state.selected_group_idx = 0
                st.rerun()

        if clear_clicked:
            st.session_state.guvs = []
            st.session_state.measurements = []
            st.session_state.results_df = None
            st.session_state.highlight_id = None
            st.session_state.overlap_groups = []
            st.session_state.excluded_ids = set()
            st.session_state.selected_group_idx = 0
            st.rerun()

        st.divider()
        st.link_button("Send Feedback", _FB_CONFIG["google_form_url"],
                       use_container_width=True)

    # ── LEFT COLUMN: Image + tabs ────────────────────────────────────────
    with main_col:
        # Display channel selector — above image
        display_options = ch_names + (["Composite"] if meta.num_channels > 1 else [])
        display_choice = st.radio("Display", display_options, horizontal=True,
                                  label_visibility="collapsed")

        if display_choice == "Composite":
            display_mode = "Composite"
            selected_channel = 0
        else:
            display_mode = "Selected Channel"
            selected_channel = display_options.index(display_choice)

        # Render image
        if display_mode == "Composite":
            rgb = render_composite(channel_data, meta.channel_colors, meta.display_ranges)
        else:
            ch_idx = min(selected_channel, channel_data.shape[0] - 1)
            color = meta.channel_colors[ch_idx] if ch_idx < len(meta.channel_colors) else "gray"
            lo, hi = meta.display_ranges[ch_idx] if ch_idx < len(meta.display_ranges) else (0, 255)
            rgb = render_channel(channel_data[ch_idx], color, lo, hi)

        # Draw GUV overlays
        guvs = st.session_state.guvs
        highlight_id = st.session_state.highlight_id
        excluded_ids: set[int] = st.session_state.excluded_ids
        if guvs:
            rgb = draw_guv_overlay(rgb, guvs, highlight_id, excluded_ids)

        # ── Image + Overlap side-by-side ─────────────────────────────
        overlap_groups = st.session_state.overlap_groups
        df: pd.DataFrame | None = st.session_state.results_df

        if overlap_groups:
            overlap_col, img_col = st.columns([1, 2], gap="medium")
        else:
            img_col = st.container()
            overlap_col = None

        with img_col:
            st.image(rgb, use_container_width=True)
            if guvs:
                n_excluded = len(excluded_ids)
                n_overlaps = len(overlap_groups)
                summary = f"**{len(guvs)} GUV(s) detected**"
                if n_overlaps:
                    summary += f" · {n_overlaps} overlap group(s)"
                if n_excluded:
                    summary += f" · {n_excluded} excluded"
                st.caption(summary)

        # ── Overlap panel (beside image) ─────────────────────────────
        if overlap_groups and overlap_col is not None:
            with overlap_col:
                all_overlap_ids = {gid for grp in overlap_groups for gid in grp.guv_ids}
                st.warning(
                    f"{len(overlap_groups)} overlap group(s) — "
                    f"{len(all_overlap_ids)} GUVs involved"
                )

                group_names = [f"Group {g.group_id} — GUVs {g.guv_ids}" for g in overlap_groups]
                selected_group_idx = st.selectbox(
                    "Select group", range(len(group_names)),
                    format_func=lambda i: group_names[i],
                    key="overlap_group_select",
                )
                st.session_state.selected_group_idx = selected_group_idx

                group = overlap_groups[selected_group_idx]
                x_min, y_min, x_max, y_max = group.bbox
                crop = rgb[y_min:y_max, x_min:x_max].copy()
                if crop.size > 0:
                    st.image(crop, caption=f"Group {group.group_id} magnified",
                             use_container_width=True)

                guv_map = {g.id: g for g in guvs}
                for gid in group.guv_ids:
                    guv = guv_map.get(gid)
                    if guv is None:
                        continue
                    if meta.scale_um_per_px is not None:
                        d_um = guv.diameter_px * meta.scale_um_per_px
                        label = f"GUV #{gid}  (d={d_um:.1f} \u00b5m)"
                    else:
                        label = f"GUV #{gid}  (d={guv.diameter_px:.0f} px)"
                    included = st.checkbox(
                        label,
                        value=(gid not in excluded_ids),
                        key=f"overlap_cb_{gid}",
                    )
                    if included and gid in excluded_ids:
                        st.session_state.excluded_ids.discard(gid)
                        st.rerun()
                    elif not included and gid not in excluded_ids:
                        st.session_state.excluded_ids.add(gid)
                        st.rerun()

                if st.button("Exclude All Overlapping", type="primary"):
                    st.session_state.excluded_ids |= all_overlap_ids
                    st.rerun()

        # ── Results table (below image) ──────────────────────────────
        if df is not None and not df.empty:
            st.divider()
            st.subheader(f"Results — {len(df)} GUV(s) detected")

            if not meta.quantitative:
                st.warning(
                    "This image format does not preserve quantitative fluorescence values. "
                    "Fluorescence measurements are relative only."
                )

            event = st.dataframe(
                df,
                on_select="rerun",
                selection_mode="single-row",
                use_container_width=True,
            )

            if event and event.selection and event.selection.rows:
                row_idx = event.selection.rows[0]
                selected_id = int(df.iloc[row_idx]["ID"])
                if selected_id != st.session_state.highlight_id:
                    st.session_state.highlight_id = selected_id
                    st.rerun()

            for m in st.session_state.measurements:
                m.guv.excluded = (m.guv.id in st.session_state.excluded_ids)
            active_measurements = filter_active_measurements(st.session_state.measurements)
            export_scale = meta.scale_um_per_px if meta else None
            export_colors = meta.channel_colors if meta else None
            export_df = build_dataframe(active_measurements, export_scale, export_colors)

            dl1, dl2 = st.columns(2)
            with dl1:
                csv_data = export_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv_data,
                    file_name="guv_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl2:
                buf = io.BytesIO()
                export_df.to_excel(buf, index=False, engine="openpyxl")
                st.download_button(
                    "Download Excel",
                    data=buf.getvalue(),
                    file_name="guv_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
