"""Application controller — wires model and view together."""

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import QFileDialog, QMessageBox

import numpy as np

import cv2

from ..model.image_loader import load_image, get_lif_series_list, ImageMetadata
from ..model.guv_detector import detect_guvs, DetectionParams, DetectedGUV, find_overlaps, OverlapGroup
from ..model.fluorescence import measure_all_channels
from ..model.guv_data import GUVMeasurement, build_dataframe, export_csv, export_excel, filter_active_measurements
from ..view.main_window import MainWindow
from ..view.feedback_dialog import FeedbackDialog
from ..feedback import open_feedback
from .. import __version__


class DetectionWorker(QObject):
    """Run detection in a background thread."""
    finished = pyqtSignal(list)  # list[DetectedGUV]
    error = pyqtSignal(str)

    def __init__(self, channel: np.ndarray, params: DetectionParams):
        super().__init__()
        self._channel = channel
        self._params = params

    def run(self):
        try:
            guvs = detect_guvs(self._channel, self._params)
            self.finished.emit(guvs)
        except Exception as e:
            self.error.emit(str(e))


class AppController:
    """Coordinates between model and view."""

    def __init__(self, window: MainWindow):
        self._window = window
        self._channel_data: np.ndarray | None = None
        self._metadata: ImageMetadata | None = None
        self._guvs: list[DetectedGUV] = []
        self._measurements: list[GUVMeasurement] = []
        self._overlap_groups: list[OverlapGroup] = []
        self._thread: QThread | None = None

        self._connect_signals()

    def _connect_signals(self):
        w = self._window
        w.open_action.triggered.connect(self._open_file)
        w.quit_action.triggered.connect(w.close)
        w.fit_view_action.triggered.connect(w.canvas.fit_in_view)
        w.zoom_in_action.triggered.connect(lambda: w.canvas.scale(1.15, 1.15))
        w.zoom_out_action.triggered.connect(lambda: w.canvas.scale(1/1.15, 1/1.15))

        w.detection_panel.detect_requested.connect(self._run_detection)
        w.detection_panel.clear_requested.connect(self._clear_detection)
        w.detection_panel.display_combo.currentIndexChanged.connect(self._refresh_display)
        w.detection_panel.channel_combo.currentIndexChanged.connect(self._refresh_display)

        w.results_panel.row_selected.connect(self._on_row_selected)
        w.results_panel.export_csv_requested.connect(self._export_csv)
        w.results_panel.export_excel_requested.connect(self._export_excel)

        w.canvas.guv_clicked.connect(self._on_guv_clicked)

        w.overlap_panel.guv_exclude_toggled.connect(self._on_guv_exclude_toggled)
        w.overlap_panel.exclude_all_overlapping.connect(self._on_nuke_overlapping)
        w.overlap_panel.group_selected.connect(self._on_group_selected)

        w.file_dropped.connect(self._load_file)

        w.feedback_action.triggered.connect(self._show_feedback)
        w.about_action.triggered.connect(self._show_about)

    def _open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self._window,
            "Open Microscopy Image",
            "",
            "Images (*.tif *.tiff *.lif *.jpg *.jpeg *.png *.bmp);;Leica LIF (*.lif);;All Files (*)",
        )
        if filepath:
            self._load_file(filepath)

    def _load_file(self, filepath: str):
        """Load an image file (from dialog or drag-and-drop)."""
        from pathlib import Path

        series_index = 0
        # For .lif files, show series selection if multiple series exist
        if Path(filepath).suffix.lower() == ".lif":
            try:
                series_list = get_lif_series_list(filepath)
            except Exception as e:
                QMessageBox.critical(
                    self._window, "Error", f"Failed to read .lif file:\n{e}"
                )
                return
            if len(series_list) > 1:
                from ..view.series_dialog import SeriesDialog

                dlg = SeriesDialog(series_list, self._window)
                if dlg.exec() != dlg.DialogCode.Accepted:
                    return
                series_index = dlg.selected_index

        try:
            self._channel_data, self._metadata = load_image(
                filepath, series_index=series_index
            )
        except Exception as e:
            QMessageBox.critical(self._window, "Error", f"Failed to load image:\n{e}")
            return

        self._guvs = []
        self._measurements = []

        # Update metadata panel
        self._window.metadata_panel.update_metadata(self._metadata)
        self._window.results_panel.clear()

        # Set up channel dropdown
        ch_names = []
        for i in range(self._metadata.num_channels):
            color = self._metadata.channel_colors[i] if i < len(self._metadata.channel_colors) else "?"
            ch_names.append(f"Ch{i+1} ({color})")
        self._window.detection_panel.set_channels(ch_names)

        # Set scale-aware radius defaults if scale info is available
        # Typical GUV diameter: 5–50 µm → radius 2.5–25 µm
        if self._metadata.scale_um_per_px is not None:
            scale = self._metadata.scale_um_per_px
            min_r = max(3, int(2.5 / scale))   # 5 µm diameter → 2.5 µm radius
            max_r = max(min_r + 1, int(25.0 / scale))  # 50 µm diameter → 25 µm radius
            self._window.detection_panel.set_radius_defaults(min_r, max_r)

        # Display image
        self._refresh_display()
        self._window.canvas.fit_in_view()

        self._window.status_bar.showMessage(
            f"Loaded: {self._metadata.filepath.name} — "
            f"{self._metadata.num_channels} channel(s), "
            f"{self._metadata.width}×{self._metadata.height}"
        )

    def _refresh_display(self):
        """Update the canvas display based on current display mode and channel."""
        if self._channel_data is None or self._metadata is None:
            return

        dp = self._window.detection_panel
        mode = dp.display_mode
        ch_idx = dp.selected_channel

        if mode == "Composite":
            self._window.canvas.display_composite(
                self._channel_data,
                self._metadata.channel_colors,
                self._metadata.display_ranges,
            )
        else:
            ch_idx = min(ch_idx, self._channel_data.shape[0] - 1)
            color = self._metadata.channel_colors[ch_idx] if ch_idx < len(self._metadata.channel_colors) else "gray"
            lo, hi = self._metadata.display_ranges[ch_idx] if ch_idx < len(self._metadata.display_ranges) else (0, 255)
            self._window.canvas.display_channel(
                self._channel_data[ch_idx], color, lo, hi
            )

        # Redraw GUV overlays
        if self._guvs:
            self._window.canvas.draw_guvs(self._guvs, excluded_ids=self._excluded_ids())

    def _run_detection(self):
        """Run GUV detection on the selected channel."""
        if self._channel_data is None:
            QMessageBox.warning(self._window, "No Image", "Please open an image first.")
            return

        dp = self._window.detection_panel
        ch_idx = min(dp.selected_channel, self._channel_data.shape[0] - 1)

        params = DetectionParams(
            min_radius_px=dp.min_radius,
            max_radius_px=dp.max_radius,
            blur_sigma=dp.blur_sigma,
            sensitivity=dp.sensitivity,
            min_distance_px=dp.min_distance,
            use_clahe=dp.use_clahe,
            circularity_min=dp.circularity,
        )

        self._window.status_bar.showMessage("Detecting GUVs...")

        # Run in background thread
        self._thread = QThread()
        self._worker = DetectionWorker(self._channel_data[ch_idx], params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_detection_done)
        self._worker.error.connect(self._on_detection_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_detection_done(self, guvs: list[DetectedGUV]):
        """Handle detection results."""
        self._guvs = guvs
        self._window.canvas.draw_guvs(guvs)

        # Measure fluorescence
        self._measurements = []
        membrane_width = self._window.detection_panel.membrane_width
        for guv in guvs:
            fl_results = measure_all_channels(
                self._channel_data, guv, membrane_width
            )
            diameter_um = None
            if self._metadata and self._metadata.scale_um_per_px is not None:
                diameter_um = guv.diameter_px * self._metadata.scale_um_per_px

            self._measurements.append(GUVMeasurement(
                guv=guv,
                diameter_um=diameter_um,
                fluorescence=fl_results,
            ))

        # Build and display results
        self._refresh_results_table()

        # Overlap detection
        h, w = self._channel_data.shape[1], self._channel_data.shape[2]
        threshold = self._window.detection_panel.overlap_threshold
        self._overlap_groups = find_overlaps(guvs, padding_px=20,
                                             img_width=w, img_height=h,
                                             overlap_threshold=threshold)
        if self._overlap_groups:
            scale = self._metadata.scale_um_per_px if self._metadata else None
            self._window.overlap_panel.set_groups(self._overlap_groups, guvs,
                                                   scale_um_per_px=scale)
            # Render first group crop
            self._on_group_selected(0)
        else:
            self._window.overlap_panel.clear()

        self._window.status_bar.showMessage(
            f"{len(guvs)} GUV(s) detected"
            + (f" — {len(self._overlap_groups)} overlap group(s)" if self._overlap_groups else "")
        )

    def _on_detection_error(self, msg: str):
        QMessageBox.critical(self._window, "Detection Error", msg)
        self._window.status_bar.showMessage("Detection failed")

    def _clear_detection(self):
        self._guvs = []
        self._measurements = []
        self._overlap_groups = []
        self._window.canvas.clear_guvs()
        self._window.results_panel.clear()
        self._window.overlap_panel.clear()
        self._window.status_bar.showMessage("Detection cleared")

    def _excluded_ids(self) -> set[int]:
        return {g.id for g in self._guvs if g.excluded}

    @property
    def _guv_map(self) -> dict[int, DetectedGUV]:
        return {g.id: g for g in self._guvs}

    def _refresh_after_exclusion(self):
        """Refresh canvas, results, and overlap panel after exclusion changes."""
        self._refresh_canvas()
        self._refresh_results_table()
        self._window.overlap_panel.refresh_checkboxes()
        if self._overlap_groups:
            self._on_group_selected(self._window.overlap_panel._current_idx)

    def _refresh_results_table(self):
        """Rebuild the results table with current exclusion state."""
        scale = self._metadata.scale_um_per_px if self._metadata else None
        colors = self._metadata.channel_colors if self._metadata else None
        df = build_dataframe(self._measurements, scale, colors)
        quantitative = self._metadata.quantitative if self._metadata else False
        self._window.results_panel.update_results(df, quantitative, self._excluded_ids())

    def _refresh_canvas(self, highlight_id: int | None = None):
        """Redraw GUV overlays with current exclusion state."""
        self._window.canvas.draw_guvs(self._guvs, highlight_id=highlight_id,
                                       excluded_ids=self._excluded_ids())

    def _on_row_selected(self, guv_id: int):
        """Highlight the clicked GUV on the canvas."""
        self._refresh_canvas(highlight_id=guv_id)

    def _on_guv_clicked(self, guv_id: int):
        """Handle click on a GUV circle.

        If overlap groups are visible, clicking toggles excluded state.
        Otherwise, it just highlights the GUV.
        """
        if self._overlap_groups:
            overlapping_ids = {gid for grp in self._overlap_groups for gid in grp.guv_ids}
            if guv_id in overlapping_ids:
                guv = self._guv_map.get(guv_id)
                if guv:
                    guv.excluded = not guv.excluded
                    self._refresh_after_exclusion()
                    return

        self._window.results_panel.highlight_row(guv_id)
        self._refresh_canvas(highlight_id=guv_id)

    def _on_group_selected(self, idx: int):
        """Show magnified crop for the selected overlap group."""
        if idx < 0 or idx >= len(self._overlap_groups):
            return
        group = self._overlap_groups[idx]

        # Get the crop from the canvas
        crop = self._window.canvas.get_region_rgb(group.bbox)
        if crop is not None:
            # Draw GUV circles on the crop for context
            x_min, y_min, _, _ = group.bbox
            guv_map = self._guv_map
            for gid in group.guv_ids:
                guv = guv_map.get(gid)
                if guv is None:
                    continue
                cx = int(round(guv.center_x - x_min))
                cy = int(round(guv.center_y - y_min))
                r = int(round(guv.radius))
                if guv.excluded:
                    color = (255, 80, 80)
                else:
                    color = (255, 255, 0)
                cv2.circle(crop, (cx, cy), r, color, 2)
                cv2.putText(crop, str(gid), (cx + r + 3, cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            self._window.overlap_panel.set_magnified_image(crop)

        # Redraw canvas with viewport rectangle for the current group
        self._refresh_canvas()
        self._window.canvas.draw_viewport_rect(group.bbox)

    def _on_guv_exclude_toggled(self, guv_id: int, included: bool):
        """Toggle excluded state for a single GUV."""
        guv = self._guv_map.get(guv_id)
        if guv:
            guv.excluded = not included
        self._refresh_after_exclusion()

    def _on_nuke_overlapping(self):
        """Exclude all GUVs that are in any overlap group."""
        overlapping_ids = {gid for grp in self._overlap_groups for gid in grp.guv_ids}
        for guv in self._guvs:
            if guv.id in overlapping_ids:
                guv.excluded = True
        self._refresh_after_exclusion()
        self._window.status_bar.showMessage(
            f"Excluded {len(overlapping_ids)} overlapping GUV(s)"
        )

    def _show_feedback(self):
        image_format = None
        if self._metadata and self._metadata.filepath:
            image_format = self._metadata.filepath.suffix
        dlg = FeedbackDialog(image_format=image_format, parent=self._window)
        if dlg.exec() == dlg.DialogCode.Accepted and dlg.description.strip():
            open_feedback(dlg.category, dlg.description, image_format)
            self._window.status_bar.showMessage("Feedback page opened in browser")

    def _show_about(self):
        QMessageBox.about(
            self._window,
            "About GUV Analyzer",
            f"<b>GUV Analyzer</b> v{__version__}<br><br>"
            "Automated detection and fluorescence analysis of "
            "Giant Unilamellar Vesicles from confocal microscopy images.",
        )

    def _export(self, file_filter: str, export_fn):
        """Shared export logic for CSV and Excel."""
        if not self._measurements:
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self._window, "Export", "", file_filter
        )
        if filepath:
            scale = self._metadata.scale_um_per_px if self._metadata else None
            colors = self._metadata.channel_colors if self._metadata else None
            active = filter_active_measurements(self._measurements)
            df = build_dataframe(active, scale, colors)
            export_fn(df, filepath)
            self._window.status_bar.showMessage(f"Exported to {filepath}")

    def _export_csv(self):
        self._export("CSV Files (*.csv)", export_csv)

    def _export_excel(self):
        self._export("Excel Files (*.xlsx)", export_excel)
