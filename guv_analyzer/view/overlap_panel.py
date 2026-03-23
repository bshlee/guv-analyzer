"""Overlap warning panel — shows overlapping GUV groups with exclusion controls."""

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap, QCursor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QCheckBox, QScrollArea, QGroupBox,
)


class _GUVToggleRow(QWidget):
    """A single GUV row: checkbox + clickable label that toggles the checkbox."""

    toggled = pyqtSignal(int, bool)  # guv_id, included

    def __init__(self, guv_id: int, label_text: str, included: bool, parent=None):
        super().__init__(parent)
        self._guv_id = guv_id
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)

        self._cb = QCheckBox()
        self._cb.setChecked(included)
        self._cb.stateChanged.connect(self._on_state)
        layout.addWidget(self._cb)

        self._label = QLabel(label_text)
        self._label.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._label.setStyleSheet("padding: 2px;")
        self._label.mousePressEvent = self._on_label_click
        layout.addWidget(self._label, stretch=1)

    def _on_label_click(self, _event):
        self._cb.setChecked(not self._cb.isChecked())

    def _on_state(self, state: int):
        included = (state == Qt.CheckState.Checked.value)
        self._update_style(included)
        self.toggled.emit(self._guv_id, included)

    def _update_style(self, included: bool):
        if included:
            self._label.setStyleSheet("padding: 2px;")
        else:
            self._label.setStyleSheet(
                "padding: 2px; color: #999; text-decoration: line-through;"
            )

    def set_checked(self, checked: bool):
        self._cb.blockSignals(True)
        self._cb.setChecked(checked)
        self._update_style(checked)
        self._cb.blockSignals(False)

    @property
    def checkbox(self) -> QCheckBox:
        return self._cb

import numpy as np

from ..model.guv_detector import DetectedGUV, OverlapGroup


class OverlapPanel(QWidget):
    """Panel displaying overlap warnings with magnified crops and exclusion controls."""

    guv_exclude_toggled = pyqtSignal(int, bool)  # GUV ID, new included state
    exclude_all_overlapping = pyqtSignal()
    group_selected = pyqtSignal(int)  # group index

    def __init__(self, parent=None):
        super().__init__(parent)
        self._groups: list[OverlapGroup] = []
        self._guvs: list[DetectedGUV] = []
        self._guv_map: dict[int, DetectedGUV] = {}
        self._scale_um_per_px: float | None = None
        self._current_idx = 0
        self._toggle_rows: dict[int, _GUVToggleRow] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        group_box = QGroupBox("Overlap Warning")
        group_layout = QVBoxLayout()

        # Header
        self._header = QLabel("No overlaps found!")
        self._header.setStyleSheet("color: #e6a800; font-weight: bold;")
        self._header.setWordWrap(True)
        group_layout.addWidget(self._header)

        # Magnified crop
        self._crop_label = QLabel()
        self._crop_label.setFixedHeight(250)
        self._crop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._crop_label.setStyleSheet("background: #1a1a1a; border: 1px solid #444;")
        group_layout.addWidget(self._crop_label)

        # Navigation
        nav_layout = QHBoxLayout()
        self._prev_btn = QPushButton("< Prev")
        self._prev_btn.clicked.connect(self._prev_group)
        self._group_counter = QLabel("0/0")
        self._group_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._next_btn = QPushButton("Next >")
        self._next_btn.clicked.connect(self._next_group)
        nav_layout.addWidget(self._prev_btn)
        nav_layout.addWidget(self._group_counter)
        nav_layout.addWidget(self._next_btn)
        group_layout.addLayout(nav_layout)

        # Checkbox area (scrollable)
        self._checkbox_area = QScrollArea()
        self._checkbox_area.setWidgetResizable(True)
        self._checkbox_widget = QWidget()
        self._checkbox_layout = QVBoxLayout(self._checkbox_widget)
        self._checkbox_layout.setContentsMargins(4, 4, 4, 4)
        self._checkbox_area.setWidget(self._checkbox_widget)
        self._checkbox_area.setMaximumHeight(150)
        group_layout.addWidget(self._checkbox_area)

        # Nuke button
        self._nuke_btn = QPushButton("Exclude All Overlapping")
        self._nuke_btn.setStyleSheet(
            "QPushButton { background-color: #cc3333; color: white; "
            "font-weight: bold; padding: 6px; }"
            "QPushButton:hover { background-color: #ff4444; }"
        )
        self._nuke_btn.clicked.connect(self.exclude_all_overlapping)
        group_layout.addWidget(self._nuke_btn)

        group_box.setLayout(group_layout)
        layout.addWidget(group_box)

    def set_groups(self, groups: list[OverlapGroup], guvs: list[DetectedGUV],
                   scale_um_per_px: float | None = None):
        """Populate the panel with overlap groups."""
        self._groups = groups
        self._guvs = guvs
        self._guv_map = {g.id: g for g in guvs}
        self._scale_um_per_px = scale_um_per_px
        self._current_idx = 0

        count = len(groups)
        total_guvs = len({gid for grp in groups for gid in grp.guv_ids})
        self._header.setText(
            f"\u26a0 {count} overlap group(s) found\n"
            f"({total_guvs} GUVs involved)"
        )
        self._update_navigation()
        if groups:
            self._show_group(0)

    def set_magnified_image(self, rgb: np.ndarray):
        """Update the crop display from an RGB numpy array."""
        h, w, _ = rgb.shape
        bytes_per_line = 3 * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        # Scale to fit label width while keeping aspect ratio
        scaled = pixmap.scaled(
            self._crop_label.width(), self._crop_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._crop_label.setPixmap(scaled)

    def clear(self):
        """Reset the panel."""
        self._groups = []
        self._guvs = []
        self._guv_map = {}
        self._scale_um_per_px = None
        self._current_idx = 0
        self._header.setText("No overlaps found!")
        self._crop_label.clear()
        self._group_counter.setText("0/0")
        self._clear_checkboxes()

    def _update_navigation(self):
        count = len(self._groups)
        self._prev_btn.setEnabled(self._current_idx > 0)
        self._next_btn.setEnabled(self._current_idx < count - 1)
        if count > 0:
            self._group_counter.setText(f"Group {self._current_idx + 1}/{count}")
        else:
            self._group_counter.setText("0/0")

    def _prev_group(self):
        if self._current_idx > 0:
            self._current_idx -= 1
            self._show_group(self._current_idx)

    def _next_group(self):
        if self._current_idx < len(self._groups) - 1:
            self._current_idx += 1
            self._show_group(self._current_idx)

    def _show_group(self, idx: int):
        """Display the group at the given index."""
        self._current_idx = idx
        self._update_navigation()
        self._populate_checkboxes()
        self.group_selected.emit(idx)

    def _clear_checkboxes(self):
        self._toggle_rows: dict[int, _GUVToggleRow] = {}
        while self._checkbox_layout.count():
            item = self._checkbox_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _guv_label(self, guv) -> str:
        """Build label like 'GUV #3  (d=10.4 µm)' or 'GUV #3  (d=90 px)'."""
        if self._scale_um_per_px is not None:
            d_um = guv.diameter_px * self._scale_um_per_px
            return f"GUV #{guv.id}  (d={d_um:.1f} \u00b5m)"
        return f"GUV #{guv.id}  (d={guv.diameter_px:.0f} px)"

    def _populate_checkboxes(self):
        self._clear_checkboxes()
        if not self._groups:
            return
        group = self._groups[self._current_idx]
        for gid in group.guv_ids:
            guv = self._guv_map.get(gid)
            if guv is None:
                continue
            row = _GUVToggleRow(gid, self._guv_label(guv), not guv.excluded)
            row.toggled.connect(self._on_toggle)
            self._checkbox_layout.addWidget(row)
            self._toggle_rows[gid] = row

    def _on_toggle(self, guv_id: int, included: bool):
        self.guv_exclude_toggled.emit(guv_id, included)

    def refresh_checkboxes(self):
        """Update checkbox states to match current GUV excluded flags."""
        for gid, row in self._toggle_rows.items():
            guv = self._guv_map.get(gid)
            if guv:
                row.set_checked(not guv.excluded)
