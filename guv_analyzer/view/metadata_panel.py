"""Panel displaying image metadata (Step 1 info)."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QGroupBox,
)

from ..model.image_loader import ImageMetadata


class MetadataPanel(QWidget):
    """Displays image specs: scale, channels, display ranges, laser power."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Image Info")
        form = QFormLayout()

        self._file_label = QLabel("—")
        self._file_label.setWordWrap(True)
        self._size_label = QLabel("—")
        self._channels_label = QLabel("—")
        self._scale_label = QLabel("—")
        self._ranges_label = QLabel("—")
        self._ranges_label.setWordWrap(True)
        self._laser_input = QLineEdit()
        self._laser_input.setPlaceholderText("Optional (not in TIF)")

        form.addRow("File:", self._file_label)
        form.addRow("Size:", self._size_label)
        form.addRow("Channels:", self._channels_label)
        form.addRow("Scale:", self._scale_label)
        form.addRow("Display ranges:", self._ranges_label)
        form.addRow("Laser power:", self._laser_input)

        group.setLayout(form)
        layout.addWidget(group)

    def update_metadata(self, meta: ImageMetadata):
        """Populate fields from metadata."""
        self._file_label.setText(meta.filepath.name)
        self._size_label.setText(f"{meta.width} × {meta.height} px, {meta.dtype}")

        ch_parts = []
        for i in range(meta.num_channels):
            color = meta.channel_colors[i] if i < len(meta.channel_colors) else "?"
            ch_parts.append(f"Ch{i+1} ({color})")
        self._channels_label.setText(f"{meta.num_channels}: " + ", ".join(ch_parts))

        if meta.scale_um_per_px is not None:
            self._scale_label.setText(f"{meta.scale_um_per_px:.4f} µm/px")
        else:
            self._scale_label.setText("Unknown (JPG/PNG)")

        range_parts = []
        for i, (lo, hi) in enumerate(meta.display_ranges):
            range_parts.append(f"Ch{i+1}: {lo:.0f}–{hi:.0f}")
        self._ranges_label.setText("; ".join(range_parts) if range_parts else "—")

        if meta.laser_power:
            self._laser_input.setText(meta.laser_power)

    def get_laser_power(self) -> str:
        return self._laser_input.text().strip()

    def clear(self):
        self._file_label.setText("—")
        self._size_label.setText("—")
        self._channels_label.setText("—")
        self._scale_label.setText("—")
        self._ranges_label.setText("—")
        self._laser_input.clear()
