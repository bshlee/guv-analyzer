"""Panel with detection parameter controls (Step 2)."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QSlider, QSpinBox, QLabel, QPushButton,
    QHBoxLayout, QCheckBox, QDoubleSpinBox, QScrollArea,
)
from PyQt6.QtCore import Qt


class DetectionPanel(QWidget):
    """Controls for GUV detection parameters."""

    detect_requested = pyqtSignal()
    clear_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Detection Controls")
        form = QFormLayout()

        # Channel selector
        self._channel_combo = QComboBox()
        form.addRow("Detection channel:", self._channel_combo)

        # Display mode
        self._display_combo = QComboBox()
        self._display_combo.addItems(["Selected Channel", "Composite"])
        form.addRow("Display mode:", self._display_combo)

        # Min radius
        self._min_radius = QSpinBox()
        self._min_radius.setRange(3, 2000)
        self._min_radius.setValue(8)
        self._min_radius.setSuffix(" px")
        form.addRow("Min radius:", self._min_radius)

        # Max radius
        self._max_radius = QSpinBox()
        self._max_radius.setRange(10, 5000)
        self._max_radius.setValue(500)
        self._max_radius.setSuffix(" px")
        form.addRow("Max radius:", self._max_radius)

        # Sensitivity
        self._sensitivity = QDoubleSpinBox()
        self._sensitivity.setRange(0.1, 1.0)
        self._sensitivity.setSingleStep(0.05)
        self._sensitivity.setValue(0.85)
        form.addRow("Sensitivity:", self._sensitivity)

        # Min distance
        self._min_distance = QSpinBox()
        self._min_distance.setRange(5, 1000)
        self._min_distance.setValue(15)
        self._min_distance.setSuffix(" px")
        form.addRow("Min distance:", self._min_distance)

        # Blur sigma
        self._blur_sigma = QDoubleSpinBox()
        self._blur_sigma.setRange(0.5, 20.0)
        self._blur_sigma.setSingleStep(0.5)
        self._blur_sigma.setValue(2.0)
        form.addRow("Blur sigma:", self._blur_sigma)

        # Membrane width
        self._membrane_width = QDoubleSpinBox()
        self._membrane_width.setRange(1.0, 30.0)
        self._membrane_width.setSingleStep(1.0)
        self._membrane_width.setValue(4.0)
        self._membrane_width.setSuffix(" px")
        form.addRow("Membrane width:", self._membrane_width)

        # Circularity threshold
        self._circularity = QDoubleSpinBox()
        self._circularity.setRange(0.3, 1.0)
        self._circularity.setSingleStep(0.05)
        self._circularity.setValue(0.65)
        form.addRow("Min circularity:", self._circularity)

        # CLAHE toggle
        self._clahe_check = QCheckBox("Use CLAHE (enhance contrast)")
        self._clahe_check.setChecked(True)
        form.addRow(self._clahe_check)

        # Overlap threshold (fraction of smaller GUV area)
        # TODO: move to a dedicated "Advanced Settings" dialog in a future release
        self._overlap_threshold = QDoubleSpinBox()
        self._overlap_threshold.setRange(0.0, 1.0)
        self._overlap_threshold.setSingleStep(0.05)
        self._overlap_threshold.setValue(0.30)
        self._overlap_threshold.setToolTip(
            "Minimum overlap area (as fraction of the smaller GUV) to flag a pair.\n"
            "0.0 = flag any touching, 1.0 = only fully contained."
        )
        form.addRow("Overlap threshold:", self._overlap_threshold)

        group.setLayout(form)

        # Wrap in scroll area so the panel doesn't overflow
        scroll = QScrollArea()
        scroll.setWidget(group)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        layout.addWidget(scroll)

        # Buttons
        btn_layout = QHBoxLayout()
        self._detect_btn = QPushButton("Detect GUVs")
        self._detect_btn.setStyleSheet("font-weight: bold;")
        self._clear_btn = QPushButton("Clear")
        btn_layout.addWidget(self._detect_btn)
        btn_layout.addWidget(self._clear_btn)
        layout.addLayout(btn_layout)

        self._detect_btn.clicked.connect(self.detect_requested)
        self._clear_btn.clicked.connect(self.clear_requested)

    def set_channels(self, channel_names: list[str]):
        """Populate channel dropdown."""
        self._channel_combo.clear()
        for name in channel_names:
            self._channel_combo.addItem(name)

    def set_radius_defaults(self, min_radius: int, max_radius: int):
        """Set min/max radius values (e.g. from scale-based GUV size estimates)."""
        self._min_radius.setValue(min_radius)
        self._max_radius.setValue(max_radius)

    @property
    def selected_channel(self) -> int:
        return max(0, self._channel_combo.currentIndex())

    @property
    def display_mode(self) -> str:
        return self._display_combo.currentText()

    @property
    def min_radius(self) -> int:
        return self._min_radius.value()

    @property
    def max_radius(self) -> int:
        return self._max_radius.value()

    @property
    def sensitivity(self) -> float:
        return self._sensitivity.value()

    @property
    def min_distance(self) -> int:
        return self._min_distance.value()

    @property
    def blur_sigma(self) -> float:
        return self._blur_sigma.value()

    @property
    def membrane_width(self) -> float:
        return self._membrane_width.value()

    @property
    def circularity(self) -> float:
        return self._circularity.value()

    @property
    def use_clahe(self) -> bool:
        return self._clahe_check.isChecked()

    @property
    def overlap_threshold(self) -> float:
        return self._overlap_threshold.value()

    @property
    def display_combo(self) -> QComboBox:
        return self._display_combo

    @property
    def channel_combo(self) -> QComboBox:
        return self._channel_combo
