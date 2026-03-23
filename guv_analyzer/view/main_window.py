"""Main application window layout."""

import logging
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

from PyQt6.QtCore import Qt, pyqtSignal, QByteArray
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QMenuBar, QStatusBar,
)
from PyQt6.QtGui import QAction, QDragEnterEvent, QDropEvent

from .. import __version__
from .image_canvas import ImageCanvas
from .metadata_panel import MetadataPanel
from .detection_panel import DetectionPanel
from .results_panel import ResultsPanel
from .overlap_panel import OverlapPanel

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".lif"}


class MainWindow(QMainWindow):
    """Main application window with image canvas and side panels."""

    file_dropped = pyqtSignal(str)  # Emits filepath when a file is dropped

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"GUV Analyzer v{__version__}")
        self.setMinimumSize(1200, 800)

        # Enable drag and drop
        self.setAcceptDrops(True)

        # Menu bar
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        self.open_action = QAction("Open Image...", self)
        self.open_action.setShortcut("Ctrl+O")
        file_menu.addAction(self.open_action)

        file_menu.addSeparator()

        self.quit_action = QAction("Quit", self)
        self.quit_action.setShortcut("Ctrl+Q")
        file_menu.addAction(self.quit_action)

        view_menu = menu_bar.addMenu("View")
        self.fit_view_action = QAction("Fit to Window", self)
        self.fit_view_action.setShortcut("Ctrl+0")
        view_menu.addAction(self.fit_view_action)

        self.zoom_in_action = QAction("Zoom In", self)
        self.zoom_in_action.setShortcut("Ctrl+=")
        view_menu.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction("Zoom Out", self)
        self.zoom_out_action.setShortcut("Ctrl+-")
        view_menu.addAction(self.zoom_out_action)

        help_menu = menu_bar.addMenu("Help")
        self.feedback_action = QAction("Send Feedback...", self)
        help_menu.addAction(self.feedback_action)
        help_menu.addSeparator()
        self.about_action = QAction("About GUV Analyzer", self)
        help_menu.addAction(self.about_action)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # 4-panel splitter: overlap | canvas | controls | results
        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        # [0] Overlap panel (always visible)
        self.overlap_panel = OverlapPanel()
        self._splitter.addWidget(self.overlap_panel)

        # [1] Image canvas
        self.canvas = ImageCanvas()
        self.canvas.setMinimumWidth(200)
        self._splitter.addWidget(self.canvas)

        # [2] Control panel (metadata + detection)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(0, 0, 0, 0)

        self.metadata_panel = MetadataPanel()
        self.detection_panel = DetectionPanel()

        control_layout.addWidget(self.metadata_panel)
        control_layout.addWidget(self.detection_panel)
        control_layout.addStretch()

        self._splitter.addWidget(control_panel)

        # [3] Results panel (always visible, far right)
        self.results_panel = ResultsPanel()
        self._splitter.addWidget(self.results_panel)

        # Ratio 3:5:2:3
        self._splitter.setSizes([300, 500, 200, 300])

        main_layout.addWidget(self._splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — Open or drag & drop an image to begin")

    @staticmethod
    def _has_supported_ext(path: str) -> bool:
        return any(path.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)

    def _extract_filepath(self, event) -> str | None:
        """Extract a supported file path from a drag/drop event.

        Tries multiple strategies to handle macOS quirks with unrecognized
        file types (.lif) and cloud-storage paths (OneDrive).
        """
        mime = event.mimeData()

        # Debug: log all available MIME formats
        formats = mime.formats()
        logger.debug("Drag MIME formats: %s", formats)
        if mime.hasUrls():
            logger.debug("URLs: %s", [u.toString() for u in mime.urls()])
        if mime.hasText():
            logger.debug("Text: %s", mime.text()[:300])

        # Strategy 1: standard URL approach
        if mime.hasUrls():
            for url in mime.urls():
                path = url.toLocalFile()
                if path and self._has_supported_ext(path):
                    # Resolve symlinks / OneDrive aliases
                    resolved = os.path.realpath(path)
                    if os.path.isfile(resolved):
                        return resolved
                    if os.path.isfile(path):
                        return path

        # Strategy 2: text/uri-list or text/plain (macOS fallback)
        if mime.hasText():
            for line in mime.text().strip().splitlines():
                path = line.strip()
                if path.startswith("file://"):
                    path = unquote(urlparse(path).path)
                if self._has_supported_ext(path):
                    resolved = os.path.realpath(path)
                    if os.path.isfile(resolved):
                        return resolved
                    if os.path.isfile(path):
                        return path

        # Strategy 3: raw bytes from known macOS pasteboard types
        # macOS Finder may provide paths via these non-standard formats
        for fmt in formats:
            if "filename" in fmt.lower() or "file" in fmt.lower():
                raw = mime.data(fmt)
                if raw and not raw.isEmpty():
                    try:
                        text = bytes(raw).decode("utf-8", errors="replace")
                        # May be null-terminated or newline-separated
                        for part in text.replace("\x00", "\n").splitlines():
                            part = part.strip()
                            if part.startswith("file://"):
                                part = unquote(urlparse(part).path)
                            if part and self._has_supported_ext(part):
                                resolved = os.path.realpath(part)
                                if os.path.isfile(resolved):
                                    return resolved
                                if os.path.isfile(part):
                                    return part
                    except Exception:
                        continue

        logger.debug("No supported file found in drag data")
        return None

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accept ALL proposed drops unconditionally.

        Validation happens only in dropEvent. This ensures the drop cursor
        always appears, working around macOS and Windows MIME quirks.
        """
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Handle dropped file — try every extraction method."""
        path = self._extract_filepath(event)
        if path:
            self.file_dropped.emit(path)
            event.acceptProposedAction()
        else:
            formats = event.mimeData().formats()
            logger.warning("Drop failed — formats: %s", formats)
            self.status_bar.showMessage(
                "Could not read dropped file. Try File \u2192 Open instead."
            )
            event.ignore()
