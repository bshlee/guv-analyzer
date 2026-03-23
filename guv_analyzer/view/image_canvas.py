"""QGraphicsView-based image canvas with zoom, pan, and circle overlays."""

from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import (
    QImage, QPixmap, QPen, QColor, QWheelEvent, QMouseEvent,
    QDragEnterEvent, QDropEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsRectItem,
)

import numpy as np

from ..model.guv_detector import DetectedGUV


class ImageCanvas(QGraphicsView):
    """Image display with zoom/pan and GUV circle overlays."""

    guv_clicked = pyqtSignal(int)  # Emits GUV ID when a circle is clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._guv_items: dict[int, QGraphicsEllipseItem] = {}
        self._label_items: dict[int, QGraphicsTextItem] = {}
        self._viewport_rect: QGraphicsRectItem | None = None
        self._zoom_factor = 1.0

        # Accept drops so files can be dragged onto the canvas
        self.setAcceptDrops(True)

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHints(
            self.renderHints()
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    # -- Drag-and-drop: forward to MainWindow ----------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accept all drops — let the main window validate on drop."""
        event.acceptProposedAction()

    def dragMoveEvent(self, event):
        """Keep accepting during drag-move so the cursor stays valid."""
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Forward drop to the parent MainWindow."""
        # Walk up to find MainWindow and delegate
        from .main_window import MainWindow
        parent = self.window()
        if isinstance(parent, MainWindow):
            parent.dropEvent(event)
        else:
            event.ignore()

    def display_channel(self, channel: np.ndarray, color: str = "gray",
                        display_min: float = 0, display_max: float = 255):
        """Display a single channel as a colored image."""
        h, w = channel.shape
        # Normalize to 0-255 range using display range
        img = channel.astype(np.float64)
        img = np.clip((img - display_min) / max(display_max - display_min, 1) * 255, 0, 255)
        img = img.astype(np.uint8)

        # Apply color
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        color_map = {
            "gray": (1, 1, 1),
            "red": (1, 0, 0),
            "green": (0, 1, 0),
            "blue": (0, 0, 1),
            "cyan": (0, 1, 1),
            "magenta": (1, 0, 1),
            "yellow": (1, 1, 0),
        }
        r_f, g_f, b_f = color_map.get(color, (1, 1, 1))
        rgb[:, :, 0] = (img * r_f).astype(np.uint8)
        rgb[:, :, 1] = (img * g_f).astype(np.uint8)
        rgb[:, :, 2] = (img * b_f).astype(np.uint8)

        self._set_pixmap_from_rgb(rgb)

    def display_composite(self, channels: np.ndarray, colors: list[str],
                          display_ranges: list[tuple[float, float]]):
        """Display multiple channels as a composite overlay."""
        c, h, w = channels.shape
        composite = np.zeros((h, w, 3), dtype=np.float64)

        color_map = {
            "gray": (1, 1, 1),
            "red": (1, 0, 0),
            "green": (0, 1, 0),
            "blue": (0, 0, 1),
            "cyan": (0, 1, 1),
            "magenta": (1, 0, 1),
            "yellow": (1, 1, 0),
        }

        for i in range(c):
            ch = channels[i].astype(np.float64)
            lo, hi = display_ranges[i] if i < len(display_ranges) else (0, 255)
            ch = np.clip((ch - lo) / max(hi - lo, 1) * 255, 0, 255)
            col = colors[i] if i < len(colors) else "gray"
            r_f, g_f, b_f = color_map.get(col, (1, 1, 1))
            composite[:, :, 0] += ch * r_f
            composite[:, :, 1] += ch * g_f
            composite[:, :, 2] += ch * b_f

        composite = np.clip(composite, 0, 255).astype(np.uint8)
        self._set_pixmap_from_rgb(composite)

    def _set_pixmap_from_rgb(self, rgb: np.ndarray):
        """Set the scene pixmap from an RGB array."""
        h, w, _ = rgb.shape
        bytes_per_line = 3 * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())

        if self._pixmap_item is None:
            self._pixmap_item = self._scene.addPixmap(pixmap)
        else:
            self._pixmap_item.setPixmap(pixmap)

        self._scene.setSceneRect(QRectF(pixmap.rect().toRectF()))

    def draw_guvs(self, guvs: list[DetectedGUV], highlight_id: int | None = None,
                  excluded_ids: set[int] | None = None):
        """Draw circle overlays for detected GUVs.

        Visual states:
        - Normal: cyan, solid, width 2
        - Highlighted: yellow, solid, width 3
        - Excluded: red, dashed, width 2
        """
        self.clear_guvs()
        if excluded_ids is None:
            excluded_ids = set()

        for guv in guvs:
            is_highlight = (guv.id == highlight_id)
            is_excluded = (guv.id in excluded_ids)

            if is_excluded:
                pen = QPen(QColor(255, 80, 80))
                pen.setWidth(2)
                pen.setStyle(Qt.PenStyle.DashLine)
            elif is_highlight:
                pen = QPen(QColor(255, 255, 0))
                pen.setWidth(3)
            else:
                pen = QPen(QColor(0, 255, 255))
                pen.setWidth(2)
            pen.setCosmetic(True)  # Constant width regardless of zoom

            x = guv.center_x - guv.radius
            y = guv.center_y - guv.radius
            d = guv.radius * 2
            ellipse = self._scene.addEllipse(x, y, d, d, pen)
            ellipse.setData(0, guv.id)
            self._guv_items[guv.id] = ellipse

            # ID label
            label = self._scene.addText(str(guv.id))
            if is_excluded:
                label.setDefaultTextColor(QColor(255, 80, 80))
            elif is_highlight:
                label.setDefaultTextColor(QColor(255, 255, 0))
            else:
                label.setDefaultTextColor(QColor(0, 255, 255))
            label.setPos(guv.center_x + guv.radius + 3, guv.center_y - 10)
            font = label.font()
            font.setPointSize(10)
            label.setFont(font)
            self._label_items[guv.id] = label

    def draw_viewport_rect(self, bbox: tuple[int, int, int, int]):
        """Draw a semi-transparent rectangle showing the overlap crop region."""
        self.clear_viewport_rect()
        x_min, y_min, x_max, y_max = bbox
        pen = QPen(QColor(255, 165, 0, 200))  # orange
        pen.setWidth(2)
        pen.setCosmetic(True)
        pen.setStyle(Qt.PenStyle.DashDotLine)
        from PyQt6.QtGui import QBrush
        brush = QBrush(QColor(255, 165, 0, 25))  # very faint orange fill
        self._viewport_rect = self._scene.addRect(
            x_min, y_min, x_max - x_min, y_max - y_min, pen, brush
        )

    def clear_viewport_rect(self):
        """Remove the viewport rectangle if present."""
        if self._viewport_rect is not None:
            self._scene.removeItem(self._viewport_rect)
            self._viewport_rect = None

    def clear_guvs(self):
        """Remove all GUV overlays."""
        self.clear_viewport_rect()
        for item in self._guv_items.values():
            self._scene.removeItem(item)
        for item in self._label_items.values():
            self._scene.removeItem(item)
        self._guv_items.clear()
        self._label_items.clear()

    def fit_in_view(self):
        """Fit the entire image in the view."""
        if self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom_factor = 1.0

    def wheelEvent(self, event: QWheelEvent):
        """Zoom with mouse wheel."""
        factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(factor, factor)
            self._zoom_factor *= factor
        else:
            self.scale(1 / factor, 1 / factor)
            self._zoom_factor /= factor

    def get_region_rgb(self, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        """Extract a crop from the current pixmap as an RGB numpy array.

        Args:
            bbox: (x_min, y_min, x_max, y_max) in image coordinates.

        Returns:
            RGB numpy array of the cropped region, or None if no pixmap.
        """
        if self._pixmap_item is None:
            return None
        pixmap = self._pixmap_item.pixmap()
        x_min, y_min, x_max, y_max = bbox
        w = x_max - x_min
        h = y_max - y_min
        if w <= 0 or h <= 0:
            return None
        cropped = pixmap.copy(x_min, y_min, w, h)
        qimg = cropped.toImage().convertToFormat(QImage.Format.Format_RGB888)
        ptr = qimg.bits()
        ptr.setsize(qimg.bytesPerLine() * qimg.height())
        arr = np.array(ptr).reshape(qimg.height(), qimg.bytesPerLine())
        # bytesPerLine may include padding; trim to actual width
        return arr[:, :w * 3].reshape(h, w, 3).copy()

    def mousePressEvent(self, event: QMouseEvent):
        """Detect clicks on GUV circles."""
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            items = self._scene.items(scene_pos)
            for item in items:
                if isinstance(item, QGraphicsEllipseItem):
                    guv_id = item.data(0)
                    if guv_id is not None:
                        self.guv_clicked.emit(guv_id)
                        return
        super().mousePressEvent(event)
