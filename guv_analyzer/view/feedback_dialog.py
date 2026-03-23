"""Feedback dialog for the PyQt6 desktop app."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QComboBox, QTextEdit,
    QDialogButtonBox,
)
from PyQt6.QtGui import QFont

from ..feedback import CATEGORIES, collect_system_info


class FeedbackDialog(QDialog):
    """Modal dialog that collects feedback category + description."""

    def __init__(self, image_format: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Send Feedback")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)

        # Category
        layout.addWidget(QLabel("Category:"))
        self._category_combo = QComboBox()
        self._category_combo.addItems(CATEGORIES)
        layout.addWidget(self._category_combo)

        # Description
        layout.addWidget(QLabel("Description:"))
        self._description_edit = QTextEdit()
        self._description_edit.setPlaceholderText(
            "Describe the bug, feature request, or feedback..."
        )
        self._description_edit.setMinimumHeight(120)
        layout.addWidget(self._description_edit)

        # System info (read-only)
        sys_info = collect_system_info(image_format)
        info_label = QLabel(sys_info.replace("- **", "").replace("**", ""))
        info_label.setWordWrap(True)
        small_font = QFont()
        small_font.setPointSize(10)
        info_label.setFont(small_font)
        info_label.setStyleSheet("color: gray;")
        layout.addWidget(info_label)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Submit")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def category(self) -> str:
        return self._category_combo.currentText()

    @property
    def description(self) -> str:
        return self._description_edit.toPlainText()
