"""Dialog for selecting a series from a multi-series .lif file."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QListWidget, QDialogButtonBox,
)


class SeriesDialog(QDialog):
    """Let the user pick one series from a .lif file."""

    def __init__(self, series_list, parent=None):
        """
        Args:
            series_list: list of (name, dimensions, num_channels) tuples.
        """
        super().__init__(parent)
        self.setWindowTitle("Select Image Series")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("This file contains multiple image series. Select one:"))

        self._list = QListWidget()
        for name, dims, n_ch in series_list:
            dims_str = " × ".join(str(d) for d in dims)
            self._list.addItem(f"{name}  [{dims_str}, {n_ch} ch]")
        self._list.setCurrentRow(0)
        self._list.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self._list)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def selected_index(self) -> int:
        row = self._list.currentRow()
        return row if row >= 0 else 0
