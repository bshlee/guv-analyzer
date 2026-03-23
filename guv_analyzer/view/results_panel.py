"""Results table and export controls (Step 3)."""

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QTableWidget, QTableWidgetItem,
    QHBoxLayout, QPushButton, QHeaderView, QLabel,
)

import pandas as pd


class ResultsPanel(QWidget):
    """Displays GUV measurement results in a table with export buttons."""

    row_selected = pyqtSignal(int)  # Emits GUV ID when a row is clicked
    export_csv_requested = pyqtSignal()
    export_excel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Results")
        group_layout = QVBoxLayout()

        self._count_label = QLabel("No GUVs detected")
        group_layout.addWidget(self._count_label)

        self._table = QTableWidget()
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.cellClicked.connect(self._on_cell_clicked)
        group_layout.addWidget(self._table)

        # Warning label for non-quantitative images
        self._warning_label = QLabel()
        self._warning_label.setStyleSheet("color: orange; font-weight: bold;")
        self._warning_label.setWordWrap(True)
        self._warning_label.hide()
        group_layout.addWidget(self._warning_label)

        # Export buttons
        btn_layout = QHBoxLayout()
        self._csv_btn = QPushButton("Export CSV")
        self._excel_btn = QPushButton("Export Excel")
        btn_layout.addWidget(self._csv_btn)
        btn_layout.addWidget(self._excel_btn)
        group_layout.addLayout(btn_layout)

        group.setLayout(group_layout)
        layout.addWidget(group)

        self._csv_btn.clicked.connect(self.export_csv_requested)
        self._excel_btn.clicked.connect(self.export_excel_requested)

        self._df: pd.DataFrame | None = None

    def update_results(self, df: pd.DataFrame, quantitative: bool = True,
                       excluded_ids: set[int] | None = None):
        """Populate the table from a DataFrame.

        Rows whose ID is in *excluded_ids* are grayed out.
        """
        self._df = df
        if excluded_ids is None:
            excluded_ids = set()

        active_count = sum(1 for _, row in df.iterrows() if int(row["ID"]) not in excluded_ids)
        self._count_label.setText(
            f"{len(df)} GUV(s) detected"
            + (f"  ({len(df) - active_count} excluded)" if excluded_ids else "")
        )

        self._table.setRowCount(len(df))
        self._table.setColumnCount(len(df.columns))
        self._table.setHorizontalHeaderLabels(list(df.columns))

        gray_bg = QColor(60, 60, 60)
        gray_fg = QColor(120, 120, 120)

        for row_idx in range(len(df)):
            guv_id = int(df.iloc[row_idx]["ID"])
            is_excluded = guv_id in excluded_ids
            for col_idx, col in enumerate(df.columns):
                val = df.iloc[row_idx, col_idx]
                item = QTableWidgetItem(str(val))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if is_excluded:
                    item.setBackground(gray_bg)
                    item.setForeground(gray_fg)
                self._table.setItem(row_idx, col_idx, item)

        self._table.resizeColumnsToContents()

        if not quantitative:
            self._warning_label.setText(
                "Warning: Fluorescence values are not quantitatively valid "
                "(JPG/PNG lossy format). Use raw TIF for accurate measurements."
            )
            self._warning_label.show()
        else:
            self._warning_label.hide()

    def highlight_row(self, guv_id: int):
        """Select the row corresponding to a GUV ID."""
        if self._df is None:
            return
        matches = self._df.index[self._df["ID"] == guv_id].tolist()
        if matches:
            self._table.selectRow(matches[0])

    def _on_cell_clicked(self, row: int, col: int):
        if self._df is not None and row < len(self._df):
            guv_id = int(self._df.iloc[row]["ID"])
            self.row_selected.emit(guv_id)

    def clear(self):
        self._table.setRowCount(0)
        self._table.setColumnCount(0)
        self._count_label.setText("No GUVs detected")
        self._warning_label.hide()
        self._df = None
