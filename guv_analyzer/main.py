"""Entry point for GUV Analyzer application."""

import logging
import sys

from PyQt6.QtWidgets import QApplication

from .view.main_window import MainWindow
from .controller.app_controller import AppController


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(name)s %(levelname)s: %(message)s",
    )
    app = QApplication(sys.argv)
    app.setApplicationName("GUV Analyzer")

    window = MainWindow()
    controller = AppController(window)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
