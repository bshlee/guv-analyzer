"""Feedback module — builds pre-filled GitHub issue URLs and opens them in a browser.

Shared by both the PyQt6 desktop app and the Streamlit web app.
No external dependencies (stdlib only).
"""

import platform
import sys
import urllib.parse
import webbrowser

from . import __version__

# ── Configuration ────────────────────────────────────────────────────────────
# Change "method" to switch feedback target without rebuilding:
#   "github"      → opens pre-filled GitHub issue (default)
#   "google_form" → opens a Google Form URL
#   "mailto"      → opens the user's email client

FEEDBACK_CONFIG = {
    "method": "github",
    "github_url": "https://github.com/bshlee/guv-analyzer/issues/new",
    "github_labels": "feedback",
    "google_form_url": "",
    "mailto_address": "",
}

CATEGORIES = ["Bug Report", "Feature Request", "General Feedback"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def collect_system_info(image_format: str | None = None) -> str:
    """Return a markdown-formatted string with system and app details."""
    lines = [
        f"- **App version:** {__version__}",
        f"- **OS:** {platform.system()} {platform.release()} ({platform.machine()})",
        f"- **Python:** {sys.version.split()[0]}",
    ]
    if image_format:
        lines.append(f"- **Last image format:** {image_format}")
    return "\n".join(lines)


def build_github_url(category: str, description: str, system_info: str) -> str:
    """Build a pre-filled GitHub ``issues/new`` URL."""
    title = f"[{category}] "
    body = (
        f"## Description\n\n{description}\n\n"
        f"## System Info\n\n{system_info}\n"
    )
    params = {
        "title": title,
        "body": body,
        "labels": FEEDBACK_CONFIG["github_labels"],
    }
    return FEEDBACK_CONFIG["github_url"] + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)


def build_mailto_url(category: str, description: str, system_info: str) -> str:
    """Build a ``mailto:`` URL with pre-filled subject and body."""
    subject = f"[GUV Analyzer] {category}"
    body = f"{description}\n\n---\n{system_info}"
    params = {"subject": subject, "body": body}
    return f"mailto:{FEEDBACK_CONFIG['mailto_address']}?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)


def open_feedback(category: str, description: str, image_format: str | None = None) -> str:
    """Open the configured feedback target in the user's default browser.

    Returns the URL that was opened.
    """
    sys_info = collect_system_info(image_format)
    method = FEEDBACK_CONFIG["method"]

    if method == "github":
        url = build_github_url(category, description, sys_info)
    elif method == "google_form":
        url = FEEDBACK_CONFIG["google_form_url"]
    elif method == "mailto":
        url = build_mailto_url(category, description, sys_info)
    else:
        raise ValueError(f"Unknown feedback method: {method}")

    webbrowser.open(url)
    return url
