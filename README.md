# GUV Analyzer

A web-based tool for detecting and analyzing Giant Unilamellar Vesicles (GUVs) in confocal microscopy images.

## Features

- **Image loading** — supports TIF (with metadata), LIF (Leica native), and JPG/PNG (detection only)
- **GUV detection** — hybrid HoughCircles + contour approach with tunable sensitivity, radius range, and circularity
- **Fluorescence measurement** — lumen (interior) and membrane (donut annulus) intensity per channel
- **Overlap handling** — detects overlapping GUVs with option to exclude
- **Export** — CSV and Excel with diameter, position, and per-channel fluorescence

## Web App (Streamlit)

### Run locally

```bash
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser.

### Deployed version

<!-- Update this URL after deploying to Streamlit Community Cloud -->
Coming soon — see deployment instructions below.

## Desktop App (PyQt6)

```bash
pip install -r requirements.txt
python -m guv_analyzer
```

## Deployment to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app" → select this repo
4. Set main file path to `streamlit_app.py`
5. Deploy — you'll get a shareable URL

## Microscope Compatibility

Designed for Leica SP8 confocal microscope. Exports from `.lif` (native) or `.tif` (via ImageJ).
