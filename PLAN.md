# GUV Analyzer - Implementation Plan

## Context
Build a cross-platform (Mac/Windows) desktop application that loads confocal microscopy images of Giant Unilamellar Vesicles (GUVs), detects circular vesicles, and measures their diameter and fluorescence (both lumen area and membrane "donut"). The lab uses a Leica SP8 producing `.lif` files, typically exported to `.tif` via ImageJ. A training set of 254 images (JPG/PNG/TIF) across 5 researchers is available for tuning detection.

**Key constraint:** Quantitative fluorescence measurements are only valid on raw TIF files (not JPG/PNG which are lossy). JPG/PNG images can be used for detection tuning and evaluation only.

**Decisions made:**
- **TIF-only** for initial release (.lif support deferred)
- **PyQt6** for GUI framework
- **Metadata from TIF tags** — raw TIF files contain all needed metadata (scale, channels, display ranges). No manual scale input needed.

## Architecture

**Stack:** Python + PyQt6 (GUI) + OpenCV/scikit-image (detection) + tifffile (I/O)

**MVC structure:**
```
guv_analyzer/
├── main.py                  # Entry point
├── model/
│   ├── image_loader.py      # TIFF loading + metadata extraction
│   ├── guv_detector.py      # Circle detection (HoughCircles + contour fitting)
│   ├── fluorescence.py      # Area & membrane fluorescence measurement
│   └── guv_data.py          # Data classes + CSV/Excel export
├── view/
│   ├── main_window.py       # Main window layout
│   ├── image_canvas.py      # QGraphicsView with zoom/pan + circle overlays
│   ├── metadata_panel.py    # Image specs display (Step 1)
│   ├── detection_panel.py   # Detection parameter controls (Step 2)
│   └── results_panel.py     # Measurement table + export (Step 3)
└── controller/
    └── app_controller.py    # Wires model↔view, threading
```

## Step 1: Image Loading & Metadata (`image_loader.py`)

Extract from TIFF tags (confirmed from ImageJ "Show Info" output):
- **Scale per pixel** — TIFF tags 282/283 (XRes/YRes). Sample: 2.6420 px/µm → 0.3785 µm/px. Also available as voxel size in ImageJ description
- **Number of channels** — ImageJ tag 270 (`channels=2`, `mode=composite`)
- **Display ranges** — per-channel min/max from tag 270 (e.g., `1-29` for ch1, `0-255` for ch2)
- **Laser power** — NOT in exported TIF; provide optional manual input field. Could parse from `.lif` later

For JPG/PNG (training set): no metadata → detection-only mode, no fluorescence quantification.

Use `tifffile` for ImageJ-style multi-frame TIFFs. Each frame = one channel stored as separate 2D array.

## Step 2: GUV Detection (`guv_detector.py`)

**Observations from training set that drive algorithm design:**

| Challenge | Example | Implication |
|-----------|---------|-------------|
| Wide size range | ly's tiny dots vs casana's full-frame GUV | Need broad min/max radius range |
| Variable SNR | eunjin's noisy membrane vs casana's clean ring | Need adaptive preprocessing |
| Touching/connected GUVs | Han's filament-connected clusters | HoughCircles handles this well |
| Multi-channel composites | Sohyun's red+green | Let user pick detection channel |
| Scale bars in image | Bottom-right white bars | Could interfere with detection; mask bottom edge |
| Very dense fields | ly's hundreds of vesicles | Need efficient batch processing |

**Algorithm:**
1. **Preprocessing:** Convert to grayscale if needed → Gaussian blur (σ=2-5, adjustable) → optional CLAHE for low-contrast images
2. **Detection:** `cv2.HoughCircles` with `HOUGH_GRADIENT_ALT` (more robust than classic)
   - User-adjustable: min/max radius (in µm), sensitivity, min distance between centers
   - Convert µm params to pixels using metadata scale
3. **Post-filtering:** Reject circles touching image borders, reject by intensity profile (real GUVs have bright ring + dark center pattern)
4. **Future:** Ellipse fitting via `cv2.fitEllipse` on contours, with user-set axis-ratio threshold

User selects which channel to use for detection (membrane dye channel typically best).

## Step 3: Fluorescence Measurement (`fluorescence.py`)

For each detected GUV, on each channel:
- **Lumen (area) fluorescence:** Mean intensity inside circle minus a margin from the membrane
- **Membrane fluorescence:** Mean intensity in an annular ring (donut). Width adjustable (default ~3-5 px)
- **Diameter** in µm

Masks: circular mask for lumen, annular mask for membrane. Both created with `np.ogrid` for efficiency.

**Only computed on TIF files** — for JPG/PNG, show detection overlay only with a warning that fluorescence values are not quantitatively valid.

## Step 4: GUI (`view/`)

```
┌────────────────────────────────────────────────────────┐
│ File | Analysis | View | Help                          │
├─────────────────────────────┬──────────────────────────┤
│                             │ Image Info               │
│                             │  Scale: 0.378 µm/px      │
│  Image Canvas               │  Channels: 2             │
│  (zoom/pan, circle overlays)│  Laser power: [__] (manual)│
│                             ├──────────────────────────┤
│                             │ Detection Controls       │
│                             │  Channel: [dropdown]     │
│                             │  Min/Max radius sliders  │
│                             │  Sensitivity slider      │
│                             │  [Detect] [Clear]        │
│                             ├──────────────────────────┤
│                             │ Results Table            │
│                             │  ID|Diam|AreaFl|MembFl   │
│                             │  [Export CSV] [Export XLS]│
├─────────────────────────────┴──────────────────────────┤
│ Status: "15 GUVs detected"                             │
└────────────────────────────────────────────────────────┘
```

Key interactions:
- Click GUV in image → highlights row in table (and vice versa)
- Right-click → manually add/delete GUV
- Channel toggle buttons to switch displayed channel
- Mouse wheel zoom + drag pan

## Step 5: Packaging

`PyInstaller` to create standalone `.app` (Mac) and `.exe` (Windows). Use `opencv-python-headless` to avoid Qt conflicts.

## Implementation Order

1. **`requirements.txt`** + project skeleton (already created)
2. **`model/image_loader.py`** — load TIF, parse metadata, return channel arrays
3. **`model/guv_detector.py`** — HoughCircles detection with tunable params
4. **`model/fluorescence.py`** — area + membrane measurement with masks
5. **`model/guv_data.py`** — dataclasses + export
6. **`view/image_canvas.py`** — QGraphicsView with zoom/pan
7. **`view/main_window.py`** + panels — full GUI layout
8. **`controller/app_controller.py`** — wire everything, add QThread for detection
9. **`main.py`** — entry point
10. Test against training set images, tune default parameters
11. PyInstaller packaging

## Verification

1. Run `python -m guv_analyzer.main` → GUI opens
2. File → Open → load `Sample/SampleImage-1.tif` → metadata panel shows 2 channels, 0.378 µm/px
3. Select channel, click Detect → circles drawn on GUVs
4. Results table populates with diameter + fluorescence per channel
5. Export CSV → valid file with measurements
6. Test with JPG from training set → detection works, fluorescence shows warning
7. Test zoom/pan on large images

## Training Set Summary

254 images across 5 researchers in `Sample/Training Set/`:

| Researcher | Images | Conditions |
|-----------|--------|------------|
| Han | 41 (29 JPG, 8 PNG, 4 TIF) | Osmolarity (100-500 mOsm), Mg/Na concentrations, ATP treatments |
| Sohyun | 144 (136 JPG, 8 PNG) | pH studies (4-7.4), FN/Hep-H proteins, HEPES/PBS/MES buffers |
| casana | 36 (36 JPG) | LHN/MHJ peptides, DOPG lipids, melitin, pore formation |
| eunjin | 14 (14 JPG) | Time-course (4d, 37d), z-slices, tile scans |
| ly | 18 (9 JPG, 9 PNG) | Isotonic vs hypertonic, wide-field, time-lapse |
