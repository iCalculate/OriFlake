OriFlake – Triangular MoS₂ Orientation Scanner
===========================================

Overview
--------
OriFlake automatically identifies triangular MoS₂ flakes from optical microscope images, extracts their edge orientations, computes alignment relative to 60° rotational symmetry, and visualizes distributions (rose plots and histograms). It batch-processes images and exports CSV results.

Features
--------
- Preprocessing with bilateral filter, optional gray-world white balance, gamma
- K-means segmentation in Lab color space with morphological cleanup
- Contour extraction, polygon approximation, and flake filtering
- Edge orientation angles (0–180°), Δθ to 60° symmetry about a reference
- Gaussian alignment score, overlays, rose plot and histogram
- Batch mode over a folder of images; CSV export

Project Structure
-----------------
```
oriflake/
  main.py
  seg.py
  geometry.py
  viz.py
  utils.py
  config.yaml
  images/
    testImg/
      ... your subfolders with images ...
  outputs/
```
Outputs are saved by default to `OriFlake_outputs/` in the repository root.

Installation
------------
Python 3.10+ recommended.

```bash
python -m venv .venv
. .venv/Scripts/activate  # on Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Usage
-----
```bash
# Basic usage (uses default config)
python -m oriflake.main

# With custom paths
python -m oriflake.main --input images/testImg --output OriFlake_outputs

# Using preset detection profiles
python -m oriflake.main --profile loose    # Most sensitive detection
python -m oriflake.main --profile balanced  # Default balanced settings (recommended)
python -m oriflake.main --profile strict   # Conservative detection

# Fine-tune detection parameters
python -m oriflake.main --min-area 100 --min-convexity 0.75 --max-vertices 7

# Single image analysis
python -m oriflake.main --input "images/testImg/20251016 r1 small S 140 Mo 700/7.png"
```

Flags
-----
- `--preview` to open preview windows showing overlays while processing
- `--profile {loose,balanced,strict}` to use preset detection profiles
- `--min-area N` to set minimum flake area in pixels
- `--min-convexity F` to set minimum convexity (0-1)
- `--max-vertices N` to set maximum polygon vertices
- `--approx-epsilon F` to set polygon approximation tolerance
- `--union-top-n N` to union top N k-means clusters
- `--morph-open N` and `--morph-close N` for morphology kernel sizes

CSV Schema
----------
`image, flake_id, edge_id, theta_deg, delta_deg, score, center_x, center_y, area_px`

Configuration
-------------
See `oriflake/config.yaml` for tunable parameters (k-means k, bilateral filter settings, reference angle, etc.).

Detection Parameters
--------------------
- `min_area_px`: Minimum flake area in pixels (default: 100)
- `min_convexity`: Minimum convexity ratio 0-1 (default: 0.6)
- `max_vertices`: Maximum polygon vertices (default: 7)
- `approx_epsilon_frac`: Polygon approximation tolerance (default: 0.05)
- `union_top_n`: Union top N k-means clusters (default: 2)
- `morph_open`/`morph_close`: Morphology kernel sizes (default: 2/3)
- `max_area_ratio`: Maximum flake area as fraction of image (default: 0.1)

Function Signatures
------------------
```python
# Core processing functions
def process_image(img_path: str, cfg: Dict) -> Dict
def find_candidate_polygons(mask, min_area_px=800, min_convexity=0.80, max_vertices=6, approx_epsilon_frac=0.02) -> List[Dict]
def edge_angles_from_polygon(approx) -> List[Tuple[int, float, Tuple[float, float]]]
def kmeans_segment_lab(lab_img, k=3, attempts=3) -> Tuple[np.ndarray, np.ndarray]
def preprocess_image(raw_bgr, bilateral_d, bilateral_sigC, bilateral_sigS, apply_wb=False, gamma=1.0) -> Tuple[np.ndarray, np.ndarray]
```

Notes
-----
- Place test images in `oriflake/images/testImg/` (subfolders supported) or point `--input` to your folder
- Outputs (overlays, plots, CSV) will be written to `OriFlake_outputs/`


