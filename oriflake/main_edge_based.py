"""
Edge-based processing pipeline for overlapping triangular flakes.
"""

import os
import sys
import argparse
from typing import List, Dict, Tuple

import cv2
import numpy as np
import pandas as pd

# Handle both module and direct script execution
try:
    from .utils import read_config, ensure_dir, list_images, to_rgb
    from .edge_detection import (
        detect_edges_grayscale_based,
        detect_edges_kmeans_based,
    )
    from .edge_fitting import fit_lines_to_edges, LineSegment
    from .orientation_stats import (
        extract_angles_from_orientations,
        calculate_orientation_statistics,
    )
    from .triangular_viz import save_enhanced_plots
    from .edge_viz import (
        draw_edge_map,
        draw_line_segments,
        create_debug_overlay,
        save_processing_steps,
    )
except ImportError:
    # If running as script, add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from oriflake.utils import read_config, ensure_dir, list_images, to_rgb
    from oriflake.edge_detection import (
        detect_edges_grayscale_based,
        detect_edges_kmeans_based,
    )
    from oriflake.edge_fitting import fit_lines_to_edges, LineSegment
    from oriflake.orientation_stats import (
        extract_angles_from_orientations,
        calculate_orientation_statistics,
    )
    from oriflake.triangular_viz import save_enhanced_plots
    from oriflake.edge_viz import (
        draw_edge_map,
        draw_line_segments,
        create_debug_overlay,
        save_processing_steps,
    )


def draw_edges_overlay(rgb: np.ndarray, segments: List[LineSegment], orientations: List[Tuple[float, float, float]]) -> np.ndarray:
    """
    Draw edge segments and orientations on image.
    
    Args:
        rgb: RGB image
        segments: List of line segments
        orientations: List of (angle, x, y) tuples
    
    Returns:
        Overlay image
    """
    overlay = rgb.copy()
    
    # Draw line segments
    for seg in segments:
        cv2.line(overlay, (int(seg.x1), int(seg.y1)), (int(seg.x2), int(seg.y2)), (0, 255, 0), 2)
        # Draw center point
        cv2.circle(overlay, (int(seg.center_x), int(seg.center_y)), 3, (255, 0, 0), -1)
    
    return overlay


def process_image_edge_based(img_path: str, cfg: Dict) -> Dict:
    """
    Process image using edge-based detection.
    
    Args:
        img_path: Path to input image
        cfg: Configuration dictionary
    
    Returns:
        Dictionary with results
    """
    # Read image
    img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    
    rgb = to_rgb(img_bgr)
    
    # Get image processing parameters
    img_params = cfg.get('image_processing', {})
    
    # Choose detection method
    detection_method = img_params.get('detection_method', 'grayscale')  # 'grayscale', 'kmeans', 'canny'
    
    if detection_method == 'grayscale':
        # Use grayscale distribution-based edge detection (recommended)
        gray, mask, edges, analysis = detect_edges_grayscale_based(
            rgb,
            width_ratio=float(img_params.get('roi_width_ratio', 0.7)),
            height_ratio=float(img_params.get('roi_height_ratio', 0.7)),
            use_otsu=bool(img_params.get('use_otsu', False)),
            manual_threshold=img_params.get('manual_threshold'),
            invert_mask=bool(img_params.get('invert_mask', False)),
            morph_open=int(img_params.get('morph_open', 3)),
            morph_close=int(img_params.get('morph_close', 5)),
            use_contour_edges=bool(img_params.get('use_contour_edges', True)),
            canny_low=float(img_params.get('canny_low', 50)),
            canny_high=float(img_params.get('canny_high', 150)),
            bilateral_d=int(img_params.get('bilateral_d', 7)),
            bilateral_sigC=float(img_params.get('bilateral_sigC', 50)),
            bilateral_sigS=float(img_params.get('bilateral_sigS', 50)),
            white_balance=bool(img_params.get('white_balance', False)),
            gamma=float(img_params.get('gamma', 1.0)),
        )
        edges_before_enhance = edges.copy()
        grayscale_analysis = analysis
        verbose = cfg.get('verbose', False)
        if verbose:
            print(f"  Grayscale analysis: {len(analysis['peak_positions'])} peaks found")
            if len(analysis['peak_positions']) >= 2:
                print(f"    Peak 1: {analysis['peak_positions'][0]:.1f}, Peak 2: {analysis['peak_positions'][1]:.1f}")
            print(f"    Threshold used: {analysis['threshold_used']:.1f}")
            print(f"    ROI: {analysis['roi_bbox']}")
    elif detection_method == 'kmeans':
        # Use K-means segmentation-based edge detection
        gray, mask, edges = detect_edges_kmeans_based(
            rgb,
            kmeans_k=int(img_params.get('kmeans_k', 3)),
            union_top_n=int(img_params.get('union_top_n', 1)),
            morph_open=int(img_params.get('morph_open', 3)),
            morph_close=int(img_params.get('morph_close', 5)),
            canny_low=float(img_params.get('canny_low', 50)),
            canny_high=float(img_params.get('canny_high', 150)),
            use_contour_edges=bool(img_params.get('use_contour_edges', True)),
            bilateral_d=int(img_params.get('bilateral_d', 7)),
            bilateral_sigC=float(img_params.get('bilateral_sigC', 50)),
            bilateral_sigS=float(img_params.get('bilateral_sigS', 50)),
            white_balance=bool(img_params.get('white_balance', False)),
            gamma=float(img_params.get('gamma', 1.0)),
        )
        edges_before_enhance = edges.copy()  # No enhancement for segmentation method
        grayscale_analysis = None
    else:
        # Fallback to grayscale method if unknown method
        if verbose:
            print(f"  Unknown detection method '{detection_method}', using 'grayscale'")
        gray, mask, edges, analysis = detect_edges_grayscale_based(
            rgb,
            width_ratio=float(img_params.get('roi_width_ratio', 0.7)),
            height_ratio=float(img_params.get('roi_height_ratio', 0.7)),
            use_otsu=bool(img_params.get('use_otsu', False)),
            manual_threshold=img_params.get('manual_threshold'),
            invert_mask=bool(img_params.get('invert_mask', False)),
            morph_open=int(img_params.get('morph_open', 3)),
            morph_close=int(img_params.get('morph_close', 5)),
            use_contour_edges=bool(img_params.get('use_contour_edges', True)),
            canny_low=float(img_params.get('canny_low', 50)),
            canny_high=float(img_params.get('canny_high', 150)),
            bilateral_d=int(img_params.get('bilateral_d', 7)),
            bilateral_sigC=float(img_params.get('bilateral_sigC', 50)),
            bilateral_sigS=float(img_params.get('bilateral_sigS', 50)),
            white_balance=bool(img_params.get('white_balance', False)),
            gamma=float(img_params.get('gamma', 1.0)),
        )
        edges_before_enhance = edges.copy()
        grayscale_analysis = analysis
    
    # Get fitting and statistics parameters
    fit_params = cfg.get('fitting_stats', {})
    
    # Fit lines to edges
    # Debug: check edge statistics
    edge_pixels = cv2.countNonZero(edges)
    verbose = cfg.get('verbose', False)
    if verbose:
        print(f"  Edge pixels: {edge_pixels} / {edges.size} ({100.0 * edge_pixels / edges.size:.2f}%)")
    
    segments, orientations = fit_lines_to_edges(
        edges,
        gray=gray,
        method=fit_params.get('line_detection_method', 'hough'),
        min_line_length=float(fit_params.get('hough_min_line_length', 50.0)),
        max_line_gap=float(fit_params.get('hough_max_line_gap', 10.0)),
        hough_threshold=int(fit_params.get('hough_threshold', 100)),
        merge_angle_tolerance=float(fit_params.get('merge_angle_tolerance', 5.0)),
        merge_distance_tolerance=float(fit_params.get('merge_distance_tolerance', 20.0)),
        min_segment_length=float(fit_params.get('min_segment_length', 10.0)),
        filter_min_angle=float(fit_params.get('filter_min_angle', 0.0)),
        filter_max_angle=float(fit_params.get('filter_max_angle', 180.0)),
        verbose=verbose,
    )
    
    # Extract angles
    angles = extract_angles_from_orientations(orientations)
    
    # Calculate statistics
    period = int(fit_params.get('period', 60))
    stats = calculate_orientation_statistics(angles, period=period)
    
    # Create output rows
    rows = []
    for i, (angle, x, y) in enumerate(orientations):
        rows.append({
            'segment_id': i,
            'theta_deg': float(angle),
            'center_x': float(x),
            'center_y': float(y),
            'length': segments[i].length if i < len(segments) else 0.0,
        })
    
    # Draw overlay
    overlay = draw_edges_overlay(rgb, segments, orientations)
    
    # Store intermediate results for debug output
    result = {
        'overlay': overlay,
        'rows': rows,
        'angles': angles,
        'segments': segments,
        'statistics': stats,
        'num_segments': len(segments),
        # Intermediate results for debugging
        'rgb': rgb,
        'gray': gray,
        'edges': edges,
        'edges_before_enhance': edges_before_enhance,
        'mask': mask,  # Segmentation mask if using segmentation method
        'grayscale_analysis': grayscale_analysis,  # Grayscale analysis if using grayscale method
        'orientations': orientations,
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="OriFlake Edge-Based - Triangular MoS2 Orientation Scanner")
    default_config = os.path.join(os.path.dirname(__file__), "config_edge_based.yaml")
    default_input = os.path.join(os.path.dirname(__file__), "images", "testImg")
    default_output = os.path.join(os.path.dirname(__file__), "..", "OriFlake_outputs")
    
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--input", type=str, default=default_input)
    parser.add_argument("--output", type=str, default=default_output)
    parser.add_argument("--preview", action="store_true", help="Show preview windows")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress logs")
    parser.add_argument("--debug", action="store_true", help="Save intermediate processing steps")
    
    # Image processing overrides
    parser.add_argument("--canny-low", type=float, help="Canny lower threshold")
    parser.add_argument("--canny-high", type=float, help="Canny upper threshold")
    parser.add_argument("--enhance-dilate", type=int, help="Edge dilation iterations")
    
    # Fitting overrides
    parser.add_argument("--line-method", choices=['hough', 'lsd'], help="Line detection method")
    parser.add_argument("--min-line-length", type=float, help="Minimum line length")
    parser.add_argument("--merge-angle-tol", type=float, help="Angle tolerance for merging")
    
    args = parser.parse_args()
    
    cfg = read_config(args.config)
    
    # Apply CLI overrides
    if args.canny_low is not None:
        cfg.setdefault('image_processing', {})['canny_low'] = args.canny_low
    if args.canny_high is not None:
        cfg.setdefault('image_processing', {})['canny_high'] = args.canny_high
    if args.enhance_dilate is not None:
        cfg.setdefault('image_processing', {})['enhance_dilate'] = args.enhance_dilate
    if args.line_method:
        cfg.setdefault('fitting_stats', {})['line_detection_method'] = args.line_method
    if args.min_line_length is not None:
        cfg.setdefault('fitting_stats', {})['hough_min_line_length'] = args.min_line_length
    if args.merge_angle_tol is not None:
        cfg.setdefault('fitting_stats', {})['merge_angle_tolerance'] = args.merge_angle_tol
    
    ensure_dir(args.output)
    
    image_paths = list_images(args.input, recursive=True)
    if args.verbose:
        print(f"Scanning input: {args.input}")
        print(f"Found {len(image_paths)} image(s)")
    if not image_paths:
        print(f"No images found under {args.input}")
        return
    
    # Pass verbose flag to config
    cfg['verbose'] = args.verbose
    
    suffix = "edge_based"
    
    all_rows: List[Dict] = []
    all_angles: List[float] = []
    for p in image_paths:
        if args.verbose:
            print(f"Processing: {p}")
        try:
            res = process_image_edge_based(p, cfg)
            rel = os.path.relpath(p, args.input).replace("\\", "/")
            fname = os.path.splitext(os.path.basename(p))[0]
            
            # Save main overlay
            out_overlay = os.path.join(args.output, f"{fname}_overlay_{suffix}.png")
            ok = cv2.imwrite(out_overlay, cv2.cvtColor(res["overlay"], cv2.COLOR_RGB2BGR))
            if args.verbose:
                print(f"Overlay saved: {out_overlay} ({'ok' if ok else 'failed'})")
                print(f"  Segments: {res['num_segments']}, Angles: {len(res['angles'])}")
                if len(res['angles']) == 0:
                    print(f"  WARNING: No segments detected! Check edge detection parameters.")
            
            # Save debug images if requested
            if args.debug:
                debug_dir = os.path.join(args.output, f"{fname}_debug_{suffix}")
                saved_files = save_processing_steps(
                    res['rgb'],
                    res['gray'],
                    res['edges'],
                    res['segments'],
                    res['orientations'],
                    debug_dir,
                    prefix=fname,
                    edges_before_enhance=res.get('edges_before_enhance'),
                    mask=res.get('mask'),
                    grayscale_analysis=res.get('grayscale_analysis'),
                )
                if args.verbose:
                    print(f"  Debug images saved to: {debug_dir}")
                    print(f"    Saved {len(saved_files)} debug images")
            for r in res["rows"]:
                row = {"image": rel, **r}
                all_rows.append(row)
            all_angles.extend(res["angles"])
            if args.preview:
                cv2.imshow("OriFlake Edge-Based Overlay", cv2.cvtColor(res["overlay"], cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
        except Exception as e:
            print(f"Error processing {p}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Save CSV
    if all_rows:
        df = pd.DataFrame(all_rows, columns=[
            "image", "segment_id", "theta_deg", "center_x", "center_y", "length"
        ])
        csv_path = os.path.join(args.output, f"oriflake_results_{suffix}.csv")
        df.to_csv(csv_path, index=False)
        if args.verbose:
            print(f"CSV saved: {csv_path} (rows={len(all_rows)})")
    
    # Save histogram with triple peak analysis
    if all_angles:
        hist_png = os.path.join(args.output, f"orientations_hist_{suffix}.png")
        rose_png = os.path.join(args.output, f"orientations_rose_{suffix}.png")
        
        period = int(cfg.get('fitting_stats', {}).get('period', 60))
        save_enhanced_plots(all_angles, rose_png, hist_png, period=period, bins=36)
        
        if args.verbose:
            print(f"Histogram saved: {hist_png}")
    
    if args.preview:
        print("Press any key on an image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

