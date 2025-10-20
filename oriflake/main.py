import os
import argparse
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd

from .utils import read_config, ensure_dir, list_images, preprocess_image, angle_delta_sym60, gaussian_score
from .seg import kmeans_segment_lab, select_candidate_mask, morph_cleanup, threshold_fallback
from .geometry import find_candidate_polygons, edge_angles_from_polygon
from .viz import draw_polygons_overlay, save_hist_rose


def process_image(img_path: str, cfg: Dict) -> Dict:
	img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
	if img_bgr is None:
		raise FileNotFoundError(f"Failed to read image: {img_path}")
	rgb, lab = preprocess_image(
		img_bgr,
		bilateral_d=int(cfg.get("bilateral_d", 7)),
		bilateral_sigC=float(cfg.get("bilateral_sigC", 50)),
		bilateral_sigS=float(cfg.get("bilateral_sigS", 50)),
		apply_wb=bool(cfg.get("white_balance", False)),
		gamma=float(cfg.get("gamma", 1.0)),
	)

	labels, _ = kmeans_segment_lab(lab, k=int(cfg.get("kmeans_k", 3)))
	# Evaluate each label; optionally union top-N labels
	open_ks = int(cfg.get("morph_open", 3))
	close_ks = int(cfg.get("morph_close", 5))
	min_area = int(cfg.get("min_area_px", 800))
	union_top_n = int(cfg.get("union_top_n", 1))
	label_scores = []
	cand_masks = []
	for lbl in np.unique(labels):
		m = (labels == int(lbl)).astype(np.uint8) * 255
		m = morph_cleanup(m, open_ks=open_ks, close_ks=close_ks)
		polys_lbl = find_candidate_polygons(
			m,
			min_area_px=min_area,
			min_convexity=float(cfg.get("min_convexity", 0.8)),
			max_vertices=int(cfg.get("max_vertices", 6)),
			approx_epsilon_frac=float(cfg.get("approx_epsilon_frac", 0.02)),
			max_area_ratio=float(cfg.get("max_area_ratio", 0.3)),
		)
		label_scores.append((len(polys_lbl), lbl))
		cand_masks.append((lbl, m))
	label_scores.sort(reverse=True)
	if union_top_n > 1:
		mask = np.zeros_like(labels, dtype=np.uint8)
		for i in range(min(union_top_n, len(label_scores))):
			_, lbl = label_scores[i]
			m = next(m for l, m in cand_masks if l == lbl)
			mask = cv2.bitwise_or(mask, m)
	else:
		best_lbl = label_scores[0][1] if label_scores else 0
		mask = next(m for l, m in cand_masks if l == best_lbl) if cand_masks else select_candidate_mask(labels)
	mask = morph_cleanup(mask, open_ks=open_ks, close_ks=close_ks)

	if cv2.countNonZero(mask) < int(cfg.get("min_area_px", 800)):
		gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
		mask = threshold_fallback(gray)
		mask = morph_cleanup(mask, open_ks=3, close_ks=5)

	polys = find_candidate_polygons(
		mask,
		min_area_px=int(cfg.get("min_area_px", 800)),
		min_convexity=float(cfg.get("min_convexity", 0.8)),
		max_vertices=int(cfg.get("max_vertices", 6)),
		approx_epsilon_frac=float(cfg.get("approx_epsilon_frac", 0.02)),
		max_area_ratio=float(cfg.get("max_area_ratio", 0.3)),
	)
	edge_lists: List[List] = []
	rows: List[Dict] = []
	all_angles: List[float] = []
	ref = float(cfg.get("reference_deg", 0.0))
	sigma = float(cfg.get("sigma_deg", 5.0))
	for fidx, poly in enumerate(polys):
		edges = edge_angles_from_polygon(poly["approx"])
		edge_lists.append(edges)
		cx, cy = poly["centroid"]
		for (eid, theta, (mx, my)) in edges:
			delta = angle_delta_sym60(theta, ref)
			score = gaussian_score(delta, sigma)
			rows.append({
				"flake_id": fidx,
				"edge_id": eid,
				"theta_deg": float(theta),
				"delta_deg": float(delta),
				"score": float(score),
				"center_x": float(cx),
				"center_y": float(cy),
				"area_px": float(poly["area"]),
			})
			all_angles.append(float(theta))

	over = draw_polygons_overlay(rgb, polys, edge_lists)
	return {"overlay": over, "rows": rows, "angles": all_angles, "num_flakes": len(polys)}


def main():
	parser = argparse.ArgumentParser(description="OriFlake - Triangular MoS2 Orientation Scanner")
	default_config = os.path.join(os.path.dirname(__file__), "config.yaml")
	default_input = os.path.join(os.path.dirname(__file__), "images", "testImg")
	default_output = os.path.join(os.path.dirname(__file__), "..", "OriFlake_outputs")
	parser.add_argument("--config", type=str, default=default_config)
	parser.add_argument("--input", type=str, default=default_input)
	parser.add_argument("--output", type=str, default=default_output)
	parser.add_argument("--preview", action="store_true", help="Show preview windows")
	parser.add_argument("--verbose", action="store_true", help="Print detailed progress logs")
	# Detection control overrides
	parser.add_argument("--min-area", type=int, help="Min flake area (px)")
	parser.add_argument("--min-convexity", type=float, help="Min convexity (0-1)")
	parser.add_argument("--max-vertices", type=int, help="Max polygon vertices")
	parser.add_argument("--approx-epsilon", type=float, help="Polygon approx epsilon fraction")
	parser.add_argument("--union-top-n", type=int, help="Union top N k-means clusters")
	parser.add_argument("--morph-open", type=int, help="Morphology open kernel size")
	parser.add_argument("--morph-close", type=int, help="Morphology close kernel size")
	parser.add_argument("--profile", choices=["loose", "balanced", "strict"], help="Preset detection profile")
	args = parser.parse_args()

	cfg = read_config(args.config)
	# Apply preset profile if specified
	if args.profile:
		presets = {
			"loose": {
				"min_area_px": 150,
				"min_convexity": 0.5,
				"max_vertices": 7,
				"approx_epsilon_frac": 0.06,
				"union_top_n": 2,
				"morph_open": 2,
				"morph_close": 2,
				"max_area_ratio": 0.2,
			},
			"balanced": {
				"min_area_px": 200,
				"min_convexity": 0.7,
				"max_vertices": 6,
				"approx_epsilon_frac": 0.03,
				"union_top_n": 1,
				"morph_open": 3,
				"morph_close": 4,
				"max_area_ratio": 0.2,
			},
			"strict": {
				"min_area_px": 300,
				"min_convexity": 0.8,
				"max_vertices": 5,
				"approx_epsilon_frac": 0.02,
				"union_top_n": 1,
				"morph_open": 4,
				"morph_close": 5,
				"max_area_ratio": 0.25,
			},
		}
		cfg.update(presets[args.profile])
	# Override with CLI args if provided
	if args.min_area is not None:
		cfg["min_area_px"] = args.min_area
	if args.min_convexity is not None:
		cfg["min_convexity"] = args.min_convexity
	if args.max_vertices is not None:
		cfg["max_vertices"] = args.max_vertices
	if args.approx_epsilon is not None:
		cfg["approx_epsilon_frac"] = args.approx_epsilon
	if args.union_top_n is not None:
		cfg["union_top_n"] = args.union_top_n
	if args.morph_open is not None:
		cfg["morph_open"] = args.morph_open
	if args.morph_close is not None:
		cfg["morph_close"] = args.morph_close
	ensure_dir(args.output)

	image_paths = list_images(args.input, recursive=True)
	if args.verbose:
		print(f"Scanning input: {args.input}")
		print(f"Found {len(image_paths)} image(s)")
	if not image_paths:
		print(f"No images found under {args.input}")
		return

	# Determine output suffix based on profile or custom parameters
	if args.profile:
		suffix = args.profile
	else:
		suffix = "default"
	
	all_rows: List[Dict] = []
	all_angles: List[float] = []
	for p in image_paths:
		if args.verbose:
			print(f"Processing: {p}")
		res = process_image(p, cfg)
		rel = os.path.relpath(p, args.input).replace("\\", "/")
		fname = os.path.splitext(os.path.basename(p))[0]
		out_overlay = os.path.join(args.output, f"{fname}_overlay_{suffix}.png")
		ok = cv2.imwrite(out_overlay, cv2.cvtColor(res["overlay"], cv2.COLOR_RGB2BGR))
		if args.verbose:
			print(f"Overlay saved: {out_overlay} ({'ok' if ok else 'failed'})")
		for r in res["rows"]:
			row = {"image": rel, **r}
			all_rows.append(row)
		all_angles.extend(res["angles"])
		if args.preview:
			cv2.imshow("OriFlake Overlay", cv2.cvtColor(res["overlay"], cv2.COLOR_RGB2BGR))
			cv2.waitKey(1)

	# Save CSV
	df = pd.DataFrame(all_rows, columns=[
		"image", "flake_id", "edge_id", "theta_deg", "delta_deg", "score", "center_x", "center_y", "area_px"
	])
	csv_path = os.path.join(args.output, f"oriflake_results_{suffix}.csv")
	df.to_csv(csv_path, index=False)
	if args.verbose:
		print(f"CSV saved: {csv_path} (rows={len(all_rows)})")

	# Save plots
	if all_angles:
		rose_png = os.path.join(args.output, f"orientations_rose_{suffix}.png")
		hist_png = os.path.join(args.output, f"orientations_hist_{suffix}.png")
		save_hist_rose(all_angles, rose_png, bins=36, rose=True)
		save_hist_rose(all_angles, hist_png, bins=36, rose=False)
		if args.verbose:
			print(f"Rose plot saved: {rose_png}")
			print(f"Histogram saved: {hist_png}")

	if args.preview:
		print("Press any key on an image window to close...")
		cv2.waitKey(0)
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()


