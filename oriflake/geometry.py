from typing import Dict, List, Tuple

import cv2
import numpy as np

from .utils import angle_wrap_180


def find_candidate_polygons(
	mask: np.ndarray,
	min_area_px: int = 800,
	min_convexity: float = 0.80,
	max_vertices: int = 6,
	approx_epsilon_frac: float = 0.02,
	max_area_ratio: float = 0.3,
) -> List[Dict]:
	cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	polys: List[Dict] = []
	h, w = mask.shape
	total_pixels = h * w
	max_allowed_area = total_pixels * float(max_area_ratio)
	
	for c in cnts:
		area = cv2.contourArea(c)
		if area < float(min_area_px) or area > max_allowed_area:
			continue
		peri = cv2.arcLength(c, True)
		eps = max(1e-6, float(approx_epsilon_frac)) * peri
		approx = cv2.approxPolyDP(c, eps, True)
		v = len(approx)
		if v < 3 or v > int(max_vertices):
			continue
		hull = cv2.convexHull(c)
		hull_area = cv2.contourArea(hull)
		convexity = float(area) / float(hull_area + 1e-6)
		if convexity < float(min_convexity):
			continue
		
		# Additional filtering: check if contour is too close to image border
		M = cv2.moments(c)
		cx = float(M["m10"] / (M["m00"] + 1e-6))
		cy = float(M["m01"] / (M["m00"] + 1e-6))
		
		# Skip if centroid is too close to edges (likely image border)
		border_margin = min(w, h) * 0.02  # 2% margin from edges (reduced for loose detection)
		if (cx < border_margin or cx > w - border_margin or 
			cy < border_margin or cy > h - border_margin):
			continue
			
		# Skip if contour touches image border (relaxed for loose detection)
		x, y, w_rect, h_rect = cv2.boundingRect(c)
		if (x <= 1 or y <= 1 or x + w_rect >= w - 1 or y + h_rect >= h - 1):
			continue
			
		polys.append({
			"contour": c,
			"approx": approx,
			"area": float(area),
			"centroid": (cx, cy),
			"convexity": convexity,
		})
	return polys


def edge_angles_from_polygon(approx: np.ndarray) -> List[Tuple[int, float, Tuple[float, float]]]:
	pts = approx.reshape(-1, 2).astype(np.float32)
	n = len(pts)
	angles: List[Tuple[int, float, Tuple[float, float]]] = []
	for i in range(n):
		p0 = pts[i]
		p1 = pts[(i + 1) % n]
		dx = float(p1[0] - p0[0])
		dy = float(p1[1] - p0[1])
		theta = np.degrees(np.arctan2(-dy, dx))
		theta = angle_wrap_180(theta)
		mx = float(0.5 * (p0[0] + p1[0]))
		my = float(0.5 * (p0[1] + p1[1]))
		angles.append((i, theta, (mx, my)))
	return angles


