from typing import List, Dict, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_polygons_overlay(rgb: np.ndarray, polys: List[Dict], edge_angles: List[List[Tuple[int, float, Tuple[float, float]]]]) -> np.ndarray:
	over = rgb.copy()
	for fidx, poly in enumerate(polys):
		approx = poly["approx"].reshape(-1, 2).astype(np.int32)
		cv2.polylines(over, [approx], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
		for (eid, theta, (mx, my)) in edge_angles[fidx]:
			length = 20
			rad = np.deg2rad(theta)
			x2 = int(mx + length * np.cos(rad))
			y2 = int(my - length * np.sin(rad))
			cv2.arrowedLine(over, (int(mx), int(my)), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.25)
	return over


def save_hist_rose(angles_deg: List[float], out_png: str, bins: int = 36, rose: bool = True) -> None:
	plt.figure(figsize=(5, 5), dpi=150)
	if rose:
		theta = np.deg2rad(np.array(angles_deg))
		n, b = np.histogram(theta, bins=bins, range=(0, 2 * np.pi))
		ax = plt.subplot(111, polar=True)
		ax.bar((b[:-1] + b[1:]) * 0.5, n, width=(b[1] - b[0]), bottom=0.0, color="#4C78A8", alpha=0.8)
		ax.set_theta_zero_location("E")
		ax.set_theta_direction(-1)
		ax.set_title("Edge orientation (rose)")
	else:
		plt.hist(angles_deg, bins=bins, range=(0, 180), color="#4C78A8", alpha=0.85)
		plt.xlabel("Theta (deg)")
		plt.ylabel("Count")
		plt.title("Edge orientation (hist)")
	plt.tight_layout()
	plt.savefig(out_png)
	plt.close()


