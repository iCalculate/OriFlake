from typing import Tuple

import cv2
import numpy as np


def kmeans_segment_lab(lab_img: np.ndarray, k: int = 3, attempts: int = 3) -> Tuple[np.ndarray, np.ndarray]:
	h, w, c = lab_img.shape
	data = lab_img.reshape(-1, c).astype(np.float32)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
	_, labels, centers = cv2.kmeans(
		data,
		k,
		None,
		criteria,
		attempts,
		cv2.KMEANS_PP_CENTERS,
	)
	labels = labels.reshape(h, w)
	return labels, centers


def select_candidate_mask(labels: np.ndarray) -> np.ndarray:
	mask = np.zeros_like(labels, dtype=np.uint8)
	best_area = 0
	best_label = 0
	for lbl in np.unique(labels):
		m = (labels == lbl).astype(np.uint8)
		cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		area = sum(cv2.contourArea(c) for c in cnts)
		if area > best_area:
			best_area = area
			best_label = int(lbl)
	mask = (labels == best_label).astype(np.uint8) * 255
	return mask


def morph_cleanup(mask: np.ndarray, open_ks: int = 3, close_ks: int = 5) -> np.ndarray:
	open_ks = max(1, int(open_ks))
	close_ks = max(1, int(close_ks))
	kopen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks))
	kclose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
	res = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kopen)
	res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kclose)
	return res


def threshold_fallback(gray: np.ndarray) -> np.ndarray:
	if gray.ndim == 3:
		gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
	_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return th


