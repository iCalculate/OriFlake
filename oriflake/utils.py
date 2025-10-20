import os
import glob
import math
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import yaml
from skimage import color


def read_config(path: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)
	return cfg or {}


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def list_images(folder: str, recursive: bool = True) -> List[str]:
	# Allow a single file path
	if os.path.isfile(folder):
		return [folder]
	patterns = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
	files: List[str] = []
	if recursive:
		for root, _dirs, _files in os.walk(folder):
			for p in patterns:
				files.extend(glob.glob(os.path.join(root, p)))
	else:
		for p in patterns:
			files.extend(glob.glob(os.path.join(folder, p)))
	return sorted(files)


def to_rgb(image: np.ndarray) -> np.ndarray:
	if image is None:
		raise ValueError("Input image is None")
	if len(image.shape) == 2:
		return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	if image.shape[2] == 4:
		return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def apply_bilateral(rgb: np.ndarray, d: int, sigC: float, sigS: float) -> np.ndarray:
	bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
	filtered = cv2.bilateralFilter(bgr, d, sigC, sigS)
	return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)


def white_balance_grayworld(rgb: np.ndarray) -> np.ndarray:
	avg = rgb.reshape(-1, 3).mean(axis=0)
	avg[avg == 0] = 1.0
	scale = avg.mean() / avg
	wb = np.clip((rgb.astype(np.float32) * scale).astype(np.float32), 0, 255)
	return wb.astype(np.uint8)


def preprocess_image(
	raw_bgr: np.ndarray,
	bilateral_d: int,
	bilateral_sigC: float,
	bilateral_sigS: float,
	apply_wb: bool = False,
	gamma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
	rgb = to_rgb(raw_bgr)
	if apply_wb:
		rgb = white_balance_grayworld(rgb)
	if bilateral_d > 0:
		rgb = apply_bilateral(rgb, bilateral_d, bilateral_sigC, bilateral_sigS)
	if gamma and abs(gamma - 1.0) > 1e-3:
		norm = (rgb.astype(np.float32) / 255.0) ** (1.0 / max(gamma, 1e-6))
		rgb = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
	rgb01 = rgb.astype(np.float32) / 255.0
	lab = color.rgb2lab(rgb01)
	return rgb, lab.astype(np.float32)


def angle_wrap_180(theta_deg: float) -> float:
	val = theta_deg % 180.0
	if val < 0:
		val += 180.0
	return val


def angle_delta_sym60(theta_deg: float, reference_deg: float) -> float:
	d = (theta_deg - reference_deg) % 60.0
	if d < 0:
		d += 60.0
	return min(d, 60.0 - d)


def gaussian_score(delta_deg: float, sigma_deg: float) -> float:
	s = max(sigma_deg, 1e-6)
	return float(math.exp(-((delta_deg / s) ** 2)))


