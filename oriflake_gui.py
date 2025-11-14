#!/usr/bin/env python3
"""
OriFlake GUI Application
A modern GUI for triangular MoS₂ flake orientation analysis
"""

import sys
import os
import tempfile
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import cv2
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QPushButton, QComboBox, QSlider, QSpinBox,
    QDoubleSpinBox, QGroupBox, QTabWidget, QTextEdit, QFileDialog,
    QProgressBar, QSplitter, QScrollArea, QFrame, QSizePolicy, QLayout, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QDragEnterEvent, QDropEvent

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

# Import OriFlake modules
from oriflake.main import process_image, save_rgb_histogram
from oriflake.utils import read_config


class AnalysisWorker(QThread):
    """Worker thread for image analysis to keep GUI responsive"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, image_path: str, config: Dict[str, Any], 
                 cached_result: Optional[Dict[str, Any]] = None,
                 start_from_step: str = "full"):
        super().__init__()
        self.image_path = image_path
        self.config = config
        self.cached_result = cached_result
        self.start_from_step = start_from_step  # "full", "crop", "threshold", "edges", "fitting"
        
    def run(self):
        try:
            if self.start_from_step == "full":
                self.progress.emit(10)
                result = process_image(self.image_path, self.config)
                self.progress.emit(100)
                self.finished.emit(result)
            else:
                # Incremental processing
                result = self.incremental_process()
                self.progress.emit(100)
                self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
    
    def incremental_process(self) -> Dict[str, Any]:
        """Incremental processing from a specific step"""
        from oriflake.main import (
            convert_to_8bit, crop_center_region, find_dual_color_peaks_from_histogram_full_image,
            find_dual_peaks, enhance_edges_robust, fit_line_segments
        )
        import cv2
        
        cached = self.cached_result or {}
        cfg = self.config
        
        # Step 1: Image loading and preprocessing (always needed if not cached)
        if "rgb_full" not in cached or "color1" not in cached or "color2" not in cached:
            self.progress.emit(5)
            img_bgr = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
            if img_bgr is None:
                raise FileNotFoundError(f"Failed to read image: {self.image_path}")
            
            img_bgr = convert_to_8bit(img_bgr)
            if len(img_bgr.shape) == 3:
                if img_bgr.shape[2] == 4:
                    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
                rgb_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            else:
                rgb_full = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
            
            color1, color2 = find_dual_color_peaks_from_histogram_full_image(rgb_full)
            cached["rgb_full"] = rgb_full
            cached["color1"] = color1
            cached["color2"] = color2
        else:
            rgb_full = cached["rgb_full"]
            color1 = cached["color1"]
            color2 = cached["color2"]
        
        # Step 2: Cropping (if crop_ratio changed or not cached)
        # Also need to recrop if we're starting from threshold/edges/fitting but rgb is missing
        need_crop = (self.start_from_step in ["full", "crop"] or 
                    "rgb" not in cached or
                    (self.start_from_step in ["threshold", "edges", "fitting"] and "rgb" not in cached))
        
        if need_crop:
            self.progress.emit(15)
            crop_ratio = float(cfg.get("crop_ratio", 0.3))
            img_bgr = convert_to_8bit(cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED))
            if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
            img_cropped = crop_center_region(img_bgr, crop_ratio)
            
            if len(img_cropped.shape) == 3:
                if img_cropped.shape[2] == 4:
                    img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGRA2BGR)
                rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(img_cropped, cv2.COLOR_GRAY2RGB)
            cached["rgb"] = rgb
        else:
            rgb = cached["rgb"]
        
        # Step 3: Threshold calculation (if threshold_offset changed or not cached)
        # Also need to recalculate if rgb was just recropped
        need_threshold = (self.start_from_step in ["full", "crop", "threshold"] or 
                         "blue_threshold" not in cached or need_crop)
        
        if need_threshold:
            self.progress.emit(25)
            blue_threshold = None
            if color1 is not None and color2 is not None:
                blue_channel = rgb[:, :, 2]
                hist_blue, bins_blue = np.histogram(blue_channel.flatten(), bins=256, range=(0, 256))
                bin_centers_blue = (bins_blue[:-1] + bins_blue[1:]) / 2
                peak1_idx, peak2_idx = find_dual_peaks(hist_blue, bin_centers_blue, min_distance=10)
                
                if peak1_idx is not None and peak2_idx is not None:
                    peak1_val = bin_centers_blue[peak1_idx]
                    peak2_val = bin_centers_blue[peak2_idx]
                    blue_threshold = (peak1_val + peak2_val) / 2.0
                    threshold_offset = float(cfg.get("threshold_offset", 0.0))
                    blue_threshold = blue_threshold + threshold_offset
                    blue_threshold = max(0.0, min(255.0, blue_threshold))
            cached["blue_threshold"] = blue_threshold
        else:
            blue_threshold = cached["blue_threshold"]
        
        # Step 4: Edge extraction (if edge parameters changed or not cached)
        # Also need to re-extract if threshold or rgb changed
        need_edges = (self.start_from_step in ["full", "crop", "threshold", "edges"] or 
                     "edges" not in cached or need_threshold)
        
        if need_edges:
            self.progress.emit(50)
            min_contour_area = int(cfg.get("min_contour_area", 100))
            morph_open_ks = int(cfg.get("morph_open_ks", 3))
            morph_close_ks = int(cfg.get("morph_close_ks", 0))
            boundary_kernel_size = int(cfg.get("boundary_kernel_size", 3))
            boundary_dilate_iterations = int(cfg.get("boundary_dilate_iterations", 1))
            edge_cleanup_kernel_size = int(cfg.get("edge_cleanup_kernel_size", 2))
            edge_cleanup_iterations = int(cfg.get("edge_cleanup_iterations", 1))
            border_margin = int(cfg.get("border_margin", 3))
            border_area_threshold = float(cfg.get("border_area_threshold", 0.1))
            border_width_threshold = float(cfg.get("border_width_threshold", 0.8))
            
            edges, binary_image = enhance_edges_robust(
                rgb, color1=color1, color2=color2, blue_threshold=blue_threshold,
                min_contour_area=min_contour_area,
                morph_open_ks=morph_open_ks, morph_close_ks=morph_close_ks,
                boundary_kernel_size=boundary_kernel_size, boundary_dilate_iterations=boundary_dilate_iterations,
                edge_cleanup_kernel_size=edge_cleanup_kernel_size, edge_cleanup_iterations=edge_cleanup_iterations,
                border_margin=border_margin, border_area_threshold=border_area_threshold,
                border_width_threshold=border_width_threshold
            )
            cached["edges"] = edges
            cached["binary"] = binary_image
        else:
            edges = cached["edges"]
            binary_image = cached["binary"]
        
        # Step 5: Line fitting (if fitting parameters changed or not cached)
        # Also need to re-fit if edges changed
        need_fitting = (self.start_from_step in ["full", "crop", "threshold", "edges", "fitting"] or 
                       "orientations" not in cached or need_edges)
        
        if need_fitting:
            self.progress.emit(75)
            min_line_length = float(cfg.get("min_line_length", 20.0))
            max_line_gap = float(cfg.get("max_line_gap", 10.0))
            hough_threshold = int(cfg.get("hough_threshold", 50))
            min_segment_distance = float(cfg.get("min_segment_distance", 30.0))
            
            orientations = fit_line_segments(edges, min_line_length, max_line_gap, hough_threshold, min_segment_distance)
            angles = [o[0] for o in orientations]
            cached["orientations"] = orientations
            cached["angles"] = angles
        else:
            orientations = cached["orientations"]
            angles = cached["angles"]
        
        # Step 6: Visualization (always regenerate)
        self.progress.emit(90)
        overlay = rgb.copy()
        overlay[edges > 0] = [255, 0, 0]
        
        for angle, cx, cy in orientations:
            length = 50
            angle_rad = np.radians(angle)
            x1 = int(cx - length * np.cos(angle_rad))
            y1 = int(cy + length * np.sin(angle_rad))
            x2 = int(cx + length * np.cos(angle_rad))
            y2 = int(cy - length * np.sin(angle_rad))
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        rows = []
        for i, (angle, cx, cy) in enumerate(orientations):
            rows.append({
                "segment_id": i,
                "theta_deg": float(angle),
                "center_x": float(cx),
                "center_y": float(cy),
            })
        
        blue_channel = rgb[:, :, 2]
        
        result = {
            "overlay": overlay,
            "edges": edges,
            "rgb": rgb,
            "blue_gray": blue_channel,
            "binary": binary_image,
            "color1": color1,
            "color2": color2,
            "blue_threshold": blue_threshold,
            "rows": rows,
            "angles": angles,
        }
        
        return result


class MatplotlibWidget(FigureCanvas):
    """Custom matplotlib widget for publication-quality plots"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
    def plot_histogram(self, rgb: np.ndarray, blue_threshold: Optional[float], 
                      peak1_val: Optional[float] = None, peak2_val: Optional[float] = None):
        """Plot RGB histogram with blue channel peaks and threshold"""
        self.fig.clear()
        
        # Create subplots for R, G, B channels
        axes = self.fig.subplots(3, 1)
        channel_names = ['Red', 'Green', 'Blue']
        channel_colors = ['red', 'green', 'blue']
        channels = [rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]]
        
        for i, (ax, name, color, channel) in enumerate(zip(axes, channel_names, channel_colors, channels)):
            hist, bins = np.histogram(channel.flatten(), bins=256, range=(0, 256))
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            ax.bar(bin_centers, hist, width=1.0, alpha=0.6, color=color, 
                   edgecolor='black', linewidth=0.3)
            
            # Mark blue channel peaks and threshold (only in blue channel)
            if i == 2:  # Blue channel
                if peak1_val is not None and peak2_val is not None:
                    ax.axvline(x=peak1_val, color='orange', linestyle='--', linewidth=2, 
                              label=f'Peak 1: {int(peak1_val)}')
                    ax.axvline(x=peak2_val, color='purple', linestyle='--', linewidth=2, 
                              label=f'Peak 2: {int(peak2_val)}')
                
                if blue_threshold is not None:
                    threshold_val = int(blue_threshold)
                    ax.axvline(x=threshold_val, color='green', linestyle='-', linewidth=2.5, 
                              label=f'Threshold: {threshold_val:.1f}')
            
            ax.set_xlabel(f'{name} Channel Value', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.set_title(f'{name} Channel Histogram', fontsize=11, fontweight='bold')
            if i == 2 and ((peak1_val is not None and peak2_val is not None) or blue_threshold is not None):
                ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 255)
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_orientation_histogram(self, angles: list):
        """Plot orientation distribution histogram"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        if not angles or len(angles) == 0:
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            self.draw()
            return
        
        try:
            bins_count = 36
            n_hist, bins_hist, patches = ax.hist(angles, bins=bins_count, range=(0, 180), 
                                                alpha=0.7, color='steelblue', 
                                                edgecolor='navy', linewidth=0.5)
            
            ax.set_xlabel('Orientation Angle (degrees)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_title('Edge Orientation Distribution', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f'Total Edges: {len(angles)}\n'
            stats_text += f'Mean: {np.mean(angles):.1f}°\n'
            stats_text += f'Std: {np.std(angles):.1f}°'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting: {str(e)}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=10, color='red')
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_rose_diagram(self, angles: list):
        """Create rose diagram for orientation distribution"""
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection='polar')
        
        if not angles or len(angles) == 0:
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            self.draw()
            return
        
        try:
            bins_count = 36
            theta = np.deg2rad(np.array(angles))
            theta_bins = np.linspace(0, 2*np.pi, bins_count+1)
            
            n_rose, b_rose = np.histogram(theta, bins=theta_bins)
            bin_centers_rose = (b_rose[:-1] + b_rose[1:]) / 2
            width = 2 * np.pi / bins_count
            
            bars = ax.bar(bin_centers_rose, n_rose, width=width, alpha=0.7, 
                         color='steelblue', edgecolor='navy', linewidth=0.5)
            
            ax.set_theta_zero_location('E')
            ax.set_theta_direction(-1)
            ax.set_title('Edge Orientation Rose Diagram', fontsize=12, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting: {str(e)}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=10, color='red')
        
        self.fig.tight_layout()
        self.draw()


class ParameterControl(QWidget):
    """Single parameter control with label, slider, spinbox, and range/step controls"""
    
    def __init__(self, name: str, param_type: str, min_val: float, max_val: float, 
                 default_val: float, description: str, default_step: float = None, parent=None):
        super().__init__(parent)
        self.name = name
        self.param_type = param_type  # 'int' or 'float'
        self.min_val = min_val
        self.max_val = max_val
        self.description = description
        if default_step is None:
            self.default_step = 1.0 if param_type == 'int' else 0.01
        else:
            self.default_step = default_step
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Main control row
        main_row = QHBoxLayout()
        main_row.setContentsMargins(0, 0, 0, 0)
        main_row.setSpacing(6)
        
        # Label
        self.label = QLabel(f"{name}:")
        self.label.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        self.label.setToolTip(description)
        main_row.addWidget(self.label)
        
        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        if param_type == 'int':
            self.slider.setRange(int(min_val), int(max_val))
            self.slider.setValue(int(default_val))
        else:
            # For float, scale by 100
            self.slider.setRange(int(min_val * 100), int(max_val * 100))
            self.slider.setValue(int(default_val * 100))
        self.slider.setMinimumWidth(100)
        self.slider.setToolTip(description)
        main_row.addWidget(self.slider)
        
        # SpinBox
        if param_type == 'int':
            self.spin = QSpinBox()
            self.spin.setRange(int(min_val), int(max_val))
            self.spin.setValue(int(default_val))
            self.spin.setSingleStep(int(self.default_step))
        else:
            self.spin = QDoubleSpinBox()
            self.spin.setRange(min_val, max_val)
            self.spin.setValue(default_val)
            self.spin.setDecimals(2)
            self.spin.setSingleStep(self.default_step)
        self.spin.setButtonSymbols(QSpinBox.ButtonSymbols.PlusMinus)
        self.spin.setMinimumWidth(70)
        self.spin.setMaximumWidth(70)
        self.spin.setToolTip(description)
        main_row.addWidget(self.spin)
        
        layout.addLayout(main_row)
        
        # Range and step controls row
        range_row = QHBoxLayout()
        range_row.setContentsMargins(0, 0, 0, 0)
        range_row.setSpacing(4)
        
        # Min value control
        min_label = QLabel("Min:")
        min_label.setStyleSheet("color: #b0b0b0; font-size: 9px;")
        range_row.addWidget(min_label)
        if param_type == 'int':
            self.min_spin = QSpinBox()
            self.min_spin.setRange(-999999, 999999)
            self.min_spin.setValue(int(min_val))
            self.min_spin.setSingleStep(int(self.default_step))
        else:
            self.min_spin = QDoubleSpinBox()
            self.min_spin.setRange(-999999.0, 999999.0)
            self.min_spin.setValue(min_val)
            self.min_spin.setDecimals(2)
            self.min_spin.setSingleStep(self.default_step)
        self.min_spin.setMinimumWidth(50)
        self.min_spin.setMaximumWidth(50)
        self.min_spin.setToolTip("Minimum value")
        range_row.addWidget(self.min_spin)
        
        # Max value control
        max_label = QLabel("Max:")
        max_label.setStyleSheet("color: #b0b0b0; font-size: 9px;")
        range_row.addWidget(max_label)
        if param_type == 'int':
            self.max_spin = QSpinBox()
            self.max_spin.setRange(-999999, 999999)
            self.max_spin.setValue(int(max_val))
            self.max_spin.setSingleStep(int(self.default_step))
        else:
            self.max_spin = QDoubleSpinBox()
            self.max_spin.setRange(-999999.0, 999999.0)
            self.max_spin.setValue(max_val)
            self.max_spin.setDecimals(2)
            self.max_spin.setSingleStep(self.default_step)
        self.max_spin.setMinimumWidth(50)
        self.max_spin.setMaximumWidth(50)
        self.max_spin.setToolTip("Maximum value")
        range_row.addWidget(self.max_spin)
        
        # Step control
        step_label = QLabel("Step:")
        step_label.setStyleSheet("color: #b0b0b0; font-size: 9px;")
        range_row.addWidget(step_label)
        if param_type == 'int':
            self.step_spin = QSpinBox()
            self.step_spin.setRange(1, 1000)
            self.step_spin.setValue(int(self.default_step))
        else:
            self.step_spin = QDoubleSpinBox()
            self.step_spin.setRange(0.001, 10.0)
            self.step_spin.setValue(self.default_step)
            self.step_spin.setDecimals(3)
        self.step_spin.setMinimumWidth(50)
        self.step_spin.setMaximumWidth(50)
        self.step_spin.setToolTip("Step size for fine adjustment")
        range_row.addWidget(self.step_spin)
        
        range_row.addStretch()
        layout.addLayout(range_row)
        
        # Connect signals
        if param_type == 'int':
            self.slider.valueChanged.connect(self.spin.setValue)
            self.spin.valueChanged.connect(self.slider.setValue)
            self.min_spin.valueChanged.connect(self.update_range)
            self.max_spin.valueChanged.connect(self.update_range)
            self.step_spin.valueChanged.connect(self.update_step)
        else:
            self.slider.valueChanged.connect(lambda v: self.spin.setValue(v / 100.0))
            self.spin.valueChanged.connect(lambda v: self.slider.setValue(int(v * 100)))
            self.min_spin.valueChanged.connect(self.update_range)
            self.max_spin.valueChanged.connect(self.update_range)
            self.step_spin.valueChanged.connect(self.update_step)
        
        self.setLayout(layout)
    
    def update_range(self):
        """Update slider and spinbox range"""
        min_val = self.min_spin.value()
        max_val = self.max_spin.value()
        
        if min_val >= max_val:
            return  # Invalid range
        
        self.min_val = min_val
        self.max_val = max_val
        
        current_val = self.spin.value()
        if self.param_type == 'int':
            self.slider.setRange(int(min_val), int(max_val))
            self.spin.setRange(int(min_val), int(max_val))
            # Clamp current value
            if current_val < min_val:
                self.spin.setValue(int(min_val))
            elif current_val > max_val:
                self.spin.setValue(int(max_val))
        else:
            self.slider.setRange(int(min_val * 100), int(max_val * 100))
            self.spin.setRange(min_val, max_val)
            # Clamp current value
            if current_val < min_val:
                self.spin.setValue(min_val)
            elif current_val > max_val:
                self.spin.setValue(max_val)
    
    def update_step(self):
        """Update spinbox step"""
        step = self.step_spin.value()
        self.spin.setSingleStep(step)
        self.default_step = step
    
    def get_value(self):
        """Get current parameter value"""
        return self.spin.value()
    
    def set_value(self, value):
        """Set parameter value"""
        if self.param_type == 'int':
            self.spin.setValue(int(value))
        else:
            self.spin.setValue(float(value))
    
    def set_range(self, min_val: float, max_val: float):
        """Update parameter range"""
        self.min_val = min_val
        self.max_val = max_val
        self.min_spin.setValue(min_val)
        self.max_spin.setValue(max_val)
        self.update_range()


class ParameterPanel(QWidget):
    """Parameter control panel with categorized parameters"""
    
    # Signal emitted when parameters change
    parameters_changed = pyqtSignal()
    
    # Parameter definitions with (name, type, min, max, default, description)
    PARAM_DEFINITIONS = {
        "Image Preprocessing": [
            ("crop_ratio", "float", 0.0, 0.5, 0.3, "Crop ratio, 0.3 means crop 30% (15% on each side), only process center region"),
            ("threshold_offset", "float", -50.0, 50.0, 0.0, "Threshold offset to adjust the binary threshold calculated from histogram peaks (positive = higher threshold, negative = lower threshold)"),
        ],
        "Segmentation Parameters": [
            ("morph_open_ks", "int", 0, 20, 3, "Morphological opening kernel size for cleaning segmentation results (remove noise)"),
            ("morph_close_ks", "int", 0, 20, 0, "Morphological closing kernel size (0 means no closing to avoid connecting independent structures)"),
        ],
        "Boundary Extraction Parameters": [
            ("boundary_kernel_size", "int", 1, 10, 3, "Morphological kernel size for boundary extraction"),
            ("boundary_dilate_iterations", "int", 1, 5, 1, "Dilation/erosion iterations for boundary extraction"),
        ],
        "Edge Cleanup Parameters": [
            ("edge_cleanup_kernel_size", "int", 0, 10, 2, "Kernel size for edge cleanup"),
            ("edge_cleanup_iterations", "int", 0, 5, 1, "Iterations for edge cleanup"),
        ],
        "Boundary Filtering Parameters": [
            ("min_contour_area", "int", 10, 2000, 100, "Minimum contour area (pixels) for filtering small edge connected components"),
            ("border_margin", "int", 0, 50, 3, "Image border filtering pixels (exclude how many pixels from image edge)"),
            ("border_area_threshold", "float", 0.0, 1.0, 0.1, "Image border area threshold (ratio relative to total image area)"),
            ("border_width_threshold", "float", 0.0, 1.0, 0.8, "Image border width threshold (ratio relative to image width)"),
        ],
        "Line Fitting Parameters": [
            ("min_line_length", "float", 5.0, 200.0, 20.0, "Minimum line segment length (pixels), strict threshold, short edges will be filtered"),
            ("max_line_gap", "float", 1.0, 100.0, 10.0, "Maximum line gap (pixels)"),
            ("hough_threshold", "int", 10, 500, 50, "Hough transform threshold"),
            ("min_segment_distance", "float", 5.0, 200.0, 30.0, "Minimum segment distance (pixels) to avoid duplicate counting of the same long edge"),
        ],
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.param_controls = {}  # Store parameter controls
        self.init_ui()
        self.connect_signals()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Import/Export buttons
        import_export_layout = QHBoxLayout()
        self.import_btn = QPushButton("📥 Import Config")
        self.export_btn = QPushButton("📤 Export Config")
        self.import_btn.setToolTip("Import parameter configuration from YAML file")
        self.export_btn.setToolTip("Export current parameter configuration to YAML file")
        import_export_layout.addWidget(self.import_btn)
        import_export_layout.addWidget(self.export_btn)
        import_export_layout.addStretch()
        layout.addLayout(import_export_layout)
        
        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        params_widget = QWidget()
        params_layout = QVBoxLayout()
        params_layout.setSpacing(10)
        
        # Create parameter groups
        for category, params in self.PARAM_DEFINITIONS.items():
            group = QGroupBox(category)
            group_layout = QGridLayout()
            group_layout.setSpacing(8)
            group_layout.setContentsMargins(10, 5, 10, 5)
            group_layout.setColumnStretch(0, 1)
            group_layout.setColumnStretch(1, 2)
            
            row = 0
            for name, param_type, min_val, max_val, default_val, description in params:
                # Create parameter control
                control = ParameterControl(name, param_type, min_val, max_val, default_val, description)
                self.param_controls[name] = control
                
                # Add to layout
                group_layout.addWidget(control.label, row, 0)
                group_layout.addWidget(control, row, 1)
                row += 1
            
            group.setLayout(group_layout)
            params_layout.addWidget(group)
        
        params_widget.setLayout(params_layout)
        scroll.setWidget(params_widget)
        layout.addWidget(scroll)
        
        # Auto-run checkbox
        auto_run_layout = QHBoxLayout()
        self.auto_run_checkbox = QCheckBox("Auto-run on parameter change")
        self.auto_run_checkbox.setChecked(False)
        auto_run_layout.addWidget(self.auto_run_checkbox)
        auto_run_layout.addStretch()
        layout.addLayout(auto_run_layout)
        
        # Analysis button
        self.analyze_btn = QPushButton("Analyze Image")
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        layout.addWidget(self.analyze_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        
        self.setLayout(layout)
        
    def connect_signals(self):
        """Connect parameter change signals"""
        for control in self.param_controls.values():
            control.slider.valueChanged.connect(self.on_parameter_changed)
            control.spin.valueChanged.connect(self.on_parameter_changed)
        
        self.import_btn.clicked.connect(self.import_config)
        self.export_btn.clicked.connect(self.export_config)
        
    def on_parameter_changed(self):
        """Handle parameter changes"""
        self.parameters_changed.emit()
        
    def get_config(self) -> Dict[str, Any]:
        """Get current parameter configuration"""
        config = {}
        for name, control in self.param_controls.items():
            config[name] = control.get_value()
        
        # Add default values for other parameters
        config.setdefault("verbose", False)
        config.setdefault("preview", False)
        
        return config
    
    def set_config(self, config: Dict[str, Any]):
        """Set parameter values from config"""
        for name, value in config.items():
            if name in self.param_controls:
                self.param_controls[name].set_value(value)
    
    def import_config(self):
        """Import configuration from YAML file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Config", "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self.set_config(config)
                QApplication.instance().processEvents()  # Update UI
            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Import Failed", f"Failed to import configuration file:\n{str(e)}")
    
    def export_config(self):
        """Export current configuration to YAML file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Config", "oriflake_config.yaml", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if file_path:
            try:
                config = self.get_config()
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Export Success", f"Configuration saved to:\n{file_path}")
            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Export Failed", f"Failed to export configuration file:\n{str(e)}")


class ImageInputWidget(QWidget):
    """Simple image input widget with path input and file browser"""
    
    image_loaded = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)
        
        # Path input
        path_layout = QHBoxLayout()
        path_label = QLabel("Image Path:")
        path_label.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        path_layout.addWidget(path_label)
        
        self.path_input = QTextEdit()
        self.path_input.setMaximumHeight(60)
        self.path_input.setPlaceholderText("Enter image path or click Browse to select file...")
        self.path_input.setToolTip("Supported formats: PNG, JPG, TIFF, BMP")
        path_layout.addWidget(self.path_input)
        
        # Browse button
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setToolTip("Select image file")
        self.browse_btn.clicked.connect(self.browse_file)
        path_layout.addWidget(self.browse_btn)
        
        layout.addLayout(path_layout)
        
        # Info label
        self.info_label = QLabel("No image loaded")
        self.info_label.setStyleSheet("""
            QLabel {
                color: #b0b0b0;
                font-size: 11px;
                font-style: italic;
                padding: 5px;
            }
        """)
        layout.addWidget(self.info_label)
        
        self.setLayout(layout)
        
    def browse_file(self):
        """Open file dialog to select image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*)"
        )
        if file_path:
            self.path_input.setPlainText(file_path)
            self.on_path_changed()
    
    def on_path_changed(self):
        """Handle path change"""
        path = self.path_input.toPlainText().strip()
        if path and self.is_valid_image(path):
            if os.path.exists(path):
                self.info_label.setText(f"✅ {Path(path).name}")
                self.info_label.setStyleSheet("""
                    QLabel {
                        color: #4CAF50;
                        font-size: 11px;
                        padding: 5px;
                    }
                """)
                self.image_loaded.emit(path)
            else:
                self.info_label.setText("❌ File not found")
                self.info_label.setStyleSheet("""
                    QLabel {
                        color: #f44336;
                        font-size: 11px;
                        padding: 5px;
                    }
                """)
        else:
            self.info_label.setText("No image loaded")
            self.info_label.setStyleSheet("""
                QLabel {
                    color: #b0b0b0;
                    font-size: 11px;
                    font-style: italic;
                    padding: 5px;
                }
            """)
    
    def is_valid_image(self, path: str) -> bool:
        """Check if file is a valid image"""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        return Path(path).suffix.lower() in valid_extensions
    
    def get_path(self) -> Optional[str]:
        """Get current image path"""
        path = self.path_input.toPlainText().strip()
        if path and self.is_valid_image(path) and os.path.exists(path):
            return path
        return None


class OriFlakeGUI(QMainWindow):
    """Main GUI application for OriFlake"""
    
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.analysis_result = None
        self.cached_result = None  # Cache intermediate results for incremental processing
        self.last_config = None  # Track last config to detect parameter changes
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("OriFlake - Triangular MoS₂ Orientation Analysis Tool")
        self.setFixedSize(1800, 1100)
        
        # Set dark theme style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QWidget {
                color: #e0e0e0;
                background-color: #2b2b2b;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                color: #e0e0e0;
                border: 2px solid #555555;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 15px;
                background-color: #353535;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                color: #4CAF50;
                background-color: #353535;
            }
            QLabel {
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #404040;
                color: #e0e0e0;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #4CAF50;
                border: 1px solid #4CAF50;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #404040;
                color: #e0e0e0;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px 6px;
                min-width: 70px;
                font-size: 11px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover {
                border: 1px solid #4CAF50;
                background-color: #4a4a4a;
            }
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 6px;
                background: #404040;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #45a049;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #66bb6a;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                border-radius: 5px;
                background-color: #353535;
            }
            QTabBar::tab {
                background-color: #404040;
                color: #e0e0e0;
                border: 1px solid #555555;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #353535;
                color: #4CAF50;
                border-bottom-color: #353535;
            }
            QTextEdit {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #555555;
                border-radius: 5px;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 4px;
                text-align: center;
                color: #e0e0e0;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left panel (parameters and image)
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        
        # Image input
        image_group = QGroupBox("📷 Image Input")
        image_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        image_layout = QVBoxLayout()
        
        self.image_input = ImageInputWidget()
        self.image_input.image_loaded.connect(self.on_image_loaded)
        # Connect text change signal
        self.image_input.path_input.textChanged.connect(self.image_input.on_path_changed)
        
        image_layout.addWidget(self.image_input)
        image_group.setLayout(image_layout)
        left_panel.addWidget(image_group, 0)
        
        # Parameter panel
        self.param_panel = ParameterPanel()
        self.param_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.param_panel.analyze_btn.clicked.connect(self.analyze_image)
        self.param_panel.parameters_changed.connect(self.on_parameters_changed)
        left_panel.addWidget(self.param_panel, 2)
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                font-weight: bold;
                padding: 5px;
                background-color: #1e3a1e;
                border: 1px solid #4CAF50;
                border-radius: 3px;
            }
        """)
        left_panel.addWidget(self.status_label, 0)
        
        # Right panel (results)
        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(0, 0, 0, 0)
        
        # Results tabs
        self.results_tabs = QTabWidget()
        
        # Processing results tab
        processing_widget = QWidget()
        processing_layout = QVBoxLayout()
        
        # Histogram tab
        self.histogram_widget = MatplotlibWidget(self, width=8, height=6)
        processing_layout.addWidget(self.histogram_widget)
        
        processing_widget.setLayout(processing_layout)
        self.results_tabs.addTab(processing_widget, "📊 Histogram")
        
        # Binary image tab
        binary_widget = QWidget()
        binary_layout = QVBoxLayout()
        self.binary_label = QLabel("No analysis performed")
        self.binary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.binary_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #555555;
                background-color: #2b2b2b;
                min-height: 400px;
                border-radius: 10px;
                font-size: 14px;
                color: #e0e0e0;
            }
        """)
        self.binary_label.setScaledContents(False)
        binary_layout.addWidget(self.binary_label)
        binary_widget.setLayout(binary_layout)
        self.results_tabs.addTab(binary_widget, "⚫ Binary")
        
        # Edges tab
        edges_widget = QWidget()
        edges_layout = QVBoxLayout()
        self.edges_label = QLabel("No analysis performed")
        self.edges_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.edges_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #555555;
                background-color: #2b2b2b;
                min-height: 400px;
                border-radius: 10px;
                font-size: 14px;
                color: #e0e0e0;
            }
        """)
        self.edges_label.setScaledContents(False)
        edges_layout.addWidget(self.edges_label)
        edges_widget.setLayout(edges_layout)
        self.results_tabs.addTab(edges_widget, "🔲 Edges")
        
        # Overlay tab
        overlay_widget = QWidget()
        overlay_layout = QVBoxLayout()
        overlay_controls = QHBoxLayout()
        overlay_zoom_in = QPushButton("🔍+")
        overlay_zoom_out = QPushButton("🔍-")
        overlay_reset = QPushButton("↻ Reset")
        overlay_save = QPushButton("💾 Save")
        overlay_controls.addWidget(overlay_zoom_in)
        overlay_controls.addWidget(overlay_zoom_out)
        overlay_controls.addWidget(overlay_reset)
        overlay_controls.addWidget(overlay_save)
        overlay_controls.addStretch()
        
        self.overlay_label = QLabel("📊 No analysis performed\n\nLoad an image and click 'Analyze Image' to begin")
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #555555;
                background-color: #2b2b2b;
                min-height: 400px;
                border-radius: 10px;
                font-size: 14px;
                color: #e0e0e0;
            }
        """)
        self.overlay_label.setScaledContents(False)
        self.overlay_original_pixmap = None
        self.overlay_scale_factor = 1.0
        
        overlay_zoom_in.clicked.connect(self.overlay_zoom_in_func)
        overlay_zoom_out.clicked.connect(self.overlay_zoom_out_func)
        overlay_reset.clicked.connect(self.overlay_reset_zoom)
        overlay_save.clicked.connect(self.save_overlay)
        
        overlay_layout.addLayout(overlay_controls)
        overlay_layout.addWidget(self.overlay_label)
        overlay_widget.setLayout(overlay_layout)
        self.results_tabs.addTab(overlay_widget, "🔍 Edge Fitting")
        
        # Statistics tab
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.results_tabs.addTab(self.stats_text, "📊 Statistics")
        
        # Orientation histogram tab
        self.orientation_hist_widget = MatplotlibWidget(self, width=8, height=6)
        self.results_tabs.addTab(self.orientation_hist_widget, "📈 Orientation Distribution")
        
        # Rose diagram tab
        self.rose_widget = MatplotlibWidget(self, width=8, height=6)
        self.results_tabs.addTab(self.rose_widget, "🌹 Rose Diagram")
        
        right_panel.addWidget(self.results_tabs)
        
        # Splitter with fixed sizes
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMinimumWidth(600)
        left_widget.setMaximumWidth(600)
        splitter.addWidget(left_widget)
        
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        right_widget.setMinimumWidth(1200)
        right_widget.setMaximumWidth(1200)
        splitter.addWidget(right_widget)
        
        splitter.setSizes([600, 1200])
        # Disable resizing by setting stretch factors to 0
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 0)
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        
    def on_image_loaded(self, image_path: str):
        """Handle image loading"""
        self.current_image_path = image_path
        self.param_panel.analyze_btn.setEnabled(True)
        
        # Reset cache when new image is loaded
        self.cached_result = None
        self.last_config = None
        
        self.status_label.setText("✅ Image loaded - Ready for analysis")
        
    def on_parameters_changed(self):
        """Handle parameter changes - auto-run if enabled with incremental processing"""
        image_path = self.image_input.get_path()
        if self.param_panel.auto_run_checkbox.isChecked() and image_path:
            if not hasattr(self, '_auto_run_timer'):
                self._auto_run_timer = QTimer()
                self._auto_run_timer.setSingleShot(True)
                self._auto_run_timer.timeout.connect(self.incremental_analyze)
            self._auto_run_timer.stop()
            self._auto_run_timer.start(500)
    
    def determine_start_step(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> str:
        """Determine which step to start from based on parameter changes"""
        if old_config is None:
            return "full"
        
        # Parameters that affect cropping (step 2)
        if old_config.get("crop_ratio") != new_config.get("crop_ratio"):
            return "crop"
        
        # Parameters that affect threshold (step 3)
        if old_config.get("threshold_offset") != new_config.get("threshold_offset"):
            return "threshold"
        
        # Parameters that affect edge extraction (step 4)
        edge_params = [
            "morph_open_ks", "morph_close_ks",
            "boundary_kernel_size", "boundary_dilate_iterations",
            "edge_cleanup_kernel_size", "edge_cleanup_iterations",
            "min_contour_area", "border_margin",
            "border_area_threshold", "border_width_threshold"
        ]
        if any(old_config.get(p) != new_config.get(p) for p in edge_params):
            return "edges"
        
        # Parameters that affect line fitting (step 5)
        fitting_params = [
            "min_line_length", "max_line_gap",
            "hough_threshold", "min_segment_distance"
        ]
        if any(old_config.get(p) != new_config.get(p) for p in fitting_params):
            return "fitting"
        
        # No parameter changes detected, skip processing
        return "skip"
    
    def incremental_analyze(self):
        """Perform incremental analysis based on parameter changes"""
        image_path = self.image_input.get_path()
        if not image_path:
            return
        
        # Check if we have cached results
        if self.cached_result is None:
            # No cache, do full analysis
            self.analyze_image()
            return
        
        self.current_image_path = image_path
        self.param_panel.analyze_btn.setEnabled(False)
        self.param_panel.progress_bar.setValue(0)
        
        new_config = self.param_panel.get_config()
        start_step = self.determine_start_step(self.last_config, new_config)
        
        # Skip if no changes
        if start_step == "skip":
            self.param_panel.analyze_btn.setEnabled(True)
            return
        
        self.last_config = new_config.copy()
        
        self.status_label.setText(f"🔄 Incremental analysis ({start_step})...")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #ffc107;
                font-weight: bold;
                padding: 5px;
                background-color: #3a2e1e;
                border: 1px solid #ffc107;
                border-radius: 3px;
            }
        """)
        
        self.worker = AnalysisWorker(
            self.current_image_path, 
            new_config,
            cached_result=self.cached_result,
            start_from_step=start_step
        )
        self.worker.progress.connect(self.param_panel.progress_bar.setValue)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()
        
    def analyze_image(self):
        """Perform full image analysis from scratch"""
        # Get path from image input widget
        image_path = self.image_input.get_path()
        if not image_path:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", "Please select or enter a valid image path first")
            return
        
        self.current_image_path = image_path
        self.param_panel.analyze_btn.setEnabled(False)
        self.param_panel.progress_bar.setValue(0)
        
        # Reset cache for full analysis
        self.cached_result = None
        self.last_config = None
        
        self.status_label.setText("🔄 Analyzing image (full)...")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #ffc107;
                font-weight: bold;
                padding: 5px;
                background-color: #3a2e1e;
                border: 1px solid #ffc107;
                border-radius: 3px;
            }
        """)
        
        config = self.param_panel.get_config()
        self.last_config = config.copy()
        
        self.worker = AnalysisWorker(self.current_image_path, config, start_from_step="full")
        self.worker.progress.connect(self.param_panel.progress_bar.setValue)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()
        
    def on_analysis_finished(self, result: Dict[str, Any]):
        """Handle analysis completion"""
        self.analysis_result = result
        self.param_panel.progress_bar.setValue(100)
        self.param_panel.analyze_btn.setEnabled(True)
        
        # Update cache with intermediate results for incremental processing
        # Preserve rgb_full from cache if available (not returned by process_image)
        cached_rgb_full = None
        if self.cached_result and "rgb_full" in self.cached_result:
            cached_rgb_full = self.cached_result["rgb_full"]
        
        self.cached_result = {
            "rgb_full": cached_rgb_full,  # Preserve from previous cache
            "color1": result.get("color1"),
            "color2": result.get("color2"),
            "rgb": result.get("rgb"),
            "blue_threshold": result.get("blue_threshold"),
            "edges": result.get("edges"),
            "binary": result.get("binary"),
            "orientations": result.get("orientations"),
            "angles": result.get("angles"),
        }
        
        num_segments = len(result.get('rows', []))
        self.status_label.setText(f"✅ Analysis complete - {num_segments} edge segments detected")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                font-weight: bold;
                padding: 5px;
                background-color: #1e3a1e;
                border: 1px solid #4CAF50;
                border-radius: 3px;
            }
        """)
        
        # Update results
        self.update_processing_results(result)
        self.update_statistics(result)
        self.update_orientation_plots(result)
        
    def on_analysis_error(self, error_msg: str):
        """Handle analysis error"""
        self.param_panel.progress_bar.setValue(0)
        self.param_panel.analyze_btn.setEnabled(True)
        
        self.status_label.setText("❌ Analysis failed")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #f44336;
                font-weight: bold;
                padding: 5px;
                background-color: #3a1e1e;
                border: 1px solid #f44336;
                border-radius: 3px;
            }
        """)
        
        self.stats_text.setText(f"❌ Error: {error_msg}")
        
    def update_processing_results(self, result: Dict[str, Any]):
        """Update processing result displays"""
        # Histogram - find blue channel peaks for threshold
        rgb = result.get("rgb")
        blue_threshold = result.get("blue_threshold")
        peak1_val = None
        peak2_val = None
        
        if rgb is not None:
            # Find two peaks in blue channel (same as in main.py)
            from oriflake.main import find_dual_peaks
            blue_channel = rgb[:, :, 2]
            hist_blue, bins_blue = np.histogram(blue_channel.flatten(), bins=256, range=(0, 256))
            bin_centers_blue = (bins_blue[:-1] + bins_blue[1:]) / 2
            
            peak1_idx, peak2_idx = find_dual_peaks(hist_blue, bin_centers_blue, min_distance=10)
            
            if peak1_idx is not None and peak2_idx is not None:
                peak1_val = bin_centers_blue[peak1_idx]
                peak2_val = bin_centers_blue[peak2_idx]
            
            self.histogram_widget.plot_histogram(rgb, blue_threshold, peak1_val, peak2_val)
        
        # Binary image
        binary = result.get("binary")
        if binary is not None:
            binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            h, w = binary_rgb.shape[:2]
            bytes_per_line = 3 * w
            qt_image = QImage(binary_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.binary_label.setPixmap(pixmap.scaled(
                self.binary_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        
        # Edges
        edges = result.get("edges")
        if edges is not None:
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            h, w = edges_rgb.shape[:2]
            bytes_per_line = 3 * w
            qt_image = QImage(edges_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.edges_label.setPixmap(pixmap.scaled(
                self.edges_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        
        # Overlay
        overlay = result.get("overlay")
        if overlay is not None:
            overlay_rgb = overlay
            if len(overlay_rgb.shape) == 3:
                h, w, ch = overlay_rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(overlay_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.overlay_original_pixmap = QPixmap.fromImage(qt_image)
                self.overlay_scale_factor = 1.0
                self.update_overlay_display()
        
    def update_overlay_display(self):
        """Update overlay display with current zoom"""
        if self.overlay_original_pixmap:
            scaled_pixmap_size = self.overlay_original_pixmap.size() * self.overlay_scale_factor
            scaled_pixmap = self.overlay_original_pixmap.scaled(
                int(scaled_pixmap_size.width()), int(scaled_pixmap_size.height()),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.overlay_label.setPixmap(scaled_pixmap)
            
    def overlay_zoom_in_func(self):
        """Zoom in overlay"""
        if self.overlay_original_pixmap:
            self.overlay_scale_factor *= 1.2
            self.overlay_scale_factor = min(5.0, self.overlay_scale_factor)
            self.update_overlay_display()
            
    def overlay_zoom_out_func(self):
        """Zoom out overlay"""
        if self.overlay_original_pixmap:
            self.overlay_scale_factor *= 0.8
            self.overlay_scale_factor = max(0.1, self.overlay_scale_factor)
            self.update_overlay_display()
            
    def overlay_reset_zoom(self):
        """Reset overlay zoom"""
        if self.overlay_original_pixmap:
            self.overlay_scale_factor = 1.0
            self.update_overlay_display()
            
    def save_overlay(self):
        """Save overlay image"""
        if self.overlay_original_pixmap:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Overlay Image", "overlay_result.png",
                "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
            )
            if file_path:
                self.overlay_original_pixmap.save(file_path)
        
    def update_statistics(self, result: Dict[str, Any]):
        """Update statistics display"""
        rows = result.get("rows", [])
        if not rows:
            self.stats_text.setText("No edge segments detected")
            return
            
        df = pd.DataFrame(rows)
        angles = result.get("angles", [])
        
        stats_text = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🔬 Edge Orientation Analysis Results                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 Detection Summary
────────────────────
• Edge Segments Detected: {len(rows):,}
• Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

🧭 Edge Orientation Analysis
───────────────────────────
• Mean Orientation: {df['theta_deg'].mean():.1f}°
• Median Orientation: {df['theta_deg'].median():.1f}°
• Standard Deviation: {df['theta_deg'].std():.1f}°
• Min Angle: {df['theta_deg'].min():.1f}°
• Max Angle: {df['theta_deg'].max():.1f}°
• Angle Range: {df['theta_deg'].max() - df['theta_deg'].min():.1f}°

📐 Position Statistics
──────────────────────────────
• Mean X Coordinate: {df['center_x'].mean():.1f} px
• Mean Y Coordinate: {df['center_y'].mean():.1f} px

╔══════════════════════════════════════════════════════════════════════════════╗
║  💡 Use Orientation Distribution and Rose Diagram tabs for detailed stats   ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        
        self.stats_text.setText(stats_text)
        
    def update_orientation_plots(self, result: Dict[str, Any]):
        """Update orientation visualization plots"""
        angles = result.get("angles", [])
        if not angles:
            return
        
        try:
            # Orientation histogram
            self.orientation_hist_widget.plot_orientation_histogram(angles)
            
            # Rose diagram
            self.rose_widget.plot_rose_diagram(angles)
        except Exception as e:
            print(f"Error updating plots: {e}")


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    app.setApplicationName("OriFlake")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("OriFlake Research")
    
    window = OriFlakeGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
