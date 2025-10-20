#!/usr/bin/env python3
"""
OriFlake GUI Application
A modern GUI for triangular MoS₂ flake orientation analysis
"""

import sys
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import cv2
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QPushButton, QComboBox, QSlider, QSpinBox,
    QDoubleSpinBox, QGroupBox, QTabWidget, QTextEdit, QFileDialog,
    QProgressBar, QSplitter, QScrollArea, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QDragEnterEvent, QDropEvent

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

# Import OriFlake modules
from oriflake.main import process_image
from oriflake.utils import read_config


class AnalysisWorker(QThread):
    """Worker thread for image analysis to keep GUI responsive"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, image_path: str, config: Dict[str, Any]):
        super().__init__()
        self.image_path = image_path
        self.config = config
        
    def run(self):
        try:
            self.progress.emit(10)
            result = process_image(self.image_path, self.config)
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MatplotlibWidget(FigureCanvas):
    """Custom matplotlib widget for publication-quality plots"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Set publication-quality style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
    def plot_rose_diagram(self, angles: list, title: str = "Edge Orientation Distribution"):
        """Create publication-quality rose diagram"""
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection='polar')
        
        # Convert to radians
        theta = np.deg2rad(np.array(angles))
        
        # Create histogram
        bins = np.linspace(0, 2*np.pi, 37)  # 36 bins for 10-degree intervals
        n, _, _ = ax.hist(theta, bins=bins, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        
        # Customize appearance
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(-1)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_angle = np.degrees(np.mean(theta))
        std_angle = np.degrees(np.std(theta))
        ax.text(0.02, 0.98, f'Mean: {mean_angle:.1f}°\nStd: {std_angle:.1f}°\nN: {len(angles)}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.fig.tight_layout()
        self.draw()
        
    def plot_histogram(self, angles: list, title: str = "Edge Orientation Histogram"):
        """Create publication-quality histogram"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Create histogram
        bins = np.linspace(0, 180, 37)  # 36 bins for 5-degree intervals
        n, bins, patches = ax.hist(angles, bins=bins, alpha=0.7, color='steelblue', 
                                  edgecolor='black', linewidth=0.5)
        
        # Customize appearance
        ax.set_xlabel('Orientation Angle (degrees)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        ax.axvline(mean_angle, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_angle:.1f}°')
        ax.axvline(mean_angle + std_angle, color='orange', linestyle=':', alpha=0.7, label=f'±1σ: {std_angle:.1f}°')
        ax.axvline(mean_angle - std_angle, color='orange', linestyle=':', alpha=0.7)
        ax.legend()
        
        self.fig.tight_layout()
        self.draw()


class ParameterPanel(QWidget):
    """Parameter control panel with preset and manual controls"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Preset selection
        preset_group = QGroupBox("Detection Profile")
        preset_layout = QVBoxLayout()
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["loose", "balanced", "strict", "custom"])
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(QLabel("Profile:"))
        preset_layout.addWidget(self.preset_combo)
        
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)
        
        # Manual parameters
        params_group = QGroupBox("Manual Parameters")
        params_layout = QGridLayout()
        
        # Min area
        params_layout.addWidget(QLabel("Min Area (px):"), 0, 0)
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(10, 2000)
        self.min_area_spin.setValue(150)
        params_layout.addWidget(self.min_area_spin, 0, 1)
        
        # Min convexity
        params_layout.addWidget(QLabel("Min Convexity:"), 1, 0)
        self.min_convexity_spin = QDoubleSpinBox()
        self.min_convexity_spin.setRange(0.1, 1.0)
        self.min_convexity_spin.setSingleStep(0.05)
        self.min_convexity_spin.setValue(0.5)
        params_layout.addWidget(self.min_convexity_spin, 1, 1)
        
        # Max vertices
        params_layout.addWidget(QLabel("Max Vertices:"), 2, 0)
        self.max_vertices_spin = QSpinBox()
        self.max_vertices_spin.setRange(3, 10)
        self.max_vertices_spin.setValue(7)
        params_layout.addWidget(self.max_vertices_spin, 2, 1)
        
        # Approx epsilon
        params_layout.addWidget(QLabel("Approx Epsilon:"), 3, 0)
        self.approx_epsilon_spin = QDoubleSpinBox()
        self.approx_epsilon_spin.setRange(0.01, 0.2)
        self.approx_epsilon_spin.setSingleStep(0.01)
        self.approx_epsilon_spin.setValue(0.06)
        params_layout.addWidget(self.approx_epsilon_spin, 3, 1)
        
        # Union top N
        params_layout.addWidget(QLabel("Union Top N:"), 4, 0)
        self.union_top_n_spin = QSpinBox()
        self.union_top_n_spin.setRange(1, 5)
        self.union_top_n_spin.setValue(2)
        params_layout.addWidget(self.union_top_n_spin, 4, 1)
        
        # Max area ratio
        params_layout.addWidget(QLabel("Max Area Ratio:"), 5, 0)
        self.max_area_ratio_spin = QDoubleSpinBox()
        self.max_area_ratio_spin.setRange(0.01, 0.5)
        self.max_area_ratio_spin.setSingleStep(0.01)
        self.max_area_ratio_spin.setValue(0.2)
        params_layout.addWidget(self.max_area_ratio_spin, 5, 1)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
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
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.setLayout(layout)
        
    def on_preset_changed(self, preset_name: str):
        """Update parameters based on preset selection"""
        if preset_name == "loose":
            self.min_area_spin.setValue(150)
            self.min_convexity_spin.setValue(0.5)
            self.max_vertices_spin.setValue(7)
            self.approx_epsilon_spin.setValue(0.06)
            self.union_top_n_spin.setValue(2)
            self.max_area_ratio_spin.setValue(0.2)
        elif preset_name == "balanced":
            self.min_area_spin.setValue(200)
            self.min_convexity_spin.setValue(0.7)
            self.max_vertices_spin.setValue(6)
            self.approx_epsilon_spin.setValue(0.03)
            self.union_top_n_spin.setValue(1)
            self.max_area_ratio_spin.setValue(0.2)
        elif preset_name == "strict":
            self.min_area_spin.setValue(300)
            self.min_convexity_spin.setValue(0.8)
            self.max_vertices_spin.setValue(5)
            self.approx_epsilon_spin.setValue(0.02)
            self.union_top_n_spin.setValue(1)
            self.max_area_ratio_spin.setValue(0.25)
            
    def get_config(self) -> Dict[str, Any]:
        """Get current parameter configuration"""
        return {
            "min_area_px": self.min_area_spin.value(),
            "min_convexity": self.min_convexity_spin.value(),
            "max_vertices": self.max_vertices_spin.value(),
            "approx_epsilon_frac": self.approx_epsilon_spin.value(),
            "union_top_n": self.union_top_n_spin.value(),
            "max_area_ratio": self.max_area_ratio_spin.value(),
            "reference_deg": 0,
            "sigma_deg": 5,
            "kmeans_k": 3,
            "bilateral_d": 7,
            "bilateral_sigC": 50,
            "bilateral_sigS": 50,
        }


class ImageDisplayWidget(QLabel):
    """Custom widget for image display with drag-and-drop support and scaling"""
    
    image_loaded = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumSize(500, 400)
        self.setMaximumSize(800, 600)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #4CAF50;
                border-radius: 15px;
                background-color: #f8f9fa;
                font-size: 14px;
                color: #666666;
            }
        """)
        self.setText("🖼️ Drag and drop image here\nor click to browse for files\n\nSupported formats: PNG, JPG, TIFF, BMP")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(True)
        self.original_pixmap = None
        self.scale_factor = 1.0
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            image_path = files[0]
            if self.is_valid_image(image_path):
                self.load_image(image_path)
                self.image_loaded.emit(image_path)
                
    def is_valid_image(self, path: str) -> bool:
        """Check if file is a valid image"""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        return Path(path).suffix.lower() in valid_extensions
        
    def load_image(self, image_path: str):
        """Load and display image with scaling"""
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.original_pixmap = pixmap
            self.scale_factor = 1.0
            self.update_display()
            self.setStyleSheet("""
                QLabel {
                    border: 2px solid #4CAF50;
                    border-radius: 15px;
                    background-color: white;
                }
            """)
            
    def update_display(self):
        """Update image display with current scale"""
        if self.original_pixmap:
            scaled_pixmap = self.original_pixmap.scaled(
                self.original_pixmap.size() * self.scale_factor,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
            
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if self.original_pixmap:
            delta = event.angleDelta().y()
            if delta > 0:
                self.scale_factor *= 1.1
            else:
                self.scale_factor *= 0.9
            
            # Limit zoom range
            self.scale_factor = max(0.1, min(5.0, self.scale_factor))
            self.update_display()
            
    def reset_zoom(self):
        """Reset zoom to fit"""
        if self.original_pixmap:
            self.scale_factor = 1.0
            self.update_display()
            
    def mousePressEvent(self, event):
        """Handle click to browse for image"""
        if event.button() == Qt.MouseButton.LeftButton:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Image", "", 
                "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
            )
            if file_path and self.is_valid_image(file_path):
                self.load_image(file_path)
                self.image_loaded.emit(file_path)


class OriFlakeGUI(QMainWindow):
    """Main GUI application for OriFlake"""
    
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.analysis_result = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("OriFlake - Triangular MoS₂ Orientation Scanner")
        self.setGeometry(50, 50, 1600, 1000)
        self.setMinimumSize(1200, 800)
        
        # Set modern style
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #f8f9fa, stop:1 #e9ecef);
            }
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                color: #495057;
                background-color: white;
            }
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                border-radius: 5px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom-color: white;
            }
            QTextEdit {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left panel (parameters and image)
        left_panel = QVBoxLayout()
        
        # Image display
        image_group = QGroupBox("📷 Image Input")
        image_layout = QVBoxLayout()
        
        # Create image display first
        self.image_display = ImageDisplayWidget()
        self.image_display.image_loaded.connect(self.on_image_loaded)
        
        # Add zoom controls
        zoom_layout = QHBoxLayout()
        zoom_in_btn = QPushButton("🔍+")
        zoom_out_btn = QPushButton("🔍-")
        reset_zoom_btn = QPushButton("↻ Reset")
        
        zoom_in_btn.setToolTip("Zoom In")
        zoom_out_btn.setToolTip("Zoom Out")
        reset_zoom_btn.setToolTip("Reset Zoom")
        
        zoom_in_btn.clicked.connect(lambda: self.image_display.wheelEvent(type('', (), {'angleDelta': lambda: type('', (), {'y': lambda: 120})()})()))
        zoom_out_btn.clicked.connect(lambda: self.image_display.wheelEvent(type('', (), {'angleDelta': lambda: type('', (), {'y': lambda: -120})()})()))
        reset_zoom_btn.clicked.connect(self.image_display.reset_zoom)
        
        zoom_layout.addWidget(zoom_in_btn)
        zoom_layout.addWidget(zoom_out_btn)
        zoom_layout.addWidget(reset_zoom_btn)
        zoom_layout.addStretch()
        
        image_layout.addLayout(zoom_layout)
        image_layout.addWidget(self.image_display)
        
        # Add image info label
        self.image_info_label = QLabel("No image loaded")
        self.image_info_label.setStyleSheet("""
            QLabel {
                color: #6c757d;
                font-size: 12px;
                font-style: italic;
                padding: 5px;
            }
        """)
        image_layout.addWidget(self.image_info_label)
        
        image_group.setLayout(image_layout)
        left_panel.addWidget(image_group)
        
        # Parameter panel
        self.param_panel = ParameterPanel()
        self.param_panel.analyze_btn.clicked.connect(self.analyze_image)
        left_panel.addWidget(self.param_panel)
        
        # Add status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #28a745;
                font-weight: bold;
                padding: 5px;
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                border-radius: 3px;
            }
        """)
        left_panel.addWidget(self.status_label)
        
        # Right panel (results)
        right_panel = QVBoxLayout()
        
        # Results tabs
        self.results_tabs = QTabWidget()
        
        # Overlay tab with scroll area
        overlay_widget = QWidget()
        overlay_layout = QVBoxLayout()
        
        # Add overlay controls
        overlay_controls = QHBoxLayout()
        self.overlay_zoom_in = QPushButton("🔍+")
        self.overlay_zoom_out = QPushButton("🔍-")
        self.overlay_reset = QPushButton("↻ Reset")
        self.overlay_save = QPushButton("💾 Save")
        
        overlay_controls.addWidget(self.overlay_zoom_in)
        overlay_controls.addWidget(self.overlay_zoom_out)
        overlay_controls.addWidget(self.overlay_reset)
        overlay_controls.addWidget(self.overlay_save)
        overlay_controls.addStretch()
        
        self.overlay_label = QLabel("📊 No analysis performed yet\n\nLoad an image and click 'Analyze Image' to begin")
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #dee2e6;
                background-color: #f8f9fa;
                min-height: 400px;
                border-radius: 10px;
                font-size: 14px;
                color: #6c757d;
            }
        """)
        self.overlay_label.setScaledContents(True)
        self.overlay_original_pixmap = None
        self.overlay_scale_factor = 1.0
        
        # Connect overlay controls
        self.overlay_zoom_in.clicked.connect(self.overlay_zoom_in_func)
        self.overlay_zoom_out.clicked.connect(self.overlay_zoom_out_func)
        self.overlay_reset.clicked.connect(self.overlay_reset_zoom)
        self.overlay_save.clicked.connect(self.save_overlay)
        
        overlay_layout.addLayout(overlay_controls)
        overlay_layout.addWidget(self.overlay_label)
        
        overlay_widget.setLayout(overlay_layout)
        self.results_tabs.addTab(overlay_widget, "🔍 Overlay Results")
        
        # Statistics tab
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
            }
        """)
        self.results_tabs.addTab(self.stats_text, "📊 Statistics")
        
        # Rose diagram tab
        self.rose_widget = MatplotlibWidget(self, width=7, height=6)
        self.results_tabs.addTab(self.rose_widget, "🌹 Rose Diagram")
        
        # Histogram tab
        self.hist_widget = MatplotlibWidget(self, width=7, height=6)
        self.results_tabs.addTab(self.hist_widget, "📈 Histogram")
        
        right_panel.addWidget(self.results_tabs)
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        splitter.addWidget(left_widget)
        
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        splitter.addWidget(right_widget)
        
        splitter.setSizes([600, 1000])
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        
    def on_image_loaded(self, image_path: str):
        """Handle image loading"""
        self.current_image_path = image_path
        self.param_panel.analyze_btn.setEnabled(True)
        
        # Update image info
        from pathlib import Path
        file_name = Path(image_path).name
        file_size = Path(image_path).stat().st_size / 1024  # KB
        self.image_info_label.setText(f"📁 {file_name} ({file_size:.1f} KB)")
        
        # Update status
        self.status_label.setText("✅ Image loaded - Ready for analysis")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #28a745;
                font-weight: bold;
                padding: 5px;
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                border-radius: 3px;
            }
        """)
        
    def analyze_image(self):
        """Perform image analysis"""
        if not self.current_image_path:
            return
            
        self.param_panel.analyze_btn.setEnabled(False)
        self.param_panel.progress_bar.setVisible(True)
        self.param_panel.progress_bar.setValue(0)
        
        # Update status
        self.status_label.setText("🔄 Analyzing image...")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #856404;
                font-weight: bold;
                padding: 5px;
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 3px;
            }
        """)
        
        # Get configuration
        config = self.param_panel.get_config()
        
        # Start analysis worker
        self.worker = AnalysisWorker(self.current_image_path, config)
        self.worker.progress.connect(self.param_panel.progress_bar.setValue)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()
        
    def on_analysis_finished(self, result: Dict[str, Any]):
        """Handle analysis completion"""
        self.analysis_result = result
        self.param_panel.progress_bar.setVisible(False)
        self.param_panel.analyze_btn.setEnabled(True)
        
        # Update status
        num_flakes = result.get('num_flakes', 0)
        self.status_label.setText(f"✅ Analysis complete - {num_flakes} flakes detected")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #155724;
                font-weight: bold;
                padding: 5px;
                background-color: #d1ecf1;
                border: 1px solid #bee5eb;
                border-radius: 3px;
            }
        """)
        
        # Update results
        self.update_overlay(result)
        self.update_statistics(result)
        self.update_plots(result)
        
    def on_analysis_error(self, error_msg: str):
        """Handle analysis error"""
        self.param_panel.progress_bar.setVisible(False)
        self.param_panel.analyze_btn.setEnabled(True)
        
        # Update status
        self.status_label.setText("❌ Analysis failed")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #721c24;
                font-weight: bold;
                padding: 5px;
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                border-radius: 3px;
            }
        """)
        
        self.stats_text.setText(f"❌ Error: {error_msg}")
        
    def update_overlay(self, result: Dict[str, Any]):
        """Update overlay display"""
        overlay = result["overlay"]
        
        # Convert BGR to RGB for display
        if len(overlay.shape) == 3:
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        else:
            overlay_rgb = overlay
            
        # Convert to QImage
        h, w, ch = overlay_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(overlay_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Store original pixmap for zooming
        self.overlay_original_pixmap = QPixmap.fromImage(qt_image)
        self.overlay_scale_factor = 1.0
        self.update_overlay_display()
        
    def update_overlay_display(self):
        """Update overlay display with current zoom"""
        if self.overlay_original_pixmap:
            scaled_pixmap = self.overlay_original_pixmap.scaled(
                self.overlay_original_pixmap.size() * self.overlay_scale_factor,
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
        rows = result["rows"]
        if not rows:
            self.stats_text.setText("No flakes detected")
            return
            
        df = pd.DataFrame(rows)
        
        stats_text = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           🔬 ORIENTATION ANALYSIS RESULTS                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 DETECTION SUMMARY
────────────────────
• Total Flakes Detected: {result['num_flakes']:,}
• Total Edges Analyzed: {len(rows):,}
• Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

📐 FLAKE MORPHOLOGY STATISTICS
──────────────────────────────
• Mean Area: {df['area_px'].mean():.1f} px²
• Median Area: {df['area_px'].median():.1f} px²
• Std Deviation: {df['area_px'].std():.1f} px²
• Min Area: {df['area_px'].min():.1f} px²
• Max Area: {df['area_px'].max():.1f} px²
• Area Range: {df['area_px'].max() - df['area_px'].min():.1f} px²

🧭 EDGE ORIENTATION ANALYSIS
───────────────────────────
• Mean Orientation: {df['theta_deg'].mean():.1f}°
• Median Orientation: {df['theta_deg'].median():.1f}°
• Std Deviation: {df['theta_deg'].std():.1f}°
• Min Angle: {df['theta_deg'].min():.1f}°
• Max Angle: {df['theta_deg'].max():.1f}°
• Orientation Range: {df['theta_deg'].max() - df['theta_deg'].min():.1f}°

🎯 ALIGNMENT QUALITY METRICS
────────────────────────────
• Mean Alignment Score: {df['score'].mean():.3f}
• Median Alignment Score: {df['score'].median():.3f}
• Std Deviation: {df['score'].std():.3f}
• High Quality (score > 0.5): {(df['score'] > 0.5).sum():,} edges ({(df['score'] > 0.5).mean()*100:.1f}%)
• Excellent (score > 0.8): {(df['score'] > 0.8).sum():,} edges ({(df['score'] > 0.8).mean()*100:.1f}%)
• Outstanding (score > 0.9): {(df['score'] > 0.9).sum():,} edges ({(df['score'] > 0.9).mean()*100:.1f}%)

📏 SYMMETRY DEVIATION ANALYSIS
──────────────────────────────
• Mean Δθ: {df['delta_deg'].mean():.1f}°
• Median Δθ: {df['delta_deg'].median():.1f}°
• Std Deviation: {df['delta_deg'].std():.1f}°
• Well-aligned (Δθ < 10°): {(df['delta_deg'] < 10).sum():,} edges ({(df['delta_deg'] < 10).mean()*100:.1f}%)
• Moderately aligned (Δθ < 20°): {(df['delta_deg'] < 20).sum():,} edges ({(df['delta_deg'] < 20).mean()*100:.1f}%)
• Poorly aligned (Δθ > 30°): {(df['delta_deg'] > 30).sum():,} edges ({(df['delta_deg'] > 30).mean()*100:.1f}%)

🔍 QUALITY ASSESSMENT
─────────────────────
• Detection Success Rate: {len(rows)/max(1, result['num_flakes']):.1f} edges/flake
• High-Quality Detection: {(df['score'] > 0.7).mean()*100:.1f}% of edges
• Symmetry Compliance: {(df['delta_deg'] < 15).mean()*100:.1f}% within 15° tolerance
• Overall Quality Score: {(df['score'].mean() + (1 - df['delta_deg'].mean()/60))/2:.3f}

╔══════════════════════════════════════════════════════════════════════════════╗
║  💡 For publication-quality figures, use the Rose Diagram and Histogram tabs ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        
        self.stats_text.setText(stats_text)
        
    def update_plots(self, result: Dict[str, Any]):
        """Update visualization plots"""
        angles = result["angles"]
        if not angles:
            return
            
        # Update rose diagram
        self.rose_widget.plot_rose_diagram(angles, "Edge Orientation Rose Diagram")
        
        # Update histogram
        self.hist_widget.plot_histogram(angles, "Edge Orientation Distribution")


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("OriFlake")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("OriFlake Research")
    
    # Create and show main window
    window = OriFlakeGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
