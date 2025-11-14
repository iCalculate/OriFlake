"""
OriFlake - 纳米片边缘取向统计工具
处理薄膜显微照片，统计纳米片边缘的取向分布
"""

import os
import sys
import argparse
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# Handle both module and direct script execution
try:
    from .utils import read_config, ensure_dir, list_images, preprocess_image
    from .seg import kmeans_segment_lab, morph_cleanup, threshold_fallback
    from .geometry import find_candidate_polygons
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from oriflake.utils import read_config, ensure_dir, list_images, preprocess_image
    from oriflake.seg import kmeans_segment_lab, morph_cleanup, threshold_fallback
    from oriflake.geometry import find_candidate_polygons


def convert_to_8bit(img: np.ndarray) -> np.ndarray:
    """
    将图像统一转换为8bit（每个通道0-255范围）
    支持所有图像类型：8bit、16bit、24bit RGB、32bit、浮点等
    
    Args:
        img: 输入图像（任意位深度和类型）
    
    Returns:
        8bit图像 (uint8, 每个通道0-255范围)
    """
    # 如果已经是8bit，直接返回
    if img.dtype == np.uint8:
        return img
    
    # 处理不同数据类型
    if img.dtype == np.uint16:
        # 16位图像，值范围是0-65535，归一化到0-255
        img_max = np.max(img)
        img_min = np.min(img)
        if img_max > img_min:
            # 线性归一化
            img_float = (img.astype(np.float32) - img_min) / (img_max - img_min) * 255.0
            return np.clip(img_float, 0, 255).astype(np.uint8)
        else:
            # 如果所有值相同，简单除以256
            return (img / 256).astype(np.uint8)
    
    elif img.dtype == np.uint32 or img.dtype == np.int32:
        # 32位整数图像
        img_max = np.max(img)
        img_min = np.min(img)
        if img_max > img_min:
            img_float = (img.astype(np.float32) - img_min) / (img_max - img_min) * 255.0
            return np.clip(img_float, 0, 255).astype(np.uint8)
        else:
            return np.zeros_like(img, dtype=np.uint8)
    
    elif img.dtype == np.float32 or img.dtype == np.float64:
        # 浮点图像
        img_max = np.max(img)
        img_min = np.min(img)
        
        if img_max <= 1.0 and img_min >= 0.0:
            # 范围是0-1，直接乘以255
            return np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            # 其他范围，进行归一化
            if img_max > img_min:
                img_float = (img - img_min) / (img_max - img_min) * 255.0
                return np.clip(img_float, 0, 255).astype(np.uint8)
            else:
                return np.zeros_like(img, dtype=np.uint8)
    
    elif img.dtype == np.int16 or img.dtype == np.int8:
        # 有符号整数，先转换为无符号
        if img.dtype == np.int16:
            # int16范围通常是-32768到32767，转换为0-65535
            img = (img.astype(np.int32) - np.iinfo(np.int16).min).astype(np.uint16)
            return convert_to_8bit(img)  # 递归处理
        else:  # int8
            # int8范围是-128到127，转换为0-255
            img = (img.astype(np.int16) - np.iinfo(np.int8).min).astype(np.uint8)
            return img
    
    else:
        # 其他类型，尝试直接转换或归一化
        try:
            img_max = np.max(img)
            img_min = np.min(img)
            if img_max > img_min:
                img_float = (img.astype(np.float32) - img_min) / (img_max - img_min) * 255.0
                return np.clip(img_float, 0, 255).astype(np.uint8)
            else:
                return np.zeros_like(img, dtype=np.uint8)
        except:
            # 最后的fallback：尝试直接转换
            return img.astype(np.uint8)


def crop_center_region(img: np.ndarray, crop_ratio: float = 0.3) -> np.ndarray:
    """
    裁剪图片中心区域，去除黑边
    
    Args:
        img: 输入图像
        crop_ratio: 裁剪比例，默认0.3表示裁剪掉30%（上下左右各15%）
    
    Returns:
        裁剪后的图像
    """
    h, w = img.shape[:2]
    crop_h = int(h * crop_ratio / 2)
    crop_w = int(w * crop_ratio / 2)
    
    # 裁剪中心区域
    cropped = img[crop_h:h-crop_h, crop_w:w-crop_w]
    return cropped


def subtract_background(gray: np.ndarray, method: str = 'gaussian') -> np.ndarray:
    """
    扣除整体色偏（扣背底）
    
    Args:
        gray: 灰度图像
        method: 方法，'gaussian'使用高斯模糊，'median'使用中值滤波
    
    Returns:
        扣除背底后的图像（归一化到0-255范围）
    """
    if method == 'gaussian':
        # 使用大核高斯模糊作为背底
        background = cv2.GaussianBlur(gray, (101, 101), 0)
    elif method == 'median':
        # 使用中值滤波作为背底
        background = cv2.medianBlur(gray, 51)
    else:
        background = np.mean(gray) * np.ones_like(gray)
    
    # 扣除背底
    subtracted = gray.astype(np.float32) - background.astype(np.float32)
    
    # 归一化到0-255范围，保持对比度
    min_val = np.min(subtracted)
    max_val = np.max(subtracted)
    if max_val > min_val:
        subtracted = (subtracted - min_val) / (max_val - min_val) * 255.0
    else:
        subtracted = np.zeros_like(subtracted)
    
    subtracted = np.clip(subtracted, 0, 255).astype(np.uint8)
    return subtracted


def find_dual_color_peaks_from_histogram_full_image(rgb: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    从完整RGB图像的直方图中找到两个主要的颜色峰（类似analyze_histogram.py的方法）
    
    方法：
    1. 在所有通道（R、G、B、Intensity）中尝试找两个峰
    2. 优先使用Intensity或Blue通道（通常这些通道有两个峰）
    3. 对于每个峰，找到对应的RGB颜色值
    
    Args:
        rgb: RGB图像 (H, W, 3)，完整未裁剪的图像，确保是uint8类型，值范围0-255
    
    Returns:
        (color1, color2) 两个主要颜色的RGB值，或 (None, None)
    """
    # 确保图像是uint8类型
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    
    # 方法1：优先在BLUE通道中找两个峰（默认）
    blue_channel = rgb[:, :, 2]
    hist_blue, bins_blue = np.histogram(blue_channel.flatten(), bins=256, range=(0, 256))
    bin_centers_blue = (bins_blue[:-1] + bins_blue[1:]) / 2
    
    # 对于完整图像，使用合适的参数
    peak1_idx, peak2_idx = find_dual_peaks(hist_blue, bin_centers_blue, min_distance=15)
    
    # 如果BLUE通道找不到，尝试Intensity（灰度）通道作为fallback
    if peak1_idx is None or peak2_idx is None:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hist_gray, bins_gray = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        bin_centers_gray = (bins_gray[:-1] + bins_gray[1:]) / 2
        peak1_idx, peak2_idx = find_dual_peaks(hist_gray, bin_centers_gray, min_distance=10)
        
        if peak1_idx is None or peak2_idx is None:
            return None, None
        
        # 使用Intensity通道的峰值找到对应的RGB颜色
        peak1_val = int(bin_centers_gray[peak1_idx])
        peak2_val = int(bin_centers_gray[peak2_idx])
        
        # 峰1对应的像素（灰度值在峰值附近±5范围内）
        mask1 = (gray >= max(0, peak1_val - 5)) & (gray <= min(255, peak1_val + 5))
        if np.sum(mask1) > 0:
            color1 = np.mean(rgb[mask1], axis=0).astype(np.uint8)
        else:
            mask1 = (gray == peak1_val)
            if np.sum(mask1) > 0:
                color1 = np.mean(rgb[mask1], axis=0).astype(np.uint8)
            else:
                color1 = np.array([peak1_val, peak1_val, peak1_val], dtype=np.uint8)
        
        # 峰2对应的像素
        mask2 = (gray >= max(0, peak2_val - 5)) & (gray <= min(255, peak2_val + 5))
        if np.sum(mask2) > 0:
            color2 = np.mean(rgb[mask2], axis=0).astype(np.uint8)
        else:
            mask2 = (gray == peak2_val)
            if np.sum(mask2) > 0:
                color2 = np.mean(rgb[mask2], axis=0).astype(np.uint8)
            else:
                color2 = np.array([peak2_val, peak2_val, peak2_val], dtype=np.uint8)
        
        return color1, color2
    
    # 使用BLUE通道的峰值找到对应的RGB颜色
    peak1_val = int(bin_centers_blue[peak1_idx])
    peak2_val = int(bin_centers_blue[peak2_idx])
    
    # 使用BLUE通道的峰值找到对应的RGB颜色
    mask1 = (blue_channel >= max(0, peak1_val - 5)) & (blue_channel <= min(255, peak1_val + 5))
    if np.sum(mask1) > 0:
        color1 = np.mean(rgb[mask1], axis=0).astype(np.uint8)
    else:
        mask1 = (blue_channel == peak1_val)
        if np.sum(mask1) > 0:
            color1 = np.mean(rgb[mask1], axis=0).astype(np.uint8)
        else:
            r_mean = int(np.mean(rgb[:, :, 0]))
            g_mean = int(np.mean(rgb[:, :, 1]))
            color1 = np.array([r_mean, g_mean, peak1_val], dtype=np.uint8)
    
    mask2 = (blue_channel >= max(0, peak2_val - 5)) & (blue_channel <= min(255, peak2_val + 5))
    if np.sum(mask2) > 0:
        color2 = np.mean(rgb[mask2], axis=0).astype(np.uint8)
    else:
        mask2 = (blue_channel == peak2_val)
        if np.sum(mask2) > 0:
            color2 = np.mean(rgb[mask2], axis=0).astype(np.uint8)
        else:
            r_mean = int(np.mean(rgb[:, :, 0]))
            g_mean = int(np.mean(rgb[:, :, 1]))
            color2 = np.array([r_mean, g_mean, peak2_val], dtype=np.uint8)
    
    return color1, color2


def find_dual_color_peaks(rgb: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    在RGB颜色空间中使用k-means找到两个主要的颜色（fallback方法）
    
    Args:
        rgb: RGB图像 (H, W, 3)
    
    Returns:
        (color1, color2) 两个主要颜色的RGB值，或 (None, None)
    """
    h, w = rgb.shape[:2]
    
    # 将图像reshape为像素列表
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    
    # 使用k-means找到两个主要颜色
    k = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )
    
    # 计算每个聚类的像素数量
    label_counts = np.bincount(labels.flatten())
    
    # 返回两个颜色中心（按像素数量排序，最多的在前）
    if len(centers) >= 2:
        sorted_indices = np.argsort(label_counts)[::-1]  # 降序
        color1 = centers[sorted_indices[0]].astype(np.uint8)
        color2 = centers[sorted_indices[1]].astype(np.uint8)
        return color1, color2
    
    return None, None


def find_dual_peaks(hist: np.ndarray, bins: np.ndarray, min_distance: int = 20) -> Tuple[Optional[int], Optional[int]]:
    """
    在灰度直方图中找到两个峰（薄膜和非薄膜区域）
    
    Args:
        hist: 直方图值
        bins: 直方图bin中心
        min_distance: 两个峰之间的最小距离
    
    Returns:
        (peak1_idx, peak2_idx) 或 (None, None)
    """
    # 平滑直方图以减少噪声
    from scipy.ndimage import gaussian_filter1d
    hist_smooth = gaussian_filter1d(hist.astype(np.float32), sigma=1.0)
    
    # 计算直方图的最大值，用于设置prominence
    hist_max = np.max(hist_smooth)
    
    # 使用scipy的find_peaks找峰，对于裁剪后的图像使用更宽松的参数
    # 首先尝试较宽松的参数（适合裁剪后的图像）
    peaks, properties = signal.find_peaks(hist_smooth, distance=min_distance, 
                                         prominence=hist_max * 0.02,  # 降低到2%
                                         width=1)  # 减小最小峰宽度
    
    if len(peaks) < 2:
        # 如果找不到两个峰，尝试使用更宽松的参数
        peaks, _ = signal.find_peaks(hist_smooth, distance=max(3, min_distance // 4), 
                                    prominence=hist_max * 0.01,  # 进一步降低到1%
                                    width=1)
    
    # 如果还是找不到，尝试找局部最大值（不要求prominence）
    if len(peaks) < 2:
        # 使用更简单的方法：找所有局部最大值，然后选择最高的两个
        from scipy.signal import argrelextrema
        # 找所有局部最大值
        local_maxima = argrelextrema(hist_smooth, np.greater, order=min_distance//2)[0]
        if len(local_maxima) >= 2:
            # 选择最高的两个
            peak_heights = hist_smooth[local_maxima]
            top_two = np.argsort(peak_heights)[-2:]
            peaks = local_maxima[top_two]
    
    if len(peaks) >= 2:
        # 选择最高的两个峰
        peak_heights = hist[peaks]
        top_two = np.argsort(peak_heights)[-2:]
        return int(peaks[top_two[0]]), int(peaks[top_two[1]])
    elif len(peaks) == 1:
        # 只有一个峰，尝试找第二个峰（可能是谷底）
        peak_idx = peaks[0]
        # 在峰的两侧找局部最小值
        left_min = np.argmin(hist[:peak_idx]) if peak_idx > 0 else None
        right_min = np.argmin(hist[peak_idx:]) + peak_idx if peak_idx < len(hist) - 1 else None
        
        # 选择距离峰更远的那个作为第二个峰的位置
        if left_min is not None and right_min is not None:
            if abs(left_min - peak_idx) > abs(right_min - peak_idx):
                return int(left_min), int(peak_idx)
            else:
                return int(peak_idx), int(right_min)
        elif left_min is not None:
            return int(left_min), int(peak_idx)
        elif right_min is not None:
            return int(peak_idx), int(right_min)
    
    return None, None


def enhance_edges_robust(rgb: np.ndarray, color1: Optional[np.ndarray] = None, 
                         color2: Optional[np.ndarray] = None, 
                         blue_threshold: Optional[float] = None,
                         min_contour_area: int = 100,
                         morph_open_ks: int = 3, morph_close_ks: int = 0,
                         boundary_kernel_size: int = 3, boundary_dilate_iterations: int = 1,
                         edge_cleanup_kernel_size: int = 2, edge_cleanup_iterations: int = 1,
                         border_margin: int = 3, border_area_threshold: float = 0.1,
                         border_width_threshold: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """
    重新实现的边缘检测方法：基于蓝色通道两个峰值的颜色对比进行分割和边界提取
    
    思路：
    1. 使用蓝色通道的两个颜色峰来分割图像（纳米片和衬底）
    2. 形态学清理分割结果（去除噪点，但不连接独立结构）
    3. 提取分割区域的边界（两个颜色峰区域的分界线）
    4. 过滤图像边框，保留所有内部边界
    
    Args:
        rgb: RGB图像 (H, W, 3) - 裁剪后的图像
        color1: 第一个颜色峰（RGB值）- 从完整图像提取
        color2: 第二个颜色峰（RGB值）- 从完整图像提取
        blue_threshold: 蓝色通道二值化阈值（基于完整图像的蓝色通道峰值计算）
        min_contour_area: 最小连通域面积，用于过滤小的边缘连通域
        morph_open_ks: 形态学opening的核大小，用于清理分割结果（去除噪点）
        morph_close_ks: 形态学closing的核大小（0表示不进行closing，避免连接独立结构）
        boundary_kernel_size: 边界提取的形态学核大小
        boundary_dilate_iterations: 边界提取的膨胀/腐蚀迭代次数
        edge_cleanup_kernel_size: 边缘清理的核大小
        edge_cleanup_iterations: 边缘清理的迭代次数
        border_margin: 图像边界过滤的像素数（排除图像边缘多少像素）
        border_area_threshold: 图像边框判断的面积阈值（相对于图像总面积的比值）
        border_width_threshold: 图像边框判断的宽度阈值（相对于图像宽度的比值）
    
    Returns:
        (edge_filtered, binary_output): 
        - edge_filtered: 边缘增强后的二值图像（包含几何结构边缘，无图像边界）
        - binary_output: 二值化图像（基于蓝色通道两个峰值的平均值阈值）
    """
    h, w = rgb.shape[:2]
    
    # 第一步：使用颜色峰分割识别纳米片区域
    if color1 is None or color2 is None:
        # 如果没有提供颜色峰，使用k-means找到两个主要颜色
        color1, color2 = find_dual_color_peaks(rgb)
    
    if color1 is None or color2 is None:
        # 如果找不到颜色峰，使用蓝色通道阈值作为fallback（不使用灰度）
        blue_channel = rgb[:, :, 2]
        _, binary1 = cv2.threshold(blue_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, binary2 = cv2.threshold(blue_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        area1 = np.sum(binary1 > 0)
        area2 = np.sum(binary2 > 0)
        mask = binary1 if area1 > area2 else binary2
    else:
        # 第一步：提取蓝色通道（裁剪后的图像）
        blue_channel = rgb[:, :, 2]
        
        # 第二步：使用传入的阈值进行二值化（阈值基于完整图像的蓝色通道峰值）
        if blue_threshold is not None:
            threshold_value = blue_threshold
        else:
            # 如果没有传入阈值，使用color1和color2的蓝色通道值计算（fallback）
            color1_blue = float(color1[2])
            color2_blue = float(color2[2])
            threshold_value = (color1_blue + color2_blue) / 2.0
        
        _, binary_mask = cv2.threshold(blue_channel, threshold_value, 255, cv2.THRESH_BINARY)
        
        # 保存二值化图像（用于输出）
        binary_output = binary_mask.copy()
        
        # 第三步：基于二值化结果进行分割和处理
        # 二值化图像已经是分割结果，直接使用
        mask = binary_mask
        
        # 第四步：形态学清理分割结果（去除噪点，但不连接独立结构）
        mask = morph_cleanup(mask, open_ks=morph_open_ks, close_ks=morph_close_ks)
        
        # 第五步：提取分割区域的边界（这是两个颜色峰区域的分界线）
        # 使用多种方法组合，确保提取所有边界
        
        # 方法1：使用findContours提取所有轮廓边界（最精确的方法）
        # 使用RETR_TREE获取所有层级，包括内部孔洞
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        # 创建边缘图像
        edge_img = np.zeros((h, w), dtype=np.uint8)
        
        # 绘制所有轮廓的边界（包括外部边界和内部孔洞边界）
        # 使用thickness=1确保提取所有边界像素
        for i, contour in enumerate(contours):
            # 绘制所有轮廓，包括内部孔洞
            cv2.drawContours(edge_img, [contour], -1, 255, 1)
        
        # 方法2：使用距离变换提取边界（更精确的边界定位）
        # 计算到边界的距离
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        # 距离为1的像素就是边界
        edge_dist = (dist_transform == 1).astype(np.uint8) * 255
        edge_img = cv2.bitwise_or(edge_img, edge_dist)
        
        # 方法3：使用形态学方法提取边界（补充findContours可能遗漏的边界）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (boundary_kernel_size, boundary_kernel_size))
        mask_dilated = cv2.dilate(mask, kernel, iterations=boundary_dilate_iterations)
        mask_eroded = cv2.erode(mask, kernel, iterations=boundary_dilate_iterations)
        # 边界 = 膨胀后的mask - 腐蚀后的mask
        edge_morph = cv2.subtract(mask_dilated, mask_eroded)
        edge_img = cv2.bitwise_or(edge_img, edge_morph)
        
        # 方法4：使用Sobel梯度检测边界（在原始RGB图像上）
        # 在蓝色通道上计算梯度，增强边界检测
        blue_channel = rgb[:, :, 2]
        sobelx = cv2.Sobel(blue_channel, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blue_channel, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        # 归一化到0-255
        gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)
        # 在mask边界区域内，使用梯度信息增强边界
        # 只保留mask边界附近的梯度
        mask_boundary_region = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
        mask_boundary_region = cv2.subtract(mask_boundary_region, cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2))
        gradient_in_boundary = cv2.bitwise_and(gradient_magnitude, mask_boundary_region)
        # 使用自适应阈值提取梯度边界
        _, gradient_edges = cv2.threshold(gradient_in_boundary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edge_img = cv2.bitwise_or(edge_img, gradient_edges)
        
        # 方法5：使用Canny边缘检测作为补充（检测可能遗漏的弱边界）
        # 在mask上应用Canny，使用自适应阈值
        edges_canny = cv2.Canny(mask, 50, 150)
        edge_img = cv2.bitwise_or(edge_img, edges_canny)
        
        # 方法6：使用Laplacian检测边界（对弱边界敏感）
        laplacian = cv2.Laplacian(blue_channel, cv2.CV_64F)
        laplacian_abs = np.abs(laplacian).astype(np.uint8)
        # 在mask边界区域内使用Laplacian
        laplacian_in_boundary = cv2.bitwise_and(laplacian_abs, mask_boundary_region)
        _, laplacian_edges = cv2.threshold(laplacian_in_boundary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edge_img = cv2.bitwise_or(edge_img, laplacian_edges)
        
        # 形态学清理：只去除明显的噪点，但保留所有边界
        # 使用非常小的核和很少的迭代次数，避免过度清理
        if edge_cleanup_kernel_size > 0 and edge_cleanup_iterations > 0:
            # 只清理非常小的孤立点
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_cleanup_kernel_size, edge_cleanup_kernel_size))
            # 使用opening去除小噪点，但保留所有边界
            edge_img_cleaned = cv2.morphologyEx(edge_img, cv2.MORPH_OPEN, kernel_small, iterations=1)
            # 如果清理后边界像素减少太多，说明清理过度，不应用清理
            if np.sum(edge_img_cleaned > 0) > np.sum(edge_img > 0) * 0.8:
                edge_img = edge_img_cleaned
        
        # 调试：检查边界提取结果
        edge_pixels_before_filter = np.sum(edge_img > 0)
        if edge_pixels_before_filter == 0:
            # 如果边界提取失败，尝试更宽松的方法
            # 直接使用mask的边界
            edge_img = cv2.subtract(mask_dilated, mask_eroded)
            for contour in contours:
                cv2.drawContours(edge_img, [contour], -1, 255, 1)
    
    # 确保edge_img和binary_output已定义（fallback路径）
    if 'edge_img' not in locals():
        # 如果edge_img未定义，创建一个空的
        edge_img = np.zeros((h, w), dtype=np.uint8)
    
    if 'binary_output' not in locals():
        # 如果binary_output未定义，创建一个空的
        binary_output = np.zeros((h, w), dtype=np.uint8)
    
    # 第六步：只过滤图像四个边本身（很小的区域），保留所有内部边界
    # 只排除图像最边缘的几行/几列，避免识别图像边框
    # 但是要保留延伸到边缘附近的边界（这些可能是真实的几何边界）
    # 先保存原始边缘图像
    edge_img_before_border_filter = edge_img.copy()
    
    # 创建一个更宽松的边界mask，只排除非常边缘的像素
    border_mask = np.ones((h, w), dtype=np.uint8) * 255
    # 只排除最边缘的1-2个像素，而不是整个border_margin区域
    thin_border = max(1, border_margin // 2)
    border_mask[:thin_border, :] = 0  # 上边界
    border_mask[-thin_border:, :] = 0  # 下边界
    border_mask[:, :thin_border] = 0  # 左边界
    border_mask[:, -thin_border:] = 0  # 右边界
    
    # 应用边界过滤，但保留延伸到边缘附近的边界
    edge_img_filtered = cv2.bitwise_and(edge_img, border_mask)
    
    # 对于延伸到边缘附近的边界，如果它们不是明显的图像边框，也要保留
    # 检查边缘附近的连通域，如果它们不是大面积的图像边框，就保留
    edge_img = edge_img_filtered
    
    # 第七步：过滤小的边缘连通域，只排除明显是图像边框的大轮廓
    # 使用原始边缘图像（未经过边界过滤）进行分析，确保不遗漏任何边界
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_img_before_border_filter, connectivity=8)
    
    # 创建过滤后的边缘图像
    edge_filtered = np.zeros((h, w), dtype=np.uint8)
    
    for label_id in range(1, num_labels):  # 跳过背景（label=0）
        area = stats[label_id, cv2.CC_STAT_AREA]
        
        # 过滤太小的连通域
        if area < min_contour_area:
            continue
        
        # 获取连通域的边界框
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        width = stats[label_id, cv2.CC_STAT_WIDTH]
        height = stats[label_id, cv2.CC_STAT_HEIGHT]
        
        # 只排除明显是图像边框的大轮廓（覆盖大部分图像边缘）
        # 如果连通域很大且几乎覆盖整个图像边缘，才排除
        border_check_margin = 5  # 检查边界的像素范围
        is_large_border = (
            area > w * h * border_area_threshold and  # 面积大于阈值
            (
                (x <= border_check_margin and width > w * border_width_threshold) or  # 左边缘大轮廓
                (y <= border_check_margin and height > h * border_width_threshold) or  # 上边缘大轮廓
                ((x + width) >= (w - border_check_margin) and width > w * border_width_threshold) or  # 右边缘大轮廓
                ((y + height) >= (h - border_check_margin) and height > h * border_width_threshold)  # 下边缘大轮廓
            )
        )
        
        # 如果明显是图像边框，跳过
        if is_large_border:
            continue
        
        # 保留所有其他边界（包括内部边界和延伸到边缘附近的边界）
        # 使用原始边缘图像中的连通域，确保提取所有边界
        component_mask = (labels == label_id).astype(np.uint8) * 255
        edge_filtered = cv2.bitwise_or(edge_filtered, component_mask)
    
    return edge_filtered, binary_output


def fit_line_segments(edges: np.ndarray, min_line_length: float = 20.0, 
                     max_line_gap: float = 10.0, threshold: int = 50,
                     min_segment_distance: float = 30.0) -> List[Tuple[float, float, float]]:
    """
    对边缘进行线段拟合，提取取向角度
    重要：对每个独立的连通域分别处理，避免将不同结构当作整体
    
    Args:
        edges: 二值边缘图像
        min_line_length: 最小线段长度（严格阈值，不会被过度降低）
        max_line_gap: 最大线段间隙
        threshold: Hough变换阈值
        min_segment_distance: 最小线段间距，用于避免对同一条长边重复统计（像素）
    
    Returns:
        线段列表，每个元素为 (角度, 中心x, 中心y)
    """
    # 第一步：找到所有独立的连通域（每个独立的结构）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
    
    all_orientations = []
    
    # 对每个独立的连通域分别进行线段拟合
    for label_id in range(1, num_labels):  # 跳过背景（label=0）
        # 提取当前连通域的掩码
        component_mask = (labels == label_id).astype(np.uint8) * 255
        component_edges = cv2.bitwise_and(edges, component_mask)
        
        # 统计当前连通域的边缘像素数
        edge_pixels = np.sum(component_edges > 0)
        # 严格使用min_line_length，不降低阈值
        if edge_pixels < min_line_length:
            continue
        
        # 根据连通域大小自适应调整参数（但保持min_line_length的严格性）
        area = stats[label_id, cv2.CC_STAT_AREA]
        total_pixels = edges.size
        
        # 局部参数调整（只调整threshold和max_line_gap，不降低min_line_length）
        local_threshold = threshold
        local_min_line_length = min_line_length  # 严格使用，不降低
        local_max_line_gap = max_line_gap
        
        if edge_pixels < total_pixels * 0.01:  # 边缘像素少于1%
            local_threshold = max(5, threshold // 4)
            local_max_line_gap = max_line_gap * 3.0
        elif edge_pixels < total_pixels * 0.05:  # 边缘像素少于5%
            local_threshold = max(5, threshold // 3)
            local_max_line_gap = max_line_gap * 2.0
        elif edge_pixels < total_pixels * 0.1:  # 边缘像素少于10%
            local_threshold = max(10, threshold // 2)
            local_max_line_gap = max_line_gap * 1.5
        
        # 对当前连通域使用Hough变换检测线段
        lines = cv2.HoughLinesP(
            component_edges,
            rho=1.0,
            theta=np.pi / 180.0,
            threshold=local_threshold,
            minLineLength=int(local_min_line_length),  # 严格使用最小长度
            maxLineGap=int(local_max_line_gap)
        )
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # 计算线段长度
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # 严格过滤：如果线段长度小于min_line_length，跳过
                if line_length < min_line_length:
                    continue
                
                # 计算角度（0-180度）
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0:
                    angle = 90.0
                else:
                    angle_rad = np.arctan2(-dy, dx)  # 负号因为图像坐标系
                    angle = np.degrees(angle_rad) % 180.0
                    if angle < 0:
                        angle += 180.0
                
                # 计算中心点
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                
                all_orientations.append((angle, center_x, center_y, line_length))
    
    # 在去重之前，先打印统计信息
    if len(all_orientations) > 0:
        lengths = [o[3] for o in all_orientations]
        print(f"    Before deduplication: {len(all_orientations)} segments, "
              f"length range: [{min(lengths):.1f}, {max(lengths):.1f}], "
              f"mean: {np.mean(lengths):.1f}")
    
    # 第二步：去重，避免对同一条长边重复统计
    # 合并相似角度且距离接近的线段，特别是共线的连续线段
    if len(all_orientations) == 0:
        return []
    
    # 首先按长度排序，优先处理长线段（这样长线段会先被选中作为代表）
    all_orientations_sorted = sorted(all_orientations, key=lambda x: x[3], reverse=True)
    
    # 按角度和位置分组
    filtered_orientations = []
    used = [False] * len(all_orientations_sorted)
    
    # 角度容差（度）
    angle_tolerance = 5.0
    
    # 需要存储线段的端点信息用于共线判断
    # 从all_orientations中恢复端点信息（需要从原始lines中获取）
    # 为了简化，我们使用中心点和角度来重建线段的近似端点
    def get_line_endpoints(angle, cx, cy, length):
        """根据角度、中心点和长度计算线段端点"""
        angle_rad = np.radians(angle)
        half_len = length / 2.0
        dx = half_len * np.cos(angle_rad)
        dy = -half_len * np.sin(angle_rad)  # 负号因为图像坐标系
        x1 = cx - dx
        y1 = cy - dy
        x2 = cx + dx
        y2 = cy + dy
        return (x1, y1, x2, y2)
    
    def point_to_line_distance(px, py, x1, y1, x2, y2):
        """计算点到直线的距离"""
        # 直线方程: (y2-y1)*x - (x2-x1)*y + (x2-x1)*y1 - (y2-y1)*x1 = 0
        A = y2 - y1
        B = -(x2 - x1)
        C = (x2 - x1) * y1 - (y2 - y1) * x1
        if A == 0 and B == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        distance = abs(A * px + B * py + C) / np.sqrt(A**2 + B**2)
        return distance
    
    def are_collinear_and_continuous(angle1, cx1, cy1, len1, angle2, cx2, cy2, len2, 
                                     angle_tol=angle_tolerance, dist_tol=min_segment_distance):
        """判断两条线段是否共线且连续"""
        # 检查角度相似性
        angle_diff = abs(angle1 - angle2)
        if angle_diff > 90:
            angle_diff = 180 - angle_diff
        if angle_diff > angle_tol:
            return False
        
        # 获取两条线段的端点
        x1_1, y1_1, x1_2, y1_2 = get_line_endpoints(angle1, cx1, cy1, len1)
        x2_1, y2_1, x2_2, y2_2 = get_line_endpoints(angle2, cx2, cy2, len2)
        
        # 计算中心点距离
        center_dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        
        # 方法1：如果中心点距离很近，认为是同一条边
        if center_dist < dist_tol:
            return True
        
        # 方法2：检查是否共线（一个线段的中心点到另一个线段的距离）
        # 计算线段2的中心点到线段1的距离
        dist1 = point_to_line_distance(cx2, cy2, x1_1, y1_1, x1_2, y1_2)
        # 计算线段1的中心点到线段2的距离
        dist2 = point_to_line_distance(cx1, cy1, x2_1, y2_1, x2_2, y2_2)
        
        # 如果两个中心点都在对方的直线上（距离很小），且中心点距离合理，认为是共线
        line_dist_tol = max(len1, len2) * 0.1  # 允许10%的长度误差
        if dist1 < line_dist_tol and dist2 < line_dist_tol:
            # 检查是否连续（中心点距离应该小于两个线段长度之和）
            max_gap = (len1 + len2) / 2.0 + dist_tol
            if center_dist < max_gap:
                return True
        
        return False
    
    for i, (angle1, cx1, cy1, len1) in enumerate(all_orientations_sorted):
        if used[i]:
            continue
        
        # 找到与当前线段相似且接近的线段（包括共线的连续线段）
        similar_segments = [(i, angle1, cx1, cy1, len1)]
        
        for j, (angle2, cx2, cy2, len2) in enumerate(all_orientations_sorted[i+1:], start=i+1):
            if used[j]:
                continue
            
            # 使用改进的共线判断
            if are_collinear_and_continuous(angle1, cx1, cy1, len1, angle2, cx2, cy2, len2):
                similar_segments.append((j, angle2, cx2, cy2, len2))
        
        # 从相似线段中选择最长的作为代表（避免重复统计）
        # 由于已经按长度排序，第一个就是最长的
        best_idx, best_angle, best_cx, best_cy, best_len = similar_segments[0]
        filtered_orientations.append((best_angle, best_cx, best_cy))
        
        # 标记所有相似线段为已使用
        for idx, _, _, _, _ in similar_segments:
            used[idx] = True
    
    # 打印去重统计信息
    if len(all_orientations_sorted) > 0:
        print(f"    After deduplication: {len(filtered_orientations)} segments "
              f"(merged {len(all_orientations_sorted) - len(filtered_orientations)} similar segments)")
    
    return filtered_orientations


def process_image(img_path: str, cfg: Dict) -> Dict:
    """
    处理单张图像
    
    Args:
        img_path: 图像路径
        cfg: 配置字典
    
    Returns:
        处理结果字典
    """
    # 读取图像（保持原始位深度）
    img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    
    # 统一转换为8bit（每个通道0-255范围）
    # 支持所有图像类型：8bit、16bit、24bit RGB、32bit、浮点等
    img_bgr = convert_to_8bit(img_bgr)
    
    # 第一步：转换为RGB（完整图像，用于找峰）
    if len(img_bgr.shape) == 3:
        if img_bgr.shape[2] == 4:
            # BGRA图像，转换为BGR
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
        rgb_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        # 如果是灰度图，转为RGB
        rgb_full = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    
    # 第二步：在完整RGB图像上分析直方图，找到两个颜色峰（不裁剪、不扣背底）
    # 使用完整图像找峰，因为裁剪可能去除边缘的峰
    color1, color2 = find_dual_color_peaks_from_histogram_full_image(rgb_full)
    
    # 第三步：裁剪中心区域（用于后续处理和直方图显示）
    crop_ratio = float(cfg.get("crop_ratio", 0.3))
    img_cropped = crop_center_region(img_bgr, crop_ratio)
    
    # 转换为RGB（裁剪后的图像用于边缘检测和直方图显示）
    if len(img_cropped.shape) == 3:
        if img_cropped.shape[2] == 4:
            img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGRA2BGR)
        rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(img_cropped, cv2.COLOR_GRAY2RGB)
    
    # 调试：打印图像统计信息
    if cfg.get("verbose", False):
        print(f"  Cropped image shape: {rgb.shape}, dtype: {rgb.dtype}")
        print(f"  RGB value ranges - R: [{rgb[:,:,0].min()}, {rgb[:,:,0].max()}], "
              f"G: [{rgb[:,:,1].min()}, {rgb[:,:,1].max()}], "
              f"B: [{rgb[:,:,2].min()}, {rgb[:,:,2].max()}]")
    
    # 如果找不到，使用k-means作为fallback
    if color1 is None or color2 is None:
        color1, color2 = find_dual_color_peaks(rgb)
    
    # 不再使用灰度图，所有处理都在RGB空间进行
    # rgb就是裁剪后的RGB图像，用于所有后续处理
    
    # 计算裁剪后图像的蓝色通道的两个峰值，用于二值化阈值
    blue_threshold = None
    if color1 is not None and color2 is not None:
        # 在裁剪后的蓝色通道上找两个峰值
        blue_channel_cropped = rgb[:, :, 2]
        hist_blue, bins_blue = np.histogram(blue_channel_cropped.flatten(), bins=256, range=(0, 256))
        bin_centers_blue = (bins_blue[:-1] + bins_blue[1:]) / 2
        
        # 找两个峰值
        peak1_idx, peak2_idx = find_dual_peaks(hist_blue, bin_centers_blue, min_distance=10)
        
        if peak1_idx is not None and peak2_idx is not None:
            peak1_val = bin_centers_blue[peak1_idx]
            peak2_val = bin_centers_blue[peak2_idx]
            # 使用两个峰值的平均值作为阈值
            blue_threshold = (peak1_val + peak2_val) / 2.0
            
            # 应用阈值偏移
            threshold_offset = float(cfg.get("threshold_offset", 0.0))
            blue_threshold = blue_threshold + threshold_offset
            # 限制在有效范围内
            blue_threshold = max(0.0, min(255.0, blue_threshold))
    
    # 边缘增强（使用RGB颜色峰进行分割）
    min_contour_area = int(cfg.get("min_contour_area", 100))
    
    # 分割参数
    morph_open_ks = int(cfg.get("morph_open_ks", 3))  # 形态学opening核大小，用于清理分割结果
    morph_close_ks = int(cfg.get("morph_close_ks", 0))  # 形态学closing核大小（0表示不进行closing）
    
    # 边界提取参数
    boundary_kernel_size = int(cfg.get("boundary_kernel_size", 3))  # 边界提取的形态学核大小
    boundary_dilate_iterations = int(cfg.get("boundary_dilate_iterations", 1))  # 边界提取的膨胀/腐蚀迭代次数
    edge_cleanup_kernel_size = int(cfg.get("edge_cleanup_kernel_size", 2))  # 边缘清理的核大小
    edge_cleanup_iterations = int(cfg.get("edge_cleanup_iterations", 1))  # 边缘清理的迭代次数
    
    # 边界过滤参数
    border_margin = int(cfg.get("border_margin", 3))  # 图像边界过滤的像素数
    border_area_threshold = float(cfg.get("border_area_threshold", 0.1))  # 图像边框判断的面积阈值
    border_width_threshold = float(cfg.get("border_width_threshold", 0.8))  # 图像边框判断的宽度阈值
    
    edges, binary_image = enhance_edges_robust(
        rgb, color1=color1, color2=color2, blue_threshold=blue_threshold,
        min_contour_area=min_contour_area,
        morph_open_ks=morph_open_ks, morph_close_ks=morph_close_ks,
        boundary_kernel_size=boundary_kernel_size, boundary_dilate_iterations=boundary_dilate_iterations,
        edge_cleanup_kernel_size=edge_cleanup_kernel_size, edge_cleanup_iterations=edge_cleanup_iterations,
        border_margin=border_margin, border_area_threshold=border_area_threshold,
        border_width_threshold=border_width_threshold
    )
    
    # 调试：检查边缘图像，如果为空则使用fallback方法（基于RGB颜色分割）
    edge_pixel_count = np.sum(edges > 0)
    if edge_pixel_count == 0:
        # 如果边缘图像为空，尝试基于RGB的fallback方法
        # 使用k-means分割
        from oriflake.seg import kmeans_segment_lab, morph_cleanup
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        labels, _ = kmeans_segment_lab(lab, k=2)
        # 选择面积更大的区域
        mask = (labels == 0).astype(np.uint8) * 255
        if np.sum(mask) < np.sum(labels == 1):
            mask = (labels == 1).astype(np.uint8) * 255
        mask = morph_cleanup(mask, open_ks=3, close_ks=5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        edges = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_contour_area * 0.2:
                cv2.drawContours(edges, [contour], -1, 255, 1)
    
    # 第三步：边缘取向拟合
    min_line_length = float(cfg.get("min_line_length", 20.0))
    max_line_gap = float(cfg.get("max_line_gap", 10.0))
    hough_threshold = int(cfg.get("hough_threshold", 50))
    min_segment_distance = float(cfg.get("min_segment_distance", 30.0))  # 最小线段间距，避免重复统计
    
    if cfg.get("verbose", False):
        print(f"  Edge fitting parameters:")
        print(f"    min_line_length: {min_line_length} (strict threshold, short edges will be filtered)")
        print(f"    min_segment_distance: {min_segment_distance} (merge similar segments within this distance)")
    
    orientations = fit_line_segments(edges, min_line_length, max_line_gap, hough_threshold, min_segment_distance)
    
    if cfg.get("verbose", False):
        print(f"  Found {len(orientations)} edge segments after filtering (min_length={min_line_length}, min_distance={min_segment_distance})")
    
    # 提取角度列表
    angles = [o[0] for o in orientations]
    
    # 创建可视化
    # 使用裁剪后的RGB图像
    overlay = rgb.copy()
    overlay[edges > 0] = [255, 0, 0]  # 红色边缘
    
    # 绘制线段
    for angle, cx, cy in orientations:
        # 计算线段的两个端点
        length = 50
        angle_rad = np.radians(angle)
        x1 = int(cx - length * np.cos(angle_rad))
        y1 = int(cy + length * np.sin(angle_rad))
        x2 = int(cx + length * np.cos(angle_rad))
        y2 = int(cy - length * np.sin(angle_rad))
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 准备输出数据
    rows = []
    for i, (angle, cx, cy) in enumerate(orientations):
        rows.append({
            "segment_id": i,
            "theta_deg": float(angle),
            "center_x": float(cx),
            "center_y": float(cy),
        })
    
    # 提取蓝色通道作为灰度图像（用于gray.png输出）
    blue_channel = rgb[:, :, 2]
    
    return {
        "overlay": overlay,
        "edges": edges,
        "rgb": rgb,  # 裁剪后的RGB图像（用于直方图）
        "blue_gray": blue_channel,  # 蓝色通道的灰度图像（用于gray.png输出）
        "binary": binary_image,  # 二值化图像（基于蓝色通道两个峰值的平均值阈值）
        "color1": color1,  # 添加颜色峰1
        "color2": color2,  # 添加颜色峰2
        "blue_threshold": blue_threshold,  # 蓝色通道二值化阈值
        "rows": rows,
        "angles": angles,
    }


def save_rgb_histogram(rgb: np.ndarray, color1: Optional[np.ndarray], 
                      color2: Optional[np.ndarray], output_path: str,
                      blue_threshold: Optional[float] = None):
    """
    保存RGB颜色直方图，标注两个颜色峰的位置（类似analyze_histogram.py的显示方式）
    
    Args:
        rgb: RGB图像 (H, W, 3) - 使用完整原始未处理的图像
        color1: 第一个颜色峰（RGB值）
        color2: 第二个颜色峰（RGB值）
        output_path: 输出路径
        blue_threshold: 蓝色通道二值化阈值（基于裁剪后图像的蓝色通道峰值计算）
    """
    from scipy.ndimage import gaussian_filter1d
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    channel_names = ['Red', 'Green', 'Blue']
    channel_colors = ['red', 'green', 'blue']
    channels = [rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]]
    
    # 第一行：Intensity (Grayscale) 通道
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hist_gray, bins_gray = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    bin_centers_gray = (bins_gray[:-1] + bins_gray[1:]) / 2
    
    ax_gray = axes[0]
    ax_gray.bar(bin_centers_gray, hist_gray, width=1.0, alpha=0.7, color='gray', 
               edgecolor='black', linewidth=0.3)
    
    # 在灰度直方图中找峰并标注（类似analyze_histogram.py）
    hist_smooth = gaussian_filter1d(hist_gray.astype(np.float32), sigma=1.0)
    peaks, _ = signal.find_peaks(hist_smooth, distance=15, 
                                prominence=np.max(hist_smooth) * 0.05, width=2)
    if len(peaks) < 2:
        peaks, _ = signal.find_peaks(hist_smooth, distance=5, 
                                    prominence=np.max(hist_smooth) * 0.02, width=1)
    
    # 标注所有找到的峰
    for i, peak_idx in enumerate(peaks[:2]):  # 最多显示前两个峰
        peak_val = int(bin_centers_gray[peak_idx])
        ax_gray.axvline(x=peak_val, color='red' if i == 0 else 'blue', 
                       linestyle='--', linewidth=2, label=f'Peak {i+1}: {peak_val}')
        ax_gray.plot(peak_val, hist_gray[peak_val], 
                    'ro' if i == 0 else 'bo', markersize=10)
    
    # 如果提供了颜色峰，也标注（用不同样式）
    if color1 is not None and color2 is not None:
        gray1 = int(np.mean(color1))
        gray2 = int(np.mean(color2))
        if gray1 not in [int(bin_centers_gray[p]) for p in peaks[:2]]:
            ax_gray.axvline(x=gray1, color='orange', linestyle=':', linewidth=1.5, 
                          label=f'Color Peak 1: {gray1}')
        if gray2 not in [int(bin_centers_gray[p]) for p in peaks[:2]]:
            ax_gray.axvline(x=gray2, color='purple', linestyle=':', linewidth=1.5, 
                          label=f'Color Peak 2: {gray2}')
    
    ax_gray.set_xlabel('Intensity Value', fontsize=12, fontweight='bold')
    ax_gray.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax_gray.set_title(f'Intensity (Grayscale) Histogram - Found {len(peaks)} peak(s)', 
                     fontsize=13, fontweight='bold')
    ax_gray.legend(loc='upper right', fontsize=9)
    ax_gray.grid(True, alpha=0.3)
    ax_gray.set_xlim(0, 255)
    
    # R, G, B 通道
    for i, (ax, name, color, channel) in enumerate(zip(axes[1:], channel_names, 
                                                       channel_colors, channels)):
        hist, bins = np.histogram(channel.flatten(), bins=256, range=(0, 256))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # 绘制直方图
        ax.bar(bin_centers, hist, width=1.0, alpha=0.6, color=color, 
               edgecolor='black', linewidth=0.3)
        
        # 在每个通道的直方图中找峰（类似analyze_histogram.py）
        hist_smooth = gaussian_filter1d(hist.astype(np.float32), sigma=1.0)
        peaks, _ = signal.find_peaks(hist_smooth, distance=15, 
                                    prominence=np.max(hist_smooth) * 0.05, width=2)
        if len(peaks) < 2:
            peaks, _ = signal.find_peaks(hist_smooth, distance=5, 
                                        prominence=np.max(hist_smooth) * 0.02, width=1)
        
        # 标注所有找到的峰
        for j, peak_idx in enumerate(peaks[:2]):  # 最多显示前两个峰
            peak_val = int(bin_centers[peak_idx])
            ax.axvline(x=peak_val, color='red' if j == 0 else 'blue', 
                      linestyle='--', linewidth=2, label=f'Peak {j+1}: {peak_val}')
            if peak_val < len(hist):
                ax.plot(peak_val, hist[peak_val], 
                       'ro' if j == 0 else 'bo', markersize=8)
        
        # 如果提供了颜色峰，也标注（用不同样式）
        if color1 is not None and color2 is not None:
            peak1_val = int(color1[i])
            peak2_val = int(color2[i])
            if peak1_val not in [int(bin_centers[p]) for p in peaks[:2]]:
                ax.axvline(x=peak1_val, color='orange', linestyle=':', linewidth=1.5, 
                          label=f'Color Peak 1: {peak1_val}')
            if peak2_val not in [int(bin_centers[p]) for p in peaks[:2]]:
                ax.axvline(x=peak2_val, color='purple', linestyle=':', linewidth=1.5, 
                          label=f'Color Peak 2: {peak2_val}')
        
        # 在蓝色通道直方图中标注阈值
        if i == 2 and blue_threshold is not None:  # Blue channel is index 2
            threshold_val = int(blue_threshold)
            ax.axvline(x=threshold_val, color='green', linestyle='-', linewidth=2.5, 
                      label=f'Threshold: {threshold_val:.1f}', zorder=10)
            # 在阈值线上添加文本标注
            y_max = ax.get_ylim()[1]
            ax.text(threshold_val, y_max * 0.95, f'Threshold\n{threshold_val:.1f}', 
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                   color='black')
        
        ax.set_xlabel(f'{name} Channel Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'{name} Channel Histogram - Found {len(peaks)} peak(s)', 
                    fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 255)
    
    # 添加整体标题和颜色信息
    if color1 is not None and color2 is not None:
        color1_str = f'RGB({color1[0]}, {color1[1]}, {color1[2]})'
        color2_str = f'RGB({color2[0]}, {color2[1]}, {color2[2]})'
        fig.suptitle(f'RGB Color Histogram\nColor Peak 1: {color1_str} (Red) | Color Peak 2: {color2_str} (Blue)', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        # 添加颜色块显示
        fig.text(0.02, 0.02, f'Color Peak 1: {color1_str}\nColor Peak 2: {color2_str}', 
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    else:
        fig.suptitle('RGB Color Histogram\n(No color peaks detected)', 
                    fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    # 确保文件被正确写入并刷新
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)  # 明确关闭figure
    del fig  # 删除figure对象以释放内存


def save_grayscale_histogram(hist: np.ndarray, bin_centers: np.ndarray, 
                            peak1_idx: Optional[int], peak2_idx: Optional[int],
                            peak1_val: int, peak2_val: int, output_path: str):
    """
    保存灰度直方图，标注两个峰的位置，用于debug
    
    Args:
        hist: 直方图值
        bin_centers: bin中心值
        peak1_idx: 第一个峰的索引
        peak2_idx: 第二个峰的索引
        peak1_val: 第一个峰的灰度值
        peak2_val: 第二个峰的灰度值
        output_path: 输出路径
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制直方图
    ax.bar(bin_centers, hist, width=1.0, alpha=0.7, color='lightblue', edgecolor='navy', linewidth=0.5)
    
    # 标注两个峰
    if peak1_idx is not None and peak2_idx is not None:
        # 标注峰1
        ax.axvline(x=peak1_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Peak 1: {peak1_val}')
        ax.plot(peak1_val, hist[peak1_idx], 'ro', markersize=10, label=f'Peak 1: {peak1_val}')
        
        # 标注峰2
        ax.axvline(x=peak2_val, color='green', linestyle='--', linewidth=2, 
                   label=f'Peak 2: {peak2_val}')
        ax.plot(peak2_val, hist[peak2_idx], 'go', markersize=10, label=f'Peak 2: {peak2_val}')
        
        # 标注阈值（两个峰的中点）
        threshold = (peak1_val + peak2_val) / 2
        ax.axvline(x=threshold, color='orange', linestyle=':', linewidth=2, 
                   label=f'Threshold: {threshold:.1f}')
    else:
        # 如果没有找到两个峰，标注Otsu阈值
        threshold = (peak1_val + peak2_val) / 2
        ax.axvline(x=peak1_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Peak 1 (Otsu-based): {peak1_val}')
        ax.axvline(x=peak2_val, color='green', linestyle='--', linewidth=2, 
                   label=f'Peak 2 (Otsu-based): {peak2_val}')
        ax.axvline(x=threshold, color='orange', linestyle=':', linewidth=2, 
                   label=f'Threshold: {threshold:.1f}')
    
    ax.set_xlabel('Grayscale Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Grayscale Histogram (After Background Subtraction)\n' + 
                'Red/Green: Detected Peaks, Orange: Threshold', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f'Peak 1: {peak1_val}\nPeak 2: {peak2_val}\nThreshold: {(peak1_val + peak2_val) / 2:.1f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    del fig


def save_orientation_histogram(angles: List[float], output_path: str, bins: int = 36):
    """
    保存取向分布直方图
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(angles, bins=bins, range=(0, 180), alpha=0.7, edgecolor='black')
    ax.set_xlabel('Orientation Angle (degrees)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Edge Orientation Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    del fig


def main():
    parser = argparse.ArgumentParser(description="OriFlake - 纳米片边缘取向统计工具")
    default_config = os.path.join(os.path.dirname(__file__), "config.yaml")
    
    parser.add_argument("--config", type=str, default=default_config, help="配置文件路径（覆盖config.yaml）")
    parser.add_argument("--input", type=str, default=None, help="输入图像路径或文件夹（覆盖config中的设置）")
    parser.add_argument("--output", type=str, default=None, help="输出文件夹（覆盖config中的设置）")
    parser.add_argument("--preview", action="store_true", default=None, help="显示预览窗口（覆盖config中的设置）")
    parser.add_argument("--verbose", action="store_true", default=None, help="打印详细日志（覆盖config中的设置）")
    
    args = parser.parse_args()
    
    # 读取配置
    if os.path.exists(args.config):
        cfg = read_config(args.config)
    else:
        # 使用默认配置
        cfg = {
            "crop_ratio": 0.3,
            "threshold_offset": 0.0,
            "background_method": "gaussian",
            "peak_min_distance": 20,
            "min_contour_area": 100,
            "morph_open_ks": 3,
            "morph_close_ks": 0,
            "boundary_kernel_size": 3,
            "boundary_dilate_iterations": 1,
            "edge_cleanup_kernel_size": 2,
            "edge_cleanup_iterations": 1,
            "border_margin": 3,
            "border_area_threshold": 0.1,
            "border_width_threshold": 0.8,
            "min_line_length": 20.0,
            "max_line_gap": 10.0,
            "hough_threshold": 50,
            "min_segment_distance": 30.0,
            "input": "images/testImg",
            "output": "OriFlake_outputs",
            "verbose": False,
            "preview": False,
        }
        if args.verbose is None:
            args.verbose = False
    
    # 命令行参数覆盖配置文件中的设置
    if args.input is not None:
        cfg["input"] = args.input
    if args.output is not None:
        cfg["output"] = args.output
    if args.verbose is not None:
        cfg["verbose"] = args.verbose
    if args.preview is not None:
        cfg["preview"] = args.preview
    
    # 确保所有必需的参数在配置中存在（只填充缺失的键，不覆盖已存在的值）
    defaults = {
        "crop_ratio": 0.3,
        "threshold_offset": 0.0,
        "background_method": "gaussian",
        "peak_min_distance": 20,
        "min_contour_area": 100,
        "morph_open_ks": 3,
        "morph_close_ks": 0,
        "boundary_kernel_size": 3,
        "boundary_dilate_iterations": 1,
        "edge_cleanup_kernel_size": 2,
        "edge_cleanup_iterations": 1,
        "border_margin": 3,
        "border_area_threshold": 0.1,
        "border_width_threshold": 0.8,
        "min_line_length": 20.0,
        "max_line_gap": 10.0,
        "hough_threshold": 50,
        "min_segment_distance": 30.0,
        "input": "images/testImg",
        "output": "OriFlake_outputs",
        "verbose": False,
        "preview": False,
    }
    # 只填充config中不存在的键，不覆盖已存在的值
    for key, value in defaults.items():
        if key not in cfg:
            cfg[key] = value
    
    # 调试：打印配置信息（verbose模式下）
    if cfg.get("verbose", False):
        print("=" * 60)
        print("Configuration loaded from:", args.config)
        print(f"  Input: {cfg.get('input', 'NOT SET')}")
        print(f"  Output: {cfg.get('output', 'NOT SET')}")
        print(f"  Crop ratio: {cfg.get('crop_ratio', 'NOT SET')}")
        print(f"  Min line length: {cfg.get('min_line_length', 'NOT SET')}")
        print(f"  Min segment distance: {cfg.get('min_segment_distance', 'NOT SET')}")
        print(f"  Verbose: {cfg.get('verbose', 'NOT SET')}")
        print(f"  Preview: {cfg.get('preview', 'NOT SET')}")
        print(f"  Boundary kernel size: {cfg.get('boundary_kernel_size', 'NOT SET')}")
        print(f"  Edge cleanup kernel size: {cfg.get('edge_cleanup_kernel_size', 'NOT SET')}")
        print("=" * 60)
    
    # 处理相对路径
    if not os.path.isabs(cfg["input"]):
        # 如果是相对路径，相对于项目根目录
        project_root = os.path.dirname(os.path.dirname(__file__))
        cfg["input"] = os.path.join(project_root, cfg["input"])
    if not os.path.isabs(cfg["output"]):
        # 如果是相对路径，相对于项目根目录
        project_root = os.path.dirname(os.path.dirname(__file__))
        cfg["output"] = os.path.join(project_root, cfg["output"])
    
    ensure_dir(cfg["output"])
    
    # 获取图像列表
    # 如果input是文件，直接使用；如果是文件夹，使用list_images
    if os.path.isfile(cfg["input"]):
        image_paths = [cfg["input"]]
    else:
        image_paths = list_images(cfg["input"], recursive=True)
    if cfg["verbose"]:
        print(f"Found {len(image_paths)} image(s)")
        print(f"Input: {cfg['input']}")
        print(f"Output: {cfg['output']}")
    
    if not image_paths:
        print(f"No images found under {cfg['input']}")
        return
    
    # 处理所有图像
    all_rows = []
    all_angles = []
    
    for img_path in image_paths:
        if cfg["verbose"]:
            print(f"Processing: {img_path}")
        
        try:
            result = process_image(img_path, cfg)
            
            # 保存结果
            rel_path = os.path.relpath(img_path, cfg["input"]).replace("\\", "/")
            fname = os.path.splitext(os.path.basename(img_path))[0]
            
            # 保存叠加图（先删除旧文件以确保更新）
            overlay_path = os.path.join(cfg["output"], f"{fname}_overlay.png")
            if os.path.exists(overlay_path):
                try:
                    os.remove(overlay_path)
                except Exception as e:
                    if cfg.get("verbose", False):
                        print(f"  Note: Could not remove old overlay file: {e}")
            success_overlay = cv2.imwrite(overlay_path, cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR))
            if not success_overlay:
                print(f"Warning: Failed to save overlay image to {overlay_path}")
            
            # 保存边缘图
            edges_path = os.path.join(cfg["output"], f"{fname}_edges.png")
            if os.path.exists(edges_path):
                try:
                    os.remove(edges_path)
                except Exception as e:
                    if cfg.get("verbose", False):
                        print(f"  Note: Could not remove old edges file: {e}")
            success_edges = cv2.imwrite(edges_path, result["edges"])
            if not success_edges:
                print(f"Warning: Failed to save edges image to {edges_path}")
            
            # 保存蓝色通道的灰度图像（gray.png显示蓝色通道）
            gray_path = os.path.join(cfg["output"], f"{fname}_gray.png")
            if os.path.exists(gray_path):
                try:
                    os.remove(gray_path)
                except Exception as e:
                    if cfg.get("verbose", False):
                        print(f"  Note: Could not remove old gray file: {e}")
            success_gray = cv2.imwrite(gray_path, result["blue_gray"])
            if not success_gray:
                print(f"Warning: Failed to save gray image to {gray_path}")
            
            # 保存二值化图像（基于蓝色通道两个峰值的平均值阈值）
            binary_path = os.path.join(cfg["output"], f"{fname}_binary.png")
            if os.path.exists(binary_path):
                try:
                    os.remove(binary_path)
                except Exception as e:
                    if cfg.get("verbose", False):
                        print(f"  Note: Could not remove old binary file: {e}")
            success_binary = cv2.imwrite(binary_path, result["binary"])
            if not success_binary:
                print(f"Warning: Failed to save binary image to {binary_path}")
            
            # 保存RGB直方图（标注颜色峰和阈值）
            hist_path = os.path.join(cfg["output"], f"{fname}_histogram.png")
            if os.path.exists(hist_path):
                try:
                    os.remove(hist_path)
                except Exception as e:
                    if cfg.get("verbose", False):
                        print(f"  Note: Could not remove old histogram file: {e}")
            try:
                save_rgb_histogram(
                    result["rgb"],
                    result.get("color1"),
                    result.get("color2"),
                    hist_path,
                    blue_threshold=result.get("blue_threshold")
                )
                # 确保matplotlib完全关闭并刷新文件系统
                import matplotlib
                matplotlib.pyplot.close('all')
            except Exception as e:
                print(f"Warning: Failed to save histogram: {e}")
            
            if cfg.get("verbose", False):
                print(f"  Found {len(result['angles'])} edge segments")
                if result.get("color1") is not None and result.get("color2") is not None:
                    print(f"  Color peaks: {result['color1']}, {result['color2']}")
                print(f"  Images saved:")
                print(f"    - Overlay: {overlay_path}")
                print(f"    - Edges: {edges_path}")
                print(f"    - Gray: {gray_path}")
                print(f"    - Binary: {binary_path}")
                print(f"    - Histogram: {hist_path}")
            
            # 添加到总列表
            for row in result["rows"]:
                row["image"] = rel_path
                all_rows.append(row)
            all_angles.extend(result["angles"])
            
            if cfg.get("preview", False):
                # 创建预览窗口并显示图像
                preview_img = cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)
                # 如果图像太大，缩放以便显示
                max_display_size = 1920
                h, w = preview_img.shape[:2]
                if max(h, w) > max_display_size:
                    scale = max_display_size / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    preview_img = cv2.resize(preview_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                cv2.imshow("Overlay", preview_img)
                cv2.waitKey(1)  # 非阻塞等待，允许窗口更新
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            if cfg.get("verbose", False):
                import traceback
                traceback.print_exc()
    
    # 保存CSV
    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_path = os.path.join(cfg["output"], "oriflake_results.csv")
        df.to_csv(csv_path, index=False)
        if cfg.get("verbose", False):
            print(f"CSV saved: {csv_path} ({len(all_rows)} rows)")
    
    # 保存取向分布直方图
    if all_angles:
        hist_path = os.path.join(cfg["output"], "orientations_hist.png")
        save_orientation_histogram(all_angles, hist_path)
        if cfg.get("verbose", False):
            print(f"Histogram saved: {hist_path}")
    
    if cfg.get("preview", False):
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
