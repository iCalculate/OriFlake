"""
Orientation Statistics Module
Handles orientation angle extraction, statistics, and fitting.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from .utils import angle_wrap_180


def extract_angles_from_orientations(
    orientations: List[Tuple[float, float, float]],
) -> List[float]:
    """
    Extract angle values from orientation tuples.
    
    Args:
        orientations: List of (angle, x, y) tuples
    
    Returns:
        List of angles in degrees [0, 180)
    """
    angles = []
    for angle, _, _ in orientations:
        angle = angle_wrap_180(angle)
        angles.append(angle)
    return angles


def calculate_orientation_statistics(
    angles: List[float],
    period: int = 60,
) -> Dict:
    """
    Calculate basic orientation statistics.
    
    Args:
        angles: List of orientation angles
        period: Expected periodicity (60 for triangular, 90 for square)
    
    Returns:
        Dictionary with statistics
    """
    if not angles:
        return {
            'mean': 0.0,
            'std': 0.0,
            'median': 0.0,
            'count': 0,
            'period': period,
        }
    
    angles_arr = np.array(angles)
    
    # Basic statistics
    mean_angle = float(np.mean(angles_arr))
    std_angle = float(np.std(angles_arr))
    median_angle = float(np.median(angles_arr))
    
    # Period-folded statistics
    normalized = angles_arr % period
    mean_normalized = float(np.mean(normalized))
    std_normalized = float(np.std(normalized))
    
    # Circular statistics for periodicity
    complex_angles = np.exp(1j * 2 * np.pi * normalized / period)
    mean_resultant = np.mean(complex_angles)
    mean_direction = np.angle(mean_resultant) * period / (2 * np.pi)
    mean_resultant_length = float(np.abs(mean_resultant))
    
    # Circular standard deviation
    circular_std = np.sqrt(-2 * np.log(max(mean_resultant_length, 1e-10))) * period / (2 * np.pi)
    
    return {
        'mean': mean_angle,
        'std': std_angle,
        'median': median_angle,
        'count': len(angles),
        'period': period,
        'mean_normalized': mean_normalized,
        'std_normalized': std_normalized,
        'mean_direction': float(mean_direction),
        'circular_std': float(circular_std),
        'consistency_score': mean_resultant_length,
    }



