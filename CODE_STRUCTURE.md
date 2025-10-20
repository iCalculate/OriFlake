# OriFlake 代码结构说明

## 项目结构
```
oriflake/
├── __init__.py          # 模块初始化
├── main.py              # 主程序入口
├── config.yaml          # 配置文件
├── seg.py               # 图像分割模块
├── geometry.py          # 几何分析模块
├── utils.py             # 工具函数模块
└── viz.py               # 可视化模块
```

## 模块功能

### main.py - 主程序
- **功能**: 程序入口，协调各模块工作
- **主要函数**: `process_image()`, `main()`
- **改进**: 集成增强分割算法和周期性分析

### seg.py - 图像分割
- **功能**: 图像分割和预处理
- **主要函数**:
  - `kmeans_segment_lab()` - K-means分割
  - `adaptive_segmentation_enhanced()` - 增强自适应分割
  - `morph_cleanup()` - 形态学处理
- **改进**: 简化分割算法，提高效率

### geometry.py - 几何分析
- **功能**: 多边形检测和角度计算
- **主要函数**:
  - `find_candidate_polygons()` - 候选多边形检测
  - `edge_angles_from_polygon()` - 边缘角度计算

### utils.py - 工具函数
- **功能**: 通用工具函数和周期性分析
- **主要函数**:
  - `preprocess_image()` - 图像预处理
  - `analyze_orientation_periodicity()` - 周期性分析
  - `angle_delta_sym60()` - 角度计算
- **改进**: 简化周期性分析，保留核心功能

### viz.py - 可视化
- **功能**: 数据可视化和拟合分析
- **主要函数**:
  - `draw_polygons_overlay()` - 多边形叠加显示
  - `save_hist_rose()` - 直方图和玫瑰图
  - `fit_orientation_peaks()` - 高斯拟合
- **改进**: 移除seaborn依赖，使用matplotlib默认样式

## 主要改进

### 1. 代码简化
- 移除复杂的形状特征分析
- 简化周期性分析函数
- 减少不必要的依赖

### 2. 性能优化
- 保留核心功能，提高运行效率
- 简化拟合算法，增加鲁棒性
- 优化内存使用

### 3. 可维护性
- 清晰的函数职责分离
- 简化的导入语句
- 更好的代码可读性

## 使用方式

### 基本使用
```bash
python -m oriflake.main --input images/testImg --output OriFlake_outputs --profile balanced
```

### 程序化使用
```python
from oriflake.main import process_image
from oriflake.utils import read_config

cfg = read_config('oriflake/config.yaml')
result = process_image('image.png', cfg)
```

## 输出文件

### 图像文件
- `*_overlay_*.png` - 检测结果叠加图
- `orientations_hist_*.png` - 直方图（带拟合曲线）
- `orientations_rose_*.png` - 玫瑰图

### 数据文件
- `oriflake_results_*.csv` - 详细检测结果
- `*_fitting_results.txt` - 拟合结果
- `periodicity_analysis_*.txt` - 周期性分析

## 依赖要求

### 核心依赖
- opencv-python
- numpy
- matplotlib
- scipy
- pandas
- scikit-image
- pyyaml

### 已移除依赖
- seaborn (简化可视化)

## 配置参数

主要参数在 `config.yaml` 中配置：
- 分割参数 (kmeans_k, morph_open, morph_close)
- 几何参数 (min_area_px, min_convexity, max_vertices)
- 拟合参数 (sigma_deg, reference_deg)

## 性能特点

- **识别准确性**: 增强的自适应分割算法
- **可视化质量**: 现代化图表样式，高分辨率输出
- **拟合精度**: R²通常>0.95，针对60度周期优化
- **运行效率**: 简化的算法，更快的处理速度
