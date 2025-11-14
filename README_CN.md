# OriFlake

用于分析薄膜显微照片中三角形 MoS₂ 纳米片边缘取向分布的工具。

[English README](README.md)

## 项目概述

OriFlake 旨在处理薄膜显微照片图像，并统计分析纳米片边缘的取向分布。它支持命令行和图形界面两种方式，适用于批量处理和交互式分析。

## 主要功能

- **双界面支持**：命令行工具用于批量处理，现代化 GUI 用于交互式分析
- **强大的图像处理**：支持多种图像格式（8位、16位、32位、浮点型）
- **自动颜色峰值检测**：从 RGB 直方图中识别双颜色峰值用于阈值计算
- **边缘检测与线段拟合**：先进的边缘检测，使用 Hough 变换提取线段
- **取向统计**：全面的边缘取向统计分析
- **可视化**：生成叠加图像、边缘图、直方图和统计图表

## 安装

### 前置要求

- Python 3.7 或更高版本
- pip 包管理器

### 安装依赖

```bash
pip install -r requirements.txt
```

## 配置

项目使用 YAML 配置文件（`oriflake/config.yaml`）来控制处理参数。主要配置部分包括：

### 文件路径
- `input`: 输入图像路径（文件或文件夹）
- `output`: 输出文件夹路径

### 图像预处理
- `crop_ratio`: 中心区域裁剪比例（0.3 表示裁剪掉边缘 30%，保留中心 70%）
- `threshold_offset`: 阈值偏移量，用于微调二值化阈值（正值=更高阈值，负值=更低阈值）

### 分割参数
- `morph_open_ks`: 形态学开运算核大小，用于去除噪点
- `morph_close_ks`: 形态学闭运算核大小（0 = 禁用）

### 边缘检测参数
- `boundary_kernel_size`: 边界提取的形态学核大小
- `edge_cleanup_kernel_size`: 边缘清理的核大小
- `min_contour_area`: 最小轮廓面积（像素），用于过滤小的边缘区域

### 线段拟合参数
- `min_line_length`: 最小线段长度（像素）
- `max_line_gap`: 线段之间的最大间隙（像素）
- `hough_threshold`: Hough 变换阈值
- `min_segment_distance`: 线段之间的最小距离，用于避免重复计数

### 其他选项
- `verbose`: 启用详细日志输出
- `preview`: 处理过程中显示预览窗口

## 使用方法

### 命令行界面

使用默认配置的基本用法：

```bash
python -m oriflake.main
```

使用自定义配置文件：

```bash
python -m oriflake.main --config path/to/config.yaml
```

覆盖配置参数：

```bash
python -m oriflake.main --input path/to/images --output path/to/output --verbose
```

可用的命令行参数：
- `--config`: 配置文件路径（覆盖默认 config.yaml）
- `--input`: 输入图像路径或目录（覆盖配置）
- `--output`: 输出目录（覆盖配置）
- `--preview`: 显示预览窗口（覆盖配置）
- `--verbose`: 打印详细日志（覆盖配置）

### 图形界面

启动图形用户界面：

```bash
python oriflake_gui.py
```

GUI 提供以下功能：
- 交互式图像加载和处理
- 实时参数调整
- 处理流程的逐步可视化
- 从特定步骤开始的增量处理
- 结果和可视化的导出

## 实现细节

### 处理流程

图像处理流程包括以下步骤：

1. **图像加载与预处理**
   - 以原始位深度加载图像
   - 转换为 8 位格式（每通道 0-255 范围）
   - 支持多种格式：8 位、16 位、24 位 RGB、32 位、浮点型

2. **颜色峰值检测**
   - 分析完整图像的 RGB 直方图
   - 识别双颜色峰值（color1, color2）用于阈值计算
   - 使用完整图像以避免裁剪后丢失边缘峰值

3. **中心区域裁剪**
   - 根据 `crop_ratio` 参数裁剪中心区域
   - 减少处理区域，聚焦中心区域

4. **蓝色通道提取与阈值化**
   - 从裁剪后的 RGB 图像提取蓝色通道
   - 从双峰值平均值计算阈值
   - 应用阈值偏移进行微调
   - 生成二值掩码

5. **边缘检测与增强**
   - 从二值掩码提取边界
   - 应用形态学操作进行清理
   - 过滤小轮廓和边界区域
   - 使用鲁棒算法增强边缘

6. **线段拟合**
   - 应用 Hough 变换检测线段
   - 按最小长度和最大间隙过滤线段
   - 基于距离阈值去除重复线段
   - 从线段提取取向角度

7. **统计与可视化**
   - 计算取向统计
   - 生成带检测边缘的叠加图像
   - 创建直方图和统计图表
   - 将结果导出为 CSV 格式

### 输出文件

对于每张处理的图像，工具会生成：
- `{filename}_overlay.png`: 带有检测边缘叠加的原始图像
- `{filename}_edges.png`: 边缘图可视化
- `{filename}_gray.png`: 蓝色通道灰度图像
- `{filename}_binary.png`: 二值阈值掩码
- `{filename}_histogram.png`: 带有颜色峰值和阈值的 RGB 直方图

汇总输出：
- `oriflake_results.csv`: 包含所有检测到的边缘线段及其角度的 CSV 文件
- `orientations_hist.png`: 所有图像中所有取向角度的直方图

## 项目结构

```
OriFlake/
├── oriflake/              # 主包
│   ├── __init__.py
│   ├── main.py            # 命令行界面和核心处理
│   ├── config.yaml        # 默认配置文件
│   ├── geometry.py        # 几何操作
│   ├── orientation_stats.py  # 取向统计
│   ├── seg.py             # 分割函数
│   └── utils.py           # 工具函数
├── oriflake_gui.py        # GUI 应用程序
├── requirements.txt       # Python 依赖
├── README.md             # 本文档（英文）
└── README_CN.md          # 中文文档
```

## 依赖项

完整的依赖列表请参见 `requirements.txt`。主要包包括：
- PyQt6: GUI 框架
- OpenCV (cv2): 图像处理
- NumPy: 数值运算
- Pandas: 数据处理
- Matplotlib: 可视化
- SciPy: 科学计算
- PyYAML: 配置文件解析
- scikit-image: 额外的图像处理工具
- Seaborn: 统计可视化

## 许可证

[在此添加许可证信息]

## 贡献

[在此添加贡献指南]

## 联系方式

[在此添加联系方式]

