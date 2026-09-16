"""应用配置常量。"""

import os
from pathlib import Path

# === 项目根目录 ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# === 临时文件目录 ===
TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# === 上传限制 ===
MAX_UPLOAD_SIZE_MB = 512  # 遥感影像可能较大
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# === 允许的输入格式 ===
ALLOWED_EXTENSIONS = {".tif", ".tiff"}

# === 输出配置 ===
OUTPUT_SUFFIX = "_Enhance"
OUTPUT_EXTENSION = ".tiff"

# === 噪声抑制与边缘增强 ===
DENOISE_OUTPUT_SUFFIX = "_Denoise"

# === 默认处理参数（均 [-1, 1]，0 为中性） ===
DEFAULT_BRIGHTNESS = 0.0
DEFAULT_CONTRAST = 0.0
DEFAULT_SATURATION = 0.0
DEFAULT_WATER_FACTOR = 0.2      # 内部参数，不对外暴露
DEFAULT_VEGETATION_FACTOR = 0.18  # 内部参数，不对外暴露

# === OBB 目标检测默认参数 ===
OBB_MODEL_PATH = "models/yolov8s-obb.pt"  # 相对于模块目录
OBB_DEFAULT_CONFIDENCE = 0.25
OBB_DEFAULT_IOU = 0.45
OBB_DEFAULT_DEVICE = "cpu"
OBB_TILE_SIZE = 640
OBB_OVERLAP_RATIO = 0.15
OBB_BATCH_SIZE = 8
OBB_NMS_IOU = 0.5

# === 影像拼接 ===
MOSAIC_OUTPUT_SUFFIX = "_Mosaic"

# === 精配准参数（相位相关法） ===
REGISTRATION_ENABLED = True            # 是否启用相位相关精配准
REGISTRATION_RESPONSE_THRESHOLD = 0.15 # 相位相关响应阈值，低于此值认为配准不可靠
REGISTRATION_MAX_SHIFT = 50            # 最大允许偏移（像素），超过则回退
REGISTRATION_MIN_OVERLAP = 128         # 配准所需最小重叠尺寸（像素）

# === 正射校正 ===
ORTHO_OUTPUT_SUFFIX = "_Ortho"
ORTHO_DEFAULT_HEIGHT = 0.0            # 无 DEM 时的默认高程（米）
ORTHO_DEFAULT_RESAMPLING = "bilinear" # 重采样方法



# === 全色锐化（Pan Sharpening，Gram-Schmidt） ===
PANSHARPEN_OUTPUT_SUFFIX = "_Pansharpen"
PANSHARPEN_RESAMPLING = "cubic"        # MS 重采样到 Pan 网格的方法
PANSHARPEN_MIN_COVERAGE_RATIO = 0.999    # MS 范围覆盖 Pan 范围的最小比例，低于此值视为未完全重叠

# === 图片信息提取 ===
INFO_EXTRACT_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
INFO_EXTRACT_MAX_SIDE = 2048            # 上传前等比缩放的最长边像素上限
INFO_EXTRACT_JPEG_QUALITY = 90          # 转码为 JPEG 时的质量
INFO_EXTRACT_MAX_TOKENS = 1000          # 模型返回的最大 token 数
