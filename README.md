# ImgForge

模块化遥感影像处理 API，基于 **FastAPI + OpenCV + rasterio**。

## 快速开始

```bash
# 创建环境（推荐）
conda create -n imgforge python=3.11
conda activate imgforge
conda install -c conda-forge rasterio
pip install -r requirements.txt

# 启动服务
uvicorn app.main:app --reload --port 8100
```

访问 `http://localhost:8100/docs` 查看交互式 API 文档。

## 项目结构

```
ImgForge/
├── app/
│   ├── main.py                  # FastAPI 入口，注册路由
│   ├── core/                    # 公共模块
│   │   ├── config.py            #   配置常量
│   │   ├── exceptions.py        #   自定义异常
│   │   └── responses.py         #   统一响应模型
│   └── tools/                   # 工具模块（每个工具一个文件夹）
│       ├── img_enhance/         #   影像增强（旧）
│       ├── img_enhance_2/       #   影像增强（Cesium 风格，当前启用）
│       ├── obb_detect/          #   目标检测
│       ├── img_mosaic/          #   影像拼接
│       ├── img_registration/    #   相位相关配准（内部模块）
│       └── img_ortho/           #   几何校正（正射校正 / 配准）
├── test/                        # 测试素材
├── requirements.txt
└── README.md
```

每个工具模块独立一个文件夹，内部按 `router` / `service` / `schemas` 三层分离，互不侵入。

## 工具列表

| 工具 | 路径 | 说明 |
|------|------|------|
| 影像增强 | `POST /tools/enhance/` | Cesium 风格色调增强：亮度 / 对比度 / 饱和度 |
| 目标检测 | `POST /tools/obb-detect/` | YOLOv8-OBB 旋转目标检测：飞机 / 舰船 / 港口 / 桥梁 |
| 影像拼接 | `POST /tools/mosaic/` | 多 TIFF 地理参考拼接，重叠区域后覆盖前，支持相位相关精配准校正卫星定位误差 |
| 几何校正 | `POST /tools/ortho/` | 正射校正（RPC）/ 图像配准（参考图），根据 `reference` 类型自动切换 |

### 影像增强

对遥感影像（TIFF）进行 Cesium 风格色调增强。无水体/植被区域检测，简洁高效。

```bash
curl -X POST http://localhost:8100/tools/enhance/ \
  -F "file=@input.tif" \
  -F "brightness=1.0" \
  -F "contrast=1.0" \
  -F "saturation=1.0" \
  -o output_Enhance.tiff
```

#### 参数

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `file` | file | — | .tif/.tiff | 上传的遥感影像 |
| `brightness` | float | 1.0 | [0, 3] | 亮度，1 原始值，0 全黑 |
| `contrast` | float | 1.0 | [0, 5] | 对比度，1 原始值，0 统一灰色 |
| `saturation` | float | 1.0 | [0, 4] | 饱和度，1 原始值，0 完全灰度 |

### 目标检测

对遥感影像（TIFF）进行 YOLOv8-OBB 旋转目标检测。采用滑动窗口分块策略处理大幅面影像，返回 GeoJSON 格式的检测结果（OBB 多边形 + 类别 + 置信度 + 地理坐标）。

```bash
curl -X POST http://localhost:8100/tools/obb-detect/ \
  -F "file=@input.tif" \
  -F "classes=0,1,7,8" \
  -F "confidence=0.25"
```

#### 参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `file` | — | .tif/.tiff | 上传的遥感影像 |
| `classes` | `"0,1,7,8"` | 逗号分隔 ID | 0:飞机 1:舰船 7:港口 8:桥梁 |
| `confidence` | 0.25 | [0.01, 1] | 置信度阈值，低于此值的检测结果被过滤 |

### 影像拼接

上传多个带地理参考的 TIFF 影像，根据坐标信息自动拼接为一张完整影像。重叠区域后覆盖前（上传顺序决定优先级）。支持对第 2 张及之后的影像进行**相位相关精配准**，自动校正不同时期影像之间的卫星定位误差，缓解重叠区重影/错位问题。

```bash
curl -X POST http://localhost:8100/tools/mosaic/ \
  -F "files=@a.tif" \
  -F "files=@b.tif" \
  -o output_Mosaic.tiff
```

#### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `files` | — | 多个 .tif/.tiff 文件，至少 1 个 |

### 几何校正

统一接口，根据 `reference` 的文件类型自动切换模式：

- **TIFF 文件** → 图像配准模式：将 `file` 配准到 `reference`，通过重叠区计算偏移并修正地理坐标
- **RPC 文件** → 正射校正模式：使用 RPC 系数消除地形起伏和传感器姿态引起的几何形变
- **不传 reference** → 正射校正模式：尝试读取 TIFF 内嵌 RPC 标签

```bash
# 配准模式：传参考 TIFF
curl -X POST http://localhost:8100/tools/ortho/ \
  -F "file=@new.tif" \
  -F "reference=@old.tif" \
  -o output.tiff

# 正射模式：传 RPC 文件
curl -X POST http://localhost:8100/tools/ortho/ \
  -F "file=@image.tif" \
  -F "reference=@image.rpc" \
  -F "height=0" \
  -o output_Ortho.tiff

# 正射模式：使用 TIFF 内嵌 RPC
curl -X POST http://localhost:8100/tools/ortho/ \
  -F "file=@image.tif" \
  -F "height=0" \
  -o output_Ortho.tiff
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `file` | file | — | 待处理的遥感影像（.tif / .tiff） |
| `reference` | file | 可选 | `.tif/.tiff`=配准模式，`.rpc/.rpb`=正射模式，不传=尝试内嵌 RPC |
| `dem_file` | file | 可选 | DEM 高程文件（.tif），正射模式可选 |
| `height` | float | 0 | 无 DEM 时的默认高程（米），正射模式 |
| `dst_crs` | str | `"auto"` | 输出投影，正射模式；`"auto"`=CGCS2000 高斯-克吕格 |
| `resolution` | float | 自动 | 输出分辨率（地理单位/像素），正射模式 |

## 技术栈

- **Web 框架**: FastAPI
- **影像处理**: OpenCV（含相位相关配准、稠密光流）
- **几何校正**: rasterio.warp 配合 RPC 模型（含 DEM 地形校正）
- **TIFF 读写**: rasterio (GDAL)
- **目标检测**: ultralytics (YOLOv8-OBB)
- **数据校验**: Pydantic v2
