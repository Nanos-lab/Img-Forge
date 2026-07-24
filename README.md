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
uvicorn app.main:app --reload --port 8000
```

访问 `http://localhost:8000/docs` 查看交互式 API 文档。

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
│       ├── img_enhance/         #   影像增强
│       ├── obb_detect/          #   目标检测
│       ├── img_mosaic/          #   影像拼接
│       ├── img_registration/    #   相位相关配准（内部模块，供其他模块调用）
│       └── img_ortho/           #   正射校正
├── test/                        # 测试素材
├── requirements.txt
└── README.md
```

每个工具模块独立一个文件夹，内部按 `router` / `service` / `schemas` 三层分离，互不侵入。

## 工具列表

| 工具 | 路径 | 说明 |
|------|------|------|
| 影像增强 | `POST /tools/enhance/` | 色调增强：亮度 / 对比度 / 饱和度 |
| 目标检测 | `POST /tools/obb-detect/` | YOLOv8-OBB 旋转目标检测：飞机 / 舰船 / 港口 / 桥梁 |
| 影像拼接 | `POST /tools/mosaic/` | 多 TIFF 地理参考拼接，重叠区域后覆盖前，支持相位相关精配准校正卫星定位误差 |
| 正射校正 | `POST /tools/ortho/` | RPC 有理多项式正射校正，可选 DEM 地形校正，默认 CGCS2000 高斯-克吕格输出 |

### 影像增强

对遥感影像（TIFF）进行色调增强。内部自动识别水体/植被区域做针对性色彩优化，输出影像保留原始投影、分辨率和波段数。

```bash
curl -X POST http://localhost:8000/tools/enhance/ \
  -F "file=@input.tif" \
  -o output_Enhance.tiff
```

#### 参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `file` | — | .tif/.tiff | 上传的遥感影像 |
| `brightness` | 0 | [-1, 1] | 亮度，`gamma = 2^(-brightness)`，正值提亮，负值压暗 |
| `contrast` | 0 | [-1, 1] | 对比度，正值 S 曲线增强，负值压向中灰降低 |
| `saturation` | 0 | [-1, 1] | 饱和度，`S_out = S × (1 + s)`，−1 完全去色 |

### 目标检测

对遥感影像（TIFF）进行 YOLOv8-OBB 旋转目标检测。采用滑动窗口分块策略处理大幅面影像，返回 GeoJSON 格式的检测结果（OBB 多边形 + 类别 + 置信度 + 地理坐标）。

```bash
curl -X POST http://localhost:8000/tools/obb-detect/ \
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
curl -X POST http://localhost:8000/tools/mosaic/ \
  -F "files=@a.tif" \
  -F "files=@b.tif" \
  -o output_Mosaic.tiff
```

#### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `files` | — | 多个 .tif/.tiff 文件，至少 1 个 |

### 正射校正

上传原始遥感影像进行正射校正以消除地形起伏和传感器姿态引起的几何形变。RPC 可从外置文件读取或使用 TIFF 内嵌标签。可选上传 DEM 做精确地形校正，无 DEM 时使用常量高度做平面校正。输出默认采用 **CGCS2000 高斯-克吕格 3 度带**投影，可按需指定任意 CRS。

```bash
# 方式一：TIFF 已内嵌 RPC，无需额外文件
curl -X POST http://localhost:8000/tools/ortho/ \
  -F "file=@input.tif" \
  -F "height=0" \
  -o output_Ortho.tiff

# 方式二：使用外置 RPC 文件
curl -X POST http://localhost:8000/tools/ortho/ \
  -F "file=@input.tif" \
  -F "rpc_file=@input.rpc" \
  -o output_Ortho.tiff
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `file` | file | — | 原始遥感影像（.tif / .tiff） |
| `rpc_file` | file | 可选 | RPC 系数文件（.rpc / .rpb），不传则使用 TIFF 内嵌 RPC |
| `dem_file` | file | 可选 | DEM 高程文件（.tif），用于精确地形校正 |
| `height` | float | 0 | 无 DEM 时的默认高程（米） |
| `dst_crs` | str | `"auto"` | 输出投影：`"auto"`=CGCS2000 高斯-克吕格，或 `"EPSG:xxxx"`，或 Proj4 字符串 |
| `resolution` | float | 自动 | 输出分辨率（地理单位/像素） |


## 技术栈

- **Web 框架**: FastAPI
- **影像处理**: OpenCV（含相位相关配准）
- **几何校正**: rasterio.warp 配合 RPC 模型（含 DEM 地形校正）
- **TIFF 读写**: rasterio (GDAL)
- **目标检测**: ultralytics (YOLOv8-OBB)
- **数据校验**: Pydantic v2
