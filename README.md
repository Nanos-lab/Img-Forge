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

## 环境变量（.env）

图片信息提取模块（`img_info_extract`）依赖大模型 API，需在项目根目录创建 `.env` 文件（已加入 `.gitignore`，不会被提交）：

```bash
# DeepSeek 或兼容 OpenAI 协议的大模型 API
DEEPSEEK_API_KEY=sk-xxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1   # 可选，默认官方地址
DEEPSEEK_MODEL=deepseek-chat                     # 可选，默认 deepseek-chat
```

其他工具模块不依赖 `.env`，只有使用 `/tools/info-extract/` 时才需要配置。


## 项目结构

```
ImgForge/
├── app/
│   ├── main.py                  # FastAPI 入口，注册路由
│   ├── core/                    # 应用基础设施层
│   │   ├── config.py            #   配置常量
│   │   ├── exceptions.py        #   自定义异常
│   │   └── responses.py         #   统一响应模型
│   ├── shared/                  # 共享算法层（跨工具复用，无 API 端点）
│   │   ├── img_registration/    #   相位相关配准
│   │   └── llm_client/          #   大模型调用客户端（连接/配置，不含业务逻辑）
│   └── tools/                   # 工具模块（每个工具一个文件夹）
│       ├── img_enhance/         #   影像增强（旧）
│       ├── img_enhance_2/       #   影像增强（Cesium 风格，当前启用）
│       ├── img_denoise/         #   噪声抑制与边缘增强
│       ├── obb_detect/          #   目标检测
│       ├── img_mosaic/          #   影像拼接
│       ├── img_pansharpen/      #   全色锐化（多光谱 + 全色融合）
│       ├── img_ortho/           #   几何校正（正射校正 / 配准）
│       ├── img_changedet/       #   变化检测
│       └── img_info_extract/    #   图片信息提取（基于大模型视觉理解）
├── test/                        # 测试素材
├── requirements.txt
└── README.md
```

每个工具模块独立一个文件夹，内部按 `router` / `service` / `schemas` 三层分离，互不侵入。详细的模块依赖方向、数据流、新增模块规范见 [DEVELOPMENT.md](DEVELOPMENT.md)。

## 工具列表

| 工具 | 路径 | 说明 |
|------|------|------|
| 影像增强 | `POST /tools/enhance/` | Cesium 风格色调增强：亮度 / 对比度 / 饱和度 |
| 噪声抑制与边缘增强 | `POST /tools/denoise/` | 双边滤波保边降噪 + 反锐化蒙版边缘增强，逐波段处理，任意波段数通用 |
| 目标检测 | `POST /tools/obb-detect/` | YOLOv8-OBB 旋转目标检测：飞机 / 舰船 / 港口 / 桥梁 |
| 影像拼接 | `POST /tools/mosaic/` | 多 TIFF 地理参考拼接，重叠区域后覆盖前，支持相位相关精配准校正卫星定位误差 |
| 全色锐化 | `POST /tools/pansharpen/` | 低分辨率多光谱 + 高分辨率全色融合，基于 Gram-Schmidt 算法，兼顾光谱与空间细节 |
| 几何校正 | `POST /tools/ortho/` | 正射校正（RPC）/ 图像配准（参考图），根据 `reference` 类型自动切换 |
| 变化检测 | `POST /tools/changedet/` | 基于 TinyCD 深度学习模型的双时相遥感影像变化检测，输出 GeoJSON 格式变化区域 |
| 图片信息提取 | `POST /tools/info-extract/` | 基于大模型视觉理解，提取目标名称 / 标题 / 分辨率 / 平台 / 时间 / 比例尺，图中未标注字段返回 null |

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

### 噪声抑制与边缘增强

对遥感影像（TIFF）进行噪声抑制（双边滤波，保边去噪）和边缘增强（反锐化蒙版）。逐波段独立处理，不依赖颜色语义，单波段灰度图、RGB、多光谱等任意波段数的影像均可处理。输出影像保留原始的投影、分辨率和波段数。

```bash
curl -X POST http://localhost:8100/tools/denoise/ \
  -F "file=@input.tif" \
  -F "denoise=0.5" \
  -F "sharpen=1.0" \
  -o output_Denoise.tiff
```

#### 参数

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `file` | file | — | .tif/.tiff | 上传的遥感影像 |
| `denoise` | float | 0.0 | [0, 1] | 降噪强度，0 不处理，1 最强（双边滤波） |
| `sharpen` | float | 0.0 | [0, 3] | 锐化强度，0 不处理，值越大边缘增强越明显 |

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

> **注意**：当前实现假设所有输入影像分辨率一致（画布分辨率取第一张影像的分辨率，其余影像按地理坐标计算像素偏移后直接贴入，不做重采样）。如果输入影像分辨率不一致，贴入位置会因缺少重采样而产生偏差。

### 全色锐化

上传配准好的低分辨率多光谱影像（MS）和高分辨率全色影像（Pan），基于 **Gram-Schmidt 算法**融合为高分辨率多光谱影像，兼顾光谱信息与空间细节。MS 会自动重采样对齐到 Pan 的地理网格，输出分辨率与波段数与 Pan/MS 保持一致（空间分辨率=Pan，光谱=MS）。

处理流程：
1. 校验 MS / Pan 的 CRS 一致性，Pan 必须为单波段，且 **MS 的地理范围必须完全覆盖 Pan**（不要求分辨率一致，但范围不能只有部分重叠或完全不重叠，否则直接拒绝，避免融合结果被无效区域的统计量污染）
2. 将 MS 重采样（cubic）到 Pan 的地理网格，解决分辨率对齐问题
3. 构造合成全色分量 I（MS 各波段加权平均，默认等权重）
4. 将真实 Pan 的均值/方差线性匹配到 I，避免融合后整体亮度偏移
5. 对每个 MS 波段计算 GS 增益并注入 Pan 的高频细节

```bash
curl -X POST http://localhost:8100/tools/pansharpen/ \
  -F "ms=@multispectral.tif" \
  -F "pan=@panchromatic.tif" \
  -o output_Pansharpen.tiff
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ms` | file | — | 低分辨率多光谱影像（.tif / .tiff，多波段），需与 Pan 配准到同一坐标系，地理范围必须完全覆盖 Pan |
| `pan` | file | — | 高分辨率全色影像（.tif / .tiff，单波段） |
| `weights` | str | 不传=等权重 | MS 各波段合成全色分量的权重，逗号分隔，长度需等于 MS 波段数 |

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

### 变化检测

上传两期遥感影像（变化前 + 变化后），自动进行地理配准、分块推理和变化区域提取。基于 TinyCD 轻量化深度学习模型（EfficientNet-B4 骨干网络，290K 参数），输出 GeoJSON 格式的变化区域多边形。

```bash
curl -X POST http://localhost:8100/tools/changedet/ \
  -F "old=@old.tif" \
  -F "new=@new.tif" \
  -o result.json
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `old` | file | — | 变化前遥感影像（.tif / .tiff） |
| `new` | file | — | 变化后遥感影像（.tif / .tiff） |
| `threshold` | float | Otsu 自动 | 变化概率二值化阈值 [0, 1] |
| `min_area` | int | 自动 | 最小变化区域面积（像素） |


### 图片信息提取

上传单张图片，基于大模型视觉理解提取以下 6 项信息：目标名称、标题、分辨率、平台、时间、比例尺。仅依据图片中实际可见的文字/标注作答，找不到明确依据的字段返回 `null`，不做推测或编造。

支持 `.jpg` / `.jpeg` / `.png` / `.tif` / `.tiff` 输入，内部统一转码为 JPEG（按最长边 2048px 等比缩放）后上传给模型，因此大幅遥感 TIFF 也可直接使用。

```bash
curl -X POST http://localhost:8100/tools/info-extract/ \
  -F "file=@input.jpg"
```

响应示例：

```json
{
  "target": "某机场",
  "title": "某机场卫星影像图",
  "resolution": null,
  "platform": null,
  "time": "2006年9月10日10时41分",
  "scale": "0—80米"
}
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `file` | file | — | 待提取信息的图片（.jpg/.jpeg/.png/.tif/.tiff） |

依赖 `.env` 中配置的大模型 API，见[环境变量](#环境变量env)一节。

## 技术栈

- **Web 框架**: FastAPI
- **影像处理**: OpenCV（含相位相关配准）
- **全色锐化**: rasterio.warp 重采样对齐 + Gram-Schmidt 融合
- **几何校正**: rasterio.warp 配合 RPC 模型（含 DEM 地形校正）
- **TIFF 读写**: rasterio (GDAL)
- **目标检测**: ultralytics (YOLOv8-OBB)
- **变化检测**: PyTorch + TinyCD（EfficientNet-B4）
- **图片信息提取**: OpenAI SDK（兼容 DeepSeek 等 OpenAI 协议 API）视觉理解
