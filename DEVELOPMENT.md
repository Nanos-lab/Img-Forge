# ImgForge 开发规范

> 本文档供 AI 助手和开发者阅读，涵盖项目架构、环境配置、模块规范和代码风格。新增模块时请严格遵循本文档。

---

## 1. 项目概述

ImgForge 是一个模块化遥感影像处理 API 服务，面向 TIFF 格式遥感影像，提供影像增强、目标检测、裁剪、压缩等工具。项目以 FastAPI 为 Web 层、OpenCV 为像素处理引擎、rasterio 为 TIFF 读写驱动、ultralytics 为深度学习推理引擎。

**核心原则：**
- 每个工具一个独立文件夹，内部按 `router` / `service` / `schemas` 三层分离
- 一个文件只对应一类业务，不混合无关逻辑
- 所有像素级操作在 `[0, 1]` 浮点空间内完成，输入输出归一化/反归一化集中处理

---

## 2. Python 环境

### 2.1 环境管理

使用 **Anaconda** 管理 Python 环境：

| 项目 | 值 |
|------|-----|
| 环境名称 | `imgforge` |
| Python 版本 | 3.11 |
| 环境路径 | `E:/anaconda3/envs/imgforge/python` |
| 依赖文件 | `requirements.txt` |

### 2.2 创建/重建环境

```bash
conda create -n imgforge python=3.11
conda activate imgforge
conda install -c conda-forge rasterio        # GDAL 系走 conda-forge，避免编译问题
pip install -r requirements.txt
```

### 2.3 依赖清单

| 包 | 版本 | 作用 | 安装源 |
|----|------|------|--------|
| fastapi | ≥0.115.0 | Web 框架 | pip |
| uvicorn[standard] | ≥0.30.0 | ASGI 服务器 | pip |
| python-multipart | ≥0.0.9 | 文件上传 | pip |
| opencv-python-headless | ≥4.9.0 | 像素级影像处理 | pip |
| rasterio | ≥1.3.0 | TIFF 读写 + 投影保留 | conda-forge |
| numpy | ≥1.26.0 | 数值计算 | pip |
| pydantic | ≥2.0.0 | 数据校验 | pip |
| ultralytics | ≥8.0.0 | YOLOv8-OBB 目标检测 | pip |
| pytorch | (ultralytics 自动安装) | 深度学习框架 | pip |

---

## 3. 项目架构

### 3.1 目录结构

```
ImgForge/
├── app/
│   ├── __init__.py              # 空或一行注释
│   ├── main.py                  # FastAPI 入口：创建 app、注册路由、异常处理
│   ├── core/                    # 公共层（所有模块共享）
│   │   ├── __init__.py
│   │   ├── config.py            #   全局配置常量
│   │   ├── exceptions.py        #   异常类层次
│   │   └── responses.py         #   统一响应模型
│   └── tools/                   # 工具模块目录
│       ├── __init__.py
│       └── <tool_name>/         # 每个工具一个文件夹
│           ├── __init__.py      #   导出 router（API 工具）或 导出纯函数（内部模块）
│           ├── router.py        #   API 路由定义（API 工具）
│           ├── service.py       #   核心处理算法（纯函数）
│           ├── schemas.py       #   请求/响应 Pydantic 模型（API 工具）
│           └── ...              #   可含额外内部模块（如 detector.py, tiler.py）
│
│   # 注：tools 下也包含无路由的内部模块（如 img_registration），
│   # 它们不暴露 API 端点，但可被其他工具模块 import 复用。
├── test/                        # 测试素材
├── requirements.txt
├── README.md                    # 用户文档（API 列表 + 参数说明）
└── DEVELOPMENT.md               # 本文档（开发规范）
```

### 3.2 数据流

```
用户上传 TIFF
  → router.py: 校验文件格式/大小，保存到 temp/
    → service.py: rasterio 读取 → 归一化 [0,1] → 算法处理 → 反归一化 → rasterio 写出
      → router.py: FileResponse 返回结果，清理 temp 临时文件
```

**关键约定：**
- `router.py` 只做参数校验和文件传输，**不含算法逻辑**
- `service.py` 只做纯数据处理，**不依赖 FastAPI / HTTP 对象**
- `schemas.py` 只定义数据结构，**不含任何处理逻辑**
- **返回类型不限于 TIFF 文件**：模块可返回 `FileResponse`（如影像增强）、`JSONResponse`（如目标检测 GeoJSON）或其他 Response 类型
- **模块可含额外内部文件**：复杂算法可拆成多个子模块（如 `detector.py`、`tiler.py` 等），但入口仍通过 `service.py` 对外暴露
- **内部模块（无路由）**：部分模块仅提供可复用的算法，不暴露 API 端点（如 `img_registration`）。它们没有 `router.py` 和 `schemas.py`，`__init__.py` 直接导出纯函数，供其他工具模块 import 调用。

### 3.3 路由注册

所有工具路由统一在 `app/main.py` 中注册：

```python
from app.tools.<tool_name> import router as <tool>_router
app.include_router(<tool>_router)
```

路由前缀在模块的 `__init__.py` 中通过 `APIRouter` 的 `prefix` 参数定义，`main.py` 不触碰前缀细节。

### 3.4 异常处理

所有业务异常继承自 `app.core.exceptions.ImageProcessingError`，由 `main.py` 中的全局异常处理器统一捕获并返回 JSON。每个工具模块可定义自己的异常子类（放在 `core/exceptions.py` 中）。

**现有异常层次：**

```
ImageProcessingError              # 基类
├── InvalidImageError             # 无效/损坏影像
│   └── UnsupportedFormatError    #   不支持的文件格式
├── FileTooLargeError             # 文件过大
├── EnhancementError              # 增强处理失败
├── DetectionError                # 目标检测失败
└── MosaicError                   # 影像拼接失败
```

---

## 4. 工具模块规范

### 4.1 模块骨架

新增一个工具 `my_tool`（例如 `img_compress`），需创建以下文件和内容：

#### `app/tools/my_tool/__init__.py`

```python
from app.tools.my_tool.router import router

__all__ = ["router"]
```

#### `app/tools/my_tool/schemas.py`

```python
"""my_tool 模块的数据模型。"""

from pydantic import BaseModel, Field


class MyToolParams(BaseModel):
    """请求参数。"""
    param_a: float = Field(default=0.0, ge=-1.0, le=1.0, description="参数说明")
    param_b: float = Field(default=0.0, ge=-1.0, le=1.0, description="参数说明")


class MyToolResult(BaseModel):
    """处理结果。"""
    message: str = Field(description="处理结果描述")
    original_filename: str
    output_filename: str
```

**约定：**
- 浮点参数范围统一 `[-1, 1]`，默认 `0.0`（中性/无变化），除非有特殊业务含义
- Field 的 `description` 用中文，简洁说清参数含义

#### `app/tools/my_tool/service.py`

```python
"""my_tool 服务层 —— 核心处理算法。"""

from pathlib import Path
import numpy as np
import rasterio
from app.core.exceptions import EnhancementError  # 或自定义异常


# ---- 子步骤函数 ----

def _step_a(data: np.ndarray, param: float) -> np.ndarray:
    """一个函数 = 一个处理环节，可单独测试。"""
    ...


def _step_b(data: np.ndarray, param: float) -> np.ndarray:
    """另一个处理环节。"""
    ...

# ---- 主入口 ----

def process(src_path: str, param_a: float, param_b: float) -> str:
    """主处理函数。

    Args:
        src_path: 输入 TIFF 文件路径。
        param_a: 参数 A 说明。
        param_b: 参数 B 说明。

    Returns:
        输出文件路径。

    Raises:
        EnhancementError: 处理失败。
    """
    # 1. 读取
    with rasterio.open(src_path) as src:
        image = src.read()
        profile = src.profile.copy()
    # 2. 处理
    ...
    # 3. 写出（保留原始 profile 元数据）
    profile.update(driver="GTiff", compress="lzw")
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(result)
    return str(dst_path)
```

**service 层编码规范：**
- 所有处理函数为**纯函数**：接收 `np.ndarray` + 参数，返回 `np.ndarray`，不依赖全局状态或 FastAPI 对象
- 子步骤函数以 `_` 开头标记为内部函数，在模块中自底向上排列
- 函数用 `# ----` 分节注释分组
- docstring 必须写（中文），关键参数的取值范围写清楚
- 内部浮点运算统一在 `[0, 1]` 空间完成，通过归一化/反归一化函数处理不同位深

#### `app/tools/my_tool/router.py`

```python
"""my_tool 模块的 API 路由。"""

import shutil
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse

from app.core.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES, TEMP_DIR
from app.core.exceptions import EnhancementError, FileTooLargeError, UnsupportedFormatError
from app.tools.my_tool.service import process

router = APIRouter(prefix="/tools/my_tool", tags=["工具名称"])


def _validate_upload(filename: str, file_size: int) -> None:
    """校验上传文件的格式与大小。"""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(filename, ALLOWED_EXTENSIONS)
    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise FileTooLargeError(file_size, MAX_UPLOAD_SIZE_BYTES)


@router.post("/", summary="接口说明")
async def my_tool(
    file: UploadFile = File(..., description="待处理的遥感影像（.tif / .tiff）"),
    param_a: float = Form(default=0.0, ge=-1.0, le=1.0, description="参数说明"),
    param_b: float = Form(default=0.0, ge=-1.0, le=1.0, description="参数说明"),
) -> FileResponse:
    # 1. 校验
    if not file.filename:
        raise EnhancementError("未提供文件名")
    _validate_upload(file.filename, file.size or 0)

    # 2. 暂存上传文件
    temp_input = TEMP_DIR / f"{Path(file.filename).stem}_input.tif"
    try:
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise EnhancementError(f"保存上传文件失败: {exc}")

    # 3. 调用 service 处理
    try:
        output_path = process(str(temp_input), param_a, param_b)
    except EnhancementError:
        raise
    except Exception as exc:
        raise EnhancementError(f"处理过程中发生未知错误: {exc}")
    finally:
        if temp_input.exists():
            temp_input.unlink(missing_ok=True)

    # 4. 返回文件
    output_name = Path(output_path).name
    return FileResponse(
        path=output_path,
        filename=output_name,
        media_type="image/tiff",
        headers={
            "X-Original-Filename": quote(file.filename or "unknown", safe=""),
            "X-Output-Filename": quote(output_name, safe=""),
        },
    )
```

**router 层编码规范：**
- `Form` 参数的 `default`、`ge`、`le` **必须与 schemas.py 中的 Field 定义完全一致**
- 文件名放入 HTTP 头时必须用 `urllib.parse.quote(..., safe="")` 编码，否则中文文件名会在 latin-1 编码时报 `UnicodeEncodeError`
- 临时输入文件用 `finally` 确保清理
- 已知的业务异常（`EnhancementError` 等）直接 `raise`，不二次包装；未知异常包装后抛出

### 4.2 接入 main.py

在 `app/main.py` 中添加两行：

```python
from app.tools.my_tool import router as my_tool_router
app.include_router(my_tool_router)
```

---

## 5. core 层规范

### 5.1 config.py

存放全局配置常量。模块内局部配置（如默认参数值）也应放在这里统一管理。

命名约定：`UPPER_SNAKE_CASE`，分节注释用 `# === xxx ===`。

### 5.2 exceptions.py

新工具如需自定义异常，按照现有继承层次添加子类，放在 `ImageProcessingError` 之下。

异常类格式：
```python
class XxxError(ImageProcessingError):
    """异常说明。"""
    def __init__(self, detail: str = "默认信息"):
        self.detail = detail
        super().__init__(f"前缀描述: {detail}")
```

### 5.3 responses.py

存放统一的 `APIResponse` / `ErrorResponse` 模型。工具特有的响应数据模型（如处理结果元数据）应定义在 `responses.py` 中，不放在各自模块的 `schemas.py` 里，便于跨模块复用。

---

## 6. 编码规范

### 6.1 文件级

| 规则 | 说明 |
|------|------|
| 文件编码 | UTF-8 |
| 缩进 | 4 空格 |
| 文件头 | `"""模块说明（中文）。"""` docstring |
| 导入顺序 | 标准库 → 第三方库（空一行） → 项目内模块 |

### 6.2 函数/方法

- 函数名：`snake_case`，内部函数 `_` 前缀
- 所有函数**必须写 docstring**，中文 + 关键参数取值范围
- 一个函数不超过 30 行（逻辑复杂度上限），超了拆分
- 类型注解：参数和返回值均标注类型

### 6.3 分节注释

模块内用以下格式分组：
```python
# ============================================================
#  分组标题
# ============================================================
```

子步骤用 `# ---` 分隔：
```python
# ------------------------------------------------------------------
# 1. 步骤名
# ------------------------------------------------------------------
```

### 6.4 新增依赖

仅当确实需要且现有依赖无法替代时，才添加新包到 `requirements.txt`。优先使用项目已有生态：numpy 做数值、OpenCV 做图像处理、rasterio 做 TIFF 读写。

---

## 7. 启动 & 调试

```bash
conda activate imgforge
cd e:/Project/img-det/ImgForge
uvicorn app.main:app --reload --port 8100
```

- Swagger UI: `http://localhost:8100/docs`
- 健康检查: `curl http://localhost:8100/`
- 测试某接口: 在 `Swagger UI` 直接上传文件调试，或在 `temp/` 目录放测试 tif 用 curl

---

## 8. 新增模块检查清单

完成以下步骤视为一个工具模块开发完成：

- [ ] 创建 `app/tools/<name>/` 文件夹 + `__init__.py`
- [ ] 实现 `schemas.py`（参数模型，范围根据业务含义设定）
- [ ] 实现 `service.py`（纯函数，含 docstring 和类型注解）
- [ ] 实现 `router.py`（参数校验 + 文件暂存 + 调用 service + 清理 + 返回）
- [ ] 如有额外内部模块（如 detector.py），一并创建
- [ ] 在 `main.py` 中 import 并 include_router
- [ ] 如有新异常，在 `core/exceptions.py` 中定义
- [ ] 如有新配置项，在 `core/config.py` 中定义
- [ ] 如有新依赖，加入 `requirements.txt`
- [ ] 更新 `README.md`（工具列表 + 参数表格）
- [ ] 启动服务，在 Swagger UI 上传测试文件通过
