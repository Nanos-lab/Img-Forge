"""img_info_extract 服务层 —— 基于大模型的图片信息提取。

核心管线:
1. 读取图片（.jpg/.jpeg/.png/.tif/.tiff），统一转为 RGB uint8
2. 按最长边上限等比缩放，避免超大影像/超限分辨率拖慢请求或超出模型限制
3. 编码为 JPEG，base64 后构造多模态消息，调用 llm_client.chat_json
4. 用 ImageInfoResult 校验解析模型返回的 JSON
"""

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio

from app.core.config import (
    INFO_EXTRACT_JPEG_QUALITY,
    INFO_EXTRACT_MAX_SIDE,
    INFO_EXTRACT_MAX_TOKENS,
)
from app.core.exceptions import InfoExtractError
from app.shared.llm_client import chat_json
from app.tools.img_info_extract.schemas import ImageInfoResult

TIFF_EXTENSIONS = {".tif", ".tiff"}

SYSTEM_PROMPT = """\
你是一个图片信息提取助手，负责从遥感影像图/影像附件图中提取以下 6 项信息：

- target：图片内部标注/图例直接指出的具体地物或对象名称（例如"跑道"
  "码头""桥梁"等），这是一个独立字段，与 title 是两个不同来源的信息，
  不要从 title 或图片大标题中推断或复制。请专注查看图片中心区域的
  文字标注、图框、箭头、引线等，这些通常指向画面中被具体标出的地物，
  target 的取值应直接来自这些标注文字本身，而不是标题里描述的整体场景
  名称（如"某机场""某港口"这类整体场景名称属于 title 的范畴，不应
  作为 target 的取值）。
- title：图片的标题文字，通常位于图片顶部，是对整幅图片的概括性描述
  （如"XX机场卫星影像图"）。
- resolution：影像的空间分辨率信息（例如像元大小、分辨率数值等）。
- platform：拍摄该影像所使用的卫星、传感器或平台名称。
- time：影像的拍摄时间或成图时间，保留图中原始出现的写法，不要换算或
  重新格式化。
- scale：图中标注的比例尺信息（如比例尺数值、比例尺条对应的长度）。

请仔细查看图片中出现的所有文字、标注、图例、图框信息，逐一判断以上 6 项
是否有对应依据。只依据图片中实际可见的内容作答，禁止根据常识猜测或编造。
如果某一项在图中找不到明确依据，对应字段必须填 null，不要留空字符串，
也不要用"未知""无"等文字代替 null。

只输出一个 JSON 对象，键名固定为：target, title, resolution, platform,
time, scale。不要输出任何其他说明文字。
"""


# ============================================================
#  图片读取与预处理
# ============================================================


def _read_tiff_as_rgb(path: str) -> np.ndarray:
    """读取 TIFF 并转为 RGB uint8 数组（供编码为 JPEG 使用）。

    多波段取前 3 个波段，单波段复制为灰度 RGB；非 uint8 数据按
    min-max 归一化后拉伸到 [0, 255]。
    """
    with rasterio.open(path) as src:
        data = src.read()

    bands = data.shape[0]
    if bands >= 3:
        rgb = data[:3]
    else:
        rgb = np.repeat(data[:1], 3, axis=0)

    rgb = np.transpose(rgb, (1, 2, 0)).astype(np.float32)
    if rgb.dtype != np.uint8:
        vmin, vmax = rgb.min(), rgb.max()
        if vmax > vmin:
            rgb = (rgb - vmin) / (vmax - vmin) * 255.0
        else:
            rgb = np.zeros_like(rgb)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _read_image_as_rgb(path: str) -> np.ndarray:
    """读取图片（任意支持格式）并统一转为 RGB uint8 数组。

    Raises:
        InfoExtractError: 无法读取图片。
    """
    ext = Path(path).suffix.lower()
    if ext in TIFF_EXTENSIONS:
        try:
            return _read_tiff_as_rgb(path)
        except Exception as exc:
            raise InfoExtractError(f"无法读取 TIFF 影像: {exc}")

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise InfoExtractError(f"无法读取图片: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resize_to_max_side(rgb: np.ndarray, max_side: int) -> np.ndarray:
    """按最长边上限等比缩放，长边已小于上限则不放大。"""
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return rgb

    scale = max_side / longest
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _encode_jpeg_base64(rgb: np.ndarray, quality: int) -> str:
    """将 RGB uint8 数组编码为 JPEG，返回 base64 字符串。"""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise InfoExtractError("图片编码为 JPEG 失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ============================================================
#  消息组装与解析
# ============================================================


def _build_messages(image_b64: str) -> list[dict]:
    """组装发送给大模型的多模态消息（system + user）。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请提取这张图片中的信息。"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        },
    ]


def _parse_result(raw_content: str) -> ImageInfoResult:
    """解析并校验模型返回的 JSON。

    Raises:
        InfoExtractError: 返回内容不是合法 JSON，或字段校验失败。
    """
    try:
        return ImageInfoResult.model_validate_json(raw_content)
    except (json.JSONDecodeError, ValueError) as exc:
        raise InfoExtractError(f"模型返回内容解析失败: {exc}；原始内容: {raw_content!r}")


# ============================================================
#  主入口
# ============================================================


def extract_image_info(src_path: str) -> ImageInfoResult:
    """从图片中提取目标名称/标题/分辨率/平台/时间/比例尺信息。

    统一将输入图片转为 JPEG 后上传给大模型，找不到依据的字段填 None。

    Args:
        src_path: 输入图片路径（.jpg/.jpeg/.png/.tif/.tiff）。

    Returns:
        提取结果，6 个字段均可能为 None。

    Raises:
        InfoExtractError: 图片读取/编码失败，或模型返回内容解析失败。
        LLMClientError:   大模型调用失败（未配置 API Key、请求出错等）。
    """
    rgb = _read_image_as_rgb(src_path)
    rgb = _resize_to_max_side(rgb, INFO_EXTRACT_MAX_SIDE)
    image_b64 = _encode_jpeg_base64(rgb, INFO_EXTRACT_JPEG_QUALITY)

    messages = _build_messages(image_b64)
    raw_content = chat_json(messages, max_tokens=INFO_EXTRACT_MAX_TOKENS)

    return _parse_result(raw_content)
