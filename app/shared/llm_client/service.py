"""llm_client 服务层 —— 与 OpenAI 兼容协议大模型的通信封装。

API Key / Base URL / Model 均从环境变量读取（.env 文件），不感知具体
业务用途（图片提取、文本分类等），只提供最基础的"发消息拿 JSON"能力。
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from app.core.exceptions import LLMClientError

load_dotenv()

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


# ============================================================
#  客户端构建
# ============================================================


def get_client() -> OpenAI:
    """构建大模型客户端，API Key 从环境变量 DEEPSEEK_API_KEY 读取。

    Raises:
        LLMClientError: 未配置 DEEPSEEK_API_KEY。
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise LLMClientError("未找到环境变量 DEEPSEEK_API_KEY，请在 .env 中配置")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url, timeout=180, max_retries=2)


# ============================================================
#  对外接口
# ============================================================


def chat_json(
    messages: list[dict],
    model: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """调用大模型，强制要求返回 JSON，返回原始文本内容。

    model 默认从环境变量 DEEPSEEK_MODEL 读取，未配置时使用 DEFAULT_MODEL。
    messages 的 content 可为纯文本字符串，也可为多模态 content block 列表
    （如包含 image_url 的视觉输入），由调用方按业务需要组装。

    Args:
        messages:   聊天消息列表，格式与 OpenAI Chat Completions API 一致。
        model:      模型名称，未传时使用环境变量或默认值。
        max_tokens: 最大返回 token 数，未传时使用接口默认值。

    Returns:
        模型返回的原始 JSON 文本（未解析）。

    Raises:
        LLMClientError: 未配置 API Key，或请求失败。
    """
    client = get_client()
    resolved_model = model or os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL)
    kwargs = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    try:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
            **kwargs,
        )
    except Exception as exc:
        raise LLMClientError(f"大模型请求失败: {exc}")

    content = response.choices[0].message.content
    if not content:
        raise LLMClientError("大模型返回内容为空")
    return content
