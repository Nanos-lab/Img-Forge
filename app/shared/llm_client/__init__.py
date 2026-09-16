"""llm_client —— 大模型调用客户端（共享基建层，Layer 1）。

提供纯函数 chat_json()，封装与 OpenAI 兼容协议的大模型 API 通信
（API Key / Base URL / Model 均从环境变量读取），返回原始 JSON 字符串。
不感知任何业务语义（图片、字段提取等），供 tools/ 下的业务模块复用。
"""

from app.shared.llm_client.service import chat_json

__all__ = ["chat_json"]
