# coding=utf-8
"""Configuration loading and validation.

The config is intentionally small: which bot posts, which WeChat groups
are watched (and into which linked Telegram topic formatted texts go),
and the AI section. Keys from older layouts are ignored on purpose so an
existing config file keeps working.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError("{} must be an integer".format(name))
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be an integer, got {!r}".format(name, value))


def _float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError("{} must be a number".format(name))
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be a number, got {!r}".format(name, value))


@dataclass
class AIConfig:
    """AI analysis. Inactive unless ``enabled`` is true and an API key is
    available (``ai.api_key`` or the ``DEEPSEEK_API_KEY`` environment
    variable). While inactive the middleware only archives messages."""
    enabled: bool = False
    api_key: Optional[str] = None
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    timeout: int = 15
    proxy: Optional[str] = None
    post_min_confidence: float = 0.6
    ask_min_confidence: float = 0.4
    prompts: Dict[str, str] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return bool(self.enabled and self.api_key)


@dataclass
class Config:
    bot_token: str
    telegram_group_id: int
    groups: Dict[str, Optional[int]]  # wechat group id -> linked Telegram topic id
    telegram_admins: Any = field(default_factory=list)  # Telegram user ids allowed to command the bot
    owner_wxid: Optional[str] = None  # the account owner's wxid; their in-group word outranks presets
    db_path: Optional[str] = None
    api_base_url: str = "https://api.telegram.org"
    proxy: Optional[str] = None
    request_timeout: int = 20
    ai: AIConfig = field(default_factory=AIConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        if not isinstance(data, dict):
            raise ValueError("config must be a mapping")
        token = data.get("bot_token")
        if not token:
            raise ValueError("bot_token is required")
        group_id = data.get("telegram_group_id")
        if group_id is None:
            raise ValueError("telegram_group_id is required")
        raw_groups = data.get("groups")
        if not isinstance(raw_groups, list) or not raw_groups:
            raise ValueError("groups must list at least one {wechat_id, topic_id} entry")
        groups: Dict[str, Optional[int]] = {}
        for item in raw_groups:
            if not isinstance(item, dict) or not item.get("wechat_id"):
                raise ValueError("Each groups entry needs wechat_id and topic_id")
            topic = item.get("topic_id")
            groups[str(item["wechat_id"])] = _int(topic, "groups.topic_id") if topic is not None else None

        raw_ai = data.get("ai") or {}
        if not isinstance(raw_ai, dict):
            raise ValueError("ai must be a mapping")
        ai_key = raw_ai.get("api_key") or os.environ.get("DEEPSEEK_API_KEY")
        ai_proxy = raw_ai.get("proxy")
        if ai_proxy is None:
            ai_proxy = data.get("proxy")
        ai_prompts = raw_ai.get("prompts") or {}
        if not isinstance(ai_prompts, dict):
            raise ValueError("ai.prompts must be a mapping")
        ai_config = AIConfig(
            enabled=bool(raw_ai.get("enabled", False)),
            api_key=str(ai_key) if ai_key else None,
            base_url=str(raw_ai.get("base_url", "https://api.deepseek.com")).rstrip("/"),
            model=str(raw_ai.get("model", "deepseek-flash")),
            timeout=_int(raw_ai.get("timeout", 15), "ai.timeout"),
            proxy=str(ai_proxy) if ai_proxy else None,
            post_min_confidence=_float(raw_ai.get("post_min_confidence", 0.6),
                                       "ai.post_min_confidence"),
            ask_min_confidence=_float(raw_ai.get("ask_min_confidence", 0.4),
                                      "ai.ask_min_confidence"),
            prompts={str(k): str(v) for k, v in ai_prompts.items() if k in ("converse",)},
        )
        return cls(
            bot_token=str(token),
            telegram_group_id=_int(group_id, "telegram_group_id"),
            groups=groups,
            telegram_admins=[_int(v, "telegram_admins") for v in (data.get("telegram_admins") or [])],
            owner_wxid=str(data["owner_wxid"]) if data.get("owner_wxid") else None,
            db_path=data.get("db_path"),
            api_base_url=str(data.get("api_base_url", "https://api.telegram.org")).rstrip("/"),
            proxy=str(data["proxy"]) if data.get("proxy") else None,
            request_timeout=_int(data.get("request_timeout", 20), "request_timeout"),
            ai=ai_config,
        )
