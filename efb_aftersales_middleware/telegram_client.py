# coding=utf-8
"""Minimal Telegram Bot API client for the after-sales bot.

Uses raw HTTP so it does not interfere with the python-telegram-bot version
pinned by ETM.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class TelegramError(RuntimeError):
    pass


class TelegramClient:
    def __init__(self, token: str, base_url: str = "https://api.telegram.org", timeout: int = 20,
                 proxy: Optional[str] = None):
        self.token = token
        self.api_root = base_url.rstrip("/")
        self.base = "{}/bot{}".format(self.api_root, token)
        self.timeout = timeout
        self.session = requests.Session()
        if proxy:
            self.session.proxies.update({"http": proxy, "https": proxy})

    def call(self, method: str, http_timeout: Optional[int] = None, files=None, **params: Any) -> Any:
        payload = {k: v for k, v in params.items() if v is not None}
        if files:
            # multipart form fields must be strings
            for key, value in list(payload.items()):
                if not isinstance(value, str):
                    payload[key] = json.dumps(value, ensure_ascii=False)
        try:
            if files:
                resp = self.session.post("{}/{}".format(self.base, method), data=payload, files=files,
                                         timeout=http_timeout or self.timeout)
            else:
                resp = self.session.post("{}/{}".format(self.base, method), json=payload,
                                         timeout=http_timeout or self.timeout)
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            # Never include the URL: it contains the bot token.
            raise TelegramError("{} failed: {}".format(method, type(exc).__name__)) from None
        if not data.get("ok"):
            raise TelegramError("{} failed: {}".format(method, data.get("description")))
        return data.get("result")

    def get_me(self) -> Dict[str, Any]:
        return self.call("getMe")

    def send_message(self, chat_id: int, text: str, thread_id: Optional[int] = None,
                     reply_markup: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.call("sendMessage", chat_id=chat_id, text=text, message_thread_id=thread_id,
                         reply_markup=reply_markup, disable_web_page_preview=True)

    def edit_message_text(self, chat_id: int, message_id: int, text: str,
                          reply_markup: Optional[Dict[str, Any]] = None) -> Any:
        return self.call("editMessageText", chat_id=chat_id, message_id=message_id, text=text,
                         reply_markup=reply_markup, disable_web_page_preview=True)

    def answer_callback(self, callback_id: str, text: str = "") -> Any:
        return self.call("answerCallbackQuery", callback_query_id=callback_id, text=text[:200])

    def send_document(self, chat_id: int, filename: str, content: bytes,
                      thread_id: Optional[int] = None, caption: str = "") -> Any:
        return self.call("sendDocument", chat_id=chat_id, message_thread_id=thread_id, caption=caption,
                         files={"document": (filename, content)}, http_timeout=max(self.timeout, 60))

    def get_updates(self, offset: Optional[int], timeout: int = 50) -> List[Dict[str, Any]]:
        return self.call("getUpdates", http_timeout=timeout + 10, offset=offset, timeout=timeout,
                         allowed_updates=["message", "callback_query"])

    def download_file(self, file_id: str) -> bytes:
        """Download a file sent to the bot (Bot API limit: 20 MB)."""
        info = self.call("getFile", file_id=file_id)
        path = info.get("file_path") if isinstance(info, dict) else None
        if not path:
            raise TelegramError("getFile returned no file_path")
        # Never log this URL: it contains the bot token.
        url = "{}/file/bot{}/{}".format(self.api_root, self.token, path)
        try:
            resp = self.session.get(url, timeout=max(self.timeout, 60))
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise TelegramError("file download failed: {}".format(type(exc).__name__)) from None
        return resp.content
