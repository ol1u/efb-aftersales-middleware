# coding=utf-8
"""EFB middleware: a conversational after-sales assistant. It follows the
group conversation, summarises each problem, suggests how to handle it,
asks when unsure, and learns from imported chat history."""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from ruamel.yaml import YAML
from ehforwarderbot import Message, Middleware, coordinator
from ehforwarderbot.chat import GroupChat, SelfChatMember, SystemChatMember
from ehforwarderbot.types import InstanceID, ModuleID
from ehforwarderbot.utils import get_config_path, get_data_path

from .__version__ import __version__ as version
from .ai import DeepSeekClient
from .bot import CommandBot, TelegramOutbox
from .config import Config
from .db import Database
from .service import AfterSalesService, Incoming
from .telegram_client import TelegramClient


class AfterSalesMiddleware(Middleware):
    middleware_id: ModuleID = ModuleID("ol1u.aftersales")
    middleware_name: str = "After-sales Assistant Middleware"
    __version__: str = version
    logger = logging.getLogger("plugins.ol1u.aftersales")

    def __init__(self, instance_id: Optional[InstanceID] = None):
        super().__init__(instance_id)
        config_path: Path = get_config_path(self.middleware_id)
        if not config_path.exists():
            raise FileNotFoundError("After-sales config not found: {}".format(config_path))
        with config_path.open(encoding="utf-8") as f:
            self.config = Config.from_dict(YAML(typ="safe").load(f) or {})
        db_path = self.config.db_path or str(get_data_path(self.middleware_id) / "aftersales.db")
        self.db = Database(db_path)
        self.client = TelegramClient(self.config.bot_token, self.config.api_base_url,
                                     self.config.request_timeout, self.config.proxy)
        ai_client = None
        if self.config.ai.active:
            ai_client = DeepSeekClient(
                self.config.ai.api_key, self.config.ai.base_url, self.config.ai.model,
                self.config.ai.timeout, self.config.ai.proxy, self.config.ai.prompts)
        self.service = AfterSalesService(self.config, self.db, TelegramOutbox(self.client, self.config),
                                         ai=ai_client)
        self.bot = CommandBot(self.client, self.config, self.service)
        # Single worker keeps per-group ordering and never blocks EFB's pipeline.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aftersales")
        self.bot.start()
        if ai_client is not None:
            self.logger.info("AI assist enabled (model=%s)", self.config.ai.model)
        self.logger.info("After-sales middleware ready, db=%s, groups=%d", db_path, len(self.config.groups))

    def process_message(self, message: Message) -> Optional[Message]:
        try:
            incoming = self.to_incoming(message)
            if incoming is not None:
                self.executor.submit(self._handle, incoming)
        except Exception:
            self.logger.exception("Failed to queue after-sales message")
        return message

    def _handle(self, incoming: Incoming) -> None:
        try:
            self.service.handle(incoming)
        except Exception:
            self.logger.exception("Failed to process after-sales message %s", incoming.msg_id)

    def to_incoming(self, message: Message) -> Optional[Incoming]:
        """Convert WeChat -> Telegram group messages; ignore everything else."""
        master = getattr(coordinator, "master", None)
        if master is None or message.deliver_to is not master or message.edit:
            return None
        chat, author = message.chat, message.author
        if not isinstance(chat, GroupChat) or str(chat.uid) not in self.config.groups:
            return None
        if isinstance(author, SystemChatMember):
            return None
        target = getattr(message, "target", None)
        quoted_msg_id = None
        if target is not None and getattr(target, "uid", None):
            quoted_msg_id = str(target.uid)
        quoted_record_id = None
        if quoted_msg_id:
            group_id = str(chat.uid)
            record = self.db.find_record_by_msg_id(group_id, quoted_msg_id)
            if record is None and master is not None:
                record = self._resolve_via_master(master, chat, group_id, quoted_msg_id)
            if record is not None:
                quoted_record_id = record["id"]
        return Incoming(
            group_id=str(chat.uid),
            group_name=chat.display_name,
            sender_id=str(author.uid),
            sender_name=author.name,
            sender_alias=author.alias,
            is_self=isinstance(author, SelfChatMember),
            msg_type=message.type.name,
            text=message.text or "",
            msg_id=str(message.uid) if message.uid else None,
            quoted_msg_id=quoted_msg_id,
            quoted_record_id=quoted_record_id,
        )

    def _resolve_via_master(self, master, chat, group_id: str, quoted_msg_id: str):
        """Resolve a quote target the archive does not know.

        The target may be a message the master itself sent (our posts never
        come back through process_message, so their WeChat ids are missing
        from the archive). Ask the master what it sent under that id, match
        the text against the archive, and remember the id for next time.
        """
        try:
            sent = master.get_message_by_id(chat=chat, msg_id=quoted_msg_id)
        except Exception:
            return None
        text = getattr(sent, "text", None) if sent is not None else None
        if not text:
            return None
        record = (self.db.find_posted_record(group_id, text)
                  or self.db.find_record_by_text(group_id, text))
        if record is not None and not record["efb_msg_id"]:
            self.db.set_record_msg_id(record["id"], quoted_msg_id)
        return record
