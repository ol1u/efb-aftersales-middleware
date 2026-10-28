# coding=utf-8
"""The aftersales bot: posts assistant texts into linked topics, and —
for admins only — accepts management commands and knowledge-base imports.

Commands (Telegram admins, any chat with the bot):
  /mode                  show modes and the active one
  /mode <name>           switch the active strategy mode
  /prompt <name>         show a mode's preset prompt
  /setprompt <name> | <text>   replace a mode's preset prompt
  /kb                    knowledge-base statistics
  /tiers                 show the emotion tiers and their authority limits
  /distill               distil the latest imported batch into standard cases
  /cases [page]          list the standardised cases, page by page
  /settier <tier> | <text>   replace a tier's authority text
  /threads               list problems currently in flight
Sending a Telegram export file (result.json) as a document imports it
into the knowledge base.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

from .config import Config
from .service import AfterSalesService
from .telegram_client import TelegramClient, TelegramError

logger = logging.getLogger(__name__)


class TelegramOutbox:
    def __init__(self, client: TelegramClient, config: Config):
        self.client = client
        self.config = config

    def post_group(self, group_id: str, text: str) -> None:
        topic_id: Optional[int] = self.config.groups[group_id]
        sent = self.client.send_message(self.config.telegram_group_id, text, thread_id=topic_id)
        logger.info("Assistant text sent to Telegram group=%s topic=%s message_id=%s",
                    self.config.telegram_group_id, topic_id,
                    sent.get("message_id") if isinstance(sent, dict) else None)


class CommandBot:
    def __init__(self, client: TelegramClient, config: Config, service: AfterSalesService):
        self.client = client
        self.config = config
        self.service = service
        self.admins = set(config.telegram_admins)
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.username: Optional[str] = None

    def start(self) -> None:
        if not self.admins:
            logger.info("No telegram_admins configured; command polling disabled")
            return
        self.thread = threading.Thread(target=self._poll_loop, name="aftersales-commands",
                                       daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _poll_loop(self) -> None:
        offset = None
        me_checked = False
        while not self.stop_event.is_set():
            try:
                if not me_checked:
                    me = self.client.get_me() or {}
                    self.username = me.get("username")
                    me_checked = True
                updates = self.client.get_updates(offset) or []
            except Exception as exc:
                # 原只捕获 TelegramError:get_me 回包无 result 时的 AttributeError、
                # get_updates 返回 None 后的 TypeError 等都会杀死守护线程,之后
                # 所有 bot 命令与导入永久无响应且无告警。故全量捕获并继续轮询。
                logger.warning("After-sales bot polling failed: %s", type(exc).__name__)
                self.stop_event.wait(5)
                continue
            for update in updates:
                try:
                    offset = update["update_id"] + 1
                except (KeyError, TypeError):
                    logger.warning("Skipping malformed update without update_id")
                    continue
                try:
                    self.handle_update(update)
                except Exception:
                    logger.exception("Failed to handle after-sales bot update")

    # -- dispatch ---------------------------------------------------------

    def handle_update(self, update: Dict[str, Any]) -> None:
        message = update.get("message") or {}
        user = message.get("from") or {}
        if user.get("is_bot") or user.get("id") not in self.admins:
            return
        chat_id = message["chat"]["id"]
        thread_id = message.get("message_thread_id") if message.get("is_topic_message") else None
        document = message.get("document")
        if document:
            self._import_document(chat_id, thread_id, document)
            return
        text = (message.get("text") or "").strip()
        if not text.startswith("/"):
            return
        command, _, args = text.partition(" ")
        command, _, mention = command[1:].partition("@")
        if mention and self.username and mention.lower() != self.username.lower():
            return
        reply = self.run_command(command.lower(), args.strip(), user)
        if reply:
            self.client.send_message(chat_id, reply, thread_id=thread_id)

    def _import_document(self, chat_id: int, thread_id: Optional[int],
                         document: Dict[str, Any]) -> None:
        name = document.get("file_name") or ""
        if not name.lower().endswith(".json"):
            self.client.send_message(chat_id, "只支持导入 Telegram 导出的 result.json 文件。",
                                     thread_id=thread_id)
            return
        try:
            data = self.client.download_file(document["file_id"])
            batch = "import-{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
            stats = self.service.import_export(data, batch)
        except Exception as exc:
            logger.exception("Knowledge-base import failed")
            self.client.send_message(chat_id, "导入失败：{}".format(type(exc).__name__),
                                     thread_id=thread_id)
            return
        lo, hi = stats["range"]
        span = ""
        if lo and hi:
            span = "，覆盖 {} 至 {}".format(
                datetime.fromtimestamp(lo).strftime("%Y-%m-%d"),
                datetime.fromtimestamp(hi).strftime("%Y-%m-%d"))
        note = "正在后台蒸馏标准案例，完成后用 /kb 查看。" if stats.get("distilling") else ""
        self.client.send_message(
            chat_id,
            "知识库导入完成：新增 {} 条消息（{} 个会话{}）。库内共 {} 条。{}".format(
                stats["count"], len(stats["chats"]), span, stats["total"], note),
            thread_id=thread_id)

    # -- commands ----------------------------------------------------------

    def run_command(self, command: str, args: str, user: Dict[str, Any]) -> Optional[str]:
        if command == "mode":
            return self._cmd_mode(args)
        if command == "prompt":
            modes = self.service.modes()
            if args in modes:
                return "【{}】的预设 prompt：\n{}".format(args, modes[args])
            return "没有这个模式。现有模式：" + "、".join(modes)
        if command == "setprompt":
            name, sep, prompt = args.partition("|")
            name, prompt = name.strip(), prompt.strip()
            if not sep or not name or not prompt:
                return "用法：/setprompt 模式名 | prompt 全文"
            self.service.set_mode_prompt(name, prompt)
            return "已更新模式【{}】的预设 prompt。".format(name)
        if command == "tiers":
            return self._cmd_tiers()
        if command == "settier":
            tier, sep, authority = args.partition("|")
            tier, authority = tier.strip(), authority.strip()
            tiers = self.service.tiers()
            if tier not in tiers:
                by_name = {t["name"]: key for key, t in tiers.items()}
                tier = by_name.get(tier, tier)
            if not sep or not authority or tier not in tiers:
                return "用法：/settier 档位数字或名称 | 放权说明全文"
            self.service.set_tier_authority(tier, authority)
            return "已更新第{}档【{}】的放权：{}".format(tier, tiers[tier]["name"], authority)
        if command == "distill":
            return self._cmd_distill(args)
        if command == "cases":
            return self._cmd_cases(args)
        if command == "kb":
            stats = self.service.db.kb_stats()
            if not stats["n"]:
                return "知识库是空的。把 Telegram 导出的 result.json 作为文件发给我即可导入。"
            span = ""
            if stats["lo"] and stats["hi"]:
                span = "，覆盖 {} 至 {}".format(
                    datetime.fromtimestamp(stats["lo"]).strftime("%Y-%m-%d"),
                    datetime.fromtimestamp(stats["hi"]).strftime("%Y-%m-%d"))
            standard = self.service.db.kb_standard_count()
            extra = "其中标准案例 {} 条，用 /cases 查看内容。".format(standard) if standard else ""
            return "知识库共 {} 条历史消息{}。{}".format(stats["n"], span, extra)
        if command == "threads":
            lines = []
            for group_id in self.service.config.groups:
                for t in self.service.db.recent_threads(group_id, 72 * 3600, limit=10):
                    lines.append("#{} {}：{}（v{}）".format(
                        t["id"], t["opener_name"], (t["summary"] or "")[:40], t["version"]))
            return "\n".join(lines) if lines else "当前没有进行中的问题。"
        if command == "whereami":
            return "已就绪。发 result.json 文件给我导入知识库；/mode 查看和切换策略模式；/tiers 查看情绪档位与放权。"
        return None

    def _cmd_cases(self, args: str) -> str:
        per_page = 8
        total = self.service.db.kb_standard_count()
        if not total:
            return "还没有标准案例。导入数据后会自动蒸馏，也可发新问题等其解决后自动沉淀。"
        pages = (total + per_page - 1) // per_page
        try:
            page = int(args) if args else 1
        except ValueError:
            page = 1
        page = min(max(page, 1), pages)
        rows = self.service.db.kb_standard_cases(per_page, (page - 1) * per_page)
        lines = ["标准案例（共 {} 条，第 {}/{} 页）：".format(total, page, pages)]
        for i, row in enumerate(rows, (page - 1) * per_page + 1):
            lines.append("{}. {}".format(i, row["text"]))
        if pages > 1:
            lines.append("用 /cases 页码 翻页。")
        return "\n".join(lines)

    def _cmd_distill(self, args: str) -> str:
        if not self.service._distill_available():
            return "AI 未启用或自动蒸馏已关闭，无法蒸馏。"
        batch = args or self.service.db.kb_latest_import_batch()
        if not batch:
            return "还没有导入过数据，先把 result.json 发给我导入。"
        rows = self.service.db.kb_messages_by_batch(batch)
        if not rows:
            return "找不到批次 {}。".format(batch)
        threading.Thread(target=self.service._distill_quietly, args=(batch,),
                         daemon=True, name="aftersales-distill").start()
        return "开始蒸馏批次 {}（{} 条原始记录），完成后用 /kb 查看标准案例数。".format(
            batch, len(rows))

    def _cmd_tiers(self) -> str:
        tiers = self.service.tiers()
        lines = ["情绪档位与放权（用 /settier 档位 | 放权说明 修改）："]
        for key in sorted(tiers):
            t = tiers[key]
            lines.append("第{}档【{}】（{}）：{}".format(key, t["name"], t["stance"], t["authority"]))
        return "\n".join(lines)

    def _cmd_mode(self, args: str) -> str:
        modes = self.service.modes()
        active, _ = self.service.active_mode()
        if not args:
            listing = "\n".join("{} {}".format("▶" if name == active else " ", name)
                                for name in modes)
            return "当前模式：{}\n全部模式：\n{}\n用 /mode 模式名 切换，/prompt 模式名 查看预设，/setprompt 模式名 | 内容 修改预设。".format(
                active, listing)
        if self.service.set_active_mode(args):
            return "已切换到模式【{}】，立即生效。".format(args)
        return "没有这个模式。现有模式：" + "、".join(modes)
