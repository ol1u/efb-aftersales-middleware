# coding=utf-8
"""The conversational after-sales assistant.

Every message in the configured group is archived and shown to the AI
together with the active strategy (mode), similar historical cases and
the problems currently in flight. The AI then does exactly one of:

- open a new problem thread: post 【问题总结】+【建议方案】 quoting the
  problem message;
- update a thread (the summary was corrected, key facts arrived): post
  the revised summary + plan quoting the thread's opening message;
- ask: it cannot tell what the message belongs to — post a short
  clarifying question quoting the message itself;
- stay silent (the message changes nothing).

Attribution leans on the traces the group already produces — native
quotes, the copy-the-full-text-and-append habit, the sender's identity —
and the AI arbitrates the rest. Several problems can be in flight at
once, from several people. Our own posts are recognised on return and
never reprocessed.
"""

import json
import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

from . import distill as distill_module
from . import kb as kb_module
from .config import Config
from .db import Database
from .rules import split_quote

logger = logging.getLogger(__name__)

TEXTUAL_TYPES = {"Text", "Link"}
QUOTE_DIVIDER = " - - - - - - - - - - - - - - - "
QUOTE_HINT_FOOTER = "（补充或纠正时，请引用本条消息回复，我能准确对应到这个问题。）"
DUPLICATE_RESEND = "__duplicate_resend__"
THREAD_WINDOW_SECONDS = 72 * 3600

DEFAULT_MODES: Dict[str, str] = {
    "常规": (
        "当前是常规售后策略：先核实问题再给方案；能维修、补发小配件解决的，不整单退款；"
        "责任在买家时婉转说明并给台阶，责任在我方时爽快承担。兼顾客户感受和处理成本，不激化矛盾。"
    ),
    "保评分": (
        "当前阶段首要目标是保住店铺评分：客户满意、不给差评优先于单笔成本，这类单子可以亏给客户。"
        "倾向于主动提出退款、补发或补偿，先安抚情绪再处理问题；避免让客户升级到平台投诉或差评。"
    ),
}
_MODES_KEY = "modes_v1"

_TIERS_KEY = "emotion_tiers"

DEFAULT_TIERS: Dict[str, Dict[str, str]] = {
    "1": {"name": "平稳", "stance": "温和", "signals": "语气正常，咨询或小问题",
          "authority": "不额外让步，按正常流程：核实问题、给明确处理时限。"},
    "2": {"name": "不满", "stance": "温和", "signals": "抱怨、催促、语气变差",
          "authority": "可送小礼品或耗材安抚、可免邮补件；不退款、不折价。"},
    "3": {"name": "愤怒", "stance": "激进", "signals": "明确生气，提到差评、投诉",
          "authority": "可部分退款或折价（不超过订单金额的 30%）、可补发且旧件不退回；超出此范围须请示老板。"},
    "4": {"name": "激烈", "stance": "激进", "signals": "威胁平台投诉或曝光、已经差评、扬言退货退款",
          "authority": "可全额退款、退货运费由我方承担、额外补偿不超过 100 元；超出此范围须请示老板。"},
}


@dataclass
class Incoming:
    group_id: str
    group_name: str
    sender_id: str
    sender_name: str
    sender_alias: Optional[str]
    is_self: bool
    msg_type: str
    text: str
    msg_id: Optional[str] = None
    quoted_msg_id: Optional[str] = None
    quoted_record_id: Optional[int] = None

    @property
    def display_name(self) -> str:
        return self.sender_alias or self.sender_name


class Outbox(Protocol):
    def post_group(self, group_id: str, text: str) -> None: ...


class AIAssist(Protocol):
    def converse(self, *, mode_name: str, mode_prompt: str, cases: str, threads: str,
                 hint: str, group_name: str, sender_label: str,
                 text: str, tiers: str = "") -> Optional[Dict[str, Any]]: ...


def quote_block(name: str, quoted_text: str, body: str) -> str:
    """A text-form quote in the exact shape ComWeChat renders quotes in."""
    return "「{}：{}」\n{}\n{}".format(name, quoted_text, QUOTE_DIVIDER, body)


def _hint_tag(hint: str, quoted_msg_id: Optional[str]) -> str:
    """Short label for a trace outcome, for the decision log line."""
    if hint == DUPLICATE_RESEND:
        return "duplicate"
    if "提问帖" in hint:
        return "quote-ask"
    if "尚未成问题" in hint:
        return "quote-unanchored"
    if "引用了问题编号" in hint:
        return "quote-id" if quoted_msg_id else "quote-text"
    if "复制了问题编号" in hint:
        return "copy"
    if "案例卡一致" in hint:
        return "card-match"
    if "进行中的问题编号" in hint:
        return "own-thread"
    return "none" if not hint else "other"



EMPTY_CARD: Dict[str, Any] = {"customer": "", "phone": "", "order_no": "", "platform": "",
                              "emotion": "", "spec": "", "issue": "", "demand": "",
                              "status": "", "facts": []}

RESOLUTION_SIGNALS = ("已解决", "解决了", "处理好了", "搞定了", "客户接受了", "客户同意了",
                      "已补发", "已退款", "已换新", "没问题了", "已经处理好")


def parse_card(raw: Optional[str]) -> Dict[str, Any]:
    """Parse a stored thread card; anything unusable becomes an empty card."""
    if raw:
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            value = None
        if isinstance(value, dict):
            card = dict(EMPTY_CARD)
            for key in EMPTY_CARD:
                if key in value:
                    card[key] = value[key]
            if not isinstance(card["facts"], list):
                card["facts"] = []
            return card
    return dict(EMPTY_CARD)


def card_json(card: Dict[str, Any]) -> str:
    return json.dumps(card, ensure_ascii=False)


def _order_tail(order_no: Optional[str]) -> str:
    text = (order_no or "").strip()
    return text[-4:] if len(text) > 4 else text


def assemble_summary(card: Dict[str, Any]) -> str:
    """The one-line problem summary, assembled from the case card."""
    bits = []
    head = card.get("customer") or ""
    tail = _order_tail(card.get("order_no"))
    if head and tail:
        bits.append("{}（订单尾号 {}）".format(head, tail))
    elif head:
        bits.append(head)
    elif tail:
        bits.append("订单尾号 " + tail)
    if card.get("spec"):
        bits.append(card["spec"])
    if card.get("issue"):
        bits.append(card["issue"])
    if card.get("demand"):
        bits.append("诉求：" + card["demand"])
    return "；".join(bits) or "（信息待补充）"


def card_line(card: Dict[str, Any]) -> str:
    """Compact one-line rendering of a card for the AI context."""
    fields = []
    for key, label in (("customer", "客户"), ("phone", "电话"), ("order_no", "订单"),
                       ("platform", "平台"), ("spec", "规格"), ("issue", "问题"),
                       ("demand", "诉求"), ("emotion", "情绪"), ("status", "状态")):
        if card.get(key):
            fields.append("{}：{}".format(label, card[key]))
    line = "｜".join(fields)
    if card.get("facts"):
        line += "｜已确认：" + "；".join(str(f) for f in card["facts"])
    return line


def quote_excerpt(text: Optional[str], limit: int = 44) -> str:
    """A short human-facing excerpt for a post's quote block.

    Attribution is id-based, so the quote only needs to tell people which
    message a post answers. Order dumps end with the actual problem, so the
    last paragraph is the most informative part; quoting it also stops the
    assistant from rebroadcasting full addresses and phone numbers.
    """
    whole = " ".join((text or "").split())
    if len(whole) <= limit:
        return whole
    paragraphs = [p for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    if paragraphs:
        last = " ".join(paragraphs[-1].split())
        if len(last) >= 10:
            return last[:limit] + ("…" if len(last) > limit else "")
    return whole[:limit] + "…"


def _resolution_signal(text: Optional[str]) -> bool:
    return any(signal in (text or "") for signal in RESOLUTION_SIGNALS)

class AfterSalesService:
    def __init__(self, config: Config, db: Database, outbox: Outbox,
                 ai: Optional[AIAssist] = None):
        self.config = config
        self.db = db
        self.outbox = outbox
        self.ai = ai
        self.lock = threading.RLock()

    # -- modes -------------------------------------------------------------

    def _modes_state(self) -> Dict[str, Any]:
        raw = self.db.get_setting(_MODES_KEY)
        if raw:
            try:
                state = json.loads(raw)
                if isinstance(state, dict) and state.get("modes"):
                    return state
            except ValueError:
                pass
        state = {"modes": dict(DEFAULT_MODES), "active": "常规"}
        self.db.set_setting(_MODES_KEY, json.dumps(state, ensure_ascii=False))
        return state

    def modes(self) -> Dict[str, str]:
        return dict(self._modes_state()["modes"])

    def active_mode(self) -> Tuple[str, str]:
        state = self._modes_state()
        name = state.get("active") or "常规"
        return name, state["modes"].get(name) or DEFAULT_MODES["常规"]

    def set_active_mode(self, name: str) -> bool:
        state = self._modes_state()
        if name not in state["modes"]:
            return False
        state["active"] = name
        self.db.set_setting(_MODES_KEY, json.dumps(state, ensure_ascii=False))
        return True

    def set_mode_prompt(self, name: str, prompt: str) -> None:
        state = self._modes_state()
        state["modes"][name] = prompt
        self.db.set_setting(_MODES_KEY, json.dumps(state, ensure_ascii=False))

    # -- emotion tiers -------------------------------------------------------

    def _tiers_state(self) -> Dict[str, Any]:
        raw = self.db.get_setting(_TIERS_KEY)
        if raw:
            try:
                state = json.loads(raw)
                if isinstance(state, dict) and state.get("tiers"):
                    return state
            except ValueError:
                pass
        state = {"tiers": {k: dict(v) for k, v in DEFAULT_TIERS.items()}}
        self.db.set_setting(_TIERS_KEY, json.dumps(state, ensure_ascii=False))
        return state

    def tiers(self) -> Dict[str, Dict[str, str]]:
        return self._tiers_state()["tiers"]

    def tiers_text(self) -> str:
        lines = []
        tiers = self.tiers()
        for key in sorted(tiers):
            t = tiers[key]
            lines.append("第{}档【{}】（{}）：信号——{}；放权——{}".format(
                key, t["name"], t["stance"], t["signals"], t["authority"]))
        return "\n".join(lines)

    def set_tier_authority(self, tier: str, authority: str) -> bool:
        state = self._tiers_state()
        if tier not in state["tiers"]:
            return False
        state["tiers"][tier]["authority"] = authority
        self.db.set_setting(_TIERS_KEY, json.dumps(state, ensure_ascii=False))
        return True

    # -- knowledge base --------------------------------------------------------

    def import_export(self, data: bytes, batch: str) -> Dict[str, Any]:
        messages = kb_module.parse_export(data)
        if messages:
            self.db.kb_add_messages(batch, messages)
        chats = sorted({m["chat_key"] for m in messages})
        stamps = [m["sent_at"] for m in messages if m["sent_at"]]
        result = {
            "count": len(messages),
            "chats": chats,
            "range": (min(stamps), max(stamps)) if stamps else (None, None),
            "total": self.db.kb_stats()["n"],
        }
        if messages and self._distill_available():
            threading.Thread(target=self._distill_quietly, args=(batch,),
                             daemon=True, name="aftersales-distill").start()
            result["distilling"] = True
        return result

    # -- distillation ----------------------------------------------------------

    def _distill_available(self) -> bool:
        return (self.ai is not None and self.config.ai.auto_distill
                and callable(getattr(self.ai, "chat_json", None)))

    def _distill_quietly(self, batch: str) -> None:
        try:
            self.distill_batch(batch)
        except Exception:
            logger.exception("distillation failed for batch %s", batch)

    @staticmethod
    def distilled_name(batch: str) -> str:
        suffix = batch[len("import-"):] if batch.startswith("import-") else batch
        return "distilled-" + suffix

    def distill_batch(self, batch: str) -> int:
        """Distill one raw batch into standard cases; returns the new count."""
        if not self._distill_available():
            return 0
        rows = self.db.kb_messages_by_batch(batch)
        if not rows:
            return 0
        cases = distill_module.distill(self.ai.chat_json, [dict(r) for r in rows])
        fresh = distill_module.dedupe_cases(cases, self.db.kb_standard_texts())
        if fresh:
            self.db.kb_add_messages(self.distilled_name(batch), fresh)
        logger.info("batch %s distilled: %s new standard cases", batch, len(fresh))
        return len(fresh)

    # -- inbound ---------------------------------------------------------------

    def handle(self, msg: Incoming) -> None:
        if msg.group_id not in self.config.groups:
            return
        with self.lock:
            self._process(msg)

    def _process(self, msg: Incoming) -> None:
        raw = msg.text or ""
        if raw:
            posted = self.db.find_posted_record(msg.group_id, raw)
            if posted is not None:
                # Our own post relayed back; never reprocess. Remember its
                # WeChat message id so later quotes of it can be resolved.
                backfilled = bool(msg.msg_id and not posted["efb_msg_id"])
                if backfilled:
                    self.db.set_record_msg_id(posted["id"], msg.msg_id)
                logger.info("own post returned: record=%s msg_id=%s backfilled=%s",
                            posted["id"], msg.msg_id, backfilled)
                return
        if msg.msg_type not in TEXTUAL_TYPES:
            self._archive(msg, raw or "[{}]".format(msg.msg_type))
            return

        hint_thread, hint_text, content = self._trace(msg, raw)
        record = self._archive(msg, raw)
        logger.info("trace msg=%s quoted=%s qrec=%s kind=%s thread=%s",
                    msg.msg_id, msg.quoted_msg_id, msg.quoted_record_id,
                    _hint_tag(hint_text, msg.quoted_msg_id), hint_thread)
        if hint_text == DUPLICATE_RESEND and hint_thread is not None:
            self.db.set_record_thread(record["id"], hint_thread)
            self.db.touch_thread(hint_thread)
            body = "这条已经处理过了（问题 #{}），是有什么需要补充的吗？".format(hint_thread)
            self._post(msg, quote_block(msg.display_name, quote_excerpt(raw), body), hint_thread)
            logger.info("decision msg=%s -> duplicate notice for thread %s",
                        msg.msg_id, hint_thread)
            return
        if self.ai is None:
            return

        threads = self.db.recent_threads(msg.group_id, THREAD_WINDOW_SECONDS, limit=8)
        if hint_thread is not None and all(t["id"] != hint_thread for t in threads):
            extra = self.db.get_thread(hint_thread)
            if extra is not None:
                threads = [extra] + list(threads)
        mode_name, mode_prompt = self.active_mode()
        cases = kb_module.search_cases(self.db, content) if self.db.kb_stats()["n"] else ""
        sender_label = msg.display_name
        if msg.is_self or (self.config.owner_wxid and msg.sender_id == self.config.owner_wxid):
            sender_label += " [群主]"
        try:
            result = self.ai.converse(
                mode_name=mode_name, mode_prompt=mode_prompt, tiers=self.tiers_text(), cases=cases,
                threads=self._threads_text(threads), hint=hint_text,
                group_name=msg.group_name, sender_label=sender_label, text=content)
        except Exception:
            logger.exception("AI converse failed; message kept as archive only")
            return
        if result is None:
            logger.info("decision msg=%s -> silent (no ai result)", msg.msg_id)
            return
        threshold = (self.config.ai.ask_min_confidence if result["action"] == "ask"
                     else self.config.ai.post_min_confidence)
        if result["confidence"] < threshold:
            logger.info("decision msg=%s -> silent (action=%s confidence=%.2f < %.2f)",
                        msg.msg_id, result["action"], result["confidence"], threshold)
            return
        logger.info("decision msg=%s action=%s confidence=%.2f",
                    msg.msg_id, result["action"], result["confidence"])

        action = result["action"]
        card = result["card"]
        summary = assemble_summary(card)
        if action == "new_problem":
            thread = self.db.create_thread(
                group_id=msg.group_id, opener_id=msg.sender_id, opener_name=msg.display_name,
                summary=summary, plan=result["plan"], card=card_json(card))
            self.db.set_record_thread(record["id"], thread["id"])
            body = "【问题总结】\n{}\n【建议方案】\n{}".format(summary, result["plan"])
            self._post(msg, quote_block(msg.display_name, quote_excerpt(raw), body), thread["id"])
            if card.get("status") == "已解决":
                self._archive_case(thread["id"], card, result["plan"], msg, announce=None)
        elif action in ("update", "note"):
            thread = self._resolve_thread(msg.group_id, result.get("thread_id"), hint_thread, threads)
            if thread is None:
                return
            self.db.set_record_thread(record["id"], thread["id"])
            if thread["status"] != "open":
                self.db.reopen_thread(thread["id"])
            if action == "note":
                self.db.touch_thread(thread["id"])
                if _resolution_signal(content):
                    self._archive_case(thread["id"], parse_card(thread["card"]),
                                       thread["plan"], msg, announce=msg)
                return
            self.db.update_thread(thread["id"], summary, result["plan"], card=card_json(card))
            opening = self.db.thread_opening(thread["id"])
            q_name = opening["sender_name"] if opening is not None else thread["opener_name"]
            q_text = opening["raw_text"] if opening is not None else raw
            body = "【进展】{}\n【当前方案】\n{}".format(result["progress"], result["plan"])
            self._post(msg, quote_block(q_name, quote_excerpt(q_text), body), thread["id"])
            if card.get("status") == "已解决" or _resolution_signal(content):
                self._archive_case(thread["id"], card, result["plan"], msg, announce=msg)
        elif action == "ask":
            thread_id = result.get("thread_id") or hint_thread
            body = "【确认一下】\n{}".format(result["question"])
            self._post(msg, quote_block(msg.display_name, quote_excerpt(raw), body),
                       thread_id, target_id=record["id"])

    # -- resolution ------------------------------------------------------------

    def _archive_case(self, thread_id: int, card: Dict[str, Any], plan: Optional[str],
                      msg: Incoming, announce: Optional[Incoming]) -> None:
        """Close a thread and distill it into a knowledge-base case."""
        self.db.resolve_thread(thread_id)
        case_text = "已解决案例：{}｜规格：{}｜最终方案：{}".format(
            assemble_summary(card), card.get("spec") or "—", plan or "—")
        self.db.kb_add_messages("resolved-cases", [{
            "chat_key": "resolved:{}".format(msg.group_id),
            "sender_name": "已解决案例", "sent_at": self.db.now(), "text": case_text}])
        logger.info("thread #%s resolved; distilled into the knowledge base", thread_id)
        if announce is not None:
            body = "问题 #{} 已处理完，处理结果已存入案例库。".format(thread_id)
            self._post(announce,
                       quote_block(announce.display_name, quote_excerpt(announce.text), body),
                       thread_id)

    # -- helpers -----------------------------------------------------------------

    def _trace(self, msg: Incoming, raw: str) -> Tuple[Optional[int], str, str]:
        """Deterministic attribution traces: (thread_id, hint text, content)."""
        record = None
        if msg.quoted_record_id is not None:
            record = self.db.get_record(msg.quoted_record_id)
        if record is None and msg.quoted_msg_id:
            record = self.db.find_record_by_msg_id(msg.group_id, msg.quoted_msg_id)
        if record is not None:
            if record["thread_id"]:
                return record["thread_id"], "这条消息引用了问题编号 {} 中的消息。".format(
                    record["thread_id"]), raw
            if record["kind"] == "posted":
                # One of our posts with no thread (an ask): recover the
                # message it asked about so the AI can use the answer.
                original = ""
                if record["target_id"]:
                    target = self.db.get_record(record["target_id"])
                    if target is not None:
                        original = target["raw_text"] or ""
                if not original:
                    _, asked_about, _ = split_quote(record["raw_text"] or "")
                    original = asked_about or (record["raw_text"] or "")
                return None, (
                    "这条消息引用了你之前发出的一条提问帖，你当时是就下面这条消息提问的，"
                    "对方这条是回答。请结合原消息和回答判断：信息足够就按 new_problem "
                    "给出总结和方案；仍不足就继续 ask。被提问的原消息：「{}」".format(
                        original[:600])), raw
            return None, (
                "这条消息引用了下面这条尚未成问题的消息，请结合它判断："
                "「{}」".format((record["raw_text"] or "")[:600])), raw
        _, quoted_text, body = split_quote(raw)
        if quoted_text:
            record = self.db.find_record_by_text(msg.group_id, quoted_text)
            if record is not None and record["thread_id"]:
                return record["thread_id"], "这条消息引用了问题编号 {} 中的消息。".format(
                    record["thread_id"]), body
            return None, "", body
        found = self.db.find_copy_source(msg.group_id, raw)
        if found is not None:
            source, delta = found
            if source["thread_id"]:
                if not delta:
                    # Verbatim resend of a message already in a thread:
                    # handled deterministically in _process, no AI call.
                    return source["thread_id"], DUPLICATE_RESEND, raw
                return source["thread_id"], "这条消息开头复制了问题编号 {} 中消息的全文，正文是追加的新内容。".format(
                    source["thread_id"]), delta
            # Copy of a message that never became a thread: appended text is
            # judged on its own; a verbatim resend is judged as the full text.
            return None, "", delta if delta else raw
        card_thread = self._card_match_thread(msg.group_id, raw)
        if card_thread is not None:
            return card_thread, "这条消息里的订单号/电话与问题编号 {} 的案例卡一致。".format(
                card_thread), raw
        own = [t for t in self.db.recent_threads(msg.group_id, THREAD_WINDOW_SECONDS, limit=8)
               if t["opener_id"] == msg.sender_id]
        if own:
            return None, "发送者自己有一个进行中的问题编号 {}，请判断这条是它的补充还是新问题，分不清就 ask。".format(
                own[0]["id"]), raw
        return None, "", raw

    def _resolve_thread(self, group_id: str, chosen: Optional[int], hint_thread: Optional[int],
                        threads: List[Any]) -> Optional[Any]:
        wanted = chosen or hint_thread
        if wanted is None:
            return None
        for thread in threads:
            if thread["id"] == wanted:
                return thread
        thread = self.db.get_thread(wanted)
        if thread is not None and thread["group_id"] == group_id and thread["status"] == "open":
            return thread
        return None

    def _card_match_thread(self, group_id: str, text: str) -> Optional[int]:
        """Deterministic attribution: the message carries the phone number
        or order number of an open thread's case card."""
        phones = set(re.findall(r"1\d{10}", text or ""))
        orders = {m.upper() for m in re.findall(r"SF\d{8,}", text or "", flags=re.I)}
        orders |= set(re.findall(r"\d{15,20}", text or ""))
        if not phones and not orders:
            return None
        for thread in self.db.recent_threads(group_id, THREAD_WINDOW_SECONDS, limit=8):
            card = parse_card(thread["card"])
            card_phones = set(re.findall(r"1\d{10}", card.get("phone") or ""))
            if phones and card_phones and phones & card_phones:
                return thread["id"]
            card_order = (card.get("order_no") or "").upper()
            if card_order and any(card_order == o or card_order in o or o in card_order
                                  for o in orders if len(o) >= 8):
                return thread["id"]
        return None

    def _threads_text(self, threads: List[Any]) -> str:
        blocks = []
        for t in threads:
            line = card_line(parse_card(t["card"]))
            lines = [
                "问题编号 {}（提出者：{}，已更新 {} 版）".format(t["id"], t["opener_name"], t["version"]),
                "案例卡：{}".format(line) if line else "当前总结：{}".format(t["summary"] or ""),
                "当前方案：{}".format(t["plan"] or ""),
            ]
            msgs = self.db.thread_messages(t["id"], limit=3)
            if msgs:
                lines.append("最近消息：" + "；".join(
                    "{}：{}".format(m["sender_name"], (m["raw_text"] or "")[:60]) for m in msgs))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def _archive(self, msg: Incoming, raw: str):
        return self.db.add_record(
            group_id=msg.group_id, sender_id=msg.sender_id, sender_name=msg.display_name,
            kind="message", raw_text=raw, efb_msg_id=msg.msg_id)

    def _post(self, msg: Incoming, text: str, thread_id: Optional[int],
              target_id: Optional[int] = None) -> None:
        if QUOTE_HINT_FOOTER not in text:
            text = text.rstrip() + "\n" + QUOTE_HINT_FOOTER
        self.outbox.post_group(msg.group_id, text)
        self.db.add_record(
            group_id=msg.group_id, sender_id=None, sender_name="AI助理", kind="posted",
            raw_text=text, thread_id=thread_id, target_id=target_id)
        logger.info("Posted assistant text to group %s (thread %s)", msg.group_id, thread_id)


__all__ = ["AfterSalesService", "Incoming", "quote_block", "DEFAULT_MODES"]
