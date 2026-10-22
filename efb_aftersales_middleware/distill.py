# coding=utf-8
"""Distill raw imported chat logs into standardised knowledge-base cases.

The owner imports Telegram exports of after-sales chats; the raw stream is
colloquial and inconsistently worded, which makes keyword retrieval miss.
Distillation runs the stream through the AI in chunks and keeps only
complete cases (a problem plus how it was handled), rewritten with
canonical terms and a keyword list of colloquial variants. Phone numbers
and personal names are redacted, near-duplicates are dropped, and the
result lands in the knowledge base as a ``distilled-*`` batch, which
retrieval ranks above the raw logs.
"""

import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

DISTILL_PROMPT = """你负责把一段原始售后聊天流水蒸馏成标准案例。流水里是客服与客户的交错对话，消息很短，很多条合起来才是一件事。

只提取【完整案例】：能看出客户遇到了什么问题、并且能看出处理办法或处理结果的，才提取。只有提问没有处理、闲聊、发货单据确认、打招呼的，不要提取。

每个案例输出一个对象：
{"problem": "类别：这个案例的具体情形，一整句话（如：发音：客户反映娃娃没有发音器，已签收三天）", "detail": "背景：规格/平台/订单阶段等关键信息，没有就留空字符串", "handling": "实际处理办法与结果，一到两句", "keywords": "这个问题可能出现的其他说法，空格分隔"}

隐私要求：problem、detail、handling 里不要出现客户姓名和电话号码，涉及客户一律写“客户”。

规范词要求（重要，决定以后能不能检索到）：问题描述统一用这些词——发音（不要写发声、声音功能）、夹吸、果冻胸、无钉站立、手指骨、全身加热、植发、骨架异响、掉色、开裂、漏气、发错货、物流破损。keywords 里要放客户的口语说法，例如发音类问题写：发声 不出声 没声音 音响 喇叭。

只输出一个 JSON 对象 {"cases": [...] }，不要任何其他文字。没有完整案例就输出 {"cases": []}。

聊天流水（格式：日期 发送者：内容）：
"""

CHUNK_SIZE = 60
CHUNK_OVERLAP = 10

_PHONE_RE = re.compile(r"1\d{10}(-\d+)?")
_NAME_RE = re.compile(r"[\u4e00-\u9fa5]{1,2}(先生|女士|小姐)")


def redact(text: str) -> str:
    text = _PHONE_RE.sub("〔电话〕", text or "")
    return _NAME_RE.sub("客户", text)


def compose_case_text(case: Dict[str, Any]) -> str:
    text = "标准案例：{}".format(redact(str(case.get("problem") or "").strip()))
    detail = redact(str(case.get("detail") or "").strip())
    if detail:
        text += "｜{}".format(detail)
    text += "｜处理：{}".format(redact(str(case.get("handling") or "").strip()))
    keywords = redact(str(case.get("keywords") or "").strip())
    if keywords:
        text += "｜关键词：{}".format(keywords)
    return text


def _char_set(text: str) -> set:
    return set(re.sub(r"\s+", "", text or ""))


def _similar(a: set, b: set) -> bool:
    if not a or not b:
        return False
    return len(a & b) / max(1, min(len(a), len(b))) >= 0.8


def dedupe_cases(cases: List[Dict[str, Any]], existing_texts: List[str]) -> List[Dict[str, Any]]:
    """Drop cases too similar to an existing standard case or to each other."""
    seen = [_char_set(t) for t in existing_texts]
    fresh = []
    for row in cases:
        chars = _char_set(row["text"])
        if any(_similar(chars, other) for other in seen):
            continue
        seen.append(chars)
        fresh.append(row)
    return fresh


def chunk_rows(rows: List[Dict[str, Any]], size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[List[Dict[str, Any]]]:
    chunks = []
    start = 0
    while start < len(rows):
        chunks.append(rows[start:start + size])
        if start + size >= len(rows):
            break
        start += size - overlap
    return chunks


def render_chunk(rows: List[Dict[str, Any]]) -> str:
    return "\n".join(
        "{} {}：{}".format(
            time.strftime("%m-%d", time.localtime(row.get("sent_at") or 0)),
            row.get("sender_name") or "?", (row.get("text") or "").replace("\n", " "))
        for row in rows)


def distill(chat_json: Callable[[str], Optional[Dict[str, Any]]],
            rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run the AI over raw kb rows; return composed standard-case kb rows."""
    out: List[Dict[str, Any]] = []
    for chunk in chunk_rows(rows):
        try:
            result = chat_json(DISTILL_PROMPT + render_chunk(chunk))
        except Exception:
            logger.exception("distill chunk failed; skipped")
            continue
        cases = (result or {}).get("cases")
        if not isinstance(cases, list):
            continue
        sent_at = max((row.get("sent_at") or 0) for row in chunk) if chunk else 0
        for case in cases:
            if not isinstance(case, dict) or not case.get("problem") or not case.get("handling"):
                continue
            out.append({
                "chat_key": "distilled",
                "sender_name": "标准案例",
                "sent_at": sent_at,
                "text": compose_case_text(case),
            })
    logger.info("distilled %s standard cases from %s raw messages", len(out), len(rows))
    return out


__all__ = ["DISTILL_PROMPT", "redact", "compose_case_text", "dedupe_cases",
           "chunk_rows", "render_chunk", "distill"]
