# coding=utf-8
"""Knowledge base: imported Telegram chat exports.

The owner exports the after-sales chats from Telegram (Desktop export,
``result.json``) and sends the file to the aftersales bot; the messages
land in ``kb_messages``. When a new problem arrives, similar historical
exchanges are retrieved as snippets and handed to the AI as reference
for how such problems were summarised and solved before.

Retrieval is deliberately simple: character n-gram overlap scoring over
candidate rows, then a small context window around each hit. Good
enough for a single shop's history and fully offline.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")


def _flatten_text(value: Any) -> str:
    """Telegram export ``text`` is a string or a list of text entities."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return ""


def _parse_date(value: Any) -> Optional[int]:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value:
        try:
            return int(datetime.fromisoformat(value).timestamp())
        except ValueError:
            return None
    return None


def parse_export(data: bytes) -> List[Dict[str, Any]]:
    """Parse a Telegram Desktop export into flat message dicts.

    Accepts both a full account export (``{"chats": {"list": [...]}}``)
    and a single-chat export (``{"name": ..., "messages": [...]}``).
    Returns [{chat_key, sender_name, sent_at, text}].
    """
    root = json.loads(data.decode("utf-8"))
    chats: List[Dict[str, Any]] = []
    if isinstance(root, dict) and isinstance(root.get("chats"), dict):
        chats = root["chats"].get("list") or []
    elif isinstance(root, dict) and isinstance(root.get("messages"), list):
        chats = [root]
    out: List[Dict[str, Any]] = []
    for chat in chats:
        if not isinstance(chat, dict):
            continue
        chat_key = str(chat.get("name") or chat.get("id") or "export")
        for msg in chat.get("messages") or []:
            if not isinstance(msg, dict) or msg.get("type") != "message":
                continue
            text = _flatten_text(msg.get("text")).strip()
            if not text:
                continue
            out.append({
                "chat_key": chat_key,
                "sender_name": str(msg.get("from") or ""),
                "sent_at": _parse_date(msg.get("date")),
                "text": text,
            })
    out.sort(key=lambda m: (m["sent_at"] or 0))
    return out


def query_terms(text: str, limit: int = 12) -> List[str]:
    """Distinctive 3-grams of the query, longest-information first."""
    grams = set()
    for run in _TOKEN_RE.findall(text or ""):
        if len(run) < 3:
            grams.add(run)
            continue
        for i in range(len(run) - 2):
            grams.add(run[i:i + 3])
    # Prefer rarer/longer-signal grams: keep deterministic order by length
    # of source run then lexicographic, capped.
    return sorted(grams)[:limit * 4][:limit] if grams else []


def _score(text: str, terms: List[str]) -> int:
    return sum(1 for term in terms if term and term in text)


def search_cases(db, query: str, max_cases: int = 4, context: int = 3,
                 max_chars: int = 3500) -> str:
    """Return formatted snippets of similar historical exchanges."""
    terms = query_terms(query)
    if not terms:
        return ""
    candidates = db.kb_candidates(terms)

    def rank(pair):
        row, score = pair
        # Standardised cases (AI-distilled history, auto-archived resolved
        # threads) outrank raw imported chat logs at equal relevance.
        batch = row["batch"] or ""
        boost = 3 if batch.startswith(("distilled", "resolved")) else 0
        return (-(score + boost), -row["id"])

    scored = sorted(
        ((row, _score(row["text"], terms)) for row in candidates), key=rank)
    hits = []
    for row, score in scored:
        # Standardised cases are terse and normalised: a single distinctive
        # 3-gram match (e.g. the category word) is a real hit there, while
        # raw chat rows need two matches to stay above the noise floor.
        standard = (row["batch"] or "").startswith(("distilled", "resolved"))
        if score >= 2 or (standard and score >= 1):
            hits.append(row)
        if len(hits) >= max_cases:
            break
    if not hits:
        return ""
    blocks: List[str] = []
    used = 0
    for idx, hit in enumerate(hits, 1):
        window = db.kb_window(hit["chat_key"], hit["id"], before=context, after=context)
        lines = []
        for row in window:
            mark = ">>" if row["id"] == hit["id"] else "  "
            lines.append("{}{}：{}".format(mark, row["sender_name"] or "?", row["text"]))
        block = "【历史案例{}】\n{}".format(idx, "\n".join(lines))
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


__all__ = ["parse_export", "search_cases", "query_terms"]
