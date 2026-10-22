# coding=utf-8
"""AI conversation engine for the after-sales assistant (DeepSeek).

One call per group message. The AI sees the current strategy (the active
mode's preset prompt), similar historical cases from the knowledge base,
the problems currently being worked on (threads), any deterministic
trace (the message quotes/copies a known message), and the new message
itself. It decides:

- ``new_problem``: a new problem — return a summary and a suggested plan.
- ``update``: the message corrects or materially adds to a known problem
  — return the full revised summary and plan.
- ``note``: belongs to a known problem but changes nothing — stay silent.
- ``ask``: cannot tell what it belongs to (a supplement? a new problem?
  which problem?) or lacks the facts to judge — return a short question
  to ask the sender instead of guessing.

The prompt lives in ``ai.prompts.converse`` so it can be tuned without
code changes. Any failure returns ``None``; the caller then just keeps
the archived message. The API key is never logged.
"""

import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_CONVERSE_PROMPT = """你是一个微信售后群里的智能助理，以群主本人的身份在群里发言。这个群专门处理售后，同一时间可能有好几个客户的问题在并行处理。你的工作：看懂每个人的问题，维护每个问题的案例卡，给出简短可执行的下一步；情况变化时更新；看不懂时向对方提问确认，绝不瞎猜。

判断新消息属于哪种动作，四选一：
- new_problem：这是一个新的售后问题（哪怕提出者还有别的问题在处理中，只要内容是另一件事）。
- update：属于某个进行中的问题，并且纠正了之前的理解、补充了关键信息，或给出了会影响方案的进展。
- note：属于某个进行中的问题，但不改变卡片和方案（普通进展、确认收到等）—— 不发言。
- ask：分不清属于哪个问题、是补充还是新问题、或信息不足以判断 —— 给出一句简短的提问，向发送者确认。

案例卡（card）：new_problem 和 update 时必须给出该问题目前的完整卡片，在上下文已有卡片的基础上更新；没有的信息留空字符串，绝不编造。
- customer：客户称呼；phone：电话；order_no：订单号或运单号；spec：规格一句话（30 字内）；issue：问题一句话（30 字内）；demand：客户诉求（15 字内）；status：只能是 待核实、处理中、等客户、已解决 之一；facts：已确认事实的数组，每条 20 字内、最多 6 条，只记被确认过的（核对结果、客户答复、已发出的处理），不记猜测。

篇幅纪律（严格遵守，群里看的人很忙）：
- plan：1 到 3 条，每条一行、40 字内，写清"谁做什么"；不要话术全文，不要安抚、登记、时限这类套话，除非是这个单子特有的动作。
- progress：update 时必填，40 字内一句话：本轮新增了什么、确认了什么。
- question：ask 时必填，50 字内，只问一个最关键的问题。

规则：
- 只写已确认的信息，不编造规格、订单等细节；缺的关键信息用 ask 去问。
- 方案要结合下面的当前策略和历史相似案例。
- 标记为[群主]的发言是老板本人的指示：与预设策略冲突时以他的指示为准，并体现在方案里。
- 对方纠正时（"不是X，是Y"这类），以纠正后的信息为准更新卡片。

只输出一个 JSON 对象，不要任何其他文字：
{{"action": "new_problem"、"update"、"note" 或 "ask", "thread_id": 归属的问题编号（new_problem 时填 null）, "card": {{"customer": "", "phone": "", "order_no": "", "spec": "", "issue": "", "demand": "", "status": "", "facts": []}}, "plan": "建议方案（new_problem/update 时必填，1-3 条短句，每行一条）", "progress": "本轮进展（update 时必填，一句话）", "question": "要问对方的问题（ask 时必填）", "confidence": 0 到 1 的小数}}

当前策略（模式：{mode_name}）：
{mode_prompt}

历史相似案例（供参考，不要照抄与本问题无关的内容）：
{cases}

进行中的问题：
{threads}

痕迹提示（引用/复制检测的结果，通常可靠）：{hint}

新消息（群：{group_name}，发送者：{sender_label}）：
{text}"""

_JSON_RE = re.compile(r"\{.*\}", re.S)
_ACTIONS = ("new_problem", "update", "note", "ask")


def _render(template: str, **fields: Any) -> str:
    values = defaultdict(str, {k: ("" if v is None else str(v)) for k, v in fields.items()})
    return template.format_map(values)


def _confidence(value: Any) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, conf))


def _clean(value: Any, limit: int) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()[:limit]
    return text or None


_CARD_LIMITS = {"customer": 20, "phone": 24, "order_no": 32, "spec": 60,
                "issue": 60, "demand": 30, "status": 10}
_CARD_STATUSES = ("待核实", "处理中", "等客户", "已解决")


def _clean_card(value: Any) -> Dict[str, Any]:
    """Coerce an AI-returned case card into the stored shape."""
    card: Dict[str, Any] = {key: "" for key in _CARD_LIMITS}
    card["facts"] = []
    if isinstance(value, dict):
        for key, limit in _CARD_LIMITS.items():
            card[key] = _clean(value.get(key), limit) or ""
        facts = value.get("facts")
        if isinstance(facts, list):
            for item in facts[:6]:
                item = _clean(item, 40)
                if item:
                    card["facts"].append(item)
    if card["status"] and card["status"] not in _CARD_STATUSES:
        card["status"] = ""
    return card


class DeepSeekClient:
    """Minimal chat-completions client (also works for OpenAI-compatible APIs)."""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-flash", timeout: int = 15,
                 proxy: Optional[str] = None, prompts: Optional[Dict[str, str]] = None):
        self.base = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.prompts = prompts or {}
        self.session = requests.Session()
        self.session.headers.update({"Authorization": "Bearer {}".format(api_key)})
        if proxy:
            self.session.proxies.update({"http": proxy, "https": proxy})

    def _chat_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        result = self._chat_json_once(prompt)
        if result is None:
            logger.warning("AI reply unusable; retrying once")
            result = self._chat_json_once(prompt)
        return result

    def _chat_json_once(self, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            resp = self.session.post(
                "{}/chat/completions".format(self.base),
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
                timeout=self.timeout,
            )
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            logger.warning("AI request failed: %s", type(exc).__name__)
            return None
        match = _JSON_RE.search(content or "")
        if not match:
            logger.warning("AI reply is not JSON: %r", (content or "")[:400])
            return None
        try:
            parsed = json.loads(match.group(0))
        except ValueError:
            logger.warning("AI reply JSON is invalid: %r", (content or "")[:400])
            return None
        return parsed if isinstance(parsed, dict) else None

    def converse(self, *, mode_name: str, mode_prompt: str, cases: str, threads: str,
                 hint: str, group_name: str, sender_label: str,
                 text: str) -> Optional[Dict[str, Any]]:
        """Return {action, thread_id, card, plan, progress, question, confidence}."""
        prompt = _render(self.prompts.get("converse") or DEFAULT_CONVERSE_PROMPT,
                         mode_name=mode_name, mode_prompt=mode_prompt,
                         cases=cases or "（暂无）", threads=threads or "（暂无）",
                         hint=hint or "（无）", group_name=group_name,
                         sender_label=sender_label, text=text)
        result = self._chat_json(prompt)
        if result is None:
            return None
        action = str(result.get("action") or "").strip().lower()
        if action not in _ACTIONS:
            return None
        card = _clean_card(result.get("card"))
        # Tolerate a legacy prompt that still returns a free-text summary:
        # fold it into the card so the pipeline keeps working.
        legacy_summary = _clean(result.get("summary"), 120)
        if legacy_summary and not card["issue"]:
            card["issue"] = legacy_summary[:60]
        plan = _clean(result.get("plan"), 360)
        progress = _clean(result.get("progress"), 120)
        question = _clean(result.get("question"), 200)
        if action in ("new_problem", "update") and not plan:
            return None
        if action == "update" and not progress:
            progress = "有新的进展，方案已更新。"
        if action == "ask" and not question:
            return None
        thread_id = result.get("thread_id")
        if thread_id is not None:
            try:
                thread_id = int(thread_id)
            except (TypeError, ValueError):
                thread_id = None
        return {
            "action": action,
            "thread_id": thread_id,
            "card": card,
            "plan": plan,
            "progress": progress,
            "question": question,
            "confidence": _confidence(result.get("confidence")),
        }


__all__ = ["DeepSeekClient", "DEFAULT_CONVERSE_PROMPT"]
