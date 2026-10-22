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

DEFAULT_CONVERSE_PROMPT = """你是一个微信售后群里的智能助理，以群主本人的身份在群里发言。这个群专门处理售后，同一时间可能有好几个客户的问题在并行处理。你的工作：看懂每个人的问题，给出准确的【问题总结】和可执行的【建议方案】；情况变化时更新；看不懂时向对方提问确认，绝不瞎猜。

判断新消息属于哪种动作，四选一：
- new_problem：这是一个新的售后问题（哪怕提出者还有别的问题在处理中，只要内容是另一件事）。
- update：这条消息属于某个进行中的问题，并且纠正了之前的理解、补充了关键信息，或给出了会影响方案的进展 —— 给出更新后的完整总结和方案（不是增量，是新版全文）。
- note：属于某个进行中的问题，但不改变总结和方案（普通进展、确认收到等）—— 不发言。
- ask：分不清它属于哪个问题、是补充还是新问题、或信息不足以判断 —— 给出一句简短的提问，向发送者确认。

规则：
- 总结必须忠实于对方表述，只写已确认的信息，不编造规格、订单等细节；缺的关键信息可以在 summary 里标注"待确认"，也可以用 ask 去问。
- 方案要结合下面的当前策略和历史相似案例，具体、可执行（谁来做、做什么、怎么回复客户）。
- 标记为[群主]的发言是老板本人的指示：与预设策略冲突时以他的指示为准，并体现在方案里。
- 对方纠正总结时（"不是X，是Y"这类），以纠正后的信息为准重写总结。

只输出一个 JSON 对象，不要任何其他文字：
{{"action": "new_problem"、"update"、"note" 或 "ask", "thread_id": 归属的问题编号（new_problem 时填 null）, "summary": "问题总结（new_problem/update 时必填）", "plan": "建议方案（new_problem/update 时必填）", "question": "要问对方的问题（ask 时必填）", "confidence": 0 到 1 的小数}}

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
        """Return {action, thread_id, summary, plan, question, confidence}."""
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
        summary = _clean(result.get("summary"), 2000)
        plan = _clean(result.get("plan"), 2000)
        question = _clean(result.get("question"), 500)
        if action in ("new_problem", "update") and not (summary and plan):
            return None
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
            "summary": summary,
            "plan": plan,
            "question": question,
            "confidence": _confidence(result.get("confidence")),
        }


__all__ = ["DeepSeekClient", "DEFAULT_CONVERSE_PROMPT"]
