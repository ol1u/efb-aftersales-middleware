# coding=utf-8
"""The one piece of message structure still parsed by rules: quotes.

ComWeChat renders a native quote reply as a text block
``「name:quoted」`` + divider + reply body. Splitting it back apart is how
a reply is traced to the problem message it answers.
"""

import re
from typing import Optional, Tuple

QUOTE_RE = re.compile(r"^「(?P<name>[^:：」]*)[:：]?(?P<quoted>.*?)」\s*\n\s*(- )+-?\s*\n(?P<reply>.*)$", re.S)


def split_quote(text: str) -> Tuple[Optional[str], str, str]:
    """Return (quoted author name, quoted text, reply body)."""
    match = QUOTE_RE.match(text or "")
    if not match:
        return None, "", (text or "").strip()
    name = match.group("name").strip() or None
    return name, match.group("quoted").strip(), match.group("reply").strip()
