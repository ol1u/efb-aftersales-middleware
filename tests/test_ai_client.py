"""DeepSeekClient reply handling: one retry on unusable replies + passthrough."""
import json

from efb_aftersales_middleware.ai import DeepSeekClient

GOOD = json.dumps({"action": "note", "thread_id": 1, "summary": None,
                   "plan": None, "question": None, "confidence": 0.9})
BAD = "some prose {broken json"


class FakeResp:
    def __init__(self, content):
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content},
                             "finish_reason": "stop"}]}


class FakeSession:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = 0
        self.headers = {}
        self.proxies = {}

    def post(self, url, json=None, timeout=None):
        self.calls += 1
        return FakeResp(self.contents.pop(0))


def make_client(contents):
    client = DeepSeekClient("test-key")
    client.session = FakeSession(contents)
    return client


def converse(client):
    return client.converse(mode_name="常规", mode_prompt="p", cases="", threads="",
                           hint="", group_name="g", sender_label="s", text="hi")


def test_retry_recovers_from_bad_reply():
    client = make_client([BAD, GOOD])
    result = converse(client)
    assert result is not None and result["action"] == "note"
    assert client.session.calls == 2


def test_two_bad_replies_give_up():
    client = make_client([BAD, BAD])
    assert converse(client) is None
    assert client.session.calls == 2


def test_good_reply_not_retried():
    client = make_client([GOOD])
    result = converse(client)
    assert result is not None and result["action"] == "note"
    assert client.session.calls == 1
