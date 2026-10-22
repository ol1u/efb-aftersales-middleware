# coding=utf-8
"""The conversational assistant: threads, summaries, plans, questions,
knowledge-base import and strategy modes."""

import json
import re
from types import SimpleNamespace

from ehforwarderbot import Message, MsgType, coordinator
from ehforwarderbot.chat import GroupChat

from efb_aftersales_middleware import AfterSalesMiddleware
from efb_aftersales_middleware.bot import CommandBot
from efb_aftersales_middleware.config import Config
from efb_aftersales_middleware.db import Database
from efb_aftersales_middleware.kb import parse_export, search_cases
from efb_aftersales_middleware.service import AfterSalesService, Incoming

GROUP = "123@chatroom"
PROBLEM_TEXT = "我的手臂坏了，抬不起来，订单号 2026102012345678"


class FakeOutbox:
    def __init__(self):
        self.posts = []

    def post_group(self, group_id, text):
        self.posts.append((group_id, text))


class Clock:
    def __init__(self):
        self.t = 1_800_000_000

    def __call__(self):
        return self.t


def result(action, thread_id=None, summary=None, plan=None, question=None, confidence=0.9):
    return {"action": action, "thread_id": thread_id, "summary": summary, "plan": plan,
            "question": question, "confidence": confidence}


class FakeAI:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return self.handler(kwargs)


def thread_id_for(kwargs, opener_name):
    match = re.search(r"问题编号 (\d+)（提出者：{}".format(re.escape(opener_name)),
                      kwargs["threads"])
    return int(match.group(1)) if match else None


def default_handler(kwargs):
    text = kwargs["text"]
    if "不对" in text or "补充" in text:
        tid = thread_id_for(kwargs, kwargs["sender_label"].replace(" [群主]", ""))
        return result("update", thread_id=tid, summary="更新总结：" + text, plan="更新方案")
    if "手臂" in text or "腿" in text:
        return result("new_problem", summary="总结：" + text, plan="方案：先安抚再处理")
    if "已发货" in text:
        tid = thread_id_for(kwargs, kwargs["sender_label"].replace(" [群主]", ""))
        return result("note", thread_id=tid)
    return result("ask", question="请问这是补充刚才的问题，还是一个新问题？")


def make_config(**extra):
    data = {
        "bot_token": "1:abc",
        "telegram_group_id": -1001,
        "telegram_admins": [42],
        "groups": [{"wechat_id": GROUP, "topic_id": 10}],
        "ai": {"enabled": True, "api_key": "sk-test"},
    }
    data.update(extra)
    return Config.from_dict(data)


def make_env(tmp_path, ai, config=None):
    db = Database(tmp_path / "t.db", clock=Clock())
    outbox = FakeOutbox()
    service = AfterSalesService(config or make_config(), db, outbox, ai=ai)
    return service, db, outbox


def incoming(text, sender="cust1", name="张三", is_self=False):
    return Incoming(group_id=GROUP, group_name="售后群", sender_id=sender, sender_name=name,
                    sender_alias=None, is_self=is_self, msg_type="Text", text=text)


# -- conversation ---------------------------------------------------------------


def test_new_problem_posts_summary_and_plan(tmp_path):
    ai = FakeAI(default_handler)
    service, db, outbox = make_env(tmp_path, ai)
    service.handle(incoming(PROBLEM_TEXT))
    assert len(outbox.posts) == 1
    post = outbox.posts[0][1]
    assert post.startswith("「张三：{}」".format(PROBLEM_TEXT))
    assert "【问题总结】" in post and "【建议方案】" in post
    thread = db.recent_threads(GROUP, 10 ** 9)[0]
    assert thread["version"] == 1
    record = db.thread_messages(thread["id"])[0]
    assert record["raw_text"] == PROBLEM_TEXT


def test_correction_updates_thread_and_reposts(tmp_path):
    ai = FakeAI(default_handler)
    service, db, outbox = make_env(tmp_path, ai)
    service.handle(incoming(PROBLEM_TEXT))
    service.handle(incoming("不对，是左手不是右手"))
    assert len(outbox.posts) == 2
    # The update quotes the thread's opening problem, not the correction.
    assert outbox.posts[1][1].startswith("「张三：{}」".format(PROBLEM_TEXT))
    assert "更新总结" in outbox.posts[1][1]
    thread = db.recent_threads(GROUP, 10 ** 9)[0]
    assert thread["version"] == 2


def test_ask_when_ambiguous(tmp_path):
    ai = FakeAI(default_handler)
    service, db, outbox = make_env(tmp_path, ai)
    service.handle(incoming(PROBLEM_TEXT))
    service.handle(incoming("那个好了"))  # unclear: supplement or new?
    assert len(outbox.posts) == 2
    assert "【确认一下】" in outbox.posts[1][1]
    assert outbox.posts[1][1].startswith("「张三：那个好了」")


def test_note_is_silent_but_attached(tmp_path):
    ai = FakeAI(default_handler)
    service, db, outbox = make_env(tmp_path, ai)
    service.handle(incoming(PROBLEM_TEXT))
    service.handle(incoming("已发货，单号在后台"))
    assert len(outbox.posts) == 1
    thread = db.recent_threads(GROUP, 10 ** 9)[0]
    texts = [m["raw_text"] for m in db.thread_messages(thread["id"])]
    assert "已发货，单号在后台" in texts


def test_two_people_two_threads(tmp_path):
    ai = FakeAI(default_handler)
    service, db, outbox = make_env(tmp_path, ai)
    service.handle(incoming(PROBLEM_TEXT, sender="cust1", name="张三"))
    service.handle(incoming("我的腿坏了，动不了", sender="cust2", name="李四"))
    service.handle(incoming("补充一下，是左腿", sender="cust2", name="李四"))
    threads = {t["opener_name"]: t for t in db.recent_threads(GROUP, 10 ** 9)}
    assert threads["张三"]["version"] == 1
    assert threads["李四"]["version"] == 2


def test_own_post_never_reprocessed(tmp_path):
    ai = FakeAI(default_handler)
    service, db, outbox = make_env(tmp_path, ai)
    service.handle(incoming(PROBLEM_TEXT))
    posted = outbox.posts[0][1]
    before = len(db.list_records(GROUP))
    service.handle(incoming(posted, sender="wxid_self", name="我", is_self=True))
    assert len(outbox.posts) == 1
    assert len(db.list_records(GROUP)) == before


def test_owner_messages_labelled(tmp_path):
    ai = FakeAI(default_handler)
    service, db, outbox = make_env(tmp_path, ai)
    service.handle(incoming("这个直接给客户退全款", sender="wxid_self", name="我", is_self=True))
    assert ai.calls[-1]["sender_label"].endswith("[群主]")


def test_low_confidence_stays_silent(tmp_path):
    ai = FakeAI(lambda kw: result("new_problem", summary="s", plan="p", confidence=0.2))
    service, db, outbox = make_env(tmp_path, ai)
    service.handle(incoming(PROBLEM_TEXT))
    assert outbox.posts == []


# -- knowledge base ---------------------------------------------------------------


EXPORT = {
    "name": "售后群", "type": "public_supergroup", "id": 1,
    "messages": [
        {"id": 1, "type": "message", "date": "2026-10-01T10:00:00", "from": "张三",
         "from_id": "user1", "text": "我的手臂坏了，抬不起来"},
        {"id": 2, "type": "message", "date": "2026-10-01T10:05:00", "from": "客服",
         "from_id": "user2", "text": [{"type": "bold", "text": "已补发"}, "，请查收"]},
        {"id": 3, "type": "service", "date": "2026-10-01T10:06:00", "text": "张三 joined"},
    ],
}


def test_parse_export_shapes():
    messages = parse_export(json.dumps(EXPORT).encode("utf-8"))
    assert len(messages) == 2
    assert messages[1]["text"] == "已补发，请查收"
    wrapped = {"chats": {"list": [EXPORT]}}
    assert len(parse_export(json.dumps(wrapped).encode("utf-8"))) == 2


def test_import_and_search_cases(tmp_path):
    service, db, outbox = make_env(tmp_path, ai=None)
    stats = service.import_export(json.dumps(EXPORT).encode("utf-8"), "batch-1")
    assert stats["count"] == 2
    assert stats["total"] == 2
    snippet = search_cases(db, "手臂坏了抬不起来")
    assert "我的手臂坏了" in snippet
    assert "已补发，请查收" in snippet  # context window included


# -- modes ------------------------------------------------------------------------


def test_modes_seed_switch_and_prompt(tmp_path):
    captured = {}

    def handler(kwargs):
        captured.update(kwargs)
        return result("new_problem", summary="s", plan="p")

    ai = FakeAI(handler)
    service, db, outbox = make_env(tmp_path, ai)
    assert service.active_mode()[0] == "常规"
    assert service.set_active_mode("保评分") is True
    assert service.set_active_mode("不存在") is False
    service.handle(incoming(PROBLEM_TEXT))
    assert captured["mode_name"] == "保评分"
    assert "评分" in captured["mode_prompt"]
    service.set_mode_prompt("激进", "无条件退款")
    assert service.set_active_mode("激进") is True


# -- bot commands -------------------------------------------------------------------


class FakeClient:
    def __init__(self, export_bytes=b""):
        self.sent = []
        self.export_bytes = export_bytes

    def send_message(self, chat_id, text, thread_id=None, **kwargs):
        self.sent.append((chat_id, text))
        return {"message_id": len(self.sent)}

    def download_file(self, file_id):
        return self.export_bytes


def test_bot_commands_and_import(tmp_path):
    service, db, outbox = make_env(tmp_path, ai=None)
    client = FakeClient(json.dumps(EXPORT).encode("utf-8"))
    bot = CommandBot(client, make_config(), service)
    assert "常规" in bot.run_command("mode", "", {"id": 42})
    assert "保评分" in bot.run_command("mode", "保评分", {"id": 42})
    assert service.active_mode()[0] == "保评分"
    update = {"message": {"from": {"id": 42}, "chat": {"id": 7},
                          "document": {"file_id": "f1", "file_name": "result.json"}}}
    bot.handle_update(update)
    assert client.sent and "导入完成" in client.sent[-1][1]
    assert db.kb_stats()["n"] == 2
    # Non-admin is ignored.
    update["message"]["from"] = {"id": 999}
    before = len(client.sent)
    bot.handle_update(update)
    assert len(client.sent) == before


# -- config & adapter -----------------------------------------------------------------


def test_config_tolerates_legacy_keys():
    cfg = Config.from_dict({
        "bot_token": "1:abc", "telegram_group_id": -1001,
        "groups": [{"wechat_id": GROUP, "topic_id": 10}],
        "staff": ["wxid_a"], "internal_topic_id": 99,
        "rules": {"categories": {}}, "notify_customer_statuses": ["new"],
    })
    assert cfg.groups == {GROUP: 10}
    assert cfg.telegram_admins == []
    assert cfg.ai.active is False


def test_to_incoming_group_message(monkeypatch):
    mw = AfterSalesMiddleware.__new__(AfterSalesMiddleware)
    mw.config = make_config()
    master = SimpleNamespace(channel_id="blueset.telegram")
    monkeypatch.setattr(coordinator, "master", master, raising=False)
    chat = GroupChat(module_id="honus.comwechat", module_name="ComWeChat", uid=GROUP, name="售后群")
    author = chat.add_member(name="张三", alias="张总", uid="wxid_a")
    msg = Message(chat=chat, author=author, type=MsgType.Text, text="快递没到", uid="m1",
                  deliver_to=master)
    res = mw.to_incoming(msg)
    assert res is not None
    assert res.group_id == GROUP
    assert res.display_name == "张总"
