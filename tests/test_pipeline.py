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
from efb_aftersales_middleware.service import AfterSalesService, Incoming, QUOTE_HINT_FOOTER

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


def incoming(text, sender="cust1", name="张三", is_self=False, msg_id=None,
             quoted_msg_id=None, quoted_record_id=None):
    return Incoming(group_id=GROUP, group_name="售后群", sender_id=sender, sender_name=name,
                    sender_alias=None, is_self=is_self, msg_type="Text", text=text,
                    msg_id=msg_id, quoted_msg_id=quoted_msg_id,
                    quoted_record_id=quoted_record_id)


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


def test_posts_carry_quote_hint_footer(tmp_path):
    ai = FakeAI(default_handler)
    service, db, outbox = make_env(tmp_path, ai)
    service.handle(incoming(PROBLEM_TEXT))                 # new_problem post
    service.handle(incoming("不对，是左臂，补充一下"))        # update post
    service.handle(incoming("在吗"))                         # ask post
    assert len(outbox.posts) == 3
    for _, post in outbox.posts:
        assert post.rstrip().endswith(QUOTE_HINT_FOOTER)
    # the archived copy carries the footer too, so quotes of it still match
    posted = db.conn.execute(
        "SELECT raw_text FROM records WHERE kind = 'posted'").fetchall()
    assert len(posted) == 3
    assert all(QUOTE_HINT_FOOTER in (r["raw_text"] or "") for r in posted)


# -- quote targets (message.target from the slave) ------------------------------


def test_quote_by_msg_id_routes_to_thread(tmp_path):
    def handler(kwargs):
        if "引用了问题编号" in kwargs["hint"]:
            return result("update", thread_id=None, summary="按引用更新", plan="新方案")
        return result("new_problem", summary="总结：" + kwargs["text"], plan="方案")

    ai = FakeAI(handler)
    service, db, outbox = make_env(tmp_path, ai)
    service.handle(incoming(PROBLEM_TEXT, msg_id="1001"))
    service.handle(incoming("补充一句", quoted_msg_id="1001"))
    assert len(outbox.posts) == 2
    assert "按引用更新" in outbox.posts[1][1]
    thread = db.recent_threads(GROUP, 10 ** 9)[0]
    assert thread["version"] == 2


def test_quote_of_ask_post_recovers_original(tmp_path):
    original = "ORIGINAL 客户说电机不转了"

    def handler(kwargs):
        if "提问帖" in kwargs["hint"]:
            assert original in kwargs["hint"]
            return result("new_problem", summary="结合原消息的总结", plan="处理方案")
        return result("ask", question="这是新问题还是补充？")

    ai = FakeAI(handler)
    service, db, outbox = make_env(tmp_path, ai)
    service.handle(incoming(original, msg_id="3001"))
    assert len(outbox.posts) == 1  # the ask post
    ask_text = outbox.posts[0][1]
    # the ask post comes back into the group under the owner's name
    service.handle(incoming(ask_text, sender="wxid_boss", name="老板",
                            is_self=True, msg_id="3002"))
    posted = db.find_posted_record(GROUP, ask_text)
    assert posted["efb_msg_id"] == "3002"  # backfilled on return
    # the customer answers by quoting the ask post
    service.handle(incoming("是新问题，电机烧了", quoted_msg_id="3002"))
    assert len(outbox.posts) == 2
    assert "结合原消息的总结" in outbox.posts[1][1]
    assert db.recent_threads(GROUP, 10 ** 9)


def test_exact_resend_gets_already_handled_notice(tmp_path):
    ai = FakeAI(default_handler)
    service, db, outbox = make_env(tmp_path, ai)
    service.handle(incoming(PROBLEM_TEXT, msg_id="4001"))
    assert len(outbox.posts) == 1
    calls_before = len(ai.calls)
    service.handle(incoming(PROBLEM_TEXT, msg_id="4002"))  # verbatim resend
    assert len(outbox.posts) == 2
    notice = outbox.posts[1][1]
    assert "已经处理过了" in notice and "#1" in notice
    assert len(ai.calls) == calls_before  # the notice needs no AI call
    assert len(db.thread_messages(1, limit=10)) == 2  # resend filed under #1
    # a resend WITH additions still goes through the copy+append AI path
    service.handle(incoming(PROBLEM_TEXT + " 另外补充：已经拍了视频"))
    assert len(ai.calls) > calls_before


def test_decision_logging(tmp_path, caplog):
    import logging
    ai = FakeAI(default_handler)
    service, db, outbox = make_env(tmp_path, ai)
    with caplog.at_level(logging.INFO, logger="efb_aftersales_middleware.service"):
        service.handle(incoming(PROBLEM_TEXT, msg_id="5001"))
    texts = [r.getMessage() for r in caplog.records]
    assert any("trace msg=5001" in t and "kind=none" in t for t in texts)
    assert any("action=new_problem" in t for t in texts)


def test_low_confidence_silence_is_logged(tmp_path, caplog):
    import logging
    ai = FakeAI(lambda kwargs: result("note", confidence=0.3))
    service, db, outbox = make_env(tmp_path, ai)
    with caplog.at_level(logging.INFO, logger="efb_aftersales_middleware.service"):
        service.handle(incoming("随便一句", msg_id="5002"))
    texts = [r.getMessage() for r in caplog.records]
    assert any("silent" in t and "confidence=0.30" in t for t in texts)
    assert not outbox.posts


def test_quote_by_record_id_routes_to_thread(tmp_path):
    def handler(kwargs):
        if "引用了问题编号" in kwargs["hint"]:
            return result("update", thread_id=None, summary="按引用更新", plan="新方案")
        return result("new_problem", summary="总结：" + kwargs["text"], plan="方案")

    ai = FakeAI(handler)
    service, db, outbox = make_env(tmp_path, ai)
    service.handle(incoming(PROBLEM_TEXT, msg_id="6001"))
    opener = db.find_record_by_msg_id(GROUP, "6001")
    assert opener is not None
    service.handle(incoming("补充一句", quoted_record_id=opener["id"]))
    assert len(outbox.posts) == 2
    assert "按引用更新" in outbox.posts[1][1]
    assert db.recent_threads(GROUP, 10 ** 9)[0]["version"] == 2


def test_quote_ask_post_by_record_id_recovers(tmp_path):
    original = "ORIGINAL 客户说电机不转了"

    def handler(kwargs):
        if "提问帖" in kwargs["hint"]:
            assert original in kwargs["hint"]
            return result("new_problem", summary="结合原消息的总结", plan="处理方案")
        return result("ask", question="这是新问题还是补充？")

    ai = FakeAI(handler)
    service, db, outbox = make_env(tmp_path, ai)
    service.handle(incoming(original, msg_id="7001"))
    assert len(outbox.posts) == 1  # the ask post
    posted = db.find_posted_record(GROUP, outbox.posts[0][1])
    assert posted is not None
    service.handle(incoming("是新问题，电机烧了", quoted_record_id=posted["id"]))
    assert len(outbox.posts) == 2
    assert "结合原消息的总结" in outbox.posts[1][1]
