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


EMPTY_CARD = {"customer": "", "phone": "", "order_no": "", "platform": "", "emotion": "",
              "spec": "", "issue": "", "demand": "", "status": "处理中", "facts": []}


def result(action, thread_id=None, summary=None, plan=None, question=None, confidence=0.9,
           card=None, progress=None):
    if card is None:
        card = dict(EMPTY_CARD, issue=summary or "")
    # The old free-text "summary" now surfaces as the update's progress line,
    # so assertions on it keep working against the incremental post format.
    if progress is None:
        progress = summary or ""
    return {"action": action, "thread_id": thread_id, "card": card, "plan": plan,
            "progress": progress, "question": question, "confidence": confidence}


class FakeAI:
    def __init__(self, handler, json_handler=None):
        self.handler = handler
        self.json_handler = json_handler
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return self.handler(kwargs)

    def chat_json(self, prompt):
        if self.json_handler is None:
            return None
        return self.json_handler(prompt)


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


def test_case_card_stored_and_summary_assembled(tmp_path):
    card = {"customer": "王女士", "phone": "13800138000", "order_no": "SF1234567890",
            "spec": "硅胶 160cm", "issue": "腿上有胶水痕迹", "demand": "换货",
            "status": "处理中", "facts": ["客户已发照片"]}

    def handler(kwargs):
        return result("new_problem", card=card, plan="联系厂家换货")

    service, db, outbox = make_env(tmp_path, FakeAI(handler))
    service.handle(incoming("腿上有胶水，怎么办"))
    thread = db.recent_threads(GROUP, 10 ** 9)[0]
    import json as _json
    stored = _json.loads(thread["card"])
    assert stored["customer"] == "王女士"
    assert stored["facts"] == ["客户已发照片"]
    assert "订单尾号 7890" in thread["summary"]
    assert "腿上有胶水痕迹" in thread["summary"]
    assert "诉求：换货" in thread["summary"]
    assert "订单尾号 7890" in outbox.posts[0][1]


def test_quote_excerpt_not_full_order_dump(tmp_path):
    filler = "收货地址某某省某某市某某区某某路某某号某某小区某某栋某某室，收件人某某某"
    text = filler + "\n\n客户反馈腿上有胶水，这个怎么处理"

    def handler(kwargs):
        return result("new_problem", summary="腿上有胶水", plan="先擦拭")

    service, db, outbox = make_env(tmp_path, FakeAI(handler))
    service.handle(incoming(text))
    post = outbox.posts[0][1]
    assert filler not in post
    assert "腿上有胶水" in post


def test_card_match_by_phone_from_another_sender(tmp_path):
    card = dict(EMPTY_CARD, phone="18472206530", issue="发错货了")

    def handler(kwargs):
        if "案例卡一致" in kwargs["hint"]:
            return result("update", summary="厂家确认发错", plan="补发正确款")
        return result("new_problem", card=card, plan="先核对")

    service, db, outbox = make_env(tmp_path, FakeAI(handler))
    service.handle(incoming("发错货了，客户电话18472206530"))
    service.handle(incoming("厂家回了，单号18472206530确实发错了", sender="colleague1", name="小李"))
    assert db.recent_threads(GROUP, 10 ** 9)[0]["version"] == 2
    assert len(outbox.posts) == 2
    assert "【进展】" in outbox.posts[1][1]
    assert "【问题总结】" not in outbox.posts[1][1]


def test_resolution_archives_case_and_notifies(tmp_path):
    def handler(kwargs):
        if kwargs["text"].startswith("已经处理好了"):
            return result("note", thread_id=1)
        return result("new_problem", summary="关节异响", plan="返厂检修")

    service, db, outbox = make_env(tmp_path, FakeAI(handler))
    service.handle(incoming("关节有异响"))
    service.handle(incoming("已经处理好了，客户接受了"))
    assert db.get_thread(1)["status"] == "resolved"
    assert db.kb_stats()["n"] == 1
    assert any("已存入案例库" in text for _, text in outbox.posts)


def test_ask_uses_lower_threshold(tmp_path):
    def handler(kwargs):
        return result("ask", question="这是新问题还是补充？", confidence=0.45)

    service, db, outbox = make_env(tmp_path, FakeAI(handler))
    service.handle(incoming("那个弄一下"))
    assert len(outbox.posts) == 1
    assert "确认一下" in outbox.posts[0][1]


def test_new_problem_still_needs_confidence(tmp_path):
    def handler(kwargs):
        return result("new_problem", summary="含糊的问题", plan="方案", confidence=0.45)

    service, db, outbox = make_env(tmp_path, FakeAI(handler))
    service.handle(incoming("有点问题"))
    assert outbox.posts == []


def test_tiers_defaults_and_update(tmp_path):
    service, db, outbox = make_env(tmp_path, FakeAI(lambda kwargs: result("note")))
    tiers = service.tiers()
    assert set(tiers) == {"1", "2", "3", "4"}
    assert tiers["4"]["stance"] == "激进"
    assert "第3档" in service.tiers_text()
    assert service.set_tier_authority("3", "可退一半") is True
    assert service.tiers()["3"]["authority"] == "可退一半"
    assert service.set_tier_authority("9", "x") is False


def test_card_platform_emotion_and_tiers_context(tmp_path):
    seen = {}
    card = dict(EMPTY_CARD, platform="拼多多", emotion="愤怒", issue="发错货")

    def handler(kwargs):
        seen.update(kwargs)
        return result("new_problem", card=card, plan="先核实")

    service, db, outbox = make_env(tmp_path, FakeAI(handler))
    service.handle(incoming("拼多多的单发错货了，客户很生气"))
    import json as _json
    stored = _json.loads(db.recent_threads(GROUP, 10 ** 9)[0]["card"])
    assert stored["platform"] == "拼多多"
    assert stored["emotion"] == "愤怒"
    assert "第4档" in seen["tiers"]
    assert "放权" in seen["tiers"]


def test_kb_prefers_distilled_cases(tmp_path):
    from efb_aftersales_middleware import kb as kb_module
    service, db, outbox = make_env(tmp_path, FakeAI(lambda kwargs: result("note")))
    db.kb_add_messages("distilled-test", [{
        "chat_key": "d", "sender_name": "标准案例", "sent_at": 1,
        "text": "标准案例：腿上有胶水痕迹，用酒精棉擦拭即可去除"}])
    db.kb_add_messages("import-raw", [{
        "chat_key": "r", "sender_name": "原始记录", "sent_at": 2,
        "text": "那个腿上有胶水啊咋办呢问了下要用酒精擦"}])
    found = kb_module.search_cases(db, "腿上有胶水怎么处理")
    assert "标准案例" in found
    assert found.index("标准案例") < found.index("原始记录") or "原始记录" not in found


def test_kb_standard_case_single_gram_hit(tmp_path):
    from efb_aftersales_middleware import kb as kb_module
    service, db, outbox = make_env(tmp_path, FakeAI(lambda kwargs: result("note")))
    db.kb_add_messages("distilled-test", [{
        "chat_key": "d", "sender_name": "标准案例", "sent_at": 1,
        "text": "标准案例：发错货：客户要求仿真口腔，实际漏做且模具损坏｜处理：协商换头型重新发货｜关键词：错发 漏做"}])
    db.kb_add_messages("import-raw", [{
        "chat_key": "r", "sender_name": "原始记录", "sent_at": 2,
        "text": "昨天那个发错货的单子你看了没，客户催了两次了快点处理一下吧"}])
    found = kb_module.search_cases(db, "发错货了怎么办")
    assert "标准案例" in found


def _export_bytes(texts):
    import json as _json
    messages = [{"type": "message", "from": "Say", "date": 1730000000 + i * 60,
                 "text": t} for i, t in enumerate(texts)]
    return _json.dumps({"name": "t", "messages": messages}, ensure_ascii=False).encode()


def test_distill_redaction_and_compose():
    from efb_aftersales_middleware.distill import compose_case_text, redact
    assert "13800138000" not in redact("电话13800138000-123")
    assert "〔电话〕" in redact("电话13800138000-123")
    text = compose_case_text({
        "problem": "发音：客户13800138000反映没有发音器",
        "detail": "王先生订单",
        "handling": "已安排补发",
        "keywords": "发声 不出声"})
    assert "13800138000" not in text
    assert "王先生" not in text
    assert text.startswith("标准案例：发音")
    assert "关键词：发声 不出声" in text


def test_distill_batch_end_to_end(tmp_path):
    import time as _time
    case = {"problem": "发音：客户反映没有发音器", "detail": "",
            "handling": "核实后补发发音器", "keywords": "发声 不出声"}
    ai = FakeAI(lambda kwargs: result("note"),
                json_handler=lambda prompt: {"cases": [case]})
    service, db, outbox = make_env(tmp_path, ai)
    stats = service.import_export(_export_bytes(["客户说没有发音器", "给他补发一个发音器"]), "import-t1")
    assert stats["count"] == 2
    assert stats.get("distilling") is True
    deadline = _time.time() + 5
    while _time.time() < deadline and db.kb_standard_count() == 0:
        _time.sleep(0.05)
    assert db.kb_standard_count() == 1
    texts = db.kb_standard_texts()
    assert "标准案例：发音" in texts[0]
    # Re-running distillation over the same batch adds nothing new.
    assert service.distill_batch("import-t1") == 0


def test_distill_dedupe_against_existing(tmp_path):
    from efb_aftersales_middleware.distill import dedupe_cases
    existing = ["标准案例：发音：客户反映没有发音器｜处理：核实后补发发音器"]
    rows = [{"chat_key": "distilled", "sender_name": "标准案例", "sent_at": 1,
             "text": "标准案例：发音：客户反映没有发音器｜处理：核实后补发发音器"},
            {"chat_key": "distilled", "sender_name": "标准案例", "sent_at": 1,
             "text": "标准案例：开裂：客户反映腿部开裂｜处理：安排返厂维修"}]
    fresh = dedupe_cases(rows, existing)
    assert len(fresh) == 1
    assert "开裂" in fresh[0]["text"]


def test_cases_command_lists_standard_cases(tmp_path):
    service, db, outbox = make_env(tmp_path, ai=None)
    bot = CommandBot(FakeClient(b""), make_config(), service)
    assert "还没有标准案例" in bot.run_command("cases", "", {"id": 42})
    db.kb_add_messages("distilled-x", [
        {"chat_key": "d", "sender_name": "标准案例", "sent_at": i,
         "text": "标准案例：发音：案例{}｜处理：补发".format(i)} for i in range(10)])
    page1 = bot.run_command("cases", "", {"id": 42})
    assert "共 10 条" in page1 and "第 1/2 页" in page1 and "案例0" in page1
    page2 = bot.run_command("cases", "2", {"id": 42})
    assert "第 2/2 页" in page2 and "案例8" in page2
    assert "/cases" in bot.run_command("kb", "", {"id": 42})


def _gate_handler(plan, **card_extra):
    def handler(kwargs):
        card = dict(EMPTY_CARD, issue="加热无效果", **card_extra)
        return result("new_problem", plan=plan, card=card)
    return handler


def test_no_return_gate_asks_amount_when_unknown(tmp_path):
    ai = FakeAI(_gate_handler("补发新件，旧件不用寄回", platform="淘宝"))
    service, db, outbox = make_env(tmp_path, ai)
    service.handle(incoming("加热没效果，客户要换新"))
    assert len(outbox.posts) == 1
    post = outbox.posts[0][1]
    assert "销售额是多少" in post and "哪个平台" not in post
    assert "补发新件" not in post


def test_no_return_gate_merges_platform_question(tmp_path):
    ai = FakeAI(_gate_handler("全额退款，不用退货"))
    service, db, outbox = make_env(tmp_path, ai)
    service.handle(incoming("客户要求退货退款"))
    post = outbox.posts[0][1]
    assert "哪个平台" in post and "销售额是多少" in post


def test_no_return_gate_allows_under_1000(tmp_path):
    ai = FakeAI(_gate_handler("补发新件，旧件不用寄回", platform="淘宝", amount="800"))
    service, db, outbox = make_env(tmp_path, ai)
    service.handle(incoming("小件坏了要换新"))
    post = outbox.posts[0][1]
    assert "【建议方案】" in post and "补发新件" in post and "老板确认" not in post


def test_no_return_gate_annotates_over_1000(tmp_path):
    ai = FakeAI(_gate_handler("补发新件，旧件不用寄回", platform="淘宝", amount="1680"))
    service, db, outbox = make_env(tmp_path, ai)
    service.handle(incoming("大件坏了要换新"))
    post = outbox.posts[0][1]
    assert "补发新件" in post and "1680" in post and "需老板确认" in post


def test_no_return_helpers():
    from efb_aftersales_middleware.service import _card_amount, _no_return_hit
    assert _no_return_hit("先寄回旧件，验货后退款") is False
    assert _no_return_hit("换新不收回旧件") is False  # phrasing not covered, by design
    assert _no_return_hit("旧件无需寄回，直接补发") is True
    assert _card_amount({"amount": "1,280"}) == 1280.0
    assert _card_amount({"amount": "1.2万"}) == 12000.0
    assert _card_amount({"amount": ""}) is None
    assert _card_amount({}) is None
