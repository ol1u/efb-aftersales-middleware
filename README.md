# efb-aftersales-middleware

EFB 中间件：一个跟着微信售后群对话走的智能助理。它监控群里全部消息，把每个客户问题维护成一条线程：识别到新问题就给出【问题总结】+【建议方案】；有人纠正总结或补充关键信息，就更新出新版；拿不准（是补充还是新问题、属于哪个问题）就直接在群里向对方提问，不瞎猜。方案会参考从 Telegram 导出的历史售后记录（知识库）和当前启用的策略模式（如保评分阶段可亏本安抚客户）。

## 消息流

```text
微信售后群 -> ComWeChat 从端 -> [本中间件：存档 -> AI 判断/总结/建议]
建议帖：售后 bot -> ETM 话题 -> ETM（trusted_bots 放行）以你的微信号转发回微信群（文字引用形式）
```

- 群里可能同时有多个客户的问题在走：归属靠微信引用、复制全文的习惯痕迹、发送者身份和内容关键词，AI 做最终仲裁。
- 【问题总结】【建议方案】引用问题原消息发帖；更新版引用线程的第一条问题消息；提问引用让它困惑的那条消息。
- 群主（你）在群里的现场指示优先于预设策略。
- 自己发出的帖子经 ETM 转回来时会被认出并跳过，不会回环。

## 知识库

把 Telegram Desktop 导出的聊天记录 `result.json` **作为文件直接发给售后 bot**（需是配置里的 Telegram 管理员），bot 自动下载、解析、入库；随时可再发新的导出增量导入。出方案时自动检索相似历史片段（含上下文）喂给 AI。注意 Bot API 下载上限 20MB。

## 策略模式

管理员命令（私聊 bot 或在话题里发均可）：

- `/mode`：查看全部模式和当前模式
- `/mode 保评分`：切换模式，立即生效
- `/prompt 模式名`：查看该模式的预设 prompt
- `/setprompt 模式名 | prompt 全文`：修改（或新建）模式的预设 prompt
- `/kb`：知识库统计；`/threads`：进行中的问题列表

内置两个模式：**常规**（兼顾成本与体验）和**保评分**（保店铺评分优先，可亏本处理）。

## 前置条件

1. ETM fork 需包含 `trusted_bots` 支持，且在 @BotFather 给 ETM bot 打开 Bot-to-Bot Communication、关闭 Privacy Mode（或设为群管理员）。
2. 售后 bot 拉进 ETM 话题群；ETM 配置里把售后 bot 的 ID 加入 `trusted_bots`、话题群 ID 加入 `trusted_bot_chats`。不要把售后 bot 放进 `admins`。
3. 在 EFB 主配置的 `middlewares` 里启用 `ol1u.aftersales`（排在语音识别中间件之后）。
4. 复制 `config-example.yaml` 到 `~/.ehforwarderbot/profiles/ComWeChat/ol1u.aftersales/config.yaml` 并填写（含 DeepSeek 的 `ai.api_key`，或环境变量 `DEEPSEEK_API_KEY`；`telegram_admins` 填可以给 bot 发命令/导入文件的人）。
5. 重启 `ehforwarderbot -p ComWeChat`。

## 数据

SQLite，默认在 `~/.ehforwarderbot/profiles/ComWeChat/ol1u.aftersales/aftersales.db`：

- `records`：群里每条原始消息 + 本助理发出的每条帖子（带 thread_id 归属）
- `threads`：问题线程（提出者、当前版总结与方案、版本号）
- `kb_messages`：导入的历史聊天记录（知识库）
- `settings`：模式与预设 prompt

## 测试

```bash
pip3 install -e ".[tests]"
python3 -m pytest tests
```

## 已知限制

- 每条消息一次 AI 调用；AI 不可用或置信度不够时只存档、不发言。
- 非文字消息只存档。
- 助理帖在微信里显示为你本人发送。
