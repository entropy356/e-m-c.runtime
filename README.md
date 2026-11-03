# Demo 3：Emotion-free Long-term Memory Agent

把 Demo 2 的记忆动态机制放进一个真正连续对话的 Agent 中：先保证模型能看到按顺序保存的用户与助手消息，再让长期记忆从对话中形成、被检索和强化。

**不引入 Emotion，不做前端。** 保留真实 Chat、Embedding API 接入和离线测试。

## 要解决的问题

Demo 2 已经有记忆检索、权重衰减和激活恢复，但模型每轮只收到当前输入与 Top-K 记忆，没有收到之前的对话。Demo 3 把 Conversation Context 纳入运行时：

- 每条用户消息和助手回答都按原始内容、角色和顺序持久化。
- 每次认知都收到当前会话此前的完整消息历史，以及当前用户输入。
- 不再要求记忆库重建刚刚发生过的对话。
- 长期记忆从对话中提取，跨会话共享；每个会话的对话历史彼此隔离。
- 长期记忆继续使用 Demo 2 的权重衰减、激活恢复和稳定性机制。
- 一轮认知、消息写入、记忆更新和 Trace 在同一个 SQLite 提交中完成。

## 运行

需要 Python 3.10+；没有第三方 Python 依赖。

### 测试

在本目录运行：

~~~bash
python -m unittest discover -v
~~~

### 离线模式

离线模式使用确定性哈希字符向量和固定模板回答，只验证消息顺序、持久化、检索链路和状态提交，不代表真实语义检索或真实对话能力。

~~~bash
python -m long_term_memory --mode offline --db data/offline.sqlite3 turn "你好"
python -m long_term_memory --mode offline --db data/offline.sqlite3 history
python -m long_term_memory --mode offline --db data/offline.sqlite3 state
~~~

### 真实 Chat + Embedding API

Chat 使用 OpenAI-compatible /chat/completions；Embedding 支持 OpenAI-compatible /embeddings 或 Gemini 原生 :embedContent。Chat 和 Embedding 可以使用不同服务商、模型、密钥、代理和超时。

~~~bash
export AGENT_CHAT_BASE='https://your-chat.example/v1'
export AGENT_CHAT_MODEL='your-chat-model'
export AGENT_CHAT_KEY='your-chat-key'

export AGENT_EMBED_PROTOCOL='openai'  # 或 gemini
export AGENT_EMBED_BASE='https://your-embedding.example/v1'
export AGENT_EMBED_MODEL='your-embedding-model'
export AGENT_EMBED_KEY='your-embedding-key'
export AGENT_EMBED_DIMENSIONS='768'   # 可选，按服务商支持情况设置

# 可选：分别配置代理与超时
export AGENT_CHAT_PROXY='http://127.0.0.1:8080'
export AGENT_EMBED_PROXY='http://127.0.0.1:8080'
export AGENT_CHAT_TIMEOUT='120'
export AGENT_EMBED_TIMEOUT='60'
~~~

启动交互式聊天：

~~~bash
python -m long_term_memory --mode api --db data/agent.sqlite3 --conversation default chat
~~~

输入 /exit 或 /quit 退出。后续重新启动并使用相同数据库与会话 ID，会恢复之前的对话。

聊天里可以随时看历史，不打断对话、不调用模型：

~~~text
you> /history          # 最近 10 条
conversation default · 6 messages · 3 rounds · step 3
--------------------------------------------------------------------------
SEQ  ROLE       STEP  CONTENT
  1  user       1     我更喜欢简短的书面进度，每周五发一次。
  2  assistant  1     好的，我记住了。以后我会每周五给你发一次简短的书面进度…
  3  user       2     我在做一个长期记忆 Agent，用 Python 3.14 和 PowerShell 7 开发。
  4  assistant  2     好的，我记住了。你在用 Python 3.14 和 PowerShell 7 开发一个长期…
  5  user       3     你还记得我喜欢什么样的进度汇报吗？
  6  assistant  3     记得，你偏好简短的书面进度更新，每周五发送一次。
you> /history 2        # 只显示最近 2 条
you> /history all      # 全部
you> /help             # 命令说明
~~~

`/history` 以只读方式打开同一个 SQLite，读的是已提交的消息（`SEQ` 即 `messages.seq`，`STEP` 是写入时的轮次）；命令本身不写库、不产生消息。超过显示条数时头部会提示「只显示最近 N 条，共 M 条」。

也可以逐轮调用：

~~~bash
python -m long_term_memory --mode api --db data/agent.sqlite3 --conversation default turn "我更喜欢简短的书面进度。"
python -m long_term_memory --mode api --db data/agent.sqlite3 --conversation default turn "你还记得我喜欢什么样的进度吗？"
~~~

用另一个会话 ID 开始新对话：

~~~bash
python -m long_term_memory --mode api --db data/agent.sqlite3 --conversation project-x chat
~~~

同一数据库中的长期记忆跨会话共享，但不同会话的消息历史不会混在一起。

### 认知协议（显式配置，不自动降级）

一轮认知固定拆成**两个请求**：

1. **记忆操作**：模型报告本轮回答依赖了哪些记忆（`used_memory_ids`）、有哪些新记忆值得保存（`memories`）。用哪种线格式由 `AGENT_COGNITION_PROTOCOL` 决定。
2. **回答**：把第一阶段的提交回填进上下文（`tool` 协议回填 `assistant.tool_calls` + `tool` 消息；`json`/`text` 回填 assistant 的 JSON 原文 + 一条衔接消息），再请求一次，**取 `choices[0].message.content` 作为回答**。

也就是主流形态：**工具/结构化字段只承载元数据，面向用户的回复走正文**。协议不被支持就直接失败，不做自动重试或回滚到另一种协议。

| 取值 | 第一阶段 | 第二阶段 |
|---|---|---|
| `tool`（默认） | `tools` + 强制 `tool_choice` 调用 `submit_cognition`，参数由 JSON Schema 约束，`used_memory_ids.items.enum` 每轮填入当前提供的记忆 ID，`memories` 最多三条 | `tool_choice: "none"`，正文即回答 |
| `json` | 文本协议 + `response_format: json_object`，输出 `{used_memory_ids, memories}` | 无 tools，正文即回答 |
| `text` | 纯文本协议，模型按提示词自行输出 JSON；解析时剥代码围栏、截取首尾大括号 | 同上 |

~~~bash
export AGENT_COGNITION_PROTOCOL='tool'        # 默认值
python -m long_term_memory --mode api --protocol json ...   # 命令行覆盖
~~~

`run-api.ps1` 对应参数：`.\run-api.ps1 chat -Protocol json`。

`text` 是最不稳的一档：不发 `response_format` 时模型可能直接回自然语言（实测 DeepSeek-V4-Flash 在多轮上下文里就会这样），第一阶段随即报错。它只是给"连 `response_format` 都拒绝"的网关留的兜底开关，能用 `tool` 或 `json` 就不要用。

`tool` 协议下若第一阶段没有调用工具，或网关返回 400/404/422/501，本轮直接失败并回滚（消息与记忆都不提交），错误信息会提示切换 `AGENT_COGNITION_PROTOCOL`。第二阶段返回空正文同样报错，不会拿工具参数里的内容顶替。429 与 5xx 仍按原有退避重试——那是同一请求的网络重试，不涉及协议切换。

每轮 Trace 与 `state` 里的 `cognition_protocol` 记录本次使用的协议。

### 提示词（外置，改文案不用动代码）

系统提示词放在 `prompts/` 下，`providers.py` 只在启动时读取：

| 文件 | 作用 |
|---|---|
| `context.md` | 共享部分：一轮两个请求的分工、上下文三部分（对话历史 / 长期记忆 / 当前输入） |
| `protocol-tool.md` | 接在 `context.md` 之后，描述 `submit_cognition` 的两个字段 |
| `protocol-json.md` | 接在 `context.md` 之后，描述 JSON 协议的两个字段 |
| `answer.md` | 第二阶段的衔接消息（`json`/`text` 协议用；`tool` 协议靠 `tool` 结果衔接） |

拼装规则：`SYSTEM = context + "\n\n" + protocol-json`，`SYSTEM_TOOL = context + "\n\n" + protocol-tool`。文件缺失或为空会在启动时报错并给出路径。改措辞直接编辑这几个 `.md`，测试只断言文件结构（前缀、工具名、拼装规则），不校验具体句子。

### 查看记忆与上下文（只读）

`memories` 与 `context` 都以只读方式打开 SQLite，不需要 API Key，不创建、不迁移、不写库。也可以直接跑脚本（`viewer.py` 的相对导入已兼容直接执行）：`python long_term_memory/viewer.py --db data/agent.sqlite3 context`。

**长期记忆**

~~~bash
python -m long_term_memory --db data/agent.sqlite3 memories
python -m long_term_memory --db data/agent.sqlite3 memories --decay          # 多一列下一轮衰减预测
python -m long_term_memory --db data/agent.sqlite3 memories 2                # 单条记忆的全部字段
python -m long_term_memory --db data/agent.sqlite3 memories --sort age --conversation project-x
python -m long_term_memory --db data/agent.sqlite3 memories --json           # 机器可读
~~~

~~~text
data\agent.sqlite3
revision 3 · step 3 · 2 memories · 1 conversations · 3 rounds
embedding: api:openai:jina-embeddings-v3:native
config: two-state · decay 0.12 · protect 0.85 · recover 0.65 · gain 0.12 · top_k 5
------------------------------------------------------------------------------------
ID  WEIGHT   STAB  ACT  AGE  LAST     CONV  CONTENT
 1   0.949  0.031    2    2     3  default  用户偏好简短的书面进度更新，每周五发送一次。
 2   0.887  0.000    0    1     -  default  用户正在用 Python 3.14 和 PowerShell 7 …
------------------------------------------------------------------------------------
WEIGHT/STAB 是上一轮提交时的存储值；AGE = step - created_step；LAST = 最后使用轮次
~~~

- 列含义：`WEIGHT`/`STAB` 是上一轮提交时写入的值（衰减发生在下一轮开始时，库里不预先衰减）；`ACT` 激活次数；`AGE` = 当前 step - 创建 step；`LAST` 最后一次被引用的轮次；`CONV` 记忆来源会话。
- `--sort`：`weight`（默认，降序）/ `id` / `age` / `activations`。
- `--decay` 只在展示时算一次衰减，不写库。

**对话上下文**

~~~bash
python -m long_term_memory --db data/agent.sqlite3 context                    # 默认会话
python -m long_term_memory --db data/agent.sqlite3 --conversation project-x context
python -m long_term_memory --db data/agent.sqlite3 context --payload          # 最近一轮第 1 个请求体
python -m long_term_memory --db data/agent.sqlite3 context --full --json
~~~

~~~text
data\agent.sqlite3
conversation default · 6 messages · 3 rounds · step 3
conversations: default(6)
----------------------------------------------------------------------------------------------------
SEQ  ROLE       STEP  CONTENT
  1  user       1     我更喜欢简短的书面进度，每周五发一次。
  2  assistant  1     好的，我记住了。以后我会每周五给你发一次简短的书面进度更新。
  ...
----------------------------------------------------------------------------------------------------
STEP  USED  WRITTEN  SEL  INPUT                                   ANSWER
   1  -     1        0/0  我更喜欢简短的书面进度，每周五发一次。  好的，我记住了。以后我会每周五给你发一次简…
   2  1     2        1/1  我在做一个长期记忆 Agent，用 Pyt…       好的，我记住了。你在用 Python 3.…
   3  1     -        2/2  你还记得我喜欢什么样的进度汇报吗？      记得，你偏好简短的书面进度更新，每周五发送…
----------------------------------------------------------------------------------------------------
SEL = 检索命中 Top-K / 候选总数；USED 是本轮回答实际引用的记忆 ID；WRITTEN 是本轮新写入的记忆 ID；ANSWER 是本轮 assistant 回答（过长截断）
~~~

- 上半张表是按 `seq` 排序的原始消息（`STEP` 是写入时的轮次），也就是每次认知都会完整重发的那份上下文。user 与 assistant 两侧原文都在，默认按宽度截断，`--full` 显示完整回答。
- 下半张表是每轮的记忆活动：`SEL` 检索命中数 / 候选数，`USED` 回答实际引用的记忆 ID，`WRITTEN` 新写入的记忆 ID，`ANSWER` 是本轮 assistant 回答（与 `INPUT` 同样按 22 字截断加 `…`，`--full` 不截断）。
- `--payload` 打印**最近一轮第 1 个请求体**（记忆操作），顺序就是模型收到的顺序：`[1] system` 提示词（随 `--protocol` 切换）、`[2] system` 长期记忆 JSON（**只含该轮检索命中并注入的 Top-K**，未命中的不出现，每条只有 `id` + `content`）、`[3..]` 该轮的历史消息 + 本轮输入（末条标 `<- 本轮输入`）。tool 协议下还列出 `used_memory_ids.enum`，与实际注入的 ID 一致。每条消息带 `(seq N · step N)`，方便和 `messages` 表对账。第 2 个请求不重复渲染，末尾用一行说明它在这份 messages 上追加了什么。库里还没有已完成轮次时，退化为「下一轮请求骨架」。

这两个子命令在 `run-api.ps1` 里同样可用，识别到它们时会跳过 Key 与 embedding profile 校验，因此离线库、换过模型的库都能直接看：

~~~powershell
.\run-api.ps1 memories -Database data/agent.sqlite3 -Decay -Sort id
.\run-api.ps1 context  -Database data/agent.sqlite3 -Payload
~~~

Windows 下若把 `py viewer.py` 的输出接进管道或 `Select-Object` 时中文变乱码，先执行 `[Console]::OutputEncoding = [Text.Encoding]::UTF8`（`run-api.ps1` 已自动设置）；直接输出到终端不受影响。

### 监控模式（主进程推送，不轮询）

`watch`（别名 `monitor`）不在循环里读库。主进程用 `chat --events` 开一个只绑 `127.0.0.1` 的事件端口，每轮提交后推一条 JSON；查看器阻塞在 `recv()` 上，收到一条才读一次库、刷一次屏。空闲时两边都不动。

两个窗口，顺序无所谓（查看器先连上会先渲染当前状态）：

~~~powershell
# 窗口 A：主进程
.\run-api.ps1 chat -Events
# [events] 监听 127.0.0.1:63469 · 端口文件 data\agent.sqlite3.events.json
# [events] 另开一个窗口: python -m long_term_memory --db data/agent.sqlite3 --conversation default watch

# 窗口 B：查看器，端口从 <db>.events.json 自动读取
.\run-api.ps1 watch
~~~

~~~bash
python -m long_term_memory --mode api --db data/agent.sqlite3 chat --events
python -m long_term_memory --db data/agent.sqlite3 watch                 # 等推送
python -m long_term_memory --db data/agent.sqlite3 watch --rounds 10 --decay
python -m long_term_memory --db data/agent.sqlite3 watch --payload        # 最近一轮第 1 个请求体
python -m long_term_memory --db data/agent.sqlite3 watch --json --no-clear > watch.jsonl
~~~

实测（真实 DeepSeek 三轮对话，`revision 3` 的初始快照 + 每次提交刷新一次）：

~~~text
data\viewer-demo.sqlite3
● 监控中 127.0.0.1:63469 · 01:55:49 · revision 3 · step 3 · 2 memories · 1 conversations · 3 rounds
embedding: api:openai:jina-embeddings-v3:native
config: two-state · decay 0.12 · protect 0.85 · recover 0.65 · gain 0.12 · top_k 5
conversation default · 6 messages
---------------------------------------------------------------------------------------------------
ID  WEIGHT   STAB  ACT  AGE  LAST     CONV  CONTENT
 1   0.949  0.031    2    2     3  default  用户偏好简短的书面进度更新，每周五发送一次。
 2   0.887  0.000    0    1     -  default  用户正在用 Python 3.14 和 PowerShell 7 开发一个…
---------------------------------------------------------------------------------------------------
轮次活动（最近 3 轮，会话 default）
STEP  USED  WRITTEN  SEL  INPUT                                   ANSWER
   1  -     1        0/0  我更喜欢简短的书面进度，每周五发一次。  好的，我记住了。以后我会每周五给你发一次简…
   2  1     2        1/1  我在做一个长期记忆 Agent，用 Pyt…       好的，我记住了。你在用 Python 3.…
   3  1     -        2/2  你还记得我喜欢什么样的进度汇报吗？      记得，你偏好简短的书面进度更新，每周五发送…
~~~

- 事件体：`type`（`turn` / `error`）、`step`、`revision`、`conversation_id`、`input`、`answer`、`used_memory_ids`、`written_ids`、`activated_ids`。失败的一轮推 `error`，查看器顶部显示「上一轮失败：…」，不刷新成空白。
- 表格视图只显示记忆表 + 轮次活动表，不再重复渲染最后一轮的问答原文；每轮的问答都在轮次表里（`INPUT` 与 `ANSWER` 两列，各按 22 字截断加 `…`，`--full` 不截断），更早的完整消息用 `context` 看。
- `--payload` 把整个视图换成**最近一轮第 1 个请求体**（`context --payload` 的实时版）：`[1] system` 提示词、`[2] system` 记忆 JSON（只含该轮注入的 Top-K，id + content）、`[3..]` 该轮历史消息 + 本轮输入，顺序即发送顺序，每次推送后整块重绘。`--protocol json|text` 切换提示词版本，`--json` 时正文放在 `payload` 字段里。
- `publish()` 只入队（上限 256 条，满了丢最旧的），单次发送超时 5s 并剔除坏连接：慢或卡住的查看器不会拖住对话。
- 端口文件 `<db>.events.json` 由主进程写入、退出时删除（端口匹配才删）。agent 没在跑时 `watch` 直接报 `cannot reach the agent at 127.0.0.1:…`，不会空转。
- `--json` 每次刷新输出一行 JSON（jsonl，适合落盘分析），其中 `recent[].answer` 就是 assistant 回答；`--no-clear` 不清屏；终端是 TTY 时默认清屏重绘，Windows 下会先尝试开启 VT 处理。
- 可以同时开多个 `watch`：hub 把每条事件广播给**所有**连接，各自按自己的 `--conversation` / `--rounds` / `--payload` 渲染（实测 3 个并存：两个表格视图 + 一个 payload 视图，每个都是 1 次初始快照 + 3 次推送）。同一 db 只跑一个 `chat --events`；不同 db 各有自己的端口文件，互不影响。
- 边界：推送是「提交后」通知，连上之前的事件不补发（但连上立刻渲染当前状态，所以不会显示过期数据）；只支持本机。agent 若被强杀（没走正常退出），`<db>.events.json` 会残留，`watch` 会报 `cannot reach`，删掉该文件或重开 chat 即可。

### 场景运行

~~~bash
python -m long_term_memory --mode api --db data/scenario.sqlite3 run scenarios/continuity-and-recall.json --output results/continuity-and-recall.json
~~~

每个场景使用独立数据库。真实 API 模式下，LLM 会从对话中提出值得长期保存的记忆；离线模式不会模拟记忆提取。

## 记忆动态

每轮用户交互时，已有记忆先衰减：

\[
w^- = w \exp[-\lambda(1-\gamma s)]
\]

只有 LLM 明确报告本轮实际使用某条已提供的记忆，才执行激活：

\[
w' = w^- + (1-w^-)r
\]

\[
s' = s + (1-s)\beta(1-w^-)
\]

检索分数：

\[
score_i = \frac{1+\cos(q,m_i)}{2} \cdot w_i
\]

- weight 表示记忆当前的可访问程度。
- stability 影响后续衰减速度，不直接增加当前检索分数。
- weight-only 与 two-state 保留 Demo 2 的两种对照模式。
- 现实时间不参与衰减；每次成功提交的用户交互推进一轮。
- 新记忆以 weight=1、stability=0 写入，不在创建轮立即衰减。
- 精确重复的记忆文本不会重复写入。

## 数据组织

SQLite 中有三类主要数据：

- messages：按 conversation_id + seq 保存完整有序对话，包含 user 和 assistant。
- runtime.state：保存跨会话共享的长期记忆及其动态状态。
- rounds：保存每轮检索、认知、激活和写入 Trace。

监控模式下主进程还会在旁边写一个 `<db>.events.json`（端口 + pid），退出时删除；它只是连接信息，不属于数据本体。

当前 Context 组装方式：

~~~text
System Instructions
Long-term Memory Candidates
Earlier User Message
Earlier Assistant Message
...
Current User Message
        |
        v
       LLM
        |
        +--> submit_cognition 工具调用（或按配置走文本 JSON）
              +--> Assistant Message
              +--> Memory Proposals
              +--> Used Memory IDs
~~~

读取历史不会推进 Runtime 轮次，也不会改变记忆权重。认知失败、记忆 Embedding 失败或并发状态冲突时，本轮消息和记忆状态都不会提交。

## 当前限制与下一步

- 当前每次请求都会传递该会话的全部原始消息，不做摘要或截断。这让连续上下文行为可以先被直接观察，但长对话最终会受到模型 Context Window 限制。
- 当前消息角色只有 user 和 assistant，尚未处理多模态消息或流式中断恢复。
- 认知只用到单个工具（`submit_cognition`），没有多工具编排；工具 schema 每轮重复发送，是一笔固定的 token 开销。
- 认知协议不做自动降级，换模型或换网关若不支持当前协议，需要显式改 `AGENT_COGNITION_PROTOCOL`。
- 长期记忆由 LLM 提出，used_memory_ids 也是 LLM 自报的使用指标；需要人工抽查提取质量与实际引用情况。
- 记忆检索仍然遍历当前记忆集合，尚未引入向量索引。
- 记忆来源目前记录会话 ID，不做细粒度消息级溯源。
- 监控通道只绑 `127.0.0.1`，只推给已经连上的查看器：连接前的事件不补发，也不支持跨机观察。

先用真实聊天观察：原始历史是否足以保持对话连续性，长期记忆是否能在新会话中提供有用信息，以及记忆是否会因反复激活而过度垄断上下文。之后再决定上下文压缩、记忆去重与索引优化的实现方式。
