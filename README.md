# e-m-c.runtime

用于 e-m-c 研究的轻量 Python 运行环境：持久化状态、原始经历、模型调用、时间与延迟事件、实验记录和快照分支。Python ≥ 3.10，零第三方运行依赖。

认知流程由普通 Python 类实现，可以自行安排读取经历、检索、模型调用、状态更新和调度的顺序。研究方向见 [e-m-c.workbench](https://github.com/entropy356/e-m-c.workbench)。

## 运行

```bash
pip install -e .
python -m emc_runtime --config examples/basic.json turn "你好"
python -m emc_runtime --config examples/counter.json chat
python -m emc_runtime --config examples/research.json turn "记录一次经历"
python -m emc_runtime --config examples/research.json attempts
```

也可从仓库根目录设置 `PYTHONPATH=src` 运行。`examples` 中的类通过仓库根目录导入。

## 研究入口

```python
from emc_runtime import AssistantReply

class Workflow:
    def __init__(self, config):
        self.options = config.get("options", {})

    def run(self, ctx):
        state = ctx.state.get("experiment", scope="agent", default={})
        sources = ctx.experiences.query(text="某个细节", limit=10)
        ctx.observe("retrieved_sources", [m.message_id for m in sources])
        messages = ctx.assemble(history_limit=20)
        reply = ctx.models.chat(messages, options=self.options, purpose="reply")
        ctx.state.set("experiment", {**state, "last_time": ctx.now})
        return reply
```

配置 `cognition.kind="python"`，`entrypoint="package.module:Workflow"`，`config` 传给构造器。完整示例见 [examples/research_workflow.py](examples/research_workflow.py)。`run(ctx)` 可返回 `AssistantReply`，或返回 `None` 完成无对外回复的状态更新。

| 接口 | 用途 |
| --- | --- |
| `ctx.state.get/set` | agent / conversation 作用域的版本化 JSON 状态；写入随整轮提交 |
| `ctx.models.chat` | 选择命名模型，逐次传入推理参数、结构化输出参数，记录调用与 usage |
| `ctx.models.embed` | 按需调用命名 embedding 服务并记录结果 |
| `ctx.experiences.query` | 跨会话查询已提交的原文，返回消息 ID、会话、时间和 turn 来源 |
| `ctx.assemble` | 使用指定历史与数据块组装输入，原文存储独立保留 |
| `ctx.now` / `ctx.event.occurred_at` | 本轮开始时间 / 事件发生时间，统一 UTC |
| `ctx.scheduler.schedule/cancel` | 暂存一次性延迟事件或取消请求，随本轮原子提交 |
| `ctx.observe` | 保存研究自定义观察量 |
| `Storage.fork` | 复制一致的数据库快照，开展独立对照实验 |

## 模型配置

根级 `provider` 是默认聊天模型；`models` 和 `embeddings` 分别声明其他命名服务，首次使用时加载。

```json
{
  "schema_version": 1,
  "database": "data/experiment.sqlite3",
  "provider": {
    "kind": "openai-compatible",
    "base_url": "https://api.example.com/v1",
    "model": "your-model",
    "api_key_env": "MODEL_API_KEY",
    "options": {"temperature": 0.2},
    "max_attempts": 1
  },
  "models": {},
  "embeddings": {
    "index": {
      "kind": "openai-compatible",
      "base_url": "https://api.example.com/v1",
      "model": "your-embedding-model",
      "api_key_env": "MODEL_API_KEY"
    }
  },
  "cognition": {"kind": "direct", "history_limit": 40},
  "modules": []
}
```

调用示例：`ctx.models.chat(messages, model="extractor", options={...})`、`ctx.models.embed(text, model="index")`。每次聊天调用的参数覆盖该模型的默认参数，不修改后续调用的默认值。OpenAI 兼容端点接收 JSON 参数，如 `temperature`、`seed`、`response_format`；具体参数支持由端点决定。文本回复接口固定单条、非流式响应。

## 调度与实验

```python
from datetime import datetime, timedelta
from emc_runtime import Event

ctx.scheduler.schedule(
    Event("internal_trigger", conversation_id=ctx.event.conversation_id,
          metadata={"reason": "experiment"}),
    due_at=datetime.fromisoformat(ctx.now) + timedelta(minutes=5),
)
```

由实验宿主定期调用 `engine.run_due()`，或执行：

```bash
python -m emc_runtime --config examples/research.json schedules
python -m emc_runtime --config examples/research.json run-due --limit 10
python -m emc_runtime --config examples/research.json fork data/control.sqlite3
```

分支实验配置中的 `database` 指向复制后的数据库。快照包括状态、经历、观察记录和待执行事件；模型配置与代码由实验者分别保存。`build_engine(config, clock=...)` 接受实现 `now() -> timezone-aware datetime` 的时钟，用于控制时间推进。

## 持久化与验证

数据库 schema 为 **v2**；配置文件 schema 为 **1**。请为新版配置新的数据库路径，版本不匹配时会拒绝打开，已有文件不会被迁移或覆盖。

状态、消息、turn 和调度变更整轮原子提交；调用观察记录独立持久化，失败后仍可查看。模型调用不处于数据库事务中，已产生的外部调用不会因状态回滚而撤销。

```bash
python -m unittest discover -v
```

测试使用离线模型和注入 transport。架构与研究接口见 [docs/architecture.md](docs/architecture.md)；默认流程的状态模块见 [docs/module-development.md](docs/module-development.md)。开发遵循工作台的[代码维护约定](https://github.com/entropy356/e-m-c.workbench/blob/main/rules/code-maintenance.md)。
