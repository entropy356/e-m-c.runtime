# e-m-c.runtime

轻量、单进程、Python 模块化的 Agent 实验运行环境（Python ≥ 3.10，零第三方依赖）。

Runtime 是可复用的基础设施：输入事件、模块调度、原始上下文、模型调用编排、状态事务、运行记录与 CLI。认知数学模型（Memory / Emotion / Clock 等）以独立模块形式接入，可以完全不装任何模块运行。

## 开发约定

- [文档与文件约定](https://github.com/entropy356/e-m-c.workbench/blob/main/rules/documentation.md)
- [代码维护约定](https://github.com/entropy356/e-m-c.workbench/blob/main/rules/code-maintenance.md)
- [e-m-c.* 日常整理](https://github.com/entropy356/e-m-c.workbench/blob/main/rules/workflows/daily-maintenance.md)

## 安装

~~~bash
git clone https://github.com/entropy356/e-m-c.runtime
cd e-m-c.runtime
pip install -e .        # 或：export PYTHONPATH=src 后从仓库根目录运行
~~~

## 快速开始（离线，无网络）

~~~bash
# 最小配置：modules 为空
python -m emc_runtime --config examples/basic.json turn "你好"

# 计数器模块：每轮加一，agent 作用域（跨会话共享）
python -m emc_runtime --config examples/counter.json turn "你好" --conversation c1
python -m emc_runtime --config examples/counter.json turn "再见" --conversation c2
python -m emc_runtime --config examples/counter.json inspect --conversation c1 --json
# counter.state.count == 2：两个会话共享同一 agent 级状态

# 内部事件触发
python -m emc_runtime --config examples/counter.json trigger --conversation c1 --reason clock

# 交互式聊天
python -m emc_runtime --config examples/counter.json chat --conversation c1

# 批量剧本
python -m emc_runtime --config examples/basic.json scenario scenarios/smoke.json
~~~

## 真实 API 配置

Provider 只负责 OpenAI 兼容的 `/chat/completions` 传输，不内置任何记忆协议：

~~~json
{
  "schema_version": 1,
  "database": "data/agent.sqlite3",
  "provider": {
    "kind": "openai-compatible",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini",
    "api_key_env": "OPENAI_API_KEY",
    "timeout": 60,
    "proxy": null,
    "options": { "temperature": 0.2 }
  },
  "cognition": { "kind": "direct" },
  "system_prompt": "You are a helpful conversational agent.",
  "modules": [
    {
      "id": "counter",
      "entrypoint": "examples.counter_module:CounterModule",
      "scope": "agent",
      "config": {}
    }
  ]
}
~~~

~~~bash
export OPENAI_API_KEY=sk-...
python -m emc_runtime --config my.json chat
~~~

## 模块

模块是实现五个方法的普通 Python 类（`examples/counter_module.py` 是完整参考实现）：

| 方法 | 职责 |
| --- | --- |
| `initial_state()` | 无历史状态时的新建状态 |
| `expose(state)` | 只读公共信号，供其他模块经 `ctx.exports` 读取 |
| `prepare(event, state, ctx)` | 在深拷贝工作状态上计算，返回 `Prepared`（工作状态 + ContextBlock + trace） |
| `finalize(event, prepared, reply, ctx)` | 模型回复后返回 `Finalized`（最终状态 + trace） |
| `inspect(state)` | 只读呈现状态 |

- 作用域：`agent`（跨会话共享）或 `conversation`（按会话隔离）。
- 状态提交是全轮原子的：本轮所有消息、所有模块状态、turn 记录和全局 revision 在一个 SQLite 事务里落库；任何阶段失败则整轮回滚。
- 模块只贡献带来源标注的数据块（ContextBlock），不改写系统提示词，不接管普通对话上下文。

详见 [docs/module-development.md](docs/module-development.md) 与 [docs/architecture.md](docs/architecture.md)。

## 测试

~~~bash
python -m unittest discover -v   # 38 个离线测试，覆盖规格 T01–T17
~~~

涉及真实网络的请求（真实 API 对话）未在 CI 中执行；Provider 传输层以注入 transport 的方式离线测试。

## 仓库结构

~~~text
src/emc_runtime/
  contracts.py        事件、消息、模块协议、RunResult、异常体系
  config.py           JSON 配置加载与校验
  storage.py          SQLite：快照 / 单事务提交 / 只读查询 / 乐观并发
  context.py          确定性上下文组装（固定顺序）
  cognition.py        可替换认知策略（首版 direct：单次调用）
  engine.py           单轮编排 [1]-[7]
  modules.py          显式模块加载、注册与作用域
  events.py           提交后观察事件总线（失败隔离）
  providers/          offline 确定性回放 + OpenAI 兼容传输 + 可选 embedding
  cli.py              turn / chat / trigger / history / inspect / scenario
examples/             计数器模块与两份配置
scenarios/smoke.json  冒烟剧本（含 internal_trigger）
tests/                T01–T17 离线测试
~~~
