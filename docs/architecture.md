# 架构说明

## 设计原则

> Runtime 只提供稳定的运行协议，不提供固定的认知流程。

- Core（engine/context/storage/cognition/providers/cli）不依赖任何具体模块实现；模块只能通过配置进入。
- 原始会话历史独立保存、按原顺序供 LLM 使用；Memory 类模块不接管普通上下文。
- 所有状态变更走统一事务；LLM 调用永远发生在事务之外。
- 失败要么整轮生效，要么整轮无效；没有半提交状态。

## 单轮生命周期（engine.py）

固定顺序，模块不可改变：

1. **事件校验与快照** — `validate_event` 检查事件契约；`Storage.snapshot` 在一个读事务内返回全局 revision、按 seq 排列的原始历史、各模块状态。
2. **模块 expose + prepare** — 先对同一快照计算所有模块的 `expose`（只读公共信号），再按配置顺序运行 `prepare`。模块拿到的是深拷贝工作状态与冻结的 `ModuleContext`（原始历史、当前事件、exports）。返回值必须是 `Prepared`；ContextBlock 的 `source` 必须等于模块自身 id（防伪造）。
3. **上下文组装**（context.py）— 固定顺序：基础 system 指令 → 模块数据块（包裹在"参考数据"框架中，带来源标注）→ 原始历史 → 当前输入。internal_trigger 用内部触发说明代替用户消息，且永不入库。
4. **认知策略**（cognition.py）— 当前 `direct` 策略经 Provider 发起一次调用，返回纯文本 `AssistantReply`；认知策略可独立替换。
5. **模块 finalize** — 在同一份工作副本上运行，返回 `Finalized`。
6. **原子提交**（storage.py）— 单个 `BEGIN IMMEDIATE` 事务：校验 revision 未变（乐观并发控制），写入本轮消息、所有模块状态、turn 记录，全局 revision + 1。revision 冲突抛 `ConcurrencyConflict`，引擎不自动重试（重放已付费的模型请求既昂贵又不确定）。
7. **观察事件**（events.py）— 提交成功后发布 `turn_committed`（含 reply、revision、traces 摘要）；失败时尽力发布 `turn_failed`。观察者异常被隔离，绝不影响已提交结果。

## 持久化（SQLite，schema v1）

| 表 | 内容 |
| --- | --- |
| `runtime_meta` | 全局 revision（单调递增）与 schema 版本 |
| `conversations` | 会话元数据 |
| `messages` | 原始对话（user/assistant），`(conversation_id, seq)` 唯一 |
| `module_states` | `(module_id, scope_key)` → 状态 + state_version；scope_key 为会话 id 或保留键 `__agent__` |
| `turns` | 每轮记录：trace（含完整发送给 LLM 的消息序列）、事件、revision |

- 只读操作（history / inspect / recent_turns）不产生任何写入，也不会为从未运行过的模块创造状态。

## 模块状态版本

模块声明 `state_version`。加载已存状态时版本不一致 → `ModuleStateVersionError`，存量状态保持原样，由使用者决定处理方式。

## 并发

- 单写多读：任何写事务 `BEGIN IMMEDIATE`；读操作短事务。
- 乐观并发：提交时 revision 不匹配即冲突失败，不做自动重试或排队。
- 单进程设计；不做跨进程消息总线。

## 可选 Embedding

`providers/embeddings.py` 提供 OpenAI 兼容与 Gemini 适配器，供模块使用；默认不配置，基础运行不依赖它。
