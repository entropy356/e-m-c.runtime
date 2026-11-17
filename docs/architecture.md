# 研究运行基础设施

## 执行边界

Engine 校验输入、读取一致快照，调用 `workflow.run(ctx)`，最后提交该流程暂存的状态、原始消息和调度变更。研究流程控制内部步骤及顺序。

`direct` 是一个内置流程：prepare 模块 → 组装上下文 → 一次模型调用 → finalize 模块。自定义流程使用 `cognition.kind="python"` 和 `entrypoint` 加载；构造签名为 `(config)`，执行签名为 `run(ctx)`，返回 `AssistantReply | None`。

状态通过 `ctx.state.set()` 暂存；`get()` 返回副本。实例字段适合保存配置，跨轮研究状态应放入状态服务，才能获得回滚、重启恢复与实验分支语义。若使用模块助手，应成对调用 `prepare_modules()` 与 `finalize_modules(reply)`。

## 经历与上下文

原始 user/assistant 消息单独保存；内部事件保存在 turn 的 event 中，不伪造用户历史。返回 `None` 时不会生成 assistant 消息。查询结果带消息 ID、conversation_id、turn_id 和记录时间；事件发生时间另见 `event.occurred_at`。

`ctx.experiences.query(conversation_id=None, text=None, before_id=None, after_id=None, limit=100)` 可跨会话查询。`text` 是字面子串过滤；结果为最新匹配项按原始顺序排列，ID 游标可分页。读取限制在本轮快照 revision，排除本轮开始后的提交。派生索引可以根据原文及 ID 独立构建、重建。

`ctx.assemble(blocks=..., history=..., history_limit=...)` 是可选便利接口。研究流程也可以直接构造模型消息；上下文选择不改写原文。默认 direct 流程的 `history_limit` 按消息条数计，0 表示不带历史，省略表示全部历史；token 预算由研究流程按所用模型处理。

## 模型调用与观察

`ctx.models.chat(messages, model="default", options={}, purpose=None)` 和 `embed(text, model=..., purpose=None)` 支持多次调用。所有模型按首次调用加载；命名配置独立，可分别使用不同模型。单次 options 覆盖模型默认 options；聊天返回内容、usage、实际模型名和结束原因。

每次逻辑调用前保存输入，返回或失败后更新输出、状态和耗时。attempt 在执行前标为 running，成功状态与 turn 原子提交，异常时单独标为 failed。进程被强制终止时，running 表示结果尚未确定，不代表请求未执行。`attempts` CLI 可查看这些记录。

记录不包含 Provider 的凭据字段，包含提交给模型的文本、参数及返回数据。一次逻辑调用可能包含 Provider 配置允许的传输重试；需要限制重复请求的实验可设置 `max_attempts: 1`。记录不保证远程请求恰好执行一次，也不保证相同配置可产生相同模型输出。

`ctx.observe(name, value)` 保存 JSON 观察量；成功 turn 保存该 trace。失败尝试保留调用记录和错误，未提交的自定义 trace 随状态撤销。当前模型输入及 embedding 输出完整记录，长实验需考虑记录体积。

## 时间与调度

时钟在 Engine 创建时注入。`ctx.now` 是本轮开始时刻，事件发生时间与处理时刻分别保留；所有时间带时区并归一化为 UTC。

`schedule(event, due_at=..., schedule_id=...)`、`cancel(schedule_id)` 暂存变更。一次性任务完成标记与对应 turn 同事务提交。`run_due(limit=...)` 执行调用时已经到期的一批任务，新创建的任务留给下一批；某任务失败则停止本批，保留该任务待调用方明确重试。唤醒条件由研究流程决定，实验宿主负责定期驱动。

事件 ID 在每个数据库内唯一；重复提交已完成事件会在模型调用前被拒绝。每个独立待执行事件应使用独立 event_id。新事件默认自动生成 ID。

## 状态、分支与并发

一个数据库承载一个实验 Agent。workflow 状态与模块状态分别存储，均支持 agent / conversation 作用域和状态版本检查。业务状态提交使用全局 revision 乐观并发控制；冲突不自动重跑模型。观察写入不推进业务 revision。

同一 Engine 实例串行执行轮次。多个实例或进程竞争时可能在模型调用后冲突；实验宿主应使用单一执行者，避免无谓调用。

`Storage.fork(destination)` 使用 SQLite backup 复制一致快照，拒绝覆盖已有路径。分支拥有独立状态、历史和待执行任务；实验配置与代码需另行记录。分支不会更改原实验。

数据库 schema v2 包含 runtime_meta、conversations、messages、module_states、workflow_states、turns、attempts、model_calls、scheduled_events。版本不匹配时拒绝打开，请指定新路径。
