# 模块开发指南

模块是实现五个方法的普通 Python 类，通过配置文件显式注册。没有目录扫描、没有自动发现、没有插件平台。

## 完整参考实现

`examples/counter_module.py`（agent 作用域轮次计数器）：

~~~python
from emc_runtime import ContextBlock, Event, Finalized, ModuleContext, Prepared


class CounterModule:
    state_version = 1

    def __init__(self, config: dict, module_id: str, scope: str):
        self.module_id = module_id
        self.scope = scope

    def initial_state(self):
        return {"count": 0}

    def expose(self, state):
        # 只读公共信号：其他模块在 ctx.exports["counter"] 读到它
        return {"count": state["count"]}

    def prepare(self, event: Event, state, ctx: ModuleContext) -> Prepared:
        working = {"count": state["count"] + 1}
        return Prepared(
            working_state=working,
            context_blocks=[
                ContextBlock(source=self.module_id,
                             content=f"Counter value: {working['count']}")
            ],
            trace={"event_kind": event.kind},
        )

    def finalize(self, event, prepared: Prepared, reply, ctx) -> Finalized:
        return Finalized(state=prepared.working_state,
                         trace={"committed_count": prepared.working_state["count"]})

    def inspect(self, state):
        return {"count": state["count"]}
~~~

配置注册（作用域 `agent` 跨会话共享；`conversation` 按会话隔离）：

~~~json
{
  "modules": [
    {"id": "counter", "entrypoint": "examples.counter_module:CounterModule",
     "scope": "agent", "config": {}}
  ]
}
~~~

## 契约要点

- **构造签名** `(config, module_id, scope)`；实例声明的 `module_id` / `scope` 必须与配置一致，否则启动失败。
- **状态**是任意可 JSON 序列化数据（NaN/Infinity 被拒绝）。`initial_state` 只在无存量状态时使用；首次成功提交后才落库。
- **prepare** 在深拷贝上工作；可读 `ctx.history`（只读原始历史）、`ctx.event`（当前事件）、`ctx.exports`（同快照其他模块的 expose 结果）。返回 `Prepared`；`context_blocks` 的 `source` 必须是自己的 module_id。
- **finalize** 在模型回复后运行，返回 `Finalized`；其 `state` 即将原子提交的最终状态。
- **ContextBlock** 只是带来源标注的数据，被包裹在"参考数据"框架中送入 LLM，排在基础系统指令之后、原始历史之前；模块不能改写系统提示词或历史。
- **state_version**：状态结构变化时递增；与存量不一致时运行时报错且不覆盖旧状态。
- **错误即整轮失败**：任何阶段抛异常，本轮不写入任何内容（消息、状态、turn 记录全部不落库），已扣费的模型调用不重放。
- **模块不得**：接触引擎对象、存储连接、其他模块的可变状态、Provider 或网络。

## 失败与并发

- 提交时检测到 revision 变化（另一写入者先提交）→ `ConcurrencyConflict`，由调用方决定重试与否；引擎不会自动重发模型请求。
- 观察者（events）故障不影响已提交的轮次。

## 测试建议

参考 `tests/test_modules.py`：作用域隔离、启停与状态保留、状态版本升级，均可用离线 Provider（`tests/helpers.py` 里的桩）在无网络环境验证。
