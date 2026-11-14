# hermes-context-engine

Hermes Agent 的**缓存稳定型上下文引擎**插件（magic-context 血统、无记忆层）。

## 它解决什么

Hermes 的会话越长，请求越大；传统的"到阈值就摘要"每一步都在改写历史字节——而 provider 的
prompt 缓存是**前缀匹配**的，改一个字节 = 整段重算。本引擎把"决定丢什么"和"什么时候真的丢"
分开：

- **打标（随时、零成本）**：每个可回收单元（工具结果、文件 part）带持久编号 `§N§`；
- **盖章（模型主动）**：`ctx_reduce` 只是把 tag **排队**——不是删除，原文永远可用 `ctx_expand` 取回；
- **落地（择时）**：排队的内容只在"本来就要重建"的 pass（用量过阈值 / 排队量够大 /
  对账发现变更 / 连续多趟挂住）才离开请求，其余 pass **逐字节复现**上一趟输出 → 缓存命中率不受影响。
  空闲时间**不是**重建理由：重发同样的字节不会让 provider 忘掉缓存，而重建会提前丢掉模型
  还需要的内容。

外加：`<!-- +Xm -->` 时间标记（冻结回放）、`compress()` 紧急车道（provider 溢出 / gateway
hygiene / 手动 /compress 时兜底：强制落地 + 最老优先骨架化）。所有跨 pass 状态按 session
落库于 `$HERMES_HOME/context-engine/context.db`（gateway 每轮重建引擎实例，存储即真相）。

## 安装与启用

```bash
hermes plugins install rays1d/hermes-context-engine --enable
```

在 config.yaml（对应 profile）设置：

```yaml
context:
  engine: context-engine
```

无需 `plugins.enabled` 条目（context engine 有专用发现通道）。安装/启用后需**重启 gateway**
（平台插件同理：不重启不物化）。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `HCE_ENABLED` | `true` | 总开关。关闭需重启生效（会话中途永不改字节） |
| `HCE_EXECUTE_THRESHOLD` | `65` | 允许重建的用量百分比 |
| `HCE_ESTIMATE_RATIO` | `1.0` | token 估算的静态比例（provider 侧每 1 个“估算单位”对应多少真实 token），**每会话冻结一次**。模型的分词器系统性多于启发式估算时调大（保护尾巴更长），反之调小。不从真实用量学习——`prompt_tokens` 含系统提示与工具 schemas，拿它学会把固定开销当成漂移 |

行为性常量（保护窗口、骨架阈值 1024 字节、最新 20 次调用等）**刻意不暴露**——见设计文档第七部分。

## 模型可见面

- `ctx_reduce(tags: "3-5" | "1,2,9")` —— 盖章排队；保护窗口内的接受但暂挂；
  `ctx_reduce` 自身的旧调用自动清理，无需盖章。
- `ctx_expand(tags)` —— 取回被丢弃内容的原文（参数 + 结果 / 文件 part）。
- 工具结果内容前带 `§N§` 前缀；文件 part 后跟 `§N§` 标记；丢弃后为 `[dropped §N§]`。

## 减少提醒（reduction nudge）

模型不会主动盖章——7 天生产数据（281 个 tag、0 次 `ctx_reduce`）证实了这点。所以引擎会在
**值得时**附一段尾部提醒（`<system-reminder>` 块），措辞随压力升高：gentle / firm / urgent。

规则（都来自实测教训）：

- **按尾巴算压力**：severity = U/T，U 是可回收量、T 是尾巴自身的内容质量（文本、工具入参与结果、
  文件 part，不含 reasoning 与已丢弃内容），档位 20% / 40% / 60%。窗口不参与——1M 窗口上的脏尾巴
  同样值得提（用窗口做分母会让提醒在 1M 窗口上永远够不着）；
- **地板**：尾巴低于 60k 或可回收量低于 25k，任何比例都不提；
- **不提不可能的事**：保护窗口内、或已经是骨架/占位符的内容不计入 U 与 T；
- **遵守了就不再唠叨**：刚落地内容的 pass 不提；同档再提需要真实增长（≥ max(25k, 8% 尾巴)）；
  升档立即提；
- **只动尾部**：提醒只附加在最新 user 消息上，并**冻结**——tail 移走后必须逐字节回放；
  只有在"本来就要重建"的 pass 才清理旧提醒并重新决定，所以它永远不会让缓存前缀漂移。

## 边界

- M0 不做：无摘要层 / 无 historian / 无 `m[0]`+`m[1]`（M2）；无紧急分层回收（M3）。
- 不碰宿主记忆系统：召回注入是追加式的，本引擎把它当普通尾部内容打标回收，零桥接。
- 引擎永不抛出：内部故障降级为 LKG 重放（校验锚点），锚点不匹配则原样放行请求并大声记日志。

## 开发

```bash
# 在仓库根（hermes-projects）下，用 hermes-agent 的 venv 跑测试：
cd hermes-context-engine
../hermes-agent/.venv/Scripts/python.exe -m pytest tests/ -q
```

测试通过 `tests/conftest.py` 以合成包名加载插件（与 hermes 用户引擎加载器同款），
并用 `tests/host_sim.py` 忠实模拟宿主请求装配契约。
