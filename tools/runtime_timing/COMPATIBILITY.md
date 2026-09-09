# Runtime Timing 版本兼容说明

本工具面向 vLLM 0.23 及以上版本，按实际导入的模块、方法及参数安装打点，不按版本号强制切换实现。模型侧只使用标准库和本机 UDP；collector 可选择 JSONL 日志或 Langfuse v3，二者使用相同的请求关联信息。

## 已核对的接口

| 官方版本 | 请求头提取入口 | 验证程度 |
| --- | --- | --- |
| [0.23.0](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/entrypoints/openai/engine/serving.py) | `openai.engine.serving.OpenAIServing` | 官方源码核对、模拟接口跨进程测试 |
| [0.24.0](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/entrypoints/openai/engine/serving.py) | `openai.engine.serving.OpenAIServing` | 官方源码核对、模拟接口跨进程测试 |
| [0.25.0](https://github.com/vllm-project/vllm/blob/v0.25.0/vllm/entrypoints/generate/base/serving.py) | `generate.base.serving.GenerateBaseServing` | 官方源码核对、模拟接口跨进程测试 |
| [0.26.0](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/entrypoints/generate/base/serving.py) | `generate.base.serving.GenerateBaseServing` | 官方源码核对、模拟接口跨进程测试 |

这些版本的 `AsyncLLM.add_request` 都接受 `request_id`、`prompt`、`trace_headers`，`SchedulerOutput` 是允许附加属性的 dataclass。测试覆盖 Ascend v1/v2 runner 的两种模块布局。**上述测试使用模拟 vLLM 接口，不表示已在四套真实 vLLM/NPU 环境运行模型。**

0.23.1、其他补丁版、开发版及更高版本如果保留这些接口，会使用同一套打点；接口迁移到未知模块或改变调用约定时，需要继续适配，不能保证所有未来版本自动兼容。升级验收以目标环境的实际请求数据为准。

## 适配方式

默认 `--sample-rate 1 --every-n-steps 1 --detail full`，覆盖全部被允许采样的请求及实际可用的内部阶段。collector 默认 `--report request`：所有收到的 decode 步骤参与本机汇总，每个 HTTP 请求只输出一条摘要，不逐步生成 Langfuse span。`--detail core` 是可选的缩小打点范围模式，会省略内部阶段。

诊断日志默认关闭。排障时两端开启 `--diagnostic-log`，必要时设置 `--diagnostic-every 1` 查看每次发送/接收；诊断频率不改变请求摘要的上报策略。

| 能力 | 实现与降级 |
| --- | --- |
| 请求根 span | 在 `openai.api_server.build_app` 注册 ASGI 中间件；`entrypoints.launcher.serve_http` 再检查并补装，覆盖通过 `python -m ...api_server` 启动的情况；同一个 app 只安装一次 |
| 请求路径 | `/v1/chat/completions`、`/v1/completions`、`/v1/responses`、`/v1/embeddings`；支持流式响应和 ASGI `root_path` 前缀 |
| Trace 提取 | 同时监听旧版 OpenAI、新版生成及 pooling 模块中实际定义的 `_get_trace_headers`；保留标准 trace 头，无须启用 vLLM 自带 OTLP tracing |
| 引擎关联 | 按函数签名绑定参数，传递请求根 span 的 `traceparent`；输入已经是携带 `trace_headers` 的请求对象时，浅复制后更新，保留调用者原对象 |
| 调度与 worker | 调度器附加普通字典到 `SchedulerOutput`；worker 恢复关联并记录实际存在的方法。自定义 executor 若丢弃额外属性，worker 将没有关联数据 |
| 入口与排队时间 | 中间件记录首次进入引擎和首个非空响应 body 的时间；支持的调度器 `add_request` 成功入队后到首次调度开始的时间单独记录，接口缺失时保留缺失状态 |
| 请求摘要 | 按 `trace_id` 和 HTTP 根 span ID 汇总实际收到的阶段；同一 trace 的不同 HTTP 请求不会合并。必须收到根记录才能输出摘要，只有 worker 记录时不会伪造请求 |
| 未知接口 | 缺少类、方法、签名不匹配或 wrapper 安装失败时跳过该目标，其他打点继续工作 |
| 运行期故障 | 普通打点执行异常仍会关闭当前进程的观测以保留业务执行；诊断模式输出 `disabled operation=... error=...`，不打印异常文本 |

采集的是 Python 方法的 Host inclusive 耗时，批次执行时间可被多个请求共享。摘要保留各阶段的调用数、均值、最大值和最慢调用的 step/rank/pid，不能把嵌套阶段或并行 rank 相加，也不能解释成单请求独占 NPU 时间。没有使用 profiler 或强制 NPU 同步。首个响应 body 可能只是 SSE 角色信息，不能直接当作 TTFT。

## 升级后必须重新生成注入目录

`run.py` 把代码复制到固定目录。更新 Git 工作区不会更新旧目录，也不会更新已经运行的模型进程。

在 Linux 模型环境中执行，目录名必须尚不存在：

```bash
INJECT_DIR="$PWD/observe-inject-compat-v2"
python tools/runtime_timing/run.py \
  --output-dir "$INJECT_DIR" \
  --sample-rate 1 --every-n-steps 1 --detail full \
  --collector-port 18765 --diagnostic-log

export PYTHONPATH="$INJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
python -c "import timing_probe; print(timing_probe.__file__)"
```

最后一条应打印新目录中的 `timing_probe.py`。随后用当前环境和原有模型启动参数重启服务，收集 stderr。所有 API、scheduler、worker 进程必须加载新目录。当前数据通过本机 UDP 传输，各节点 collector 不会自动合并摘要；先在单节点、同一网络命名空间完成验收。

collector 暂时使用日志模式即可，在独立终端、同一网络命名空间启动：

```bash
python tools/runtime_timing/collector.py --port 18765 --output log --report request \
  --diagnostic-log > timing.jsonl 2> timing-collector.log
```

按 [TESTING.md](TESTING.md) 发送固定 Trace ID 的请求，等待响应完成，然后检查模型日志和结果。不要只看到 `patched=` 就认为上报成功。

| 日志 | 能确认什么 |
| --- | --- |
| `middleware_installed` | 请求中间件已注册到实际 app |
| `request ... sampled=true` | HTTP 请求命中中间件并被采样 |
| `engine_request ... trace_id=...` | 请求关联信息已经到达 AsyncLLM |
| `engine_request trace_context=missing` | 引擎入口命中，但无有效 trace；先检查中间件和请求入口 |
| `scheduler_active ... contexts=...` | 第一次调度观察结果；可能来自空批次，只打印一次 |
| `runner_active carrier=...` | 第一次执行观察结果；可能来自预热，只打印一次 |
| `patch_skipped ...` | 指定接口缺失、不支持或安装失败；其他能力仍可工作 |
| `patch_existing ...` | 已经包装，避免重复打点；此时模块的 `patched=none` 不代表失败 |
| `carrier_unsupported` | 无法给调度输出附加关联信息；保留调度记录，worker 关联不可用 |
| `disabled operation=...` | 当前进程发生了运行期打点故障，后续观测已停止 |
| `timing-send` / `timing-recv` | 分别表示本机发送尝试成功和 collector 收到数据；发送成功不是送达确认 |

这些诊断需要生成目录时开启 `--diagnostic-log`，写到相应进程的 stderr；collector 的诊断也需单独开启。诊断日志本身可能阻塞，正式性能验收应重新生成关闭诊断的目录，仍保留 `sample_rate=1`、`every_n_steps=1`、`detail=full`。

最终应在 `timing.jsonl` 中找到该 HTTP 请求的一条 `name=vllm.request` 摘要，在 `.metadata.timing_summary.stages` 中查找实际收到的 `scheduler.schedule`、`runner.execute_model` 和准备输入等阶段；阶段名称不再各占一行。请求根记录在响应结束后发送，collector 默认再等待 1 秒以收集晚到的阶段，再输出摘要。

没有阶段记录时仍可输出根摘要，但 `stage_data_status=missing`，不能理解为阶段耗时为零。检查 `history_evicted`、`truncated_packets`、`omitted_context_packets` 和 `finish_reason`，确认是否发生已知的容量或生命周期截断；UDP 传输仍可能丢包。Langfuse v3 使用相同的 trace ID 和 parent span ID，每个请求一个 observation，统计放在 metadata 中。安装和字段读法见 [README.md](README.md)，完整验收见 [TESTING.md](TESTING.md)。

只有短时排查原始阶段时才使用 `--report spans`，该模式仍逐步输出，不符合日常每请求一条摘要的目标。

## 无硬件回归测试

```bash
python -m unittest discover -s tests/ut -p 'test_runtime_timing*.py' -v
```

新增兼容测试会启动独立 API 与 worker 子进程，通过真实 import hook、pickle、UDP 和 JSONL 验证请求关联，覆盖旧/新请求模块、普通导入/`-m` 启动、v1/v2 runner，以及接口缺失时的降级。它不使用已安装的 vLLM，不测试设备执行。Langfuse SDK 不存在时仅跳过 SDK 专用测试。

真实服务验收仍须执行 [TESTING.md](TESTING.md) 的请求、collector 故障与性能对比步骤。记录完整版本及开发提交：

```bash
python -c "import importlib.metadata as m; print('vllm=', m.version('vllm')); print('vllm-ascend=', m.version('vllm-ascend'))"
```
