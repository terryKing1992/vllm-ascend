# 以模型服务可用性为优先的 Langfuse 打点

完整的测试用例、联调命令和验收标准见 [TESTING.md](TESTING.md)。没有日志或 JSONL 为空时，按照
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) 逐层排查。
多版本支持范围和升级验证方法见 [COMPATIBILITY.md](COMPATIBILITY.md)。

连加载/patch 日志都没有时，先执行排查文档开头的 `check.py --inject-dir ...` 检查。新版在开启 `--diagnostic-log` 后提供 `[timing-bootstrap]` 启动诊断，并跳过 PYTHONPATH 中旧打点副本，避免新旧注入目录互相加载；更新后必须重新生成注入目录。

**暂未安装 Langfuse 时，collector 默认输出 JSON 日志，不需要 SDK 或密钥。**

## 日常定界：每个请求一条摘要

每个请求都保留总耗时；prefill/decode 的阶段记录先在本机 collector 聚合，请求结束后只上报一条 `vllm.request` 摘要。decode 即使执行几百步，也不会在 Langfuse 中展开成几百个 span。不按慢请求筛选、不抽取普通请求，也不设置每分钟上传条数预算。

保持每步采集才能统计慢步骤。推荐配置如下，诊断日志默认关闭：

```bash
# 模型环境：生成新目录，再用同一环境和原参数重启模型
INJECT_DIR="$PWD/observe-inject-summary-v1"
python tools/runtime_timing/run.py \
  --output-dir "$INJECT_DIR" \
  --detail full --sample-rate 1 --every-n-steps 1
export PYTHONPATH="$INJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
```

在已经安装 Langfuse v3 SDK、配置好服务器地址和密钥的独立环境中启动 collector：

```bash
python tools/runtime_timing/collector.py --output langfuse --report request
```

暂未安装 SDK 时使用相同摘要格式的 JSON 日志：

```bash
python tools/runtime_timing/collector.py --output log --report request \
  > timing.jsonl 2> timing-collector.log
```

`--report request` 是 collector 默认值。请求完成后默认再等 `--summary-grace 1` 秒，接收稍晚到达的阶段包，然后输出摘要；请求进行中不会持续上传 decode。模型侧仍执行每步计时和本机 UDP 发送，减少的是日志及 Langfuse 输出量，这部分本机开销仍需实测。

在 Langfuse 打开请求的 metadata，或解析 JSONL 的 `metadata.timing_summary`：

| 字段 | 如何使用 |
| --- | --- |
| `request_ms` | HTTP 请求总耗时；流式请求包含完整响应过程 |
| `stages` | 按阶段 `name` 和 `phase` 分组，查看 `calls`、`mean_host_ms`、`max_host_ms` |
| `slowest_step`、`slowest_rank`、`slowest_pid` | 每组最慢观测发生在哪一步、哪个 worker；字段位于对应 stage 中 |
| `max_queue_to_first_schedule_ms` | 接口支持时记录的入队至首次调度间隔；缺失不能当作 0 |
| `stage_data_status`、`finish_reason` | 先检查阶段是否收到、是否被截断，以及摘要是否因正常结束、超时或容量限制而结束 |
| `coverage` | `best_effort_received_records`：统计 collector 实际收到的记录，不宣称覆盖所有 worker 和步骤 |

请求根 metadata 的 `api_to_engine_ms` 可帮助观察进入引擎前的耗时，`response_first_body_ms` 是首个响应 body 的时刻，**不是首 token 时间 TTFT**。先找请求总耗时异常，再比较 prefill/decode 各阶段的均值和最大值，定位应进一步检查的阶段。

阶段是 Host inclusive 时间，父子阶段会重叠，多 rank 可能并行，同一 batch 还被多个请求共享。**不能把这些时间相加或扣差推算排队，更不能仅凭它们断言 NPU 计算或通信硬件 bound。**

摘要不根据快慢删掉请求，但仍尊重上游 `traceparent` 的 `00` 禁采样标记；要采集每个请求，入口需允许采样并使用 `--sample-rate 1`。本机传输、阶段/批次关联上限、collector 有界队列仍可能造成缺失；没有阶段数据时保留已收到的请求总耗时并标注状态。

### 可选：短时查看逐阶段 span

collector 使用 `--report spans` 可恢复旧版逐阶段日志和 Langfuse span。只有需要逐步时间线时才短时使用；可配合 `--detail core --every-n-steps 50` 减少本机采集，但它会遗漏未采样步骤的慢点，不作为日常摘要配置。`--log-format full` 保留 JSONL 原始时间戳；`--diagnostic-log --diagnostic-every 1` 开启逐条收发诊断，均与摘要模式独立。

## 基础日志与部署

完成下面第 1、2 步的注入和服务启动后，在另一个终端执行：

```bash
python tools/runtime_timing/collector.py \
  --output log --report request \
  --port 18765 \
  --diagnostic-log \
  > timing.jsonl 2> timing-collector.log
```

省略 `--output log` 也是日志模式。不加重定向则直接显示在 collector 终端。
默认每行一个请求摘要，包含 trace 关联、请求总耗时和 `metadata.timing_summary`；`--report spans` 才逐阶段输出。
`--log-format full` 还保留 Unix 纳秒开始/结束时间。
进程启动和故障提示写 stderr，stdout 只输出 JSON 数据。
日志写在独立 collector 中，日志管道堵塞时由有界队列丢弃新数据，模型不会等待日志写入。
日志文件容量和轮转由部署方的日志收集器管理。

设计目标是：**collector、网络、认证、SDK 或普通打点异常发生时，丢弃观测数据，继续业务执行。**
这不是“绝对零影响”的承诺：进程内 Python wrapper 仍消耗 CPU、内存，也无法隔离解释器崩溃、系统级 OOM 或任意死锁。
NPU 实机性能和故障场景仍需验收。

## 与旧方案的区别

```text
原有服务管理器 ──启动──> vLLM / workers
                            │
                      选定阶段计时
                            │
                  本机非阻塞 UDP；发送一次
                            ↓
独立服务管理器 ──启动──> collector 聚合请求 ── 一条摘要 ──> Langfuse v3
```

模型不等待 collector 的启动、响应或健康检查。collector 不启动、监督、终止模型进程。
run.py 只在部署前生成固定的注入目录，生成后退出；不再作为 vLLM 的父进程。

| 位置 | 运行内容 | 依赖 |
| --- | --- | --- |
| 模型进程 | 少数函数 wrapper、时间戳、本机 UDP 发送 | Python 标准库 |
| collector 进程 | 有界队列、按请求聚合、JSON 日志或 Langfuse 上报 | 日志模式仅标准库；Langfuse 模式需独立 SDK 环境 |
| 部署工具 run.py | 生成固定注入文件 | Python 标准库；不需要服务器密钥 |

模型进程中不导入 Langfuse / OpenTelemetry，不创建上报线程，不执行 DNS、远程 HTTP、文件落盘或退出 flush；默认不打印诊断日志，显式开启 `--diagnostic-log` 时会写 stderr。
非阻塞发送遇到系统缓冲区不足就丢弃，不等待 ACK，不重传。
JSON 编码仍在模型线程中执行，大小与数量受限；它不是零开销。

## 1. 部署前生成固定注入目录

以下命令在 Linux 昇腾服务环境执行；路径按你的部署调整：

```bash
python tools/runtime_timing/run.py \
  --output-dir ./observe-inject-v1 \
  --detail full \
  --sample-rate 1 \
  --every-n-steps 1 \
  --collector-port 18765 \
  --diagnostic-log
```

`--diagnostic-log` 会打印 `[timing-bootstrap]` 启动诊断、`[timing-probe]` 安装/请求诊断及 `[timing-send]` 采样包发送结果，均写入相应进程的 stderr。
高频事件默认仅打印首次及每 1000 次；要逐条排查可加 `--diagnostic-every 1`。诊断限频只减少日志，不丢弃观测数据。
它只适合联调；确认链路正常后应重新生成生产注入目录时移除此参数。
每请求摘要配置保持 `--sample-rate 1 --every-n-steps 1`，由 collector 聚合控制远端数据量。

run.py 完成后就退出。目标目录应位于本地磁盘且必须是新目录；工具拒绝覆盖既有目录，避免破坏正在使用的版本。
其中仅包含 timing_probe.py、trace_transport.py、config.json、sitecustomize.py 和 enabled 标记。
不要把此生成命令设为模型服务必须成功的 ExecStartPre。

## 2. 模型仍按原方式启动

只在原服务的环境中追加标准 PYTHONPATH：

```bash
PYTHONPATH="$PWD/observe-inject-v1${PYTHONPATH:+:$PYTHONPATH}" \
  vllm serve /path/to/model --tensor-parallel-size 8
```

没有 run.py 常驻父进程，没有 --profiler-config，也不改变 eager / 图模式。
模型环境不需要安装本工具的 requirements.txt，不需要配置 LANGFUSE 密钥。

启动时，sitecustomize 尝试加载可选打点。配置错误、打点文件缺失、依赖加载失败时跳过打点。
既有 sitecustomize 代码先执行，其自身行为保持原样。
Python -S / -I 会跳过这种注入。支持 `vllm serve`；`python -m vllm.entrypoints.openai.api_server` 通过 `launcher.serve_http` 补装请求中间件，实际注册成功时打印 `middleware_installed`（需开启诊断）。

## 3. 独立启动 collector

默认日志模式使用开头的命令即可。以下步骤仅在准备切换到 Langfuse 时执行，模型侧配置不需要改变。

建议在另一个 Python 虚拟环境安装依赖，避免改变模型环境的 SDK 版本：

```bash
python3 -m venv ./observe-venv
./observe-venv/bin/python -m pip install -r tools/runtime_timing/requirements.txt

# 这些既有 SDK 变量只配置在 collector 环境。
export LANGFUSE_BASE_URL='https://your-langfuse-server'
export LANGFUSE_PUBLIC_KEY='pk-lf-...'
export LANGFUSE_SECRET_KEY='sk-lf-...'

./observe-venv/bin/python tools/runtime_timing/collector.py --port 18765 --output langfuse --report request
```

collector 可以晚于模型启动，也可以单独停止、重启。离线期间的数据丢失，不补发历史数据。
只监听 127.0.0.1，必须和相应模型进程处于同一网络命名空间。
容器部署建议独立 collector 容器并共享网络命名空间；禁止把它作为模型容器的必需健康依赖。
每个推理节点部署一个 collector，所有节点使用一致的项目配置。当前摘要只聚合本节点 collector 收到的数据；API 根记录不在本节点时，仅有远端 worker 阶段无法形成请求摘要。本版本不提供跨节点聚合，不能把单节点摘要当作全 rank 结果。

## 如何确认数据已经收集

完成一次 `/v1/chat/completions` 或 `/v1/completions` 请求后，按顺序检查：

```bash
# 1. 模型日志：确认发送端已经把采样包交给本机 UDP
grep '\[timing-send\]' /path/to/vllm-service.log | tail -20

# 2. collector 诊断日志：确认包已收到并成功解析
tail -f timing-collector.log

# 3. JSONL 数据：请求结束并等待约 1 秒后，查看请求摘要
tail -20 timing.jsonl

# 4. 按 trace_id 查找摘要（替换 TRACE_ID）
grep 'TRACE_ID' timing.jsonl
```

发送端日志形如 `sent ... names=scheduler.schedule`，只表示本机内核接收了 UDP 包；接收端日志
形如 `received ... names=scheduler.schedule`，表示 collector 已接收并解析。默认输出的重点是 `duration_ms` 和 `metadata.timing_summary`；`--report spans` 才输出每步的 `metadata.phase/step`。
同一 trace 可以包含多个 HTTP 请求，需结合 `span_id` 区分请求；只用 `trace_id` 查询可能得到多条请求摘要。

如果发送端有 `sent` 而接收端没有 `received`，检查两边端口以及是否共享网络命名空间。如果发送端
完全没有日志，确认模型是用注入目录所在的 `PYTHONPATH` 重启的，并使用上述采样率 1 的联调配置。
如果接收端有 `received` 而 JSONL 为空，先等待请求结束和 grace 窗口，再检查是否收到 `vllm.request` 根记录，以及 `timing-collector.log` 中的导出错误和聚合计数。

提供 [systemd 示例](deploy/vllm-langfuse-collector.service)，其中有独立 CPU、内存及线程预算。
路径和预算是示例，需要按部署调整；文件未自动安装。
模型服务不要配置指向 collector 的 Requires、BindsTo 或 PartOf。两者独立管理，collector 的失败不能触发模型重启。
资源限制需由实际运行平台实施；仅拆成进程，仍不能防止共享主机的资源竞争。

## 请求 trace 如何贯通

支持 /v1/chat/completions、/v1/completions、/v1/responses 和 /v1/embeddings，包括流式响应及 ASGI root_path 前缀。
上游传入标准 W3C version 00 traceparent：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'traceparent: 00-12345678901234567890123456789012-1234567890123456-01' \
  -d '{"model":"/path/to/model","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

trace ID 为 32 位十六进制；parent span ID 为 16 位。上游 span 应在同一个 Langfuse 项目中。
01 允许采样，00 禁用采样。本工具还会按 --sample-rate 进一步确定性筛选。
没有有效 traceparent 时生成新 trace；为保持业务响应原样，本版本不再添加 x-langfuse-trace-id 响应头。
联调时可先生成采样率为 1、every-n-steps 为 1 的注入目录，再用固定 traceparent 查验。

关联路径：

```text
HTTP request span
  → AsyncLLM.add_request 的标准 trace_headers
  → EngineCore Request
  → SchedulerOutput 上可选的普通 dict/list/str/int 元数据
  → worker 恢复本轮关联
  → collector 聚合阶段，在请求结束后生成一条 Langfuse 摘要
```

跨进程消息不再包含本工具的 Packet 等自定义类。即使接收 worker 没有安装打点模块，也能反序列化普通元数据。
元数据缺失时 worker 直接执行原函数。自定义 executor 若过滤额外属性，只会丢失关联数据。
不改 request ID，不依赖 ContextVar 自动跨进程传播，不采集 prompt、输出 token 或异常文本。

`--detail core` 记录请求、调度、execute 和 sample 边界。推荐的 `--detail full` 还收集下面这些实际执行的内部阶段，摘要模式将它们统计到 metadata；只有 `--report spans` 才在 Langfuse 中逐条显示：

```text
vllm.request
├─ scheduler.schedule
├─ runner.execute_model
│  ├─ runner._update_states
│  ├─ runner._prepare_inputs
│  ├─ runner._build_attention_metadata
│  └─ runner._model_forward
└─ runner.sample_tokens
   ├─ runner._sample
   └─ runner.propose_draft_token_ids
```

v2 runner 对应方法可能为 prepare_inputs、postprocess 等，只记录实际执行的方法。
每条记录包含请求 ID、batch ID、step、调度 token 数、来源 PID 和可用的 rank。
phase 以是否已有输出 token 区分 prefill / decode，首轮 prefix-cache 命中也记为 prefill。

**一次 batch 的阶段时间是多个请求共享的。** 同一段时间关联到各个选中的请求，并标注 shared_batch_time=true。
不能相加或当作单请求独占计算时间。
这是 Host inclusive 耗时，包含调用和等待，不能作为 NPU 内部算子的实际执行耗时。
本版本不提供要求关闭图模式的 Attention 深层打点，避免改变模型执行方式。

## 故障处理约定

| 故障 | 模型侧行为 |
| --- | --- |
| run.py 退出、失败 | 它不是服务父进程，不参与服务运行 |
| collector 未启动、崩溃、重启 | 尝试发送一次，不等待；数据可能直接丢失 |
| 本机 UDP 缓冲区满 | 当前观测包丢弃，无重试 |
| Langfuse 超时、认证错误、SDK 异常 | 故障发生在 collector 中；模型不等待 |
| 打点起止、序列化等普通 Python 异常 | 本进程打点熔断，后续 wrapper 直接调用原函数 |
| 打点清理失败，同时业务抛出异常 | 保留业务异常，不用观测异常覆盖它 |
| 注入目录文件缺失或配置损坏 | 新启动进程跳过打点；已加载代码仍在内存中 |
| 部分 worker 未注入 | 业务消息仍可解析，丢失该 worker 的阶段数据 |

每个 wrapper 的原业务函数只执行一次。fallback 不重试业务函数。
默认情况下观测错误不写模型 stderr，避免日志管道阻塞；开启 `--diagnostic-log` 后会写发送结果。
本进程熔断状态只存在内存中，重启后重新尝试。
SystemExit、KeyboardInterrupt、任务取消等正常控制流程不作为可忽略的观测错误吞掉。

此约定覆盖已测试的普通异常和通信故障，不覆盖 native 崩溃、解释器损坏、系统级 OOM、操作系统停顿或任意无限循环。
若要求模型进程完全不执行任何新增代码，就无法同时获得本方案这种细粒度请求级内部 span。

## 采样、容量及关闭

下表是生成工具的默认值；“每个请求一条摘要”应显式使用开头的 `--sample-rate 1 --every-n-steps 1 --detail full`。

| 生成参数 | 默认 | 上限或含义 |
| --- | --- | --- |
| --sample-rate | 0.01 | 0 到 1；为 0 时启动不安装 hooks |
| --every-n-steps | 10 | 每个采样请求记录第 0、10、20……步 |
| --max-records | 16 | 每个 runner 调用最多 32 个阶段，含根 span |
| --max-requests | 4 | 每个 batch 最多 8 个已采样请求 |
| --collector-port | 18765 | 本机 collector 端口 |
| --detail | core | core 仅记录请求/调度/执行/采样；full 增加内部阶段 |
| --diagnostic-every | 1000 | 开启诊断时，各类高频事件打印首次及每 N 次；1 为逐条打印 |

每个 UDP 包最多 8192 字节，超长整包丢弃，不分片、不重试。
超过阶段 / 请求上限会在已发送元数据中标记 truncated_stage_calls / omitted_sampled_requests。
collector 的接收队列默认 256 包，满时丢新包；SDK 也使用有界队列和有限重试。
UDP 不保证送达：模型端 sent 仅代表本机内核接受，collector 无法统计所有在途丢失。
collector 退出时报告 received、invalid、dropped；网络导出的错误由 collector 的 SDK 日志报告。

摘要聚合另有以下边界；它们用于限制本机内存，不是上传速率预算：

| collector 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--report` | request | request 聚合为每请求摘要；spans 逐阶段导出 |
| `--summary-grace` | 1 秒 | 根记录到达后等待迟到阶段包的时间 |
| `--summary-ttl` | 300 秒 | pending 请求的最长聚合生命周期；超过后结束已有状态 |
| `--summary-max-requests` | 2048 | 同时 pending 的请求上限；超出时驱逐旧状态 |

超时、容量驱逐或退出时，有根记录的状态输出已收到的摘要；缺少根记录的状态丢弃并计数。长请求应按实际时长调大 TTL。晚于摘要结束的阶段不会补写到 Langfuse；观察 `finish_reason` 和阶段状态，避免把不完整摘要误认作完整推理过程。

仅停止 collector 即可停止日志输出或向 Langfuse 上传，但模型仍会做采样计时。
彻底关闭：从模型启动环境移除注入 PYTHONPATH，按原部署流程重启。
也可以部署 sample-rate=0 的新目录，或移除 enabled 标记后重启。
enabled 只在进程启动时读取，不是运行中的热开关。不要在热路径反复读配置文件。

## 验证与适配边界

```bash
python tests/ut/test_runtime_timing_standalone.py
python tools/runtime_timing/benchmark.py
```

测试使用 CPU 模拟业务，包括 collector 缺失/被杀、前后置打点异常、时钟失败、序列化失败、队列拥塞、
流式响应不变、无 SDK 启动、缺失注入文件、普通类型 IPC，以及真实 SDK 的时间戳和父子关联。
SDK 远程网络出口被测试替换；另有本机真实 UDP 发送/接收测试。
benchmark 只统计空函数 wrapper 路径，未计入真实模型执行、UDP 编码发送和 collector 导出，不能据此承诺 NPU 吞吐不变。

当前适配本仓库 v1/v2 runner、常见 scheduler 和 vllm serve。
每个本机 Python 子进程需继承固定注入目录；远程节点要部署自己的持久目录，已有 Ray worker 不会自动补装。
离线 LLM.generate、Responses API、定制 IPC schema 和跨 PD 代理的额外传播需要单独适配。

上线验收应使用相同模型和负载比较原服务与启用打点后的输出、成功率、TTFT、TPOT、吞吐；
再测试停止/杀死 collector、错误认证、网络超时和资源限额。尚未完成昇腾实机及你的服务器联调。

参考：[Python 非阻塞 socket](https://docs.python.org/3/library/socket.html#socket.socket.setblocking)、
[Langfuse v3 TracerProvider 接口](https://github.com/langfuse/langfuse-python/blob/v3.10.1/langfuse/_client/client.py)、
[systemd 资源限制](https://manpages.debian.org/bookworm/systemd/systemd.resource-control.5.en.html)。
