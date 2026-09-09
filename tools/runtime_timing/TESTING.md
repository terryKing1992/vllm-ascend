# Runtime Timing 测试指南

本文验证：每个推理请求保留一条摘要、所有收到的 decode 步骤参与阶段统计但不逐步上报，以及观测组件故障时模型服务仍能处理请求。不按请求快慢筛选，不设置普通请求抽样或每分钟上报配额。

如果预期日志没有出现或 `timing.jsonl` 为空，请使用 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)。

vLLM 0.23 及以上版本的接口核对范围、新注入目录生成方法与诊断含义见 [COMPATIBILITY.md](COMPATIBILITY.md)。更新代码后必须重新生成目录并重启相关服务进程；旧目录不会自动更新。

包含兼容测试的完整独立回归命令如下，不需要运行 NPU 测试的 conftest：

```bash
python -m unittest discover -s tests/ut -p 'test_runtime_timing*.py' -v
```

兼容测试中的 API/worker 使用模拟 vLLM 接口，真实执行跨进程传输和日志输出；通过这些测试后，仍需执行下文的模型实机步骤。

启动回归还覆盖新旧注入目录并存、原有 sitecustomize 只执行一次、配置/导入失败仍可继续业务、损坏 stderr，以及 `-S/-I/-E` 禁用自动加载时检查命令正确报错。部署前按 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) 开头的命令检查，`PASS` 只证明该解释器进程的注入成功，不证明模型请求已经上报。

## 工作原理

`run.py` 不启动模型，也不收集数据。它生成一个固定的注入目录，其中的 `sitecustomize.py` 会在 Python 进程启动时自动加载打点模块。模型服务通过 `PYTHONPATH` 加载该目录后，打点模块会在目标 vLLM 模块导入时包装调度和执行方法。

请求进入时生成或继承 `trace_id`，随后通过 `trace_headers` 进入调度器。调度器把关联信息附到本轮输出，worker 从中恢复请求上下文，并为实际执行的阶段记录开始和结束时间。记录通过本机非阻塞 UDP 发给独立 collector。collector 按 HTTP 请求根 span 汇总每个阶段的次数、均值和最大值；收到请求结束记录后等待默认 1 秒收齐晚到的数据，再输出一条 `vllm.request` 摘要。

```text
HTTP 请求
  → 请求中间件生成或继承 trace_id
  → AsyncLLM 把 traceparent 传入调度请求
  → scheduler 关联 request_id、prefill/decode 和 step
  → runner wrapper 记录各阶段 Host 耗时
  → [timing-send] 非阻塞发送到 127.0.0.1:18765
  → [timing-recv] collector 接收并解析
  → collector 按请求累积各阶段的次数、均值、最大值
  → 请求结束后输出一行 JSON / 一个 Langfuse span
```

decode 执行 10 步或 1000 步，摘要里都只保留按阶段和 prefill/decode 分组的统计项，不保存逐步列表。减少的是 JSONL/Langfuse 的输出量，本机逐步采集和 UDP 的工作仍然存在。

模型进程不会等待 collector，也不会从 collector 接收确认。collector 不可用、队列已满或 UDP 丢包时，观测数据可能丢失；每请求上报是采集策略，不是持久化送达保证。

## 1. 测试前准备

在 vLLM Ascend 仓库根目录执行命令。真实服务联调需要可正常运行的 vLLM Ascend 环境；基础测试不需要 NPU、Langfuse SDK 或服务器密钥。

确认 Python 版本和工作区：

```bash
python --version
git status --short
```

测试期间使用以下固定 `traceparent`，便于在日志中查找同一次请求：

```text
00-12345678901234567890123456789012-1234567890123456-01
```

## 2. 基础自动化测试

运行全部独立回归测试：

```bash
python -m unittest discover -s tests/ut -p 'test_runtime_timing*.py' -v
```

预期结果：测试以 `OK` 结束；没有 Langfuse SDK 时允许跳过 SDK 专用测试。测试覆盖请求关联、UDP、流式响应、故障隔离，以及大量 decode 记录聚合为一条摘要、不同请求隔离、晚到记录、容量和过期清理。模拟接口测试不能替代真实 NPU 验收。

如果开发环境已安装 Ruff，再执行：

```bash
ruff check \
  tools/runtime_timing/run.py \
  tools/runtime_timing/timing_probe.py \
  tools/runtime_timing/trace_transport.py \
  tools/runtime_timing/collector.py \
  tools/runtime_timing/trace_export.py \
  tools/runtime_timing/trace_summary.py \
  tests/ut/test_runtime_timing*.py

ruff format --check \
  tools/runtime_timing/run.py \
  tools/runtime_timing/timing_probe.py \
  tools/runtime_timing/trace_transport.py \
  tools/runtime_timing/collector.py \
  tools/runtime_timing/trace_export.py \
  tools/runtime_timing/trace_summary.py \
  tests/ut/test_runtime_timing*.py
```

可选的 CPU 合成开销测试：

```bash
python tools/runtime_timing/benchmark.py --iterations 10000 --batch-size 128
```

脚本分别报告 Python wrapper 的 CPU 开销，以及 collector 汇总不同 decode 步数时的每步开销和实际 compact JSON 大小。摘要部分比较 10 步与 `--iterations` 步，输出中的 `jsonl_lines` 都应为 1；可比较 `json_bytes`，确认没有保存逐步列表。该脚本没有真实 UDP、模型执行或 Langfuse 网络发送，不能代替完整链路和 NPU 吞吐、时延测试。

## 3. 生成联调注入目录

每次代码或配置变化后都要生成一个新目录。已有注入目录是固定副本，不会随源码更新。

```bash
python tools/runtime_timing/run.py \
  --output-dir ./observe-inject-test \
  --detail full \
  --sample-rate 1 \
  --every-n-steps 1 \
  --collector-port 18765 \
  --diagnostic-log --diagnostic-every 1
```

确认目录包含以下文件：

```bash
ls -la observe-inject-test
cat observe-inject-test/config.json
```

预期至少包含 `enabled`、`config.json`、`sitecustomize.py`、`timing_probe.py` 和 `trace_transport.py`。默认值已是 `sample_rate=1`、`every_n_steps=1`、`detail=full`。联调命令额外开启逐条诊断，便于排查发送链路；性能测试和正式运行时关闭诊断，仍保留 `1/1/full`。

## 4. 启动 collector

在第一个终端执行：

```bash
python tools/runtime_timing/collector.py \
  --output log \
  --report request \
  --log-format compact \
  --port 18765 \
  --diagnostic-log --diagnostic-every 1 \
  > timing.jsonl \
  2> timing-collector.log
```

在第二个终端确认 collector 已启动：

```bash
head -20 timing-collector.log
```

预期出现：

```text
Collector listening on 127.0.0.1:18765, output=log, report=request
```

`timing.jsonl` 每个请求保存一行摘要，`timing-collector.log` 保存接收诊断，两者不应混写。默认就是 `--report request --log-format compact`；这里显式写出便于核对。`--log-format full` 只增加 JSON 字段，不会恢复逐步上报。

## 5. 启动模型服务

在第三个终端使用注入目录启动服务：

```bash
PYTHONPATH="$PWD/observe-inject-test${PYTHONPATH:+:$PYTHONPATH}" \
  vllm serve /path/to/model \
  --tensor-parallel-size 1
```

将模型路径和并行参数替换为测试环境的实际值。必须重启模型进程，给已经运行的进程修改 `PYTHONPATH` 不会生效。先在单节点、同一网络命名空间验收：当前 UDP 发往本机，分散到多台主机的 collector 不会自动合并数据；只有 worker 阶段、没有 HTTP 根记录的 collector 不会生成请求摘要。

新注入目录启用诊断日志后，模型启动日志应立即出现：

```text
[timing-probe] installed port=18765 sample_rate=1.0 every_n_steps=1
[timing-probe] module=vllm.entrypoints.openai.api_server patched=build_app
```

后续目标模块被导入时还会打印对应的 `module=... patched=...`。`patched=none` 表示本次没有新增 wrapper；结合 `patch_existing`（已包装）和 `patch_skipped`（接口缺失或不匹配）判断原因。完全没有 `[timing-probe]` 时，先核对注入目录、诊断配置和 stderr 去向。

## 6. 发送固定 Trace ID 的请求

非流式请求：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'traceparent: 00-12345678901234567890123456789012-1234567890123456-01' \
  -d '{"model":"/path/to/model","messages":[{"role":"user","content":"请回复测试成功"}],"max_tokens":16,"stream":false}'
```

再发送一次流式请求：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'traceparent: 00-22345678901234567890123456789012-1234567890123456-01' \
  -d '{"model":"/path/to/model","messages":[{"role":"user","content":"请回复流式测试成功"}],"max_tokens":16,"stream":true}'
```

两个请求都应正常返回模型结果。

## 7. 验证发送和接收链路

先查看模型服务日志：

```bash
grep '\[timing-send\]' /path/to/vllm-service.log | tail -20
```

预期先出现 `[timing-probe] request ... sampled=true`，然后出现 `[timing-send] sent`，并包含 `scheduler.schedule`、`runner.execute_model` 或 `vllm.request` 等阶段名。这里的 `sent` 只表示本机内核接受了 UDP 数据。

再查看 collector 日志：

```bash
tail -20 timing-collector.log
```

预期出现 `[timing-recv] received`。该日志表示 collector 已收到并解析数据包。

请求响应完成后再等待约 2 秒，确认 JSONL 已写入（默认汇总等待 1 秒，繁忙时还可能有队列等待）：

```bash
wc -l timing.jsonl
tail -20 timing.jsonl
```

若文件是本次启动时新建且只有上述两次请求，预期两行，每行都是 `name=vllm.request` 的完整 JSON 对象。接收诊断可能有很多行，因为它记录了本机数据包；这不等于向 Langfuse 上报了很多 span。

## 8. 查看收集结果

查看最近一条记录：

```bash
tail -1 timing.jsonl | python -m json.tool
```

按固定 Trace ID 查看请求摘要：

```bash
grep '12345678901234567890123456789012' timing.jsonl
```

如果安装了 `jq`，先看请求及入口等待：

```bash
jq -c 'select(.trace_id == "12345678901234567890123456789012") | {name,span_id,duration_ms,api_to_engine_ms:.metadata.api_to_engine_ms,response_first_body_ms:.metadata.response_first_body_ms,summary:.metadata.timing_summary}' timing.jsonl
```

再把该请求的阶段统计按最大耗时排序：

```bash
jq 'select(.trace_id == "12345678901234567890123456789012") | .metadata.timing_summary.stages | sort_by(.max_host_ms) | reverse' timing.jsonl
```

以下 `summary` 指 `.metadata.timing_summary`：

| 字段 | 验证内容 |
| --- | --- |
| `trace_id`、`span_id`、`parent_span_id` | 保留 HTTP 请求的 trace、根 span 和传入父 span；同一 trace 下的不同 HTTP 请求仍是不同摘要 |
| `name`、`duration_ms` | `vllm.request`；时长等于 `summary.request_ms`，是中间件观察到的请求时长 |
| `metadata.api_to_engine_ms` | 请求进入到首次进入引擎接口的时间；高时先检查 API 处理和输入准备 |
| `summary.max_queue_to_first_schedule_ms` | 调度器成功接收入队到首次调度开始的等待；一个 HTTP 请求有多个引擎请求时取最大值 |
| `metadata.response_first_body_ms` | 到首个非空响应 body 的时间；SSE 首包可能只是角色或错误信息，不能直接当作 TTFT |
| `summary.stages[].name`、`phase` | 实际收到的调度/runner 阶段，按 `prefill`、`decode`、`unknown` 分开统计 |
| `summary.stages[].calls` | 收到的该阶段调用数，可包含多个 rank，不能直接当作输出 token 数 |
| `summary.stages[].mean_host_ms`、`max_host_ms` | 均值持续偏高提示该阶段普遍慢；只有最大值高提示偶发慢调用，须与相同模型和负载的基线比较 |
| `summary.stages[].slowest_step`、`slowest_rank`、`slowest_pid` | 定位最大耗时对应的 step、rank 和进程，便于继续排查 |
| `summary.request_ids` | 对应的引擎请求 ID，最多保留 8 个；不能用其长度推断完整请求数 |
| `summary.stage_data_status`、`step_interval` | `observed` 仅表示收到过阶段记录；当前配置预期步频为 1，不代表传输完整 |
| `summary.history_evicted`、`truncated_packets`、`omitted_context_packets` | 标记已知的缓存淘汰、阶段截断或批次上下文省略；非零时应按部分数据解读 |
| `summary.finish_reason` | 正常为 `request_complete`；`capacity`、`ttl`、`shutdown` 说明汇总提前或在关闭时结束 |

先比较 API 入口、排队等待和请求总耗时，再查看 prefill/decode 中变慢的阶段。阶段是 Host inclusive 时间：父方法包含子方法，批次时间可关联多个请求，多个 rank 可能并行。不要相加各阶段或各 rank 充当请求耗时，也不要用请求总时长减这些统计推断排队时间。没有 profiler 或 NPU 同步，因此这些数据用于定位需要排查的环节，不能证明某个 NPU 算子或通信独占了相应时间。

字段缺失、`null`、空阶段列表或 `stage_data_status=missing` 都表示没有观测到，不能按 0 毫秒理解。即使所有缺失标记均为零，UDP 丢包也可能无法被完整检测。

### 验证 decode 增多不会增加上报条数

用新的 Trace ID，分别发一次 `max_tokens=32` 和一次 `max_tokens=128` 的请求：

```bash
for limit in 32 128; do
  trace_id=$(printf '%032x' "$limit")
  curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H "traceparent: 00-$trace_id-1234567890123456-01" \
    -d "{\"model\":\"/path/to/model\",\"messages\":[{\"role\":\"user\",\"content\":\"Please write a long story.\"}],\"max_tokens\":$limit,\"stream\":false}" \
    > "response-$limit.json"
done
```

响应完成并等待汇总后检查：

```bash
python - <<'PY'
import json
from pathlib import Path

rows = [json.loads(line) for line in Path("timing.jsonl").read_text().splitlines() if line]
for limit in (32, 128):
    matches = [row for row in rows if row["trace_id"] == f"{limit:032x}"]
    assert len(matches) == 1, (limit, "expected one request summary", len(matches))
    row = matches[0]
    assert row["name"] == "vllm.request"
    stages = row["metadata"]["timing_summary"]["stages"]
    decode = [stage for stage in stages if stage["phase"] == "decode"]
    response = json.loads(Path(f"response-{limit}.json").read_text())
    print("max_tokens=", limit, "usage=", response.get("usage"), "decode=", decode)
PY
```

两次都只产生一行。若实际生成更多 token，通常会增加 decode 的 `calls`，但不会增加 JSONL 行数或新增逐步列表。`max_tokens` 是上限，模型可能提前结束；必须结合响应中的实际 token 数判断是否确实增加了 decode。重复本测试应换新的 Trace ID，或重新启动 collector 使用新输出文件。

### 验证 Langfuse 输出相同摘要

按 [README.md](README.md) 安装 v3 SDK 并通过运行环境配置服务器和密钥，停止日志 collector，在相同端口启动：

```bash
python tools/runtime_timing/collector.py --output langfuse --report request --port 18765
```

使用新的 Trace ID 重复请求；等待汇总和 SDK 批量发送后，在 Langfuse 按 Trace ID 查找。本工具应为每个 HTTP 请求新增一个 `vllm.request` observation，其 metadata 包含 `timing_summary`，不应有本工具逐 decode 生成的 observation。上游应用可能已有自己的 span，不能把整条 trace 的所有 span 数当成本工具的上报数。

## 9. 故障隔离测试

以下测试都必须同时确认推理请求仍能成功。

### 9.1 collector 未启动

先停止 collector，再连续发送至少 10 次请求。预期模型服务持续返回成功；发送端允许丢弃数据，不会重试或等待 collector。

### 9.2 collector 中途退出并恢复

推理压测期间停止 collector，继续请求，然后重新执行第 4 节的 collector 命令。预期停机期间模型服务不受影响，停机期间的数据不补发，重启后新请求重新出现在 `timing.jsonl`。

### 9.3 无效 collector 端口

使用与注入配置不同的端口启动 collector。预期模型请求正常，但接收日志和 JSONL 没有新数据。改回一致端口并重启 collector 后，新请求恢复记录。

### 9.4 损坏或缺失注入文件

仅在一次性测试副本中进行。使用缺少 `timing_probe.py` 或包含无效 `config.json` 的测试注入目录启动模型。预期打点被跳过，模型服务仍可用。不要修改正在使用的生产注入目录。

## 10. 配置与容量验证

正式配置保留 `--sample-rate 1 --every-n-steps 1 --detail full`，collector 使用 `--report request`。生成目录时不传 `--diagnostic-log`，collector 启动时也不传此参数，即可关闭逐步诊断日志，摘要照常输出。诊断开关和 `--diagnostic-every` 只控制诊断打印，不决定哪些请求上报。

默认 `--summary-max-requests 2048` 限制同时汇总的请求状态数，`--summary-ttl 300` 是状态寿命，`--summary-grace 1` 是收到根记录后等待晚到阶段的时间；它们都不是每分钟上报限额。根记录已收到时，容量或过期清理可提前输出部分摘要；一直没有根记录时丢弃阶段状态，不伪造一个完整请求。压测时检查 collector 退出统计中的 `evicted`、`orphan_dropped`、`late_packets`、`export_failed` 和导出队列的 dropped/failed 计数。`orphan_dropped` 是状态/数据包计数，不等于精确丢失请求数。

本工具仍尊重传入 `traceparent` 的采样标志：flags 为 `00` 时不记录该请求。要求覆盖全部请求时，上游需传 `01` 或不传 trace 头。不要把 `sample-rate=0.01` 或 `every-n-steps=10` 用作本目标的正式配置：前者会漏掉请求，后者会遗漏 decode 步骤。

仅在需要查看短时原始阶段时，停止现有 collector 后换用以下命令，输出到单独文件：

```bash
python tools/runtime_timing/collector.py --output log --report spans \
  --log-format full --port 18765 > timing-spans.jsonl 2> timing-spans-collector.log
```

`--report spans` 恢复逐阶段输出，数据量会随 decode 增加；它只用于排障，不属于每请求一条摘要的正式验收。

## 11. NPU 性能验收

在相同模型、输入数据、并发、请求数和预热条件下，比较未注入基线与 `--sample-rate 1 --every-n-steps 1 --detail full`、collector `--report request` 的正式配置。两端诊断均关闭，collector 应使用计划部署的实际输出后端。至少记录：

- 请求成功率和错误类型；
- 吞吐量；
- TTFT、TPOT 和端到端时延的 P50/P95/P99；
- Host CPU、内存和 NPU 利用率；
- `timing.jsonl` 记录数和 collector 的 dropped/failed 计数。

增加输出长度再次比较：请求数相同时，JSONL/Langfuse 的摘要数量应不随 decode 步数增长；同时检查本机 CPU、UDP、collector 队列压力和阶段覆盖，不能只看外部上报条数。实际性能是否可接受以目标 NPU 环境的测量为准。

## 12. 验收标准

- 非流式和流式请求均正常完成，响应内容未因打点改变。
- 开启联调诊断时，模型端出现 `[timing-send]`，collector 端出现 `[timing-recv]`；关闭诊断后请求摘要仍正常输出。
- 健康单节点测试中每个 HTTP 请求产生一条 JSONL 摘要或一个 Langfuse observation；短请求和长请求都保留，decode 增多仅改变阶段统计。
- JSONL 每行都能被 JSON 解析，请求关联、耗时、阶段计数及缺失标志可解释。
- 停止、重启或错误配置 collector 时，模型服务仍能持续处理请求。
- 默认 `1/1/full` 配置生效，所有收到的 decode 步骤参与汇总，不逐步导出。
- NPU 实机性能下降处于项目约定的可接受范围。

测试完成后重新生成不带 `--diagnostic-log` 的注入目录并重启模型，collector 也关闭诊断。保留 `1/1/full` 和 `--report request`，避免把关闭诊断误操作成降低请求覆盖率。
