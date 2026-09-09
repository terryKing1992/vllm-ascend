# Runtime Timing 测试指南

本文用于验证三个目标：耗时记录能够从模型进程送达 collector、记录能够按请求关联，以及观测组件故障时模型服务仍然可用。

如果预期日志没有出现或 `timing.jsonl` 为空，请使用 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)。

vLLM 0.23 及以上版本的接口核对范围、新注入目录生成方法与诊断含义见 [COMPATIBILITY.md](COMPATIBILITY.md)。更新代码后必须重新生成目录并重启相关服务进程；旧目录不会自动更新。

包含兼容测试的完整独立回归命令如下，不需要运行 NPU 测试的 conftest：

```bash
python -m unittest discover -s tests/ut -p 'test_runtime_timing*.py' -v
```

兼容测试中的 API/worker 使用模拟 vLLM 接口，真实执行跨进程传输和日志输出；通过这些测试后，仍需执行下文的模型实机步骤。

## 工作原理

`run.py` 不启动模型，也不收集数据。它生成一个固定的注入目录，其中的 `sitecustomize.py` 会在 Python 进程启动时自动加载打点模块。模型服务通过 `PYTHONPATH` 加载该目录后，打点模块会在目标 vLLM 模块导入时包装调度和执行方法。

请求进入时生成或继承 `trace_id`，随后通过 `trace_headers` 进入调度器。调度器把关联信息附到本轮输出，worker 从中恢复请求上下文，并为实际执行的阶段记录开始和结束时间。采样记录编码成小型 JSON 数据包，通过本机非阻塞 UDP 发送给独立 collector。collector 解析数据包后，把每个阶段写成 `timing.jsonl` 中的一行。

```text
HTTP 请求
  → 请求中间件生成或继承 trace_id
  → AsyncLLM 把 traceparent 传入调度请求
  → scheduler 关联 request_id、prefill/decode 和 step
  → runner wrapper 记录各阶段 Host 耗时
  → [timing-send] 非阻塞发送到 127.0.0.1:18765
  → [timing-recv] collector 接收并解析
  → 每个阶段写成一行 JSON
```

模型进程不会等待 collector，也不会从 collector 接收确认。collector 不可用时，当前观测数据可能丢失，推理流程继续执行。

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

运行独立单元测试：

```bash
python tests/ut/test_runtime_timing_standalone.py
```

预期结果：全部测试通过并以 `OK` 结束。该测试覆盖采样、请求关联、UDP 收发、JSON 输出、导出队列、流式响应以及 collector 缺失或退出时的服务隔离。

如果开发环境已安装 Ruff，再执行：

```bash
ruff check \
  tools/runtime_timing/run.py \
  tools/runtime_timing/timing_probe.py \
  tools/runtime_timing/trace_transport.py \
  tools/runtime_timing/collector.py \
  tools/runtime_timing/trace_export.py \
  tests/ut/test_runtime_timing_standalone.py

ruff format --check \
  tools/runtime_timing/run.py \
  tools/runtime_timing/timing_probe.py \
  tools/runtime_timing/trace_transport.py \
  tools/runtime_timing/collector.py \
  tools/runtime_timing/trace_export.py \
  tests/ut/test_runtime_timing_standalone.py
```

可选的 CPU 合成开销测试：

```bash
python tools/runtime_timing/benchmark.py --iterations 10000 --batch-size 128
```

记录 baseline、`sample_rate=0`、`sample_rate=0.01` 和 `sample_rate=1` 的 `host_us/batch`。该结果只比较 Python wrapper 的 CPU 开销，不能代替 NPU 吞吐和时延测试。

## 3. 生成联调注入目录

每次代码或配置变化后都要生成一个新目录。已有注入目录是固定副本，不会随源码更新。

```bash
python tools/runtime_timing/run.py \
  --output-dir ./observe-inject-test \
  --sample-rate 1 \
  --every-n-steps 1 \
  --collector-port 18765 \
  --diagnostic-log
```

确认目录包含以下文件：

```bash
ls -la observe-inject-test
cat observe-inject-test/config.json
```

预期至少包含 `enabled`、`config.json`、`sitecustomize.py`、`timing_probe.py` 和 `trace_transport.py`。联调阶段使用 100% 采样和每步记录；性能测试时关闭诊断日志并改回生产采样配置。

## 4. 启动 collector

在第一个终端执行：

```bash
python tools/runtime_timing/collector.py \
  --output log \
  --port 18765 \
  --diagnostic-log \
  > timing.jsonl \
  2> timing-collector.log
```

在第二个终端确认 collector 已启动：

```bash
head -20 timing-collector.log
```

预期出现：

```text
Collector listening on 127.0.0.1:18765, output=log
```

`timing.jsonl` 保存采集结果，`timing-collector.log` 保存接收诊断，两者不应混写。

## 5. 启动模型服务

在第三个终端使用注入目录启动服务：

```bash
PYTHONPATH="$PWD/observe-inject-test${PYTHONPATH:+:$PYTHONPATH}" \
  vllm serve /path/to/model \
  --tensor-parallel-size 1
```

将模型路径和并行参数替换为测试环境的实际值。必须重启模型进程，给已经运行的进程修改 `PYTHONPATH` 不会生效。容器或多节点环境中，模型进程与 collector 必须共享网络命名空间，远程 worker 也必须拥有并加载同一版本的注入目录。

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

最后确认 JSONL 已写入：

```bash
wc -l timing.jsonl
tail -20 timing.jsonl
```

行数应大于 0，并且每一行都应是一个完整 JSON 对象。

## 8. 查看收集结果

查看最近一条记录：

```bash
tail -1 timing.jsonl | python -m json.tool
```

按固定 Trace ID 查看一次请求的全部阶段：

```bash
grep '"trace_id": "12345678901234567890123456789012"' timing.jsonl
```

如果安装了 `jq`，可以只显示阶段和耗时：

```bash
jq -c 'select(.trace_id == "12345678901234567890123456789012") | {name,request_id,duration_ms,phase:.metadata.phase,step:.metadata.step}' timing.jsonl
```

重点检查：

| 字段 | 验证内容 |
| --- | --- |
| `trace_id` | 同一请求的所有记录一致 |
| `name` | 包含请求、调度和实际执行到的 runner 阶段 |
| `duration_ms` | 为非负数，且符合该阶段的量级 |
| `request_id` | 调度和 runner 记录能够关联到 vLLM 请求 |
| `parent_span_id` | 子阶段指向请求或 runner 父阶段 |
| `metadata.phase` | 首轮通常为 `prefill`，后续通常为 `decode` |
| `metadata.step` | 从 0 开始，并符合 `every-n-steps` 配置 |

同一 batch 的耗时可能关联到多个请求，并带有 `shared_batch_time=true`。该耗时是 Host inclusive 时间，不能解释为 NPU 算子独占时间。

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

## 10. 采样验证

分别生成注入目录并重启服务验证：

- `sample-rate=0`：不安装 hook，JSONL 不应产生新记录。
- `sample-rate=1`、`every-n-steps=1`：每个合规请求都应被选中，适合联调。
- `sample-rate=0.01`、`every-n-steps=10`：用于生产候选配置，记录量应明显下降。

采样是按 Trace ID 确定性选择。同一 Trace ID 重复请求的选中结果一致；`traceparent` 最后的 flags 为 `00` 时，上游禁止采样，本工具不应记录该请求。

## 11. NPU 性能验收

在相同模型、输入数据、并发、请求数和预热条件下，分别测试未注入、生产采样配置以及联调配置。至少记录：

- 请求成功率和错误类型；
- 吞吐量；
- TTFT、TPOT 和端到端时延的 P50/P95/P99；
- Host CPU、内存和 NPU 利用率；
- `timing.jsonl` 记录数和 collector 的 dropped/failed 计数。

生产验收以未注入和 `sample-rate=0.01 --every-n-steps=10` 的差异为准。CPU 合成 benchmark 只能作为代码级回归参考。

## 12. 验收标准

- 非流式和流式请求均正常完成，响应内容未因打点改变。
- 模型端出现 `[timing-send]`，collector 端出现 `[timing-recv]`，JSONL 中存在对应 Trace ID。
- JSONL 每行都能被 JSON 解析，时间戳、耗时和父子关系有效。
- 停止、重启或错误配置 collector 时，模型服务仍能持续处理请求。
- 采样率和步频配置符合预期。
- NPU 实机性能下降处于项目约定的可接受范围。

测试完成后，生产注入目录应关闭 `--diagnostic-log` 并恢复正式采样参数，避免模型热路径持续打印诊断日志。
