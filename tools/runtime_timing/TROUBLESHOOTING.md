# Runtime Timing 排查指南

本文用于排查没有 `[timing-probe]`、`[timing-send]`、`[timing-recv]` 或 `timing.jsonl` 为空的问题。按照日志出现的位置逐层检查，不要跳过注入验证。

vLLM 0.23 及以上的接口与升级步骤见 [COMPATIBILITY.md](COMPATIBILITY.md)。特别注意：拉取代码后必须重新生成注入目录并重启服务；`PYTHONPATH` 指向旧目录时仍运行旧打点。

## 日志分别代表什么

| 日志 | 所在进程 | 含义 |
| --- | --- | --- |
| `[timing-probe] installed` | 模型进程 | Python 已加载注入目录并安装 import hook |
| `[timing-probe] module=... patched=...` | 模型进程 | 目标 vLLM 模块已导入，并完成方法包装 |
| `[timing-probe] middleware_installed` | API 进程 | 中间件已经注册到实际 app；`build_app` 被包装本身不代表注册成功 |
| `[timing-probe] patch_skipped ...` | 模型进程 | 单个接口不兼容或缺失，其他打点继续工作 |
| `[timing-probe] disabled operation=...` | 模型进程 | 运行期打点发生异常，当前进程的后续观测关闭；按 operation 定位 |
| `[timing-probe] engine_request trace_context=missing` | API 进程 | 引擎已收到请求，但关联信息缺失；检查请求中间件和实际路由 |
| `[timing-probe] request ... sampled=true` | API 进程 | 请求已进入中间件并被采样 |
| `[timing-probe] engine_request ...` | AsyncLLM 进程 | `add_request` 已收到有效 `traceparent`；可作为 HTTP 中间件未命中时的兼容诊断 |
| `[timing-probe] trace_headers ...` | API 进程 | 旧版或新版请求入口已提取关联信息并将其继续传给 AsyncLLM |
| `[timing-send] sent ...` | 模型或 worker 进程 | UDP 包已交给本机内核 |
| `[timing-recv] received ...` | collector 进程 | UDP 包已收到并成功解析 |
| JSON 行 | `timing.jsonl` | collector 已把一个阶段写入结果文件 |

`sent` 不是 collector 的接收确认。UDP 不提供 ACK，collector 未运行时发送端也可能显示 `sent`。

## 快速判断

```text
没有 timing-probe
  → 注入目录未加载、目录是旧版本，或 Python 禁用了 site

有 timing-probe，没有 request
  → API hook 未匹配，或请求没有经过支持的 OpenAI API 路径

有 request，没有 timing-send
  → 请求未采样、阶段 hook 未匹配，或打点已经熔断

有 timing-send，没有 timing-recv
  → 端口不一致，或两个进程不在同一网络命名空间

有 timing-recv，timing.jsonl 为空
  → stdout 重定向错误，或 collector 导出线程失败
```

## 1. 确认使用的是新注入目录

注入目录是 `run.py` 生成的固定副本。更新 Git 仓库不会更新已经生成的 `observe-inject-v1`。

查看配置：

```bash
cat observe-inject-v2/config.json
```

联调配置应包含：

```json
{
  "sample_rate": 1.0,
  "every_n_steps": 1,
  "collector_port": 18765,
  "diagnostic_log": true
}
```

如果没有 `diagnostic_log`，生成一个新目录：

```bash
python tools/runtime_timing/run.py \
  --output-dir ./observe-inject-v2 \
  --sample-rate 1 \
  --every-n-steps 1 \
  --collector-port 18765 \
  --diagnostic-log
```

`run.py` 会拒绝覆盖已有目录。每次工具代码变化后都应使用新的目录名，并用新目录重启模型。

确认必要文件存在：

```bash
ls -la observe-inject-v2/enabled
ls -la observe-inject-v2/config.json
ls -la observe-inject-v2/sitecustomize.py
ls -la observe-inject-v2/timing_probe.py
ls -la observe-inject-v2/trace_transport.py
```

## 2. 单独验证 Python 能否加载注入

在模型运行环境中执行：

```bash
INJECT_DIR="$(readlink -f observe-inject-v2)"

PYTHONPATH="$INJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
python -c "import sitecustomize; print('loaded:', sitecustomize.__file__)" 2>&1
```

预期同时看到：

```text
[timing-probe] installed port=18765 sample_rate=1.0 every_n_steps=1
loaded: /绝对路径/observe-inject-v2/sitecustomize.py
```

如果 `loaded:` 指向其他 `sitecustomize.py`，确认注入目录位于 `PYTHONPATH` 最前面。如果报 `ModuleNotFoundError`，检查路径和文件权限。

确认 Python 没有禁用自动加载：

```bash
PYTHONPATH="$INJECT_DIR" \
python -c "import sys; print('no_site=', sys.flags.no_site, 'isolated=', sys.flags.isolated)"
```

预期为：

```text
no_site= 0 isolated= 0
```

`no_site=1` 表示使用了 `python -S`，`isolated=1` 表示使用了 `python -I`。需要从模型启动命令中移除对应参数，否则 `sitecustomize.py` 不会自动加载。

## 3. 普通命令启动时没有 timing-probe

先在同一个终端验证：

```bash
INJECT_DIR="$(readlink -f observe-inject-v2)"

PYTHONPATH="$INJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
vllm serve /path/to/model
```

必须把注入目录放在启动命令本身的环境中。只在另一个终端执行 `export PYTHONPATH=...` 不会改变已经启动的模型进程，也不会改变其他终端的环境。

确认 `vllm` 使用的解释器：

```bash
command -v vllm
head -1 "$(command -v vllm)"
python -c "import sys; print(sys.executable)"
```

如果 `vllm` 来自另一个虚拟环境，应在那个虚拟环境中执行第 2 节的验证命令。

## 4. sudo、systemd 和 Supervisor 环境

`sudo` 可能清理 `PYTHONPATH`。临时验证可以使用：

```bash
export PYTHONPATH="$(readlink -f observe-inject-v2)${PYTHONPATH:+:$PYTHONPATH}"
sudo --preserve-env=PYTHONPATH vllm serve /path/to/model
```

systemd 服务应使用绝对路径：

```ini
[Service]
Environment="PYTHONPATH=/opt/vllm-ascend/observe-inject-v2"
ExecStart=/path/to/vllm serve /path/to/model
```

应用配置并查看日志：

```bash
sudo systemctl daemon-reload
sudo systemctl restart your-vllm-service
sudo systemctl show your-vllm-service -p Environment
sudo journalctl -u your-vllm-service -f | grep timing
```

Supervisor 示例：

```ini
[program:vllm]
environment=PYTHONPATH="/opt/vllm-ascend/observe-inject-v2"
command=/path/to/vllm serve /path/to/model
```

修改服务环境后必须重启模型进程。

## 5. Docker 或 Kubernetes 环境

宿主机路径不会自动出现在容器中。Docker 需要挂载目录并设置容器内路径：

```bash
docker run \
  -v "$PWD/observe-inject-v2:/opt/observe-inject:ro" \
  -e PYTHONPATH=/opt/observe-inject \
  ...
```

进入正在运行的容器验证：

```bash
docker exec MODEL_CONTAINER \
  python -c "import sitecustomize; print(sitecustomize.__file__)"
```

Kubernetes 需要把注入文件挂载到 Pod，并为模型容器设置：

```yaml
env:
  - name: PYTHONPATH
    value: /opt/observe-inject
```

如果 collector 是 sidecar，它可以监听 `127.0.0.1`。如果 collector 位于另一个 Pod，当前只监听回环地址的实现无法跨 Pod 接收，需要将 collector 与模型放进同一个 Pod，或者调整传输设计。

## 6. 有 installed，但没有 module patched

模型日志中查找：

```bash
grep '\[timing-probe\]' /path/to/vllm-service.log | tail -50
```

正常情况下会逐步出现：

```text
[timing-probe] module=vllm.entrypoints.openai.api_server patched=build_app
[timing-probe] module=vllm.v1.engine.async_llm patched=AsyncLLM.add_request
[timing-probe] module=...scheduler... patched=...schedule
[timing-probe] module=vllm_ascend.worker.model_runner_v1 patched=...
```

出现 `patched=none` 表示本次没有新增 wrapper，可能是已包装（`patch_existing`），也可能是类或方法结构不同（`patch_skipped`）。保存这些日志，并记录版本：

```bash
python -c "import importlib.metadata as m; print('vllm=', m.version('vllm')); print('vllm-ascend=', m.version('vllm-ascend'))"
```

如果只运行离线 `LLM.generate`、Responses API、自定义入口或非 v1 engine，现有 OpenAI API 请求关联 hook 可能不会执行，需要单独适配相应入口。

## 7. 有 patched，但没有 request 日志

当前请求中间件只处理：

```text
/v1/chat/completions
/v1/completions
```

使用固定 Trace ID 发送测试请求：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'traceparent: 00-12345678901234567890123456789012-1234567890123456-01' \
  -d '{"model":"/path/to/model","messages":[{"role":"user","content":"测试"}],"max_tokens":16}'
```

预期出现：

```text
[timing-probe] request trace_id=12345678901234567890123456789012 sampled=true
```

在 vLLM 0.23.0 中，如果 HTTP 中间件没有命中，但 vLLM 已把 `trace_headers` 传给
`AsyncLLM.add_request`，还会出现：

```text
[timing-probe] engine_request request_id=... trace_id=12345678901234567890123456789012 sampled=true
```

vLLM 0.23 默认会在自身 OpenTelemetry tracing 未启用时打印
`Received a request with trace context but tracing is disabled`，并丢弃传给 AsyncLLM 的 trace headers。
本工具会在生成和 pooling 请求入口提前提取有效的 `traceparent/tracestate`，此时应看到
`[timing-probe] trace_headers ...`，随后看到 `engine_request`。无需启用 vLLM profiler 或 OTLP tracing。

vLLM 0.23.0 的生成请求实现位于 `vllm.entrypoints.openai.engine.serving.OpenAIServing`，正常启动日志应包含：

```text
[timing-probe] module=vllm.entrypoints.openai.engine.serving patched=OpenAIServing._get_trace_headers
```

`traceparent` 最后两位为 `00` 时，上游明确禁止采样；联调时使用 `01`。同时确认新注入目录使用 `sample_rate=1`。

## 8. 有 request，但没有 timing-send

先确认日志中是否有 scheduler 和 runner 的 `patched=` 记录。请求日志只证明 API 入口生效，不代表 worker 路径已匹配。

对于多进程、多节点或 Ray 部署，每个实际执行 Python worker 都必须继承注入目录。检查 worker 日志中是否分别出现 `[timing-probe] installed`。已有 worker 不会自动加载后来添加的环境变量，必须重启。

本工具发生普通观测异常时会熔断当前进程的打点，以保证模型继续运行。熔断状态只存在内存中；修正配置或版本后重启模型进程即可重新尝试。

## 9. 有 timing-send，但没有 timing-recv

确认 collector 已启动：

```bash
python tools/runtime_timing/collector.py \
  --output log \
  --port 18765 \
  --diagnostic-log \
  > timing.jsonl \
  2> timing-collector.log

head -20 timing-collector.log
```

预期出现：

```text
Collector listening on 127.0.0.1:18765, output=log
```

比较注入配置和 collector 参数，两边端口必须相同：

```bash
cat observe-inject-v2/config.json
```

检查监听端口：

```bash
ss -lunp | grep 18765
```

由于发送目标固定为 `127.0.0.1`，模型进程和 collector 必须位于同一主机及网络命名空间。两个不同 Docker 容器默认不共享回环地址；使用同一个 Pod、host network 或共享网络命名空间。

## 10. 有 timing-recv，但 JSONL 为空

确认启动命令正确区分 stdout 和 stderr：

```bash
python tools/runtime_timing/collector.py \
  --output log \
  --port 18765 \
  --diagnostic-log \
  > timing.jsonl \
  2> timing-collector.log
```

检查文件位置和权限：

```bash
pwd
ls -l timing.jsonl timing-collector.log
tail -50 timing-collector.log
```

如果日志包含 `Export failed`、`dropped_packets` 或 `failed_packets`，保留错误类型及上下文。停止 collector 时还会打印最终的 `received`、`invalid` 和 `dropped` 计数。

验证 JSONL：

```bash
wc -l timing.jsonl
tail -1 timing.jsonl | python -m json.tool
```

日志模式每写完一个数据包都会 flush，因此正常接收后不需要等待进程退出。

## 11. 最小排查信息

如果仍无法定位，请收集以下输出。密钥、模型输入及业务数据不要包含在排查信息中。

```bash
git rev-parse HEAD
cat observe-inject-v2/config.json

INJECT_DIR="$(readlink -f observe-inject-v2)"
PYTHONPATH="$INJECT_DIR" \
python -c "import sys,sitecustomize; print(sys.version); print(sys.executable); print(sitecustomize.__file__); print(sys.flags)" 2>&1

python -c "import importlib.metadata as m; print(m.version('vllm')); print(m.version('vllm-ascend'))"
ss -lunp | grep 18765
tail -100 timing-collector.log
```

同时提供经过脱敏的模型启动命令，以及模型日志中全部 `[timing-probe]`、`[timing-send]` 行。不要提供 Langfuse secret key。
