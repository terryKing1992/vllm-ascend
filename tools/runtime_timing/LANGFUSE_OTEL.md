# Langfuse v3 OTel 上报与验收

本工具在独立 collector 内，把每个 HTTP 推理请求汇总为一个 `vllm.request` span，再交给 Langfuse SDK 批量发送。decode 的次数、均值、最大值及最慢 step/rank 留在这个 span 的 metadata 中，不新增逐步 observation。模型进程仍只负责原有计时和本机 UDP 发送。

## 配置与升级

在独立 collector 环境安装 `requirements.txt`，其中 SDK 版本范围为 `langfuse>=3.10,<4`。配置示例：

```bash
export LANGFUSE_BASE_URL='https://your-langfuse-server'
export LANGFUSE_PUBLIC_KEY='pk-lf-...'
export LANGFUSE_SECRET_KEY='sk-lf-...'

python tools/runtime_timing/collector.py \
  --output langfuse --report request --port 18765 \
  --service-name vllm-ascend-runtime \
  --environment production --release runtime-timing-otel-v1 \
  --flush-at 256 --flush-interval 2
```

`LANGFUSE_BASE_URL` 填服务器根地址，例如 `https://langfuse.example.com`。SDK 将其拼成 `https://langfuse.example.com/api/public/otel/v1/traces`，通过 OTLP HTTP/protobuf 上报，用项目 public/secret key 生成 Basic 认证。不要在根地址后重复填写完整 OTLP 路径；这不是 gRPC 配置。[Langfuse v3.15 SDK 导出实现](https://github.com/langfuse/langfuse-python/blob/v3.15.0/langfuse/_client/span_processor.py)

官方当前 OTel 文档还包含 v4 ingestion 配置。本工具保持 Python SDK v3，未添加 `x-langfuse-ingestion-version: 4`，也没有升级到 SDK v4；不要直接照搬 v4 的接入参数。[Langfuse OTel 接入说明](https://langfuse.com/integrations/native/opentelemetry)

`--service-name` 默认是 `vllm-ascend-runtime`。`--environment` 和 `--release` 可省略，分别沿用 SDK 既有的 `LANGFUSE_TRACING_ENVIRONMENT`、`LANGFUSE_RELEASE` 设置；显式 CLI 参数优先。

`--flush-at` 是 SDK 批量阈值，`--flush-interval` 是 SDK 批量发送间隔，不是请求采样率或每分钟上报限额。collector 在初始化 SDK 时兼容其 v3 批量参数读取行为，随后恢复 SDK 相关环境设置。请求结束后还需经过默认 1 秒的汇总等待和可能的队列、网络等待；这些等待不加入请求记录的时长。正常关闭 collector 时关闭 SDK 和独立 TracerProvider；强杀或故障仍可能丢失尚未导出的数据。

SDK v3 的其他内部消费者也使用该发送间隔，增大 `--flush-interval` 可能延长 collector 关闭时的等待；默认保持 2 秒。模型服务与该关闭过程独立。

升级本工具后，重启 collector 即可启用新的映射，仍兼容旧注入目录发送的数据。要新增 HTTP 方法、固定路由、响应状态码，还需重新生成注入目录并重启模型；部署方式见 [README.md](README.md)。

## 字段怎样映射

collector 使用独立的 OTel TracerProvider，不替换进程全局 provider。采样采用 `ParentBased(ALWAYS_ON)`，不在 collector 额外抽掉已经采集的请求；模型端仍尊重上游 `traceparent` 的禁采样标志。

| 本工具数据 | OTel / Langfuse 映射 | 查看方式 |
| --- | --- | --- |
| 请求 trace、span 和父 span ID | OTel 原生 `trace_id`、`span_id`、`parent_span_id` | 在上游 trace 下查看 `vllm.request`；保留请求的父子关联 |
| 请求开始、结束时间 | span 原始 `start_time`、`end_time` | 请求时长来自模型侧记录，不是 collector 的上报耗时 |
| 请求 / 原始阶段 | 请求 `SpanKind.SERVER`，阶段 `INTERNAL`；`langfuse.observation.type=span` | 默认摘要模式只导出请求；`--report spans` 才导出原始阶段 |
| `timing_summary` | `langfuse.observation.metadata.timing_summary`，保留为一个 JSON 对象 | 展开 observation 的 metadata 查看阶段统计 |
| `request_ms`、首次调度等待、状态和覆盖标志 | `langfuse.observation.metadata.<key>` | 提升为 observation metadata 顶层字段，便于筛选 |
| HTTP 方法、固定路由和状态码 | `http.request.method`、`http.route`、`http.response.status_code` | 同时保留 `http_method`、`http_route`、`http_status_code` metadata |
| 服务、主机、部署版本 | resource 的 `service.name`、`host.name`、`service.version` | 在 resource metadata 查看；部署版本同时映射 `langfuse.release` |
| 部署环境 | `langfuse.environment` 与 resource 的 `deployment.environment.name` | 按部署环境区分数据 |

摘要提升到 metadata 顶层的字段包括 `request_ms`、`max_queue_to_first_schedule_ms`、`stage_data_status`、`coverage`、`step_interval`、`history_evicted`、`truncated_packets`、`omitted_context_packets`、`finish_reason` 和 `received_stage_records`。原有 `api_to_engine_ms`、`response_first_body_ms` 也保留在顶层。缺失值不补为 0，阶段统计仍仅保存在 `timing_summary.stages` 中。

OTel 载荷中的 metadata 值遵循 SDK v3 的序列化：字符串、整数和布尔值直接写入，浮点数及复合对象编码为 JSON 字符串后由 Langfuse 解析。JSONL 日志仍使用原来的 JSON 数据类型。

Langfuse 把 `langfuse.observation.metadata.<key>` 映射到 observation metadata 的顶层，普通 OTel 属性则可能落入 `metadata.attributes`；因此本工具显式设置前缀，避免摘要只能作为不可筛选的整体属性出现。HTTP 请求存在父 span 时不设置 `langfuse.trace.name`，避免覆盖上游应用的 trace 名称。[官方 metadata 映射说明](https://langfuse.com/integrations/native/opentelemetry#attribute-mapping)

## 错误与时延的含义

| 结果 | OTel status | Langfuse level |
| --- | --- | --- |
| 正常 HTTP 响应 | `UNSET` | 默认级别 |
| HTTP 4xx，且没有抛出业务异常 | `UNSET` | `WARNING` |
| HTTP 5xx | `ERROR` | `ERROR` |
| 已捕获并继续向外传播的业务异常 | `ERROR` | `ERROR`，记录异常类型 |

HTTP server span 对 4xx 保持 `UNSET`，5xx 标记 `ERROR`，与 [OTel HTTP span 语义约定](https://opentelemetry.io/docs/specs/semconv/http/http-spans/)一致。异常只上报类型，不上报堆栈或异常正文；原业务异常仍按原方式传播。若流式响应已经发出 200 后才抛异常，以异常标记体现失败，不伪造 HTTP 状态码。

当前 observation 类型是 `span`：本工具没有读取请求/响应正文、模型名、真实 token usage 或费用，也不填造 generation 字段。`response_first_body_ms` 可能对应 SSE 角色包或错误包，不是 TTFT，不映射为 `langfuse.observation.completion_start_time`。因此不能依据这里的首个 body 时间计算短输出 TPOT，也不能期待 Langfuse 自动生成准确的 token 用量或 TTFT 图表。

`timing_summary` 里的阶段是 Host inclusive 时间，父子阶段重叠、多 rank 可能并行；阶段均值和最大值用于定位需检查的环节，不能相加得到请求耗时，也不能直接证明 NPU 计算或通信 bound。

## 验收重点

按 [TESTING.md](TESTING.md) 发送带新 Trace ID 的正常、流式及长输出请求，等待汇总和 SDK 批量发送，再检查：

1. 每个 HTTP 请求只有一个本工具生成的 `vllm.request` observation。增加 decode 步数只增加摘要里的调用计数，不增加 observation 数。
2. trace ID 和父 span 与入口 `traceparent` 一致；已有上游 trace 的名称保留。没有上游时，本工具请求 span 成为新 trace 的根。
3. observation 时长与 `timing_summary.request_ms` 对应；metadata 中可以查看完整阶段统计，并按顶层 `request_ms`、`stage_data_status` 等字段筛选。
4. 使用本次新注入目录时，正常请求有固定 HTTP 路由和实际状态码；在测试环境用无效模型请求检查 4xx 呈现为 `WARNING`。异常/5xx 分支由自动化测试覆盖，无需在生产服务制造故障。
5. 服务名、环境、部署版本符合配置；没有输入/输出正文、伪造的 token usage、费用或 TTFT。

`test_runtime_timing_otel.py` 使用真实 SDK 和模拟 HTTP 出口，解析 OTLP protobuf 核对上述协议字段。它不连接你的 Langfuse 服务器；部署后的入库、筛选和 UI 展示仍需单独验收。收发诊断只证明本机链路，不能证明远端已入库；远程认证和网络错误查看 collector 的 SDK 日志。
