# Runtime Timing 版本兼容说明

本工具不根据版本字符串选择一整套实现，而是在进程启动后监听候选模块，并按实际存在的类和方法安装 wrapper。没有出现的模块不会被主动导入，也不会影响模型启动。

## 当前能力探测

| 能力 | 候选模块或方法 |
| --- | --- |
| HTTP 请求根 span | `vllm.entrypoints.openai.api_server.build_app` |
| 请求头提取（0.23 路径） | `vllm.entrypoints.openai.engine.serving.BaseServing._get_trace_headers` |
| 请求头提取（较新路径） | `vllm.entrypoints.serve.engine.serving`、`vllm.entrypoints.generate.base.serving` 中实际定义 `_get_trace_headers` 的类 |
| Pooling 请求头 | `vllm.entrypoints.pooling.base.serving` 中实际定义 `_get_trace_headers` 的类 |
| 请求进入引擎 | `vllm.v1.engine.async_llm.AsyncLLM.add_request` |
| 调度 | vLLM v1 scheduler 及 vLLM Ascend 已知 scheduler 的 `schedule` |
| 模型执行 | vLLM Ascend v1/v2 `NPUModelRunner` 中实际存在的阶段方法 |

同一个类的方法只包装一次。版本新增、删除或移动某个候选模块时，其余能力仍可独立工作；诊断日志会显示实际命中的模块和方法。

## 版本范围

| 版本系列 | 当前状态 | 验证边界 |
| --- | --- | --- |
| vLLM 0.23.x | 已加入旧版 `openai.engine.serving` 请求头路径 | 使用 0.23.0 官方源码核对，并有模拟模块与请求链单元测试；仍需目标 NPU 环境联调 |
| vLLM 0.26.x | 已加入较新的 `serve.engine.serving` 和 `generate.base.serving` 候选路径 | 使用能力探测兼容，实际 dev commit 可能继续变化；仍需目标版本和 NPU 环境联调 |
| vLLM Ascend runner v1/v2 | 同时监听当前仓库的两个 runner 模块 | 仅包装实际存在的方法；不同分支上的方法差异会反映在 `patched=` 日志中 |

开发版版本号通常包含 `.dev` 和 Git SHA。排障或报告结果时必须保存完整版本，不能只记录 `0.23.1` 或 `0.26.1`。

```bash
python -c "import importlib.metadata as m; print('vllm=', m.version('vllm')); print('vllm-ascend=', m.version('vllm-ascend'))"
```

启用 `--diagnostic-log` 后，`installed` 日志也会打印两个完整版本：

```text
[timing-probe] installed ... vllm=0.23.1.dev... vllm_ascend=0.23.1.dev...
```

随后检查能力命中情况：

```bash
grep '\[timing-probe\].*patched=' /path/to/vllm-service.log
```

至少应看到与当前部署对应的请求头模块、`AsyncLLM.add_request`、scheduler 和 runner。模块显示 `patched=none` 表示模块存在但预期方法不存在；完全没有某个候选模块的日志通常表示该版本没有导入它，这是正常的，只需确认同一能力的另一个候选模块已命中。

## 新版本适配原则

升级 vLLM 或 vLLM Ascend 时，先用采样率 1、步频 1 和诊断日志生成全新注入目录，再运行 `TESTING.md` 的固定 Trace ID 请求。验收以下链路：

```text
trace_headers → engine_request → scheduler.schedule → runner.execute_model → timing-recv
```

如果链路在某一层中断，使用 `TROUBLESHOOTING.md` 保存完整版本和 `patched=` 日志。只有类或方法确实迁移后才增加候选模块；不要仅凭版本号分支复制整套逻辑。

兼容单元测试不等于 NPU 实机验证。每次版本升级仍应比较请求正确性、成功率、TTFT、TPOT、吞吐以及 collector 故障时的服务可用性。
