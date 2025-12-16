# aury-ai-model

> 统一、高可靠的 LLM 调用层：统一消息/事件模型，多 Provider 适配（OpenAI / OpenRouter / Doubao），结构化输出管线（Strict + Repair），工具声明与解析（含 MCP），上下文与可观测性（usage/metrics），以及基于 tenacity 的 with_retry 重试封装。面向生产调用稳定性与类型安全，不内置 Agent/工具执行。

- Python: 3.12+
- 依赖: Pydantic v2、contextvars、openai>=1.0、tenacity、json-repair、python-dotenv（开发）
- Provider 适配：
  - OpenAI（Chat/Responses）
  - OpenRouter（OpenAI 兼容，扩展 reasoning / images / provider 路由）
  - Doubao/火山方舟 Ark（ChatCompletions 风格，Responses 类特性）
- 事件：`content` / `thinking` / `tool_call` / `usage` / `completed` / `error`
- 消息：parts-only（`Text` / `Image` / `Thinking` / `FileRef`），多模态一致
- 结构化输出：Strict 优先，Repair/Extract 兜底（详见“结构化输出”）
- 工具：MCP/function/builtin 声明与解析（不执行），多轮工具链路（含 reasoning_details 透传）
- 重试：`client.with_retry(...)`（tenacity.AsyncRetrying），非流式/流式统一
- 可观测：上下文（contextvars）、instrumentation sink、usage 聚合

本 README 以“尽可能详细”为原则，覆盖设计、API、Provider 差异、测试对应关系、迁移清单。若只需入门，直接跳到“快速上手”。

---

## 安装与准备

```bash
pip install pydantic==2.* openai>=1.0 json-repair tenacity python-dotenv
# Doubao / Ark（可选）
pip install 'volcengine-python-sdk[ark]'
```

- 建议使用 .env 管理密钥：
  - `OPENROUTER_API_KEY`
  - `OPENAI_API_KEY`（如直接走 OpenAI）
  - `ARK_API_KEY`（Doubao/方舟）
  - 可选：`GEMINI_IMAGE_MODEL=google/gemini-3-pro-image-preview`

---

## 快速上手（覆盖常见能力）

### 初始化方式

直接初始化（推荐）：
```python
from aury.ai.model import ModelClient, msg

client = ModelClient(
    provider="openrouter",
    model="openai/gpt-4o-mini",
    api_key="${OPENROUTER_API_KEY}"
)
```

或使用 `bind()` 创建配置变体（适合复用基础配置）：
```python
base = ModelClient(provider="openrouter", api_key="${OPENROUTER_API_KEY}")
client_a = base.bind(model="openai/gpt-4o-mini")
client_b = base.bind(model="anthropic/claude-3-opus")
```

### 非流式调用 + usage
```python
resp = await client.ainvoke([msg.user("Hello")])
print(resp.parts)
print(client.last_usage())  # input/output/reasoning/total
```

调用时可覆盖 provider/model：
```python
resp = await client.ainvoke([msg.user("Hello")], provider="doubao", model="doubao-seed-1-6-251015")
```

### 流式调用
最终从 `last_usage()` 取用量；如需事件中带 usage：`yield_usage_event=True`
```python
async for ev in client.astream([msg.user("讲个笑话")], yield_usage_event=True):
    if ev.type == "content":
        print(ev.delta, end="")
    elif ev.type == "usage":
        print(f"用量: {ev.usage}")
print(client.last_usage())
```

### 流式 Thinking（推理过程）
```python
async for ev in client.astream([msg.user("深入分析这个问题")], return_thinking=True):
    if ev.type == "thinking":
        print(f"💭 {ev.delta}", end="")
    elif ev.type == "content":
        print(ev.delta, end="")
```

### 多模态输入（图片）
```python
# URL 或 base64 data:URL
image_url = "https://example.com/image.jpg"
# 或 image_url = "data:image/png;base64,..."

resp = await client.ainvoke([
    msg.user("描述这张图片", images=[image_url])
])
```

### 严格结构化输出
Responses/Doubao 建议配合 `text_format`：
```python
from pydantic import BaseModel

class Weather(BaseModel):
    city: str
    temp_c: float

client = ModelClient(provider="openai", model="gpt-4.1-mini", api_key="${OPENAI_API_KEY}", transport="responses")
result = await client.with_structured_output(Weather).ainvoke(
    [msg.user("请以JSON返回北京天气，包含 city 和 temp_c")],
    text_format={"type":"json_object"},
    expect_strict=True,
)
```

### 重试（with_retry）
```python
retrying = client.with_retry(max_attempts=4, base_delay=0.2)
res = await retrying.ainvoke([msg.user("稳定返回一句话")])
```

### 多图输出（OpenRouter x Gemini）
```python
client = ModelClient(provider="openrouter", model="${GEMINI_IMAGE_MODEL}", api_key="${OPENROUTER_API_KEY}")
res = await client.ainvoke([msg.user("Create two icon variations")])
# OpenRouterAdapter 会把 message.images → 多个 Image(url=...) part
```

---

## 历史记录管理

`ModelClient` 不内置对话状态，历史由调用方管理。每次调用需传入完整消息列表。

### 基本多轮对话
```python
from aury.ai.model import ModelClient, msg, Message, Role

client = ModelClient(provider="openrouter", model="openai/gpt-4o-mini", api_key="...")
history: list[Message] = []

# 第一轮
history.append(msg.user("我叫小明"))
resp = await client.ainvoke(history)
history.append(resp)

# 第二轮（模型记住上下文）
history.append(msg.user("我叫什么名字？"))
resp = await client.ainvoke(history)
history.append(resp)
```

### 工具调用多轮链路
```python
history = [msg.system("你是助手"), msg.user("查询北京天气")]

for _ in range(5):  # 最多5轮工具调用
    resp = await client.ainvoke(history, tools=TOOLS)
    history.append(resp)  # assistant 消息（含 tool_calls）
    
    if not resp.tool_calls:
        break  # 无工具调用，结束
    
    # 执行工具，把结果加入历史
    for tc in resp.tool_calls:
        result = run_tool(tc.name, tc.arguments_json)
        history.append(msg.tool(result, tool_call_id=tc.id))
```

### 流式历史管理
流式返回 `StreamEvent`，用 `StreamCollector` 聚合：
```python
from aury.ai.model import StreamCollector

history = [msg.user("你好")]
collector = StreamCollector()

async for ev in client.astream(history):
    collector.feed(ev)
    if ev.type == "content":
        print(ev.delta, end="")

# 聚合结果
collector.content           # str - 聚合文本
collector.thinking          # str - 聚合思考
collector.tool_calls        # list[ToolCall]
collector.usage             # Usage | None
collector.thinking_message  # Message | None - 仅含 thinking
collector.content_message   # Message | None - 仅含 content
collector.message           # Message - 完整消息

# 加入历史
history.append(collector.message)
```

流式工具调用：
```python
for _ in range(5):
    collector = StreamCollector()
    async for ev in client.astream(history, tools=TOOLS):
        collector.feed(ev)
        if ev.type == "content":
            print(ev.delta, end="")
    
    resp = collector.to_message()
    history.append(resp)
    
    if not resp.tool_calls:
        break
    for tc in resp.tool_calls:
        history.append(msg.tool(run_tool(tc.name, tc.arguments_json), tool_call_id=tc.id))
```

### msg 便捷构造器
```python
msg.system("你是助手")                           # system 消息
msg.user("你好", images=["url"])                   # user 消息（可带图片）
msg.assistant("OK", thinking="...")                # assistant 消息
msg.tool(result_json, tool_call_id=tc.id)          # tool 结果消息
```

### Role 枚举
```python
Role.system    # "system"
Role.user      # "user"
Role.assistant # "assistant"
Role.tool      # "tool"

# 字符串兼容（StrEnum 自动转换）
Message(role="tool", ...)  # 等同于 Message(role=Role.tool, ...)
```

---

## 核心概念

- Message / Part（强类型）：
  - `Text(text: str)`、`Image(url: str)`、`Thinking(text: str)`、`FileRef(id: str)`
  - `Message.parts` 仅包含这些 Part，避免字符串/列表二义性；多模态一致
  - `Message.tool_calls`：解析后的工具调用（`ToolCall{id,name,arguments_json,mcp_server_id}`）
  - `Message.reasoning_details`：供 OpenRouter Gemini/DeepSeek 工具链路的多轮透传
  - `Message.role`：使用 `Role` 枚举（`system`/`user`/`assistant`/`tool`）
- 事件（`StreamEvent`）：`content` 文本增量、`thinking` 思考增量、`tool_call`、`usage`、`completed`、`error`
- Usage：`input_tokens` / `output_tokens` / `reasoning_tokens` / `total_tokens`
- 上下文：`model_ctx`（with/async with/装饰器），trace_id/user_id/provider/model/extra_headers
- 可观测：instrument sink（on_request_start/end、on_stream_event、on_error），`client.last_usage()`

---

## API 参考（ModelClient）

- `bind(**updates) -> ModelClient`：不可变配置，安全复用
- `ainvoke(messages, **kw) -> Message`
- `astream(messages, **kw) -> AsyncIterator[StreamEvent]`
- `with_structured_output(schema) -> StructuredView`
- `with_retry(...) -> RetryView`
- `last_usage() -> Usage | None`

通用调用参数：
- 生成：`max_tokens`、`max_completion_tokens`、`temperature`、`top_p`、`stop`、`seed`
- 推理：`return_thinking`、`reasoning_effort`（low/medium/high）
- 结构化：`response_format`（Chat）、`text_format`（Responses/Doubao）
- 工具：`tools=[ToolSpec(...)]`（含 MCP 降级编码；详见“工具与 MCP”）
- 透传：`extra_body={...}`（Provider 特定参数原样传递）
- 绑定默认值：`default_max_tokens`、`default_temperature`、`default_top_p`、`default_reasoning_effort`

Provider 选择与 transport：
- OpenAI 默认 Chat；`transport="responses"` 使用 Responses
- OpenRouter 默认 Chat（兼容 OpenAI Chat）
- Doubao 固定 ChatCompletions 风格（内部处理兼容 Responses 特性）

---

## 结构化输出（Strict + Repair）

`client.with_structured_output(Schema).ainvoke(messages, ...)` → `Schema` 实例。策略管线：
1) StrictSchemaStep（`expect_strict=True`）：直接按 JSON/format 严格校验；
2) RepairExtractStep：先尝试 JSON 解析，其次 `json_repair` 修复，再从 markdown 代码块或最大花括号提取并校验。

要点：
- Chat API 用 `response_format={"type":"json_object"}`；Responses/Doubao 用 `text_format={"type":"json_object"}`
- 提示词尽量明确字段与扁平结构，避免多包一层（如 `{"company":{...}}`）
- Pydantic v2 支持 `validation_alias` 与 `field_validator`（支持字段别名与容错校验）
- 失败抛出 `ValueError: structured parse failed: ...`

---

## 工具与 MCP

类型：`ToolSpec(kind=function|mcp|builtin)`；MCP 用 `MCPToolSpec(server_id,name,input_schema,...)`。

编码/解码：
- 不支持 MCP 原生的后端：编码为 `function.name = "mcp::{server}::{name}"`
- 解析：`decode_maybe_mcp` / `normalize_tool_call` 还原 `mcp_server_id` 与 `name`

多轮工具链路：
- 第 1 轮 assistant 产出 `tool_calls`（以及 OpenRouter 的 `reasoning_details`）
- 第 2 轮将上一轮 assistant 消息原样回传，并附带每个 tool 的 `tool` 消息（带 `tool_call_id`）

---

## Reasoning / reasoning_details（多轮传递）

- OpenRouterAdapter：
  - 非流式：从原始 JSON `choices[0].message.reasoning_details` 抽取并挂到返回 `Message`
  - 流式：累积 chunk.delta 里的 `reasoning_details`，在 `completed` 事件回传聚合列表
- Doubao/OpenAI：按各自字段（如 `reasoning_content`）实时/最终映射到 `thinking` 事件或 `Thinking` part

用途：当模型使用函数调用（工具调用）时，将 `reasoning_details` 与 `tool_calls` 一起随 assistant 回传，以供下一轮推理。

---

## 多图输出（OpenRouter / Gemini）

- 非流式：OpenRouter 返回 `choices[0].message.images`；适配器会映射为多个 `Image(url=...)` part（支持 data:URL / https）
- 流式：目前以文本/工具/usage 为主，最终多图建议走非流式获取
- 可通过 prompt 或 `extra_body`（若后端支持，如 `n: 2`）提示多图

---

## extra_body 透传（Provider 特定参数）

- OpenRouter：`provider.order`、`transforms`、`models` 备选等
- Doubao/Ark：`previous_response_id`、`caching`、其它开关
- OpenAI：Responses/Chat 兼容的扩展参数

---

## 重试（with_retry, tenacity）

- 使用：`client.with_retry(max_attempts=3, base_delay=0.5, max_delay=5.0, backoff_factor=2.0, retry_on=..., predicate=...)`
- 默认重试：`ModelTimeoutError`、`RateLimitError`、`ModelOverloadedError`、`TransportError`
- 不重试：`InvalidRequestError`
- 适用于：`ainvoke` 与 `astream`

---

## 上下文与可观测性

- `model_ctx`：同步/异步/装饰器三用；嵌套继承策略（内层覆盖未显式字段继承外层）
- instrumentation：
  - `emit_start(provider, model)` / `emit_event(type,payload)` / `emit_end(metrics, usage|error)`
  - sink 接口：`on_request_start`、`on_stream_event`、`on_request_end`、`on_error`
- usage 聚合：
  - 流式默认不发 usage 事件；最终使用 `client.last_usage()` 读取
  - 需要事件：`yield_usage_event=True`

自定义 InstrumentSink 示例：
```python
from aury.ai.model.instrumentation import InstrumentSink, RequestMetrics, register_sink

class UsageTracker(InstrumentSink):
    def __init__(self):
        self.total_tokens = 0

    def on_request_start(self, metrics: RequestMetrics):
        print(f"请求开始: {metrics.provider}/{metrics.model}")

    def on_request_end(self, metrics: RequestMetrics):
        if metrics.total_tokens:
            self.total_tokens += metrics.total_tokens
        print(f"请求结束: 延迟={metrics.latency_ms}ms")

    def on_stream_event(self, event_type: str, payload: dict):
        pass  # 按需处理流式事件

    def on_error(self, metrics: RequestMetrics):
        print(f"请求错误: {metrics.error}")

tracker = UsageTracker()
register_sink(tracker)
```

---

## 错误模型（统一异常）

- 基类：`ModelError`
- 可重试：`ModelTimeoutError`、`RateLimitError`、`ModelOverloadedError`、`TransportError`
- 不建议重试：`InvalidRequestError`、`SchemaMismatchError`、`ProviderNotInstalledError`、`StreamBrokenError`
- Provider 适配器将 HTTP/SDK 异常归一为以上类型（具体见各适配器 `except` 分支）

---

---

## 迁移清单（Checklist）

1) Provider 选择与密钥：OpenRouter（推荐）/OpenAI/Doubao，并在 .env 配置 API Key。
2) 消息改造：使用 `msg.system/user(...)` 与 `Message.parts`（Text/Image）。
3) 结构化输出：优先 strict（Responses/Doubao 的 `text_format`），否则使用 Repair 兜底；完善 Schema（别名/容错校验）。
4) 工具：以 `ToolSpec` 声明；将第一轮 assistant 的 `tool_calls` +（OpenRouter 的）`reasoning_details` 一并回传第二轮；工具执行在业务层完成。
5) 多图：如需多图，使用 OpenRouter x Gemini，读取返回 `Image` parts；保存 data:URL。
6) 可观测：注册 sink / 使用 `client.last_usage()`；若需要流式 usage 事件，`yield_usage_event=True`。
7) 稳定性：启用 `with_retry`（默认重试限流/超时/过载/传输错误；不重试无效请求）。
8) 参数：将与业务相关的 provider 特定参数放到 `extra_body`，例如 OpenRouter 的 provider 路由与 transforms。

---

## 目录结构
```
aury/
  ai/
    model/
      __init__.py
      README.md
      client.py
      context.py
      errors.py
      instrumentation.py
      retry.py
      structured.py
      tools.py
      types.py
      providers/
        base.py
        registry.py
        openai.py
        openrouter.py
        doubao.py
```

现代化（Python 3.12+）模型调用层：统一消息/事件模型，多 Provider 适配，严格而稳健的结构化输出，MCP 工具声明与解析，可插拔可观测性，以及内置重试 with_retry（tenacity）。不内置 Agent/工具执行，仅专注“调用稳定 + 类型友好”。

- Python: 3.12+
- 依赖: Pydantic v2、contextvars、openai SDK（兼容 OpenRouter）、可选 volcengine-python-sdk[ark]
- Provider: OpenAI（Chat/Responses）、OpenRouter（OpenAI 兼容）、Doubao/火山方舟（Ark Chat）
- 事件：`content` / `thinking` / `tool_call` / `usage` / `completed` / `error`
- 消息：parts-only（`Text`/`Image`/`Thinking`/`FileRef`）
- 结构化输出：Strict 优先，Repair/Extract 兜底
- 工具：MCP/function/builtin 声明与解析（不执行）
- 可观测：上下文/指标 sink + usage 聚合
- 重试：client.with_retry(...)（tenacity）


---

## 安装

```bash
pip install pydantic==2.* openai>=1.0 json-repair tenacity python-dotenv
# Doubao / Ark 可选
pip install 'volcengine-python-sdk[ark]'
```

源码方式引入（此仓库中的 `aury/ai/model/` 目录）。

---

## 快速上手

### 初始化
直接初始化（推荐）：
```python
from aury.ai.model import ModelClient, msg

client = ModelClient(
    provider="openrouter",
    model="openai/gpt-4o-mini",
    api_key="${OPENROUTER_API_KEY}"
)
```

使用 `bind()` 复用配置：
```python
base = ModelClient(provider="openrouter", api_key="${OPENROUTER_API_KEY}")
client = base.bind(model="openai/gpt-4o-mini")
```

### 非流式 + usage
```python
m = await client.ainvoke([msg.user("Hello")])
print(m.parts)
print(client.last_usage())
```

### 流式（最终从 last_usage 取用量）
```python
async for ev in client.astream([msg.user("讲个笑话")]):
    if ev.type == "content":
        print(ev.delta, end="")
print(client.last_usage())
```

### 流式 Thinking
```python
async for ev in client.astream([msg.user("深入分析")], return_thinking=True):
    if ev.type == "thinking":
        print(f"💭 {ev.delta}", end="")
    elif ev.type == "content":
        print(ev.delta, end="")
```

### 多模态输入（图片）
```python
resp = await client.ainvoke([
    msg.user("描述这张图片", images=["https://example.com/img.jpg"])
])
```

### 严格结构化（Responses/Doubao 建议配合 `text_format`）
```python
from pydantic import BaseModel

class Weather(BaseModel):
    city: str
    temp_c: float

client = ModelClient(provider="openai", model="gpt-4.1-mini", api_key="${OPENAI_API_KEY}", transport="responses")
result = await client.with_structured_output(Weather).ainvoke(
    [msg.user("请以JSON返回北京天气，包含 city 和 temp_c")],
    text_format={"type":"json_object"},
    expect_strict=True,
)
```

### with_retry（tenacity）
```python
retrying = client.with_retry(max_attempts=4, base_delay=0.2)
res = await retrying.ainvoke([msg.user("稳定返回一句话")])
```

---

## 设计要点
- parts-only 强类型消息；多模态一致，Thinking 为一等公民。
- Provider Adapter 内部路由与参数映射（Chat/Responses、usage、reasoning、tools、多图等）。
- 结构化输出策略：StrictSchema（可选）→ Repair/Extract（json_repair + 代码块/最大花括号提取）。
- 工具：声明与解析（含 MCP 名称编码/解码），执行留给上层。
- 可观测：request start/end、stream event、usage 聚合；与上下文（trace_id、headers）协作。
- 重试：tenacity.AsyncRetrying 封装，非流式与流式一致。

---

## 消息与事件（摘要）
```python
from aury.ai.model.types import Message, Text, Image, Thinking, ToolCall, StreamEvent, Evt
# msg.system(...) / msg.user(text, images=[...]) 提供便捷构造
```
事件流：`content` 文本增量、`thinking` 思考增量、`tool_call` 工具调用、`usage` 用量、`completed` 结束。

多模态：`Image(url=...)`。OpenRouter 适配器会把 `message.images` 中的多张图片映射到 `Message.parts`（支持 data:URL / https）。

---

## ModelClient
- `bind(...)` 生成新实例（不可变配置，线程/协程安全）。
- `ainvoke(messages, **kw)` / `astream(messages, **kw)`
- `with_structured_output(schema)` → `StructuredView`
- `with_retry(...)` → RetryView（见“重试”）
- `last_usage()` 读取最近一次用量（含 reasoning_tokens）。

常用调用参数：
- 通用生成：`max_tokens`、`max_completion_tokens`、`temperature`、`top_p`、`stop`、`seed`。
- 推理：`return_thinking`、`reasoning_effort`（low/medium/high）。
- 结构化：`response_format`（Chat）、`text_format`（Responses/Doubao）。
- 工具：`tools`（支持 MCP 降级编码）。
- 透传：`extra_body`（Provider 特定参数，原样传给后端）。
- Doubao/Responses 特性：`previous_response_id`、`caching` 等。

---

## 结构化输出（with_structured_output）
- `expect_strict=True` 时优先严格校验；否则走 Repair/Extract 兜底。
- 失败会抛出 `ValueError: structured parse failed: ...`。

---

## 工具（MCP / Function / Builtin）
- 声明：`ToolSpec(kind=..., ...)`；
- 编码：`to_openai_tools` 会将 MCP 工具编码为 `mcp::{server}::{name}`；
- 解析：`normalize_tool_call` 会还原并填充 `mcp_server_id`；

---

## Reasoning / reasoning_details（多轮传递）
- OpenRouter DeepSeek/Gemini 等：适配器在非流式从原始 JSON 中提取 `reasoning_details`，在流式于 `completed` 事件聚合后回传，便于将其随 assistant tool_calls 一并带回第二轮。

---

## 多图输出（OpenRouter / Gemini）
- 非流式：OpenRouter 返回的 `choices[0].message.images` 会被映射到多个 `Image` part。
- 可用 `extra_body`（如后端支持）请求多图，否则通过 prompt 提示生成多张图。

---

## extra_body 透传
- 通过 `extra_body={...}` 传入 Provider 特定参数，如 OpenRouter 的 `provider.order`、`transforms`。

---

## 重试（with_retry, tenacity）
- `client.with_retry(max_attempts=3, base_delay=0.5, max_delay=5.0, backoff_factor=2.0)` 返回 RetryView；
- 默认重试错误：`ModelTimeoutError`、`RateLimitError`、`ModelOverloadedError`、`TransportError`；不重试 `InvalidRequestError`；
- 支持自定义 `retry_on`/`predicate`；

---

## 上下文与可观测性
- `model_ctx`：上下文管理器/装饰器（trace_id/request_id/provider/model/extra_headers）。
- instrumentation：`emit_*` + sink 注册；`client.last_usage()` 聚合读数。

---

## 错误类型
- `ModelTimeoutError` / `RateLimitError` / `ModelOverloadedError`
- `InvalidRequestError` / `TransportError` / `StreamBrokenError`
- `SchemaMismatchError` / `ProviderNotInstalledError`

---

---

## 目录
```
aury/
  ai/
    model/
      __init__.py
      README.md
      client.py
      context.py
      errors.py
      instrumentation.py
      retry.py
      structured.py
      tools.py
      types.py
      providers/
        base.py
        registry.py
        openai.py
        openrouter.py
        doubao.py
```

---

## 迁移提示
- `ModelClient`/`Message`/`StreamEvent`/`ToolSpec` 为核心 API；
- 若已有 OpenAI 兼容调用，可直接切换 provider="openrouter"；
- 对工具链路：保证将上一轮 assistant（含 tool_calls 与 reasoning_details）原样传回第二轮；
- 对结构化输出：优先尝试严格模式（Responses/Doubao 的 `text_format`），否则依赖 Repair/Extract 兜底；
- 启用 with_retry 提升稳定性（频控/瞬时网络）。
