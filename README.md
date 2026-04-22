# Image API OpenAI Proxy

**完全由AI生成、人工测试，用着还行但谨慎食用。**

将 ModelScope 和 SiliconFlow 的生图接口统一转换为 OpenAI Images API 兼容格式，便于现有 OpenAI 客户端和工具直接接入。

## 安装

虽然还没发布，但建议使用 pipx 作为独立程序安装
```bash
pipx install image-api-openai
```

或者也可以 clone 之后本地安装：

```bash
python -m venv venv
venv/bin/pip install -e .
```

## 配置

1. 选择工作目录（默认会创建和使用 `~/.config/image-api-openai/`）
2. 将包内的 `config.yaml.example` 复制为工作目录下 `config.yaml`
3. 编辑 `config.yaml`，填写：
   - `providers.modelscope.api_key`
   - `providers.siliconflow.api_key`
    - `api_keys`（其他程序访问本接口使用的鉴权 key）

## 运行

```bash
image-api-openai --dir ~/.config/image-api-openai --host 0.0.0.0 --port 8000
```

参数说明：
- `--dir`: 工作目录，读取 `config.yaml`，不指定时使用 ` ~/.config/image-api-openai`
- `--host`: 监听地址
- `--port`: 监听端口

## OpenAI 兼容接口

### 生成图片

`POST /v1/images/generations`

请求示例：

```bash
curl --request POST \
  --url http://127.0.0.1:8000/v1/images/generations \
  --header 'Authorization: Bearer your_openai_compatible_api_key' \
  --header 'Content-Type: application/json' \
  --data '{
  "model": "Kwai-Kolors/Kolors",
  "prompt": "a fox in watercolor style",
  "size": "1024x1024",
  "n": 1,
  "response_format": "b64_json",
  "provider": "siliconflow"
}'
```

响应格式：

```json
{
  "created": 1710000000,
  "data": [
    {"b64_json": "iVBORw0KGgoAAA..."}
  ]
}
```

支持参数（兼容层）：
- `model`, `prompt`, `negative_prompt`, `size`, `n`, `response_format`, `user`
- `provider`（扩展参数，可选：`modelscope` 或 `siliconflow`）

模型选择支持三种写法：
- 供应商前缀：`modelscope/Qwen/Qwen-Image`
- 别名：`Qwen Image`（按优先级选择对应供应商模型）
- 原始模型名 + `provider` 参数：`model: "Qwen/Qwen-Image", provider: "modelscope"`

**注意**：`model` 为必填参数。若未使用前缀或别名，将返回 400 错误。

当 `response_format=b64_json` 时，代理会下载图片并回传 base64。

### 查看上游提供方状态

`GET /v1/providers`

### 列出模型（OpenAI 兼容）

`GET /v1/models`

返回格式与 OpenAI 对齐：

```json
{
  "object": "list",
  "data": [
    {
      "id": "Kwai-Kolors/Kolors",
      "object": "model",
      "created": 0,
      "owned_by": "siliconflow"
    }
  ]
}
```

模型来源规则：
- SiliconFlow：优先调用上游 `GET /models?type=text-to-image`
- ModelScope：读取 `supported.static_models.modelscope`

`/v1/models` 会返回：
- 所有带 provider 前缀的模型，如 `modelscope/Qwen/Qwen-Image`
- 所有别名模型，如 `Qwen Image`

## 别名与优先级

你可以在各 provider 配置段里配置 `aliases`，为模型定义别名与优先级：

```yaml
providers:
  modelscope:
    aliases:
      Qwen/Qwen-Image:
        alias: "Qwen Image"
        priority: 20

  siliconflow:
    aliases:
      Qwen/Qwen-Image-Edit-2509:
        alias: "Qwen Image"
        priority: 10
```

当客户端请求 `model: "Qwen Image"` 时：
- 先取该别名下优先级最高的候选模型
- 若最高优先级有多个，则随机选择一个

如果你同时传了 `provider`，则会先在该 provider 内筛选别名候选。

## 路由策略

模型解析顺序：
1. 检查是否有 provider 前缀（如 `modelscope/xxx`）
2. 检查是否为别名（从 aliases 配置解析）
3. 若未提供 `provider` 参数，报错 400

配置别名可简化调用，无需每次指定 provider 或使用前缀。

## 说明

- ModelScope 目前来看对 AIGC 使用异步任务，建议打开 `providers.modelscope.async_mode: true` ，让程序自动轮询任务结果。后续直接让AI把这个设置删了，毕竟好像ModelScope全部是异步执行。
- SiliconFlow 图片 URL 存在有效期，建议调用方及时下载持久化。
