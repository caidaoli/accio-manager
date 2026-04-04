# Accio 多账号管理面板

[![LINUX DO](https://img.shields.io/badge/LINUX%20DO-社区认可-blue?style=flat-square&logo=linux)](https://linux.do)

基于 FastAPI 的本地管理面板，支持：

- 多账号本地保存
- 支持切换到 MySQL 持久化配置和账号
- 登录链接生成
- 登录成功回调保存 Token
- 独立 OAuth 页面，支持手动粘贴回调地址导入
- 账号 JSON 下载与导入
- 查看额度与重置时间
- 单账号 / 批量刷新 Token
- 配置上游代理与 API 调度策略
- 手动启用 / 禁用账号
- 额度耗尽自动禁用，额度恢复自动启用
- 单后台调度器按账号下次检查时间自动巡检额度
- Anthropic 兼容 API：`/v1/models`、`/v1/messages`
- OpenAI 兼容 API：`/v1/models`、`/v1/chat/completions`
- OpenAI Responses 兼容 API：`/v1/responses`
- Gemini 兼容 API：`/v1beta/models`、`/v1beta/models/{model}`、`/v1beta/models/{model}:generateContent`、`/v1beta/models/{model}:streamGenerateContent`
- 模型目录优先通过 `/api/llm/config` 动态拉取，并带 60 秒缓存
- 多账号 API 调度策略：优先填充 / 轮询
- 支持为每个账号设置优先填充优先级
- 支持全局上游代理：HTTP / HTTPS / SOCKS4 / SOCKS5
- 内置统计界面：模型调用次数、输入 / 输出 Tokens、账号维度统计
- 内置日志界面：记录 Anthropic / OpenAI / Gemini 兼容调用的账号选择、上游错误、空回复与图片摘要

## 启动

```bash
uv sync
uv run accio-panel
```

也可以使用：

```bash
uv sync
uv run python main.py
```

启动后访问：

- `http://127.0.0.1:4097/dashboard`
- `http://127.0.0.1:4097/oauth`
- `http://127.0.0.1:4097/login`
- `http://127.0.0.1:4097/v1/models`
- `http://127.0.0.1:4097/v1/messages`
- `http://127.0.0.1:4097/v1/chat/completions`
- `http://127.0.0.1:4097/v1/responses`
- `http://127.0.0.1:4097/v1beta/models`
- `http://127.0.0.1:4097/dashboard?view=logs`

## Docker

直接拉取镜像（推荐）：

```bash
docker pull ghcr.io/guji08233/accio-manager:latest
```

运行：

```bash
docker run -d \
  --name accio-panel \
  -p 4097:4097 \
  -v accio-panel-data:/app/data \
  -e ACCIO_CALLBACK_HOST=127.0.0.1 \
  ghcr.io/guji08233/accio-manager:latest
```

镜像由 GitHub Actions 自动构建，推送到 `main` 分支后会自动更新。

如需本地构建：

```bash
docker build -t accio-panel:latest .
docker run -d \
  --name accio-panel \
  -p 4097:4097 \
  -v accio-panel-data:/app/data \
  -e ACCIO_CALLBACK_HOST=127.0.0.1 \
  accio-panel:latest
```

说明：

- 容器内服务监听 `0.0.0.0:4097`
- 默认数据目录是 `/app/data`
- 未配置数据库连接信息时，配置和账号继续使用本地文件
- 配置了 `ACCIO_MYSQL` 或完整的 `ACCIO_MYSQL_*` 连接信息后，面板配置和账号会持久化到 MySQL；数据库空表时会在首次启动时从本地文件补种一次
- 服务器部署时，建议使用 `/oauth` 页面处理登录，并在需要时手动粘贴完整回调 URL 导入账号
- 新账号在回调导入后，会自动依次触发 `userinfo`、`invitation/query` 和 `channel/query` 完成激活

## GitHub Packages

仓库已添加 GitHub Actions 工作流：

- 文件：`.github/workflows/docker-publish.yml`
- 触发方式：`push` 到 `main`、`workflow_dispatch` 手动触发
- 推送目标：`ghcr.io/<owner>/<repo>`

首次推送成功后，可以在仓库的 `Packages` 页面看到镜像。

首次管理员密码默认值为：

```text
admin
```

可在 `data/config.json` 或面板内配置区中修改。

仓库默认只保留示例配置，不提交真实运行数据：

- `data/config.json`
- `data/stats.json`
- `data/accounts/*.json`
- `.env`

示例文件：

- `data/config.example.json`
- `.env.example`

MySQL 环境变量示例：

```text
ACCIO_MYSQL=mysql://accio:secret@127.0.0.1:3306/accio_manager
```

或者拆开写：

```text
ACCIO_MYSQL_HOST=127.0.0.1
ACCIO_MYSQL_PORT=3306
ACCIO_MYSQL_DATABASE=accio_manager
ACCIO_MYSQL_USER=accio
ACCIO_MYSQL_PASSWORD=secret
ACCIO_MYSQL_CHARSET=utf8mb4
```

兼容 API 调用时：

- 使用 `x-api-key` 或 `Authorization: Bearer`
- 值填写当前管理员密码
- Gemini 兼容接口额外支持 `x-goog-api-key`、`?key=`、`?api_key=` 这几种官方常见传法
- 如果管理员密码里包含 `+`，放到 URL Query 时请编码成 `%2B`
- `/v1/models`、`/v1/messages`、`/v1/chat/completions`、`/v1/responses` 都需要这个鉴权
- `/v1beta/models`、`/v1beta/models/{model}`、`/v1beta/models/{model}:generateContent`、`/v1beta/models/{model}:streamGenerateContent` 也使用同一套鉴权
- `/v1/models` 与 `/v1beta/models` 会优先调用 `POST /api/llm/config` 动态拉取模型目录，并缓存 60 秒
- 动态目录可用时，`/v1/messages`、`/v1/chat/completions`、`/v1/responses`、`/v1beta/models/{model}:generateContent` 的模型校验也按该目录执行
- 动态目录暂不可用时，列表接口会回退到内置静态模型；实际请求会继续按传入模型名直传上游
- 响应头 `x-accio-model-source` 会标记模型目录来源：`live`、`cache`、`stale` 或 `static-fallback`
- 模型名不会做别名改写，按请求值直接透传到上游
- 默认调度策略是 `轮询`
- 优先填充支持账号级 `fillPriority`，数值越小越优先；同优先级下会优先调用剩余额度更少的账号
- 可在面板配置区设置全局上游代理，例如：
  `http://127.0.0.1:7890`、`https://127.0.0.1:7890`、`socks5://127.0.0.1:1080`、`socks5h://127.0.0.1:1080`
- 代理只影响服务端访问 Accio 网关，不影响浏览器登录页

最小调用示例：

```bash
curl http://127.0.0.1:4097/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: admin" \
  -d "{\"model\":\"claude-sonnet-4-6\",\"max_tokens\":256,\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}"
```

Gemini 兼容调用示例：

```bash
curl http://127.0.0.1:4097/v1beta/models/gemini-3.1-pro-preview:generateContent \
  -H "content-type: application/json" \
  -H "x-api-key: admin" \
  -d "{\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":\"你好\"}]}],\"generationConfig\":{\"maxOutputTokens\":1024}}"
```

Gemini 流式示例：

```bash
curl http://127.0.0.1:4097/v1beta/models/gemini-3.1-pro-preview:streamGenerateContent \
  -H "content-type: application/json" \
  -H "x-api-key: admin" \
  -d "{\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":\"你好\"}]}]}"
```

OpenAI Chat 兼容调用示例：

```bash
curl http://127.0.0.1:4097/v1/chat/completions \
  -H "content-type: application/json" \
  -H "x-api-key: admin" \
  -d "{\"model\":\"claude-sonnet-4-6\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}"
```

OpenAI Responses 兼容调用示例：

```bash
curl http://127.0.0.1:4097/v1/responses \
  -H "content-type: application/json" \
  -H "x-api-key: admin" \
  -d "{\"model\":\"claude-sonnet-4-6\",\"input\":\"你好\",\"stream\":false}"
```

开启思考示例：

```bash
curl http://127.0.0.1:4097/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: admin" \
  -d "{\"model\":\"claude-sonnet-4-6\",\"max_tokens\":2048,\"stream\":false,\"thinking\":{\"type\":\"adaptive\",\"effort\":\"high\"},\"messages\":[{\"role\":\"user\",\"content\":\"请先思考再回答：2+2 为什么等于 4？\"}]}"
```

兼容说明：

- 如果客户端仍发送旧格式 `thinking: {\"type\":\"enabled\",\"budget_tokens\":...}`，系统也会继续兼容
- 非流式返回会保留 `thinking` 块；如果上游返回了 `signature`，也会一并透传
- `/v1/responses` 已支持基础流式事件、非流式返回，以及常见 `input` 项转换
- `Responses input` 当前支持：纯字符串、`message`/`role` 消息项、`input_text`、`input_image`、`image_url`、`input_file`、`file`、`function_call`、`function_call_output`
- `Responses` 额外字段如 `metadata`、`user`、`session_id`、`conversation_id`、`previous_response_id`、`include`、`truncation` 会继续向内部请求骨架透传

## 数据目录

当前数据结构：

```text
data/
  config.json
  stats.json
  accio-accounts.json
  accounts/
    <account_id>.json
```

- `config.json`：全局配置、管理员密码、会话密钥
- `stats.json`：兼容 API 的累计调用统计
- `api-logs.jsonl`：兼容 API 的逐条调用日志
- `accounts/*.json`：每个账号单独一个文件
- `accio-accounts.json`：旧版单文件账号列表，首次启动会自动迁移到 `accounts/` 目录
- 面板支持导入单账号 JSON，也支持直接导入旧版 `accio-accounts.json` 数组文件
- 数据库模式下，`config.json` 和 `accounts/*.json` 只用于首次补种；运行中的配置和账号以 MySQL 为准

## 自动调度

- 系统使用单个后台调度器，不为每个账号单独创建定时器
- 启用中的账号会低频巡检额度
- 自动禁用账号会基于接口返回的账单重置时间安排下次恢复检查
