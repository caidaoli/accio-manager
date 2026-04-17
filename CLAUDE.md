# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Accio 是一个基于 FastAPI 的多账号管理面板，充当 Anthropic / OpenAI / Gemini API 的本地代理网关。核心功能：多账号管理、额度自动巡检、API 请求调度（轮询/优先填充）、兼容多家 API 格式。

## Common Commands

```bash
# 安装依赖
uv sync

# 启动开发服务器 (端口 4097)
uv run accio-panel
# 或
uv run python main.py

# 运行全部测试
uv run python -m unittest

# 运行单个测试模块
uv run python -m unittest tests.test_runtime_storage
uv run python -m unittest tests.test_release_build

# 本地 Nuitka 单文件构建
uv run --with "nuitka[onefile]==4.0" python -m nuitka --onefile --include-package-data=accio_panel main.py

# 发布：推送版本标签触发 GitHub Actions 构建
git tag v0.1.3
git push origin v0.1.3
```

## Architecture

### 入口与启动

- `main.py` → 导入 `accio_panel.web:run` 启动 Uvicorn
- `pyproject.toml` 定义 CLI 入口 `accio-panel` 指向同一函数
- 版本号由 `hatch-vcs` 从 git tag 自动生成

### 核心模块 (`accio_panel/`)

**Web 层**
- `web.py` — FastAPI 主应用，会话中间件、后台调度器启动
- `panel_routes.py` — 面板路由（dashboard、oauth、settings、批量操作）
- `dashboard_views.py` — 仪表盘视图逻辑（账号列表、额度展示、分页）
- `proxy_api_routes.py` — API 代理路由注册（Anthropic/OpenAI/Gemini 端点）
- `templates/` — Jinja2 模板（dashboard、oauth、settings 等）

**API 代理层** — 三个独立模块，各自负责请求/响应格式转换：
- `anthropic_proxy.py` — `/v1/models`, `/v1/messages`
- `openai_proxy.py` — `/v1/models`, `/v1/chat/completions`, `/v1/responses`
- `gemini_proxy.py` — `/v1beta/models`, `/v1beta/models/{model}:generateContent`
- `upstream_support.py` — 上游请求封装（错误处理、重试、Token 刷新）
- `proxy_selection.py` — 账号调度策略（轮询/优先填充）

**存储层** — Repository 模式 + Factory 模式：
- `store.py` — `BaseAccountStore` 抽象基类 + `AccountStore` 文件后端实现
- `mysql_storage.py` — `MySQLAccountStore` / `MySQLPanelSettingsStore` / `MySQLGateway`（持久连接 + 自动重连）
- `app_settings.py` — `PanelSettingsStore` 文件后端配置存储
- `persistence.py` — `create_runtime_stores()` 工厂函数，按 `ACCIO_MYSQL` 环境变量决定后端；MySQL 模式下自动从文件补种

**调度与监控**
- `quota_scheduler.py` — 单后台调度器，按账号下次检查时间巡检额度
- `usage_stats.py` — API 调用统计（模型维度、账号维度）
- `api_logs.py` — API 调用日志（JSONL 格式，200 行自动截断）

**基础设施**
- `config.py` — `Settings` dataclass，从环境变量加载配置，支持 Nuitka 编译路径检测
- `models.py` — `Account` 数据模型与归一化工具函数
- `client.py` — HTTP 客户端封装，支持代理（HTTP/SOCKS）和会话复用
- `model_catalog.py` / `model_catalog_cache.py` — 动态模型目录，60 秒缓存

### 数据目录

运行时数据默认位于项目根 `data/`，可通过 `ACCIO_DATA_DIR` 环境变量覆盖：
- `config.json` — 全局配置（管理员密码、会话密钥等）
- `accounts/*.json` — 每个账号一个文件
- `stats.json` — API 调用统计
- `api-logs.jsonl` — API 调用日志

### 关键设计决策

- 所有账号操作通过 `threading.RLock` 保护并发安全
- MySQL 模式使用内存缓存 + write-through 策略减少数据库 I/O
- 后台单调度器按账号下次检查时间巡检额度（非每账号独立定时器）
- 启动时强制重置未到期账号的巡检时间，确保重启后立即触发首轮巡检
- 仪表盘使用缓存额度数据，避免页面渲染时的实时上游查询
- API 日志自动截断到 200 行，防止文件无限增长
- API 鉴权统一使用管理员密码，支持 `x-api-key` / `Authorization: Bearer` / Gemini 风格 query 参数

### 测试

测试位于 `tests/` 目录，覆盖：
- `test_runtime_storage.py` — 存储层切换（文件/MySQL）
- `test_upstream_support.py` — 上游请求与错误处理
- `test_proxy_routing.py` — 账号调度策略
- `test_api_logs.py` — 日志记录与截断
- `test_client_generate_content.py` — Gemini 客户端
- `test_release_build.py` — 版本号校验
- `test_tool_config_mapping.py` — 工具配置映射
- `test_upstream_message_id.py` / `test_upstream_turn_error.py` — 上游协议兼容性

运行测试前无需启动服务，测试使用内存存储和 mock 上游。

## Git Workflow

本仓库是 `GuJi08233/accio-manager` 的 fork。与上游差异较大，**禁止直接 merge upstream**，只能逐提交 cherry-pick。

```
origin   → caidaoli/accio-manager (push/fetch)
upstream → GuJi08233/accio-manager (fetch only)
```

## Environment Variables

关键环境变量参见 `.env.example`：
- `ACCIO_MYSQL` — MySQL 连接串，设置后启用数据库后端
- `ACCIO_DATA_DIR` — 数据目录路径
- `ACCIO_CALLBACK_PORT` — 服务端口（默认 4097）
- `ACCIO_BASE_URL` — 上游网关地址

## Development Notes

- 修改 Web 层路由时，注意 `web.py` 已拆分为 `panel_routes.py` 和 `proxy_api_routes.py`
- 新增 API 代理端点应在对应的 `*_proxy.py` 中实现，然后在 `proxy_api_routes.py` 注册
- 存储层修改需同时考虑文件后端和 MySQL 后端的兼容性
- 额度调度逻辑集中在 `quota_scheduler.py`，避免在路由层直接操作
- 仪表盘视图逻辑已提取到 `dashboard_views.py`，模板只负责渲染
- 版本号由 `hatch-vcs` 从 git tag 自动生成，不要手动修改 `__version__`
