# MySQL 配置与账号持久化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留本地文件模式的前提下，为 Accio 面板增加可切换的 MySQL 配置与账号持久化，并在数据库模式下优先从数据库加载、写回数据库。

**Architecture:** 新增一层运行时存储工厂，根据环境变量决定返回文件存储还是 MySQL 存储。MySQL 模式只接管面板配置和账号数据，统计与日志继续保留文件存储；数据库空表时，允许从现有本地文件进行一次性补种，避免升级后首启丢数据。

**Tech Stack:** Python 3.12, FastAPI, PyMySQL, unittest

---

### Task 1: 先把行为钉成测试

**Files:**
- Create: `tests/test_runtime_storage.py`

- [ ] **Step 1: 写文件模式测试**

```python
account_store, panel_store = create_runtime_stores(settings)
assert isinstance(account_store, AccountStore)
assert isinstance(panel_store, PanelSettingsStore)
```

- [ ] **Step 2: 跑测试确认当前失败**

Run: `uv run python -m unittest tests.test_runtime_storage -v`
Expected: FAIL，因为运行时存储工厂与 MySQL 存储实现尚不存在。

- [ ] **Step 3: 写数据库优先与空库补种测试**

```python
with patch("accio_panel.persistence.build_mysql_gateway", return_value=gateway):
    account_store, panel_store = create_runtime_stores(settings)
```

- [ ] **Step 4: 跑测试确认失败原因正确**

Run: `uv run python -m unittest tests.test_runtime_storage -v`
Expected: FAIL，缺少 MySQL 存储工厂或行为不匹配。

### Task 2: 抽出运行时存储工厂

**Files:**
- Create: `accio_panel/persistence.py`
- Modify: `accio_panel/config.py`
- Modify: `accio_panel/web.py`
- Modify: `accio_panel/web_bulk_delete_extension.py`

- [ ] **Step 1: 在配置对象中增加数据库连接配置**

```python
database_url: str = os.getenv("ACCIO_DATABASE_URL", "").strip()
mysql_host: str = os.getenv("ACCIO_MYSQL_HOST", "").strip()
```

- [ ] **Step 2: 新增运行时存储工厂**

```python
def create_runtime_stores(settings: Settings) -> tuple[object, object]:
    ...
```

- [ ] **Step 3: 让应用启动和扩展路由都走统一存储来源**

```python
store, panel_settings_store = create_runtime_stores(settings)
application.state.store = store
```

- [ ] **Step 4: 跑新增测试**

Run: `uv run python -m unittest tests.test_runtime_storage -v`
Expected: 仍可能 FAIL，因为 MySQL 后端尚未实现完整。

### Task 3: 实现 MySQL 配置与账号存储

**Files:**
- Create: `accio_panel/mysql_storage.py`
- Modify: `accio_panel/store.py`
- Modify: `accio_panel/app_settings.py`

- [ ] **Step 1: 抽出账号存储公共逻辑，避免文件版和数据库版重复一坨**

```python
class BaseAccountStore:
    def list_accounts(self) -> list[Account]:
        ...
```

- [ ] **Step 2: 实现 MySQL gateway 和两类 store**

```python
class MySQLGateway:
    def ensure_schema(self) -> None:
        ...
```

- [ ] **Step 3: 实现数据库空表补种**

```python
if self.gateway.count_accounts() == 0:
    for account in file_store.list_accounts():
        self.save(account)
```

- [ ] **Step 4: 跑测试确认转绿**

Run: `uv run python -m unittest tests.test_runtime_storage -v`
Expected: PASS

### Task 4: 补文档并做整体验证

**Files:**
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `compose.yaml`
- Modify: `README.md`

- [ ] **Step 1: 增加 PyMySQL 依赖和示例环境变量**

```toml
"PyMySQL>=1.1.1",
```

- [ ] **Step 2: 更新 README，说明文件模式与 MySQL 模式切换规则**

```md
- 未配置数据库连接信息：继续使用本地文件
- 配置数据库连接信息：配置与账号改为持久化到 MySQL
```

- [ ] **Step 3: 运行完整验证**

Run: `uv run python -m unittest -v`
Expected: PASS
