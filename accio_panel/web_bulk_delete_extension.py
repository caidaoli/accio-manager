from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse

from . import web as web_module


ROUTE_PATH = "/api/accounts/delete-abnormal-disabled"


def _get_account_store():
    return web_module.app.state.store


def _route_exists() -> bool:
    app = web_module.app
    for route in getattr(app, "routes", []):
        route_path = getattr(route, "path", "")
        route_methods = getattr(route, "methods", set()) or set()
        if route_path == ROUTE_PATH and "POST" in route_methods:
            return True
    return False


def _build_message(payload: dict[str, Any]) -> str:
    processed_count = int(payload.get("processedCount") or 0)
    deleted_count = int(payload.get("deletedCount") or 0)
    failure_count = int(payload.get("failureCount") or 0)

    if processed_count <= 0:
        return "没有需要删除的异常禁用账号"

    message = f"已删除 {deleted_count} 个异常禁用账号"
    if failure_count > 0:
        return f"{message}，失败 {failure_count} 个"
    return message


def register_routes() -> None:
    if _route_exists():
        return

    app = web_module.app

    @app.post(ROUTE_PATH)
    def delete_abnormal_auto_disabled_accounts() -> JSONResponse:
        store = _get_account_store()
        payload = store.delete_abnormal_auto_disabled_accounts()
        deleted_count = int(payload.get("deletedCount") or 0)
        failure_count = int(payload.get("failureCount") or 0)
        success = deleted_count > 0 and failure_count == 0
        if int(payload.get("processedCount") or 0) == 0:
            success = True
        return JSONResponse(
            {
                "success": success,
                "message": _build_message(payload),
                **payload,
            }
        )


register_routes()
