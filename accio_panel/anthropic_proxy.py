from __future__ import annotations

import json
import re
import uuid
from typing import Any
from typing import Callable
from typing import Iterator

import requests


SUPPORTED_ANTHROPIC_MODELS = (
    "claude-sonnet-4-6",
    "claude-opus-4-6",
)
SUPPORTED_ANTHROPIC_MODELS_SET = set(SUPPORTED_ANTHROPIC_MODELS)
SUPPORTED_GEMINI_MODELS = (
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3-pro-preview",
)
SUPPORTED_PROXY_MODELS = SUPPORTED_ANTHROPIC_MODELS + SUPPORTED_GEMINI_MODELS
SUPPORTED_PROXY_MODELS_SET = set(SUPPORTED_PROXY_MODELS)
DEFAULT_ANTHROPIC_MODEL = SUPPORTED_ANTHROPIC_MODELS[0]
MODEL_OWNERS = {
    "claude-sonnet-4-6": "anthropic",
    "claude-opus-4-6": "anthropic",
    "gemini-3-flash-preview": "google",
    "gemini-3.1-pro-preview": "google",
    "gemini-3-pro-preview": "google",
}


def _usage_summary() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def _normalize_thinking_level(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"low", "medium", "high"}:
        return normalized
    return ""


def _budget_to_thinking_level(value: Any) -> str:
    try:
        budget = int(value or 0)
    except (TypeError, ValueError):
        return "high"
    if budget >= 12000:
        return "high"
    if budget >= 4000:
        return "medium"
    return "low"


def _apply_thinking_config(request_body: dict[str, Any], body: dict[str, Any]) -> None:
    thinking = body.get("thinking")
    if not isinstance(thinking, dict):
        return

    thinking_type = str(thinking.get("type") or "").strip().lower()
    if thinking_type != "enabled":
        return

    request_body["include_thoughts"] = True
    request_body["thinking_level"] = "high"
    if thinking.get("budget_tokens") is not None:
        request_body["thinking_budget"] = thinking.get("budget_tokens")


def _normalize_stop_sequences(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            items.append(text)
    return items


def _guess_image_mime_type(value: str) -> str:
    lowered = value.lower()
    if lowered.endswith(".jpg") or lowered.endswith(".jpeg"):
        return "image/jpeg"
    if lowered.endswith(".webp"):
        return "image/webp"
    if lowered.endswith(".gif"):
        return "image/gif"
    return "image/png"


def build_models_payload() -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "owned_by": MODEL_OWNERS.get(model_name, "anthropic"),
            }
            for model_name in SUPPORTED_PROXY_MODELS
        ],
    }


def anthropic_error_payload(
    message: str,
    *,
    error_type: str = "api_error",
) -> dict[str, object]:
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }


def build_accio_request(
    body: dict[str, Any],
    *,
    token: str,
    utdid: str,
    version: str,
) -> dict[str, Any]:
    request_body: dict[str, Any] = {
        "utdid": utdid,
        "version": version,
        "token": token,
        "empid": str(body.get("empid") or ""),
        "tenant": str(body.get("tenant") or ""),
        "iai_tag": str(body.get("iai_tag", body.get("iaiTag")) or ""),
        "stream": True,
        "model": body.get("model") or DEFAULT_ANTHROPIC_MODEL,
        "request_id": str(
            body.get("request_id")
            or body.get("requestId")
            or f"req-{uuid.uuid4()}"
        ),
        "message_id": str(body.get("message_id", body.get("messageId")) or ""),
        "incremental": True,
        "max_output_tokens": body.get("max_tokens") or 8192,
        "contents": [],
        "stop_sequences": _normalize_stop_sequences(
            body.get("stop_sequences", body.get("stop"))
        ),
        "properties": (
            dict(body.get("properties"))
            if isinstance(body.get("properties"), dict)
            else {}
        ),
    }

    system_value = body.get("system")
    system_text = _extract_system_text(system_value)
    if system_text:
        request_body["system_instruction"] = system_text

    if body.get("temperature") is not None:
        request_body["temperature"] = body.get("temperature")
    if body.get("top_p") is not None:
        request_body["top_p"] = body.get("top_p")
    if body.get("response_format") is not None:
        request_body["response_format"] = body.get("response_format")

    _apply_thinking_config(request_body, body)

    tool_config = body.get("toolConfig", body.get("tool_config"))
    if isinstance(tool_config, dict) and tool_config:
        request_body["tool_config"] = dict(tool_config)

    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        request_body["tools"] = [
            {
                "name": str(tool.get("name") or ""),
                "description": str(tool.get("description") or ""),
                "parametersJson": json.dumps(
                    tool.get("input_schema") or {},
                    ensure_ascii=False,
                ),
            }
            for tool in tools
            if isinstance(tool, dict) and tool.get("name")
        ]

    contents = convert_messages(body.get("messages") or [])
    request_body["contents"] = ensure_alternating_roles(contents)
    return request_body


def convert_messages(messages: list[Any]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content_blocks = message.get("content")

        if role == "assistant":
            parts: list[dict[str, Any]] = []
            thought_signature: str | None = None

            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type in {"text", "input_text", "output_text"}:
                        parts.append(
                            {
                                "text": str(block.get("text") or ""),
                                "thought": False,
                            }
                        )
                    elif (
                        block_type == "image"
                        and isinstance(block.get("source"), dict)
                        and block["source"].get("type") == "base64"
                    ):
                        parts.append(
                            {
                                "thought": False,
                                "inline_data": {
                                    "mime_type": str(block["source"].get("media_type") or ""),
                                    "data": str(block["source"].get("data") or ""),
                                },
                            }
                        )
                    elif (
                        block_type == "image"
                        and isinstance(block.get("source"), dict)
                        and block["source"].get("type") == "url"
                        and block["source"].get("url")
                    ):
                        parts.append(
                            {
                                "thought": False,
                                "file_data": {
                                    "file_uri": str(block["source"].get("url") or ""),
                                    "mime_type": str(
                                        block["source"].get("media_type")
                                        or _guess_image_mime_type(
                                            str(block["source"].get("url") or "")
                                        )
                                    ),
                                },
                            }
                        )
                    elif block_type == "thinking":
                        part: dict[str, Any] = {
                            "text": str(block.get("thinking") or ""),
                            "thought": True,
                        }
                        signature = block.get("signature")
                        if signature:
                            part["thoughtSignature"] = str(signature)
                            thought_signature = str(signature)
                        parts.append(part)
                    elif block_type in {"tool_use", "tool_call", "function_call"}:
                        function_payload = block.get("function")
                        input_value = block.get("input")
                        tool_name = str(block.get("name") or "")
                        if isinstance(function_payload, dict):
                            if not tool_name:
                                tool_name = str(function_payload.get("name") or "")
                            if input_value is None:
                                input_value = function_payload.get(
                                    "arguments",
                                    function_payload.get("arguments_json"),
                                )
                        if input_value is None:
                            input_value = block.get("arguments", block.get("arguments_json"))
                        parts.append(
                            {
                                "thought": False,
                                "functionCall": {
                                    "id": sanitize_tool_call_id(
                                        block.get("id")
                                        or block.get("call_id")
                                        or block.get("tool_call_id")
                                        or uuid.uuid4().hex
                                    ),
                                    "name": tool_name,
                                    "argsJson": (
                                        input_value
                                        if isinstance(input_value, str)
                                        else json.dumps(
                                            input_value or {},
                                            ensure_ascii=False,
                                        )
                                    ),
                                },
                            }
                        )
            else:
                parts.append(
                    {
                        "text": str(content_blocks or ""),
                        "thought": False,
                    }
                )

            content: dict[str, Any] = {
                "role": "model",
                "parts": parts or [{"text": "", "thought": False}],
            }
            if thought_signature:
                content["metadata"] = {"textThoughtSignature": thought_signature}
            contents.append(content)
            continue

        if role != "user":
            continue

        tool_parts: list[dict[str, Any]] = []
        text_parts: list[dict[str, Any]] = []

        if isinstance(content_blocks, list):
            for block in content_blocks:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(
                        {
                            "text": str(block.get("text") or ""),
                            "thought": False,
                        }
                    )
                elif (
                    block_type == "image"
                    and isinstance(block.get("source"), dict)
                    and block["source"].get("type") == "base64"
                ):
                    text_parts.append(
                        {
                            "thought": False,
                            "inline_data": {
                                "mime_type": str(block["source"].get("media_type") or ""),
                                "data": str(block["source"].get("data") or ""),
                            },
                        }
                    )
                elif (
                    block_type == "image"
                    and isinstance(block.get("source"), dict)
                    and block["source"].get("type") == "url"
                    and block["source"].get("url")
                ):
                    text_parts.append(
                        {
                            "thought": False,
                            "file_data": {
                                "file_uri": str(block["source"].get("url") or ""),
                                "mime_type": str(
                                    block["source"].get("media_type")
                                    or _guess_image_mime_type(
                                        str(block["source"].get("url") or "")
                                    )
                                ),
                            },
                        }
                    )
                elif block_type == "tool_result":
                    tool_info = find_tool_info(block.get("tool_use_id"), messages)
                    function_response: dict[str, Any] = {
                        "id": sanitize_tool_call_id(
                            block.get("tool_use_id")
                            or block.get("tool_call_id")
                            or block.get("id")
                            or uuid.uuid4().hex
                        ),
                        "name": (
                            str(tool_info.get("name"))
                            if tool_info and tool_info.get("name")
                            else str(block.get("name") or "unknown")
                        ),
                        "responseJson": json.dumps(
                            {
                                "content": extract_tool_result_text(block.get("content")),
                                "is_error": bool(block.get("is_error", False)),
                            },
                            ensure_ascii=False,
                        ),
                    }
                    part: dict[str, Any] = {
                        "thought": False,
                        "functionResponse": function_response,
                    }
                    if tool_info and tool_info.get("thoughtSignature"):
                        part["thoughtSignature"] = tool_info["thoughtSignature"]
                    tool_parts.append(part)
        else:
            text_parts.append(
                {
                    "text": str(content_blocks or ""),
                    "thought": False,
                }
            )

        if tool_parts:
            content: dict[str, Any] = {"role": "tool", "parts": tool_parts}
            signature = find_last_signature(contents)
            if signature:
                content["metadata"] = {"textThoughtSignature": signature}
            contents.append(content)

        if text_parts:
            if tool_parts:
                contents.append({"role": "model", "parts": [{"text": "", "thought": False}]})
            contents.append({"role": "user", "parts": text_parts})

    return contents


def ensure_alternating_roles(contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(contents) <= 1:
        return contents

    def side(role: str) -> str:
        return "model" if role == "model" else "user"

    result = [contents[0]]
    for content in contents[1:]:
        previous = result[-1]
        if side(str(previous.get("role") or "")) == side(str(content.get("role") or "")):
            filler_role = "model" if side(str(content.get("role") or "")) == "user" else "user"
            result.append({"role": filler_role, "parts": [{"text": "", "thought": False}]})
        result.append(content)
    return result


def _as_usage_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def update_usage_summary(
    summary: dict[str, Any],
    event_name: str,
    event_payload: dict[str, Any],
) -> None:
    if event_name == "message_start":
        message = event_payload.get("message")
        if isinstance(message, dict):
            usage = message.get("usage")
            if isinstance(usage, dict):
                summary["usage"]["input_tokens"] = max(
                    summary["usage"]["input_tokens"],
                    _as_usage_int(usage.get("input_tokens")),
                )
                summary["usage"]["output_tokens"] = max(
                    summary["usage"]["output_tokens"],
                    _as_usage_int(usage.get("output_tokens")),
                )
                summary["usage"]["cache_creation_input_tokens"] = max(
                    summary["usage"]["cache_creation_input_tokens"],
                    _as_usage_int(usage.get("cache_creation_input_tokens")),
                )
                summary["usage"]["cache_read_input_tokens"] = max(
                    summary["usage"]["cache_read_input_tokens"],
                    _as_usage_int(usage.get("cache_read_input_tokens")),
                )
        return

    if event_name == "content_block_start":
        block = event_payload.get("content_block")
        if not isinstance(block, dict):
            return
        summary["content_blocks"] = int(summary.get("content_blocks") or 0) + 1
        block_type = str(block.get("type") or "")
        if block_type == "tool_use":
            summary["tool_use_blocks"] = int(summary.get("tool_use_blocks") or 0) + 1
        return

    if event_name == "content_block_delta":
        delta = event_payload.get("delta")
        if not isinstance(delta, dict):
            return
        if delta.get("text") is not None:
            summary["text_chars"] = int(summary.get("text_chars") or 0) + len(
                str(delta.get("text") or "")
            )
        if delta.get("thinking") is not None:
            summary["thinking_chars"] = int(summary.get("thinking_chars") or 0) + len(
                str(delta.get("thinking") or "")
            )
        if delta.get("partial_json") is not None:
            summary["tool_json_chars"] = int(summary.get("tool_json_chars") or 0) + len(
                str(delta.get("partial_json") or "")
            )
        return

    if event_name != "message_delta":
        return

    usage = event_payload.get("usage")
    if isinstance(usage, dict):
        summary["usage"]["input_tokens"] = max(
            summary["usage"]["input_tokens"],
            _as_usage_int(usage.get("input_tokens")),
        )
        summary["usage"]["output_tokens"] = max(
            summary["usage"]["output_tokens"],
            _as_usage_int(usage.get("output_tokens")),
        )
        summary["usage"]["cache_creation_input_tokens"] = max(
            summary["usage"]["cache_creation_input_tokens"],
            _as_usage_int(usage.get("cache_creation_input_tokens")),
        )
        summary["usage"]["cache_read_input_tokens"] = max(
            summary["usage"]["cache_read_input_tokens"],
            _as_usage_int(usage.get("cache_read_input_tokens")),
        )

    delta = event_payload.get("delta")
    if isinstance(delta, dict) and delta.get("stop_reason"):
        summary["stop_reason"] = str(delta["stop_reason"])


def iter_anthropic_sse_events(
    response: requests.Response,
    model: str,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """将上游 phoenix-gw 的 SSE 流转换为 Anthropic Messages SSE 事件序列。

    核心职责：
    - 维护 content block 的生命周期（start → 多次 delta → stop）
    - 同类型的连续 text/thinking delta 复用同一个 block，不重复 start/stop
    - tool_use block 每个工具独立（start → delta → stop）
    - 保证 message_start 在最前面，message_stop 在最后面
    """
    started = False
    next_block_index = 0
    # 当前活跃的 block 类型: None / "text" / "thinking"
    # tool_use 不保持活跃状态（每个工具独立关闭）
    active_block_type: str | None = None
    active_block_index: int = -1
    got_message_stop = False
    normalized_model = str(model or "").strip().lower()
    strict_wrapped_events = (
        normalized_model in SUPPORTED_ANTHROPIC_MODELS_SET
        or normalized_model.startswith("claude")
    )

    for raw_line in response.iter_lines(decode_unicode=True):
        line = (raw_line or "").strip()
        if not line or line.startswith(":"):
            continue

        json_text = line[5:].strip() if line.startswith("data:") else line
        if json_text == "[DONE]":
            continue

        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError:
            continue

        wrapped_raw = payload.get("raw_response_json") if isinstance(payload, dict) else None
        if strict_wrapped_events:
            if isinstance(payload, dict) and payload.get("turn_complete"):
                continue
            if wrapped_raw is None:
                continue

            # Claude 严格对齐 Worker 行为：
            # 只消费 raw_response_json 中包裹的 Anthropic 事件。
            raw_from_wrapped = _parse_raw_event(wrapped_raw)
            if not raw_from_wrapped or not raw_from_wrapped.get("type"):
                continue

            event_name = str(raw_from_wrapped["type"])
            if event_name == "message_stop":
                got_message_stop = True
            if not started and event_name != "message_start":
                started = True
                yield "message_start", _fallback_message_start_worker(model)
            if event_name == "message_start":
                started = True
                _ensure_message_start_fields_worker(raw_from_wrapped, model)
            yield event_name, raw_from_wrapped
            continue

        raw_from_wrapped = _parse_raw_event(wrapped_raw) if wrapped_raw is not None else None

        # 非 Claude 兼容路径继续保留更宽松的 fallback，
        # 兼容 Gemini / OpenAI 风格的包装载荷。
        raw_event = (
            raw_from_wrapped
            if raw_from_wrapped is not None
            else _parse_raw_event(payload)
        )
        if not raw_event:
            continue

        # 如果 payload 自身就是 Anthropic 原生格式
        if raw_event.get("type"):
            event_name = str(raw_event["type"])
            if event_name == "message_stop":
                got_message_stop = True
            if not started and event_name != "message_start":
                started = True
                yield "message_start", _fallback_message_start(model)
            if event_name == "message_start":
                started = True
                _ensure_message_start_fields(raw_event, model)
            yield event_name, raw_event
            continue

        # 上游是 Gemini/OpenAI 格式，需要提取内容片段
        fragments = _extract_content_fragments(raw_event)

        for frag in fragments:
            if not started:
                started = True
                yield "message_start", _fallback_message_start(model)

            frag_kind = frag["kind"]

            if frag_kind in ("text", "thinking"):
                # 同类型的连续 delta 复用同一个 block
                if active_block_type != frag_kind:
                    # 先关闭之前的活跃 block
                    if active_block_type is not None:
                        yield "content_block_stop", {
                            "type": "content_block_stop",
                            "index": active_block_index,
                        }
                    # 开启新的 block
                    active_block_type = frag_kind
                    active_block_index = next_block_index
                    next_block_index += 1
                    if frag_kind == "thinking":
                        yield "content_block_start", {
                            "type": "content_block_start",
                            "index": active_block_index,
                            "content_block": {"type": "thinking", "thinking": ""},
                        }
                    else:
                        yield "content_block_start", {
                            "type": "content_block_start",
                            "index": active_block_index,
                            "content_block": {"type": "text", "text": ""},
                        }

                # 发送 delta
                if frag_kind == "thinking":
                    yield "content_block_delta", {
                        "type": "content_block_delta",
                        "index": active_block_index,
                        "delta": {"type": "thinking_delta", "thinking": frag["text"]},
                    }
                else:
                    yield "content_block_delta", {
                        "type": "content_block_delta",
                        "index": active_block_index,
                        "delta": {"type": "text_delta", "text": frag["text"]},
                    }

            elif frag_kind == "signature":
                # signature 跟在 thinking block 后面
                if active_block_type == "thinking":
                    yield "content_block_delta", {
                        "type": "content_block_delta",
                        "index": active_block_index,
                        "delta": {"type": "signature_delta", "signature": frag["signature"]},
                    }

            elif frag_kind == "tool_use":
                # 先关闭活跃的 text/thinking block
                if active_block_type is not None:
                    yield "content_block_stop", {
                        "type": "content_block_stop",
                        "index": active_block_index,
                    }
                    active_block_type = None

                # tool_use 是独立的 block（start → delta → stop）
                tool_index = next_block_index
                next_block_index += 1
                yield "content_block_start", {
                    "type": "content_block_start",
                    "index": tool_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": frag["id"],
                        "name": frag["name"],
                        "input": {},
                    },
                }
                if frag.get("args_json"):
                    yield "content_block_delta", {
                        "type": "content_block_delta",
                        "index": tool_index,
                        "delta": {"type": "input_json_delta", "partial_json": frag["args_json"]},
                    }
                yield "content_block_stop", {
                    "type": "content_block_stop",
                    "index": tool_index,
                }

            elif frag_kind == "finish":
                # 先关闭活跃的 block
                if active_block_type is not None:
                    yield "content_block_stop", {
                        "type": "content_block_stop",
                        "index": active_block_index,
                    }
                    active_block_type = None

                msg_delta: dict[str, Any] = {"type": "message_delta"}
                if frag.get("stop_reason"):
                    msg_delta["delta"] = {
                        "stop_reason": frag["stop_reason"],
                        "stop_sequence": None,
                    }
                if frag.get("usage"):
                    msg_delta["usage"] = frag["usage"]
                yield "message_delta", msg_delta

    # 关闭最后一个活跃的 block
    if active_block_type is not None:
        yield "content_block_stop", {
            "type": "content_block_stop",
            "index": active_block_index,
        }

    if started and not got_message_stop:
        yield "message_stop", {"type": "message_stop"}


def iter_anthropic_sse_bytes(
    response: requests.Response,
    model: str,
    on_complete: Callable[[dict[str, Any]], None] | None = None,
) -> Iterator[bytes]:
    summary = {
        "model": model,
        "usage": _usage_summary(),
        "stop_reason": "end_turn",
        "content_blocks": 0,
        "text_chars": 0,
        "thinking_chars": 0,
        "tool_use_blocks": 0,
        "tool_json_chars": 0,
    }
    completed = False
    try:
        for event_name, event_payload in iter_anthropic_sse_events(response, model):
            update_usage_summary(summary, event_name, event_payload)
            yield format_sse(event_name, event_payload).encode("utf-8")
        completed = True
    finally:
        response.close()
        if completed and on_complete is not None:
            on_complete(summary)


def decode_non_stream_response(
    response: requests.Response,
    model: str,
) -> dict[str, Any]:
    try:
        events = list(iter_anthropic_sse_events(response, model))
    finally:
        response.close()

    content: list[dict[str, Any]] = []
    summary = {
        "model": model,
        "usage": _usage_summary(),
        "stop_reason": "end_turn",
        "content_blocks": 0,
        "text_chars": 0,
        "thinking_chars": 0,
        "tool_use_blocks": 0,
        "tool_json_chars": 0,
    }

    for event_name, event_payload in events:
        update_usage_summary(summary, event_name, event_payload)
        if event_name == "content_block_start":
            block = event_payload.get("content_block")
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                content.append({"type": "text", "text": ""})
            elif block_type == "thinking":
                content.append({"type": "thinking", "thinking": "", "signature": ""})
            elif block_type == "tool_use":
                content.append(
                    {
                        "type": "tool_use",
                        "id": str(block.get("id") or ""),
                        "name": str(block.get("name") or ""),
                        "input": {},
                    }
                )
        elif event_name == "content_block_delta":
            if not content:
                continue
            delta = event_payload.get("delta")
            if not isinstance(delta, dict):
                continue
            last_block = content[-1]
            if delta.get("text") is not None:
                last_block["text"] = str(last_block.get("text") or "") + str(delta["text"])
            if delta.get("thinking") is not None:
                last_block["thinking"] = str(last_block.get("thinking") or "") + str(
                    delta["thinking"]
                )
            if delta.get("signature") is not None:
                last_block["signature"] = str(delta["signature"] or "")
            if delta.get("partial_json") is not None:
                last_block["_input"] = str(last_block.get("_input") or "") + str(
                    delta["partial_json"]
                )
        elif event_name == "content_block_stop":
            if not content:
                continue
            last_block = content[-1]
            if last_block.get("type") == "tool_use" and last_block.get("_input"):
                try:
                    last_block["input"] = json.loads(str(last_block["_input"]))
                except json.JSONDecodeError:
                    pass
                last_block.pop("_input", None)
    for block in content:
        block.pop("_input", None)
        if block.get("type") == "thinking" and not block.get("signature"):
            block.pop("signature", None)

    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": summary["stop_reason"],
        "stop_sequence": None,
        "usage": summary["usage"],
    }


def format_sse(event_name: str, payload: dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def extract_tool_result_text(content: Any) -> str:
    value = unwrap_tool_result_content(content)
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def unwrap_tool_result_content(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return value
    if value is None:
        return ""

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            return unwrap_tool_result_content(json.loads(stripped), depth + 1)
        except json.JSONDecodeError:
            return value

    if isinstance(value, list):
        text_blocks = [item for item in value if is_text_block(item)]
        if text_blocks:
            merged_text = "\n".join(str(item.get("text") or "") for item in text_blocks)
            return unwrap_tool_result_content(merged_text, depth + 1)
        if len(value) == 1:
            return unwrap_tool_result_content(value[0], depth + 1)
        return value

    if isinstance(value, dict):
        output = value.get("output")
        if isinstance(output, dict) and (
            isinstance(output.get("content"), list)
            or value.get("toolCallId") is not None
            or value.get("input") is not None
        ):
            return unwrap_tool_result_content(output, depth + 1)

        content_value = value.get("content")
        if isinstance(content_value, list) and all(
            is_content_block_like(item) for item in content_value
        ):
            return unwrap_tool_result_content(content_value, depth + 1)

        if value.get("type") == "text" and isinstance(value.get("text"), str):
            return unwrap_tool_result_content(value["text"], depth + 1)

        return value

    return value


def is_text_block(block: Any) -> bool:
    return (
        isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )


def is_content_block_like(block: Any) -> bool:
    return isinstance(block, dict) and isinstance(block.get("type"), str)


def find_tool_info(tool_use_id: Any, messages: list[Any]) -> dict[str, str] | None:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        content_blocks = message.get("content")
        if not isinstance(content_blocks, list):
            continue

        thought_signature: str | None = None
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "thinking" and block.get("signature"):
                thought_signature = str(block["signature"])
            if block.get("type") == "tool_use" and block.get("id") == tool_use_id:
                payload = {"name": str(block.get("name") or "unknown")}
                if thought_signature:
                    payload["thoughtSignature"] = thought_signature
                return payload
    return None


def find_last_signature(contents: list[dict[str, Any]]) -> str | None:
    for content in reversed(contents):
        if content.get("role") != "model":
            continue
        metadata = content.get("metadata")
        if isinstance(metadata, dict) and metadata.get("textThoughtSignature"):
            return str(metadata["textThoughtSignature"])
    return None


def sanitize_tool_call_id(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(value or ""))


def _extract_system_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("text") is not None:
                parts.append(str(item.get("text") or ""))
        return "\n".join(part for part in parts if part)
    return ""


def _parse_raw_event(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _ensure_message_start_fields(event_payload: dict[str, Any], model: str) -> None:
    """补全 message_start 事件中可能缺少的字段。"""
    message = event_payload.get("message")
    if isinstance(message, dict):
        message.setdefault("id", f"msg_{uuid.uuid4().hex}")
        message.setdefault("type", "message")
        message.setdefault("role", "assistant")
        message.setdefault("content", [])
        message["model"] = model
        message.setdefault("stop_reason", None)
        message.setdefault("stop_sequence", None)
        usage = message.get("usage")
        if not isinstance(usage, dict):
            message["usage"] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            }
        else:
            usage.setdefault("input_tokens", 0)
            usage.setdefault("output_tokens", 0)
            usage.setdefault("cache_creation_input_tokens", 0)
            usage.setdefault("cache_read_input_tokens", 0)


def _ensure_message_start_fields_worker(event_payload: dict[str, Any], model: str) -> None:
    message = event_payload.get("message")
    if isinstance(message, dict):
        message["model"] = model


def _fallback_message_start(model: str) -> dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
    }


def _fallback_message_start_worker(model: str) -> dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
            },
        },
    }


def _map_vendor_finish_reason(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return None
    if normalized in {"stop", "stop_sequence", "end_turn"}:
        return "end_turn"
    if normalized in {"max_tokens", "length"}:
        return "max_tokens"
    if normalized in {"tool_use", "tool_calls", "function_call"}:
        return "tool_use"
    if normalized == "content_filter":
        return "content_filter"
    return None


def _usage_from_openai_payload(payload: dict[str, Any]) -> dict[str, int] | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_tokens = _as_usage_int(usage.get("prompt_tokens"))
    completion_tokens = _as_usage_int(usage.get("completion_tokens"))
    total_tokens = _as_usage_int(usage.get("total_tokens"))
    if prompt_tokens <= 0 and completion_tokens <= 0 and total_tokens <= 0:
        return None
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
    }


def _usage_from_gemini_payload(payload: dict[str, Any]) -> dict[str, int] | None:
    usage = payload.get("usageMetadata", payload.get("usage_metadata"))
    if not isinstance(usage, dict):
        return None
    prompt_tokens = _as_usage_int(
        usage.get("promptTokenCount", usage.get("prompt_token_count"))
    )
    candidate_tokens = _as_usage_int(
        usage.get("candidatesTokenCount", usage.get("candidates_token_count"))
    )
    thoughts_tokens = _as_usage_int(
        usage.get("thoughtsTokenCount", usage.get("thoughts_token_count"))
    )
    if prompt_tokens <= 0 and candidate_tokens <= 0 and thoughts_tokens <= 0:
        return None
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": candidate_tokens + thoughts_tokens,
    }


def _extract_content_fragments(
    raw_event: dict[str, Any],
) -> list[dict[str, Any]]:
    """从上游 Gemini/OpenAI 格式的 payload 中提取内容片段。

    返回一个 fragment 列表，每个 fragment 是一个 dict：
    - {"kind": "text", "text": "..."}
    - {"kind": "thinking", "text": "..."}
    - {"kind": "signature", "signature": "..."}
    - {"kind": "tool_use", "id": "...", "name": "...", "args_json": "..."}
    - {"kind": "finish", "stop_reason": "...", "usage": {...}}
    """
    fragments: list[dict[str, Any]] = []

    # --- Gemini 格式: candidates[].content.parts[] ---
    candidates = raw_event.get("candidates")
    if isinstance(candidates, list) and candidates:
        candidate = candidates[0] if isinstance(candidates[0], dict) else {}
        content = candidate.get("content") if isinstance(candidate, dict) else {}
        if not isinstance(content, dict):
            content = {}
        parts = content.get("parts")
        if not isinstance(parts, list):
            parts = []

        for part in parts:
            if not isinstance(part, dict):
                continue

            text = part.get("text")
            if text is not None:
                kind = "thinking" if bool(part.get("thought")) else "text"
                fragments.append({"kind": kind, "text": str(text or "")})
                thought_signature = part.get(
                    "thoughtSignature",
                    part.get("thought_signature"),
                )
                if thought_signature:
                    fragments.append({"kind": "signature", "signature": str(thought_signature)})
                continue

            function_call = part.get("functionCall", part.get("function_call"))
            if isinstance(function_call, dict):
                tool_id = str(
                    function_call.get("id")
                    or function_call.get("callId")
                    or function_call.get("name")
                    or uuid.uuid4().hex
                )
                tool_name = str(function_call.get("name") or "")
                args_value = function_call.get("args")
                if args_value is None:
                    args_value = function_call.get("argsJson")
                args_json = (
                    args_value
                    if isinstance(args_value, str)
                    else json.dumps(args_value or {}, ensure_ascii=False)
                )
                fragments.append({
                    "kind": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "args_json": args_json or "",
                })

        finish_reason = _map_vendor_finish_reason(
            candidate.get("finishReason", raw_event.get("finishReason"))
            if isinstance(candidate, dict)
            else raw_event.get("finishReason")
        )
        usage = _usage_from_gemini_payload(raw_event)
        if finish_reason or usage:
            frag: dict[str, Any] = {"kind": "finish"}
            if finish_reason:
                frag["stop_reason"] = finish_reason
            if usage:
                frag["usage"] = usage
            fragments.append(frag)

        return fragments

    # --- OpenAI 格式: choices[].delta / choices[].message ---
    choices = raw_event.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta") if isinstance(choice, dict) else {}
        if not isinstance(delta, dict):
            delta = {}
        message = choice.get("message") if isinstance(choice, dict) else {}
        if not isinstance(message, dict):
            message = {}

        text = delta.get("content")
        if text is None:
            text = message.get("content")
        if text is not None:
            fragments.append({"kind": "text", "text": str(text or "")})

        tool_calls = delta.get("tool_calls")
        if not isinstance(tool_calls, list):
            tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                tool_name = str(function.get("name") or "").strip()
                if not tool_name:
                    continue
                tool_id = str(tool_call.get("id") or uuid.uuid4().hex)
                args_value = function.get("arguments")
                args_json = (
                    args_value
                    if isinstance(args_value, str)
                    else json.dumps(args_value or {}, ensure_ascii=False)
                )
                fragments.append({
                    "kind": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "args_json": args_json or "",
                })

        finish_reason = _map_vendor_finish_reason(
            choice.get("finish_reason") if isinstance(choice, dict) else None
        )
        usage = _usage_from_openai_payload(raw_event)
        if finish_reason or usage:
            frag = {"kind": "finish"}
            if finish_reason:
                frag["stop_reason"] = finish_reason
            if usage:
                frag["usage"] = usage
            fragments.append(frag)

    return fragments
