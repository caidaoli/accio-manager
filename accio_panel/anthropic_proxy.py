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
DEFAULT_ANTHROPIC_MODEL = SUPPORTED_ANTHROPIC_MODELS[0]


def _usage_summary() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
    }


def build_models_payload() -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "owned_by": "anthropic",
            }
            for model_name in SUPPORTED_ANTHROPIC_MODELS
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
        "stream": True,
        "model": body.get("model") or DEFAULT_ANTHROPIC_MODEL,
        "request_id": f"req-{uuid.uuid4()}",
        "incremental": True,
        "max_output_tokens": body.get("max_tokens") or 8192,
        "contents": [],
    }

    system_value = body.get("system")
    system_text = _extract_system_text(system_value)
    if system_text:
        request_body["system_instruction"] = system_text

    if body.get("temperature") is not None:
        request_body["temperature"] = body.get("temperature")

    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        request_body["include_thoughts"] = True
        request_body["thinking_level"] = "high"
        if thinking.get("budget_tokens") is not None:
            request_body["thinking_budget"] = thinking.get("budget_tokens")

    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        request_body["tools"] = [
            {
                "name": str(tool.get("name") or ""),
                "description": str(tool.get("description") or ""),
                "parametersJson": json.dumps(tool.get("input_schema") or {}),
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
                    if block_type == "text":
                        parts.append(
                            {
                                "text": str(block.get("text") or ""),
                                "thought": False,
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
                    elif block_type == "tool_use":
                        parts.append(
                            {
                                "thought": False,
                                "functionCall": {
                                    "id": sanitize_tool_call_id(block.get("id")),
                                    "name": str(block.get("name") or ""),
                                    "argsJson": json.dumps(block.get("input") or {}),
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
                elif block_type == "tool_result":
                    tool_info = find_tool_info(block.get("tool_use_id"), messages)
                    function_response: dict[str, Any] = {
                        "id": sanitize_tool_call_id(block.get("tool_use_id")),
                        "name": (
                            str(tool_info.get("name"))
                            if tool_info and tool_info.get("name")
                            else str(block.get("name") or "unknown")
                        ),
                        "responseJson": json.dumps(
                            {
                                "content": extract_tool_result_text(block.get("content")),
                                "is_error": bool(block.get("is_error", False)),
                            }
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

    delta = event_payload.get("delta")
    if isinstance(delta, dict) and delta.get("stop_reason"):
        summary["stop_reason"] = str(delta["stop_reason"])


def iter_anthropic_sse_events(
    response: requests.Response,
    model: str,
) -> Iterator[tuple[str, dict[str, Any]]]:
    started = False

    for raw_line in response.iter_lines(decode_unicode=True):
        line = (raw_line or "").strip()
        if not line:
            continue

        json_text = line[5:].strip() if line.startswith("data:") else line
        if json_text == "[DONE]":
            continue

        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError:
            continue

        if isinstance(payload, dict) and payload.get("turn_complete"):
            continue

        wrapped_raw = payload.get("raw_response_json") if isinstance(payload, dict) else None
        raw_event = _parse_raw_event(wrapped_raw if wrapped_raw is not None else payload)
        if not raw_event:
            continue

        event_type = str(raw_event.get("type") or "message_delta")
        if not started and event_type != "message_start":
            started = True
            yield "message_start", _fallback_message_start(model)

        if event_type == "message_start":
            started = True
            message = raw_event.get("message")
            if isinstance(message, dict):
                message["model"] = model

        yield event_type, raw_event


def iter_anthropic_sse_bytes(
    response: requests.Response,
    model: str,
    on_complete: Callable[[dict[str, Any]], None] | None = None,
) -> Iterator[bytes]:
    summary = {
        "model": model,
        "usage": _usage_summary(),
        "stop_reason": "end_turn",
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
                content.append({"type": "thinking", "thinking": ""})
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

    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": summary["stop_reason"],
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
            },
        },
    }
