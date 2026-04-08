from __future__ import annotations

import json
import time
import uuid
from typing import Any
from typing import Callable
from typing import Iterator

import requests

from .anthropic_proxy import (
    build_accio_request,
    decode_non_stream_response,
    iter_anthropic_sse_events,
    update_usage_summary,
)


def openai_error_payload(
    message: str,
    *,
    error_type: str = "invalid_request_error",
    code: str | None = None,
) -> dict[str, Any]:
    error = {
        "message": message,
        "type": error_type,
        "param": None,
        "code": code,
    }
    return {"error": error}


def _stringify_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value or {}, ensure_ascii=False)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _extract_text_blocks(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if not isinstance(value, list):
        return []

    blocks: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        block_type = str(item.get("type") or "").strip().lower()
        if block_type in {"text", "input_text", "output_text"}:
            text = item.get("text")
            if text is not None:
                blocks.append({"type": "text", "text": str(text)})
    return blocks


def _parse_data_url(value: str) -> tuple[str, str] | None:
    if not value.startswith("data:"):
        return None
    marker = ";base64,"
    if marker not in value:
        return None
    header, data = value.split(marker, 1)
    mime_type = header[5:] or "application/octet-stream"
    if not data:
        return None
    return mime_type, data


def _guess_image_mime_type(value: str) -> str:
    lowered = value.lower()
    if lowered.endswith(".jpg") or lowered.endswith(".jpeg"):
        return "image/jpeg"
    if lowered.endswith(".webp"):
        return "image/webp"
    if lowered.endswith(".gif"):
        return "image/gif"
    return "image/png"


def _convert_openai_user_content(value: Any) -> str | list[dict[str, Any]]:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""

    blocks: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        block_type = str(item.get("type") or "").strip().lower()
        if block_type in {"text", "input_text", "output_text"}:
            text = item.get("text")
            if text is not None:
                blocks.append({"type": "text", "text": str(text)})
            continue

        if block_type in {"image_url", "input_image"}:
            image_value = item.get("image_url")
            image_url = ""
            media_type = ""
            if isinstance(image_value, dict):
                image_url = str(image_value.get("url") or "")
                media_type = str(
                    image_value.get("mime_type")
                    or image_value.get("media_type")
                    or ""
                )
            elif image_value is not None:
                image_url = str(image_value)
            elif item.get("url") is not None:
                image_url = str(item.get("url") or "")

            parsed = _parse_data_url(image_url)
            if parsed:
                mime_type, data = parsed
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": data,
                        },
                    }
                )
                continue

            if image_url:
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url,
                            "media_type": media_type or _guess_image_mime_type(image_url),
                        },
                    }
                )

    return blocks if blocks else ""


def _convert_openai_assistant_content(message: dict[str, Any]) -> str | list[dict[str, Any]]:
    blocks = _extract_text_blocks(message.get("content"))

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            blocks.append(
                {
                    "type": "tool_use",
                    "id": str(tool_call.get("id") or uuid.uuid4().hex),
                    "name": name,
                    "input": _parse_json_dict(function.get("arguments")),
                }
            )

    function_call = message.get("function_call")
    if isinstance(function_call, dict):
        name = str(function_call.get("name") or "").strip()
        if name:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": str(function_call.get("id") or uuid.uuid4().hex),
                    "name": name,
                    "input": _parse_json_dict(function_call.get("arguments")),
                }
            )

    return blocks if blocks else ""


def _convert_tool_result_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        text_blocks = _extract_text_blocks(value)
        if text_blocks:
            return "\n".join(str(block.get("text") or "") for block in text_blocks)
    return json.dumps(value, ensure_ascii=False)


def _normalize_responses_message_content(value: dict[str, Any]) -> str | list[dict[str, Any]]:
    content = value.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        normalized: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                normalized.append({"type": "text", "text": item})
                continue
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip().lower()
            if item_type in {"text", "input_text", "output_text"}:
                normalized.append({"type": item_type, "text": str(item.get("text") or "")})
                continue
            if item_type in {
                "image",
                "input_image",
                "image_url",
                "file",
                "input_file",
                "tool_use",
                "tool_call",
                "function_call",
                "tool_result",
                "function_call_output",
                "refusal",
            }:
                normalized.append(dict(item))
                continue
            if item.get("text") is not None:
                normalized.append({"type": "text", "text": str(item.get("text") or "")})
                continue
            normalized.append({"type": "text", "text": _json_text(item)})
        return normalized
    if isinstance(content, dict):
        return _normalize_responses_message_content({"content": [content]})
    text = value.get("text")
    if text is not None:
        return [{"type": "text", "text": str(text)}]
    return ""


def _normalize_responses_user_block(value: dict[str, Any]) -> dict[str, Any]:
    block_type = str(value.get("type") or "").strip().lower()
    if block_type in {"text", "input_text", "output_text"}:
        return {"type": "input_text", "text": str(value.get("text") or "")}
    if block_type in {"input_image", "image_url", "image"}:
        return dict(value)
    if block_type in {"input_file", "file"}:
        file_data = value.get("file_data", value.get("fileData"))
        if isinstance(file_data, dict):
            mime_type = str(
                file_data.get("mime_type")
                or file_data.get("mimeType")
                or ""
            )
            if mime_type.startswith("image/"):
                return {
                    "type": "input_image",
                    "file_data": dict(file_data),
                }
        return {
            "type": "input_text",
            "text": _json_text(value),
        }
    if value.get("image_url") is not None or value.get("url") is not None:
        return dict(value) | {"type": "input_image"}
    return {"type": "input_text", "text": _json_text(value)}


def _convert_responses_message_item(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": str(value.get("role") or "user").strip().lower() or "user",
        "content": _normalize_responses_message_content(value),
    }


def _convert_responses_tool_call_item(value: dict[str, Any]) -> dict[str, Any] | None:
    name = str(value.get("name") or value.get("tool_name") or "").strip()
    if not name:
        return None
    call_id = str(
        value.get("call_id") or value.get("tool_call_id") or value.get("id") or uuid.uuid4().hex
    )
    arguments = value.get("arguments", value.get("input"))
    if arguments is None:
        arguments = value.get("arguments_json", value.get("argumentsJson"))
    return {
        "role": "assistant",
        "content": _normalize_responses_message_content(value),
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _stringify_json(arguments or {}),
                },
            }
        ],
    }


def _convert_responses_tool_result_item(value: dict[str, Any]) -> dict[str, Any] | None:
    tool_call_id = str(
        value.get("call_id") or value.get("tool_call_id") or value.get("id") or ""
    ).strip()
    if not tool_call_id:
        return None
    output = value.get("output", value.get("content", value.get("result")))
    if output is None:
        output = value.get("text")
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": _convert_tool_result_content(output),
        "is_error": bool(value.get("is_error", False)),
    }


def _convert_openai_user_content_v2(value: Any) -> str | list[dict[str, Any]]:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""

    blocks: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue

        block_type = str(item.get("type") or "").strip().lower()
        if block_type in {"text", "input_text", "output_text"}:
            text = item.get("text")
            if text is not None:
                blocks.append({"type": "text", "text": str(text)})
            continue

        if block_type == "refusal":
            refusal = item.get("refusal")
            if refusal is not None:
                blocks.append({"type": "text", "text": str(refusal)})
            continue

        if block_type in {"image_url", "input_image", "image", "input_file", "file"}:
            image_url = ""
            media_type = ""
            image_value = item.get("image_url")
            if isinstance(image_value, dict):
                image_url = str(image_value.get("url") or "")
                media_type = str(
                    image_value.get("mime_type")
                    or image_value.get("media_type")
                    or ""
                )
            elif image_value is not None:
                image_url = str(image_value)
            elif item.get("url") is not None:
                image_url = str(item.get("url") or "")

            file_data = item.get("file_data", item.get("fileData"))
            if isinstance(file_data, dict):
                image_url = str(
                    file_data.get("file_uri")
                    or file_data.get("fileUri")
                    or file_data.get("url")
                    or image_url
                )
                media_type = str(
                    file_data.get("mime_type")
                    or file_data.get("mimeType")
                    or media_type
                    or ""
                )
                data = file_data.get("data")
                if data is not None and (media_type.startswith("image/") or block_type != "input_file"):
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type or "image/png",
                                "data": str(data or ""),
                            },
                        }
                    )
                    continue

            source = item.get("source")
            if isinstance(source, dict):
                source_type = str(source.get("type") or "").strip().lower()
                if source_type == "base64" and source.get("data") is not None:
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": str(
                                    source.get("media_type")
                                    or source.get("mime_type")
                                    or media_type
                                    or "image/png"
                                ),
                                "data": str(source.get("data") or ""),
                            },
                        }
                    )
                    continue
                if source_type == "url" and source.get("url") is not None:
                    image_url = str(source.get("url") or image_url)
                    media_type = str(
                        source.get("media_type")
                        or source.get("mime_type")
                        or media_type
                        or ""
                    )

            parsed = _parse_data_url(image_url)
            if parsed:
                mime_type, data = parsed
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": data,
                        },
                    }
                )
                continue

            if image_url and (media_type.startswith("image/") or block_type not in {"input_file", "file"}):
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url,
                            "media_type": media_type or _guess_image_mime_type(image_url),
                        },
                    }
                )
                continue

            filename = str(
                item.get("filename")
                or item.get("file_name")
                or item.get("name")
                or image_url
                or ""
            ).strip()
            if filename:
                blocks.append({"type": "text", "text": f"[file] {filename}"})
            continue

        if item:
            blocks.append({"type": "text", "text": _json_text(item)})

    if not blocks:
        return ""
    if len(blocks) == 1 and blocks[0]["type"] == "text":
        return str(blocks[0].get("text") or "")
    return blocks


def _tool_call_from_content_block(block: dict[str, Any]) -> dict[str, Any] | None:
    block_type = str(block.get("type") or "").strip().lower()
    if block_type not in {"tool_use", "tool_call", "function_call"}:
        return None
    function_payload = block.get("function")
    if isinstance(function_payload, dict):
        name_value = (
            function_payload.get("name")
            or block.get("name")
            or block.get("tool_name")
            or block.get("function_name")
            or ""
        )
        arguments = function_payload.get("arguments", function_payload.get("arguments_json"))
    else:
        name_value = (
            block.get("name")
            or block.get("tool_name")
            or block.get("function_name")
            or ""
        )
        arguments = block.get("arguments", block.get("input"))
    name = str(name_value).strip()
    if not name:
        return None
    tool_call_id = str(
        block.get("id")
        or block.get("call_id")
        or block.get("tool_call_id")
        or uuid.uuid4().hex
    )
    if arguments is None:
        arguments = block.get("arguments_json", block.get("argumentsJson"))
    return {
        "type": "tool_use",
        "id": tool_call_id,
        "name": name,
        "input": _parse_json_dict(arguments),
    }


def _convert_openai_assistant_content_v2(message: dict[str, Any]) -> str | list[dict[str, Any]]:
    content = message.get("content")
    blocks = _extract_text_blocks(content)

    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            block_type = str(item.get("type") or "").strip().lower()
            if block_type == "refusal":
                refusal = item.get("refusal")
                if refusal is not None:
                    blocks.append({"type": "text", "text": str(refusal)})
                continue
            if block_type in {"image", "input_image", "image_url", "input_file", "file"}:
                image_blocks = _convert_openai_user_content_v2([item])
                if isinstance(image_blocks, str):
                    if image_blocks:
                        blocks.append({"type": "text", "text": image_blocks})
                elif isinstance(image_blocks, list):
                    blocks.extend(image_blocks)
                continue
            tool_call = _tool_call_from_content_block(item)
            if tool_call:
                blocks.append(tool_call)
                continue
            blocks.append({"type": "text", "text": _json_text(item)})

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            blocks.append(
                {
                    "type": "tool_use",
                    "id": str(tool_call.get("id") or uuid.uuid4().hex),
                    "name": name,
                    "input": _parse_json_dict(function.get("arguments")),
                }
            )

    function_call = message.get("function_call")
    if isinstance(function_call, dict):
        name = str(function_call.get("name") or "").strip()
        if name:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": str(function_call.get("id") or uuid.uuid4().hex),
                    "name": name,
                    "input": _parse_json_dict(
                        function_call.get("arguments", function_call.get("arguments_json"))
                    ),
                }
            )

    return blocks if blocks else ""


def _convert_openai_messages(body: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return "", []

    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").strip().lower()
        if role in {"system", "developer"}:
            system_blocks = _extract_text_blocks(message.get("content"))
            if system_blocks:
                system_parts.append(
                    "\n".join(str(block.get("text") or "") for block in system_blocks)
                )
            continue

        if role == "assistant":
            converted.append(
                {
                    "role": "assistant",
                    "content": _convert_openai_assistant_content_v2(message),
                }
            )
            continue

        if role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "").strip()
            if not tool_call_id:
                continue
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": _convert_tool_result_content(message.get("content")),
                            "is_error": bool(message.get("is_error", False)),
                        }
                    ],
                }
            )
            continue

        converted.append(
            {
                "role": "user",
                "content": _convert_openai_user_content_v2(message.get("content")),
            }
        )

    return "\n".join(part for part in system_parts if part), converted


def _convert_openai_tools(body: dict[str, Any]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []

    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if str(tool.get("type") or "").strip() != "function":
                continue
            function = tool.get("function")
            if isinstance(function, dict):
                function_payload = function
            else:
                function_payload = tool
            name = str(function_payload.get("name") or "").strip()
            if not name:
                continue
            converted.append(
                {
                    "name": name,
                    "description": str(function_payload.get("description") or ""),
                    "input_schema": function_payload.get("parameters")
                    or function_payload.get("input_schema")
                    or {},
                }
            )

    if converted:
        return converted

    functions = body.get("functions")
    if isinstance(functions, list):
        for function in functions:
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            converted.append(
                {
                    "name": name,
                    "description": str(function.get("description") or ""),
                    "input_schema": function.get("parameters") or {},
                }
            )

    return converted


def build_accio_request_from_openai(
    body: dict[str, Any],
    *,
    token: str,
    utdid: str,
    version: str,
) -> dict[str, Any]:
    system_text, messages = _convert_openai_messages(body)
    anthropic_body: dict[str, Any] = {
        "model": body.get("model"),
        "messages": messages,
        "max_tokens": body.get(
            "max_completion_tokens",
            body.get("max_tokens", 8192),
        ),
    }
    if system_text:
        anthropic_body["system"] = system_text

    tools = _convert_openai_tools(body)
    if tools:
        anthropic_body["tools"] = tools

    for source_key, target_key in (
        ("request_id", "request_id"),
        ("requestId", "request_id"),
        ("message_id", "message_id"),
        ("messageId", "message_id"),
        ("session_key", "session_key"),
        ("sessionKey", "session_key"),
        ("session_id", "session_key"),
        ("sessionId", "session_key"),
        ("conversation_id", "conversation_id"),
        ("conversationId", "conversation_id"),
        ("conversation_name", "conversation_name"),
        ("conversationName", "conversation_name"),
    ):
        value = body.get(source_key)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        anthropic_body[target_key] = value

    return build_accio_request(
        anthropic_body,
        token=token,
        utdid=utdid,
        version=version,
    )


def convert_responses_input_to_messages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return [{"role": "user", "content": value}]

    if isinstance(value, dict):
        item_type = str(value.get("type") or "").strip().lower()
        if item_type in {"function_call", "tool_call"}:
            message = _convert_responses_tool_call_item(value)
            return [message] if message else []
        if item_type in {"function_call_output", "tool_result"}:
            message = _convert_responses_tool_result_item(value)
            return [message] if message else []
        if value.get("type") == "message" or value.get("role") is not None:
            return [_convert_responses_message_item(value)]
        if value.get("content") is not None or value.get("text") is not None:
            return [
                _convert_responses_message_item(
                    {
                        "role": "user",
                        "content": value.get("content"),
                        "text": value.get("text"),
                    }
                )
            ]
        if (
            item_type
            or value.get("image_url") is not None
            or value.get("file_data") is not None
            or value.get("fileData") is not None
        ):
            return [{"role": "user", "content": [_normalize_responses_user_block(value)]}]
        return [{"role": "user", "content": json.dumps(value, ensure_ascii=False)}]

    if not isinstance(value, list):
        return []

    messages: list[dict[str, Any]] = []
    pending_user_blocks: list[dict[str, Any]] = []

    def flush_pending_user_blocks() -> None:
        if not pending_user_blocks:
            return
        messages.append({"role": "user", "content": list(pending_user_blocks)})
        pending_user_blocks.clear()

    for item in value:
        if isinstance(item, str):
            pending_user_blocks.append({"type": "input_text", "text": item})
            continue
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip().lower()
        if item_type in {"function_call", "tool_call"}:
            flush_pending_user_blocks()
            message = _convert_responses_tool_call_item(item)
            if message:
                messages.append(message)
            continue
        if item_type in {"function_call_output", "tool_result"}:
            flush_pending_user_blocks()
            message = _convert_responses_tool_result_item(item)
            if message:
                messages.append(message)
            continue
        if item_type == "message" or item.get("role") is not None:
            flush_pending_user_blocks()
            messages.append(_convert_responses_message_item(item))
            continue
        if item.get("content") is not None or item.get("text") is not None:
            flush_pending_user_blocks()
            messages.append(
                _convert_responses_message_item(
                    {
                        "role": "user",
                        "content": item.get("content"),
                        "text": item.get("text"),
                    }
                )
            )
            continue
        pending_user_blocks.append(_normalize_responses_user_block(item))

    flush_pending_user_blocks()
    return messages


def build_openai_chat_payload_from_responses(body: dict[str, Any]) -> dict[str, Any]:
    messages = convert_responses_input_to_messages(body.get("input"))
    instructions = body.get("instructions")
    if instructions:
        messages = [{"role": "system", "content": instructions}, *messages]

    return {
        "model": body.get("model"),
        "messages": messages,
        "tools": body.get("tools"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "max_tokens": body.get("max_output_tokens", body.get("max_tokens")),
        "stop": body.get("stop"),
        "response_format": body.get("response_format"),
        "request_id": body.get("request_id", body.get("requestId")),
        "message_id": body.get("message_id", body.get("messageId")),
        "user": body.get("user"),
        "metadata": body.get("metadata"),
        "session_id": body.get("session_id", body.get("sessionId")),
        "conversation_id": body.get("conversation_id", body.get("conversationId")),
        "tool_choice": body.get("tool_choice"),
        "parallel_tool_calls": body.get("parallel_tool_calls"),
        "store": body.get("store"),
        "reasoning": body.get("reasoning"),
        "text": body.get("text"),
        "previous_response_id": body.get(
            "previous_response_id",
            body.get("previousResponseId"),
        ),
        "truncation": body.get("truncation"),
        "include": body.get("include"),
    }


def _map_finish_reason(stop_reason: str, has_tool_calls: bool) -> str:
    normalized = str(stop_reason or "").strip().lower()
    if has_tool_calls or normalized in {"tool_use", "function_call"}:
        return "tool_calls"
    if normalized in {"max_tokens", "length"}:
        return "length"
    if normalized == "content_filter":
        return "content_filter"
    return "stop"


def _extract_openai_message(payload: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    content = payload.get("content")
    if not isinstance(content, list):
        content = []

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "").strip().lower()
        if block_type in {"text", "output_text"}:
            text = str(block.get("text") or "")
            if text:
                text_parts.append(text)
            continue
        if block_type == "refusal":
            refusal = str(block.get("refusal") or "")
            if refusal:
                text_parts.append(refusal)
            continue
        if block_type not in {"tool_use", "tool_call", "function_call"}:
            continue
        arguments = block.get("input")
        if arguments is None:
            arguments = block.get("arguments", block.get("arguments_json"))
        tool_calls.append(
            {
                "id": str(block.get("id") or uuid.uuid4().hex),
                "type": "function",
                "function": {
                    "name": str(block.get("name") or block.get("tool_name") or ""),
                    "arguments": _stringify_json(arguments or {}),
                },
            }
        )

    text_content = "".join(text_parts)
    if tool_calls and not text_content:
        return None, tool_calls
    return text_content, tool_calls


def build_openai_chat_completion_response(
    payload: dict[str, Any],
    *,
    model: str,
    accio: dict[str, Any] | None = None,
) -> dict[str, Any]:
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    content_text, tool_calls = _extract_openai_message(payload)
    finish_reason = _map_finish_reason(
        str(payload.get("stop_reason") or "end_turn"),
        bool(tool_calls),
    )
    response_payload = {
        "id": f"chatcmpl_{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content_text,
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": int(usage.get("input_tokens") or 0),
            "completion_tokens": int(usage.get("output_tokens") or 0),
            "total_tokens": int(usage.get("input_tokens") or 0)
            + int(usage.get("output_tokens") or 0),
        },
    }
    if isinstance(accio, dict) and accio:
        response_payload["accio"] = dict(accio)
    return response_payload


def build_openai_responses_response(
    payload: dict[str, Any],
    *,
    model: str,
    response_id: str | None = None,
    message_id: str | None = None,
    created_at: int | None = None,
    status: str = "completed",
    accio: dict[str, Any] | None = None,
) -> dict[str, Any]:
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    output_text, tool_calls = _extract_openai_message(payload)
    output_items: list[dict[str, Any]] = []
    response_message_id = message_id or f"msg_{uuid.uuid4().hex}"

    if output_text is not None or not tool_calls:
        output_items.append(
            {
                "id": response_message_id,
                "type": "message",
                "role": "assistant",
                "status": status,
                "content": [
                    {
                        "type": "output_text",
                        "text": output_text or "",
                        "annotations": [],
                    }
                ],
            }
        )

    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            function = {}
        output_items.append(
            {
                "id": str(tool_call.get("id") or uuid.uuid4().hex),
                "type": "tool_call",
                "name": str(function.get("name") or ""),
                "arguments": str(function.get("arguments") or "{}"),
                "status": status,
            }
        )

    prompt_tokens = int(usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or 0)
    response_payload = {
        "id": response_id or f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(created_at or time.time()),
        "status": status,
        "model": model,
        "output": output_items,
        "output_text": output_text or "",
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    if isinstance(accio, dict) and accio:
        response_payload["accio"] = dict(accio)
    return response_payload


def decode_openai_chat_completion_response(
    response: requests.Response,
    model: str,
) -> dict[str, Any]:
    payload = decode_non_stream_response(response, model)
    return build_openai_chat_completion_response(payload, model=model)


def _build_responses_event(event_name: str, payload: dict[str, Any]) -> bytes:
    return (
        f"event: {event_name}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    ).encode("utf-8")


def _build_stream_payload(
    *,
    text: str,
    tool_calls: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if text or not tool_calls:
        content.append({"type": "text", "text": text})
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        content.append(
            {
                "type": "tool_use",
                "id": str(tool_call.get("id") or uuid.uuid4().hex),
                "name": str(tool_call.get("name") or ""),
                "input": tool_call.get("input") if isinstance(tool_call.get("input"), dict) else {},
            }
        )
    return {
        "content": content,
        "usage": summary.get("usage") if isinstance(summary.get("usage"), dict) else {},
        "stop_reason": summary.get("stop_reason") or "end_turn",
    }


def iter_openai_responses_sse_bytes(
    response: requests.Response,
    model: str,
    accio: dict[str, Any] | None = None,
    on_complete: Callable[[dict[str, Any]], None] | None = None,
) -> Iterator[bytes]:
    summary = {
        "model": model,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
        "stop_reason": "end_turn",
        "content_blocks": 0,
        "text_chars": 0,
        "thinking_chars": 0,
        "tool_use_blocks": 0,
        "tool_json_chars": 0,
    }
    response_id = f"resp_{uuid.uuid4().hex}"
    message_id = f"msg_{uuid.uuid4().hex}"
    created_at = int(time.time())
    started_response = False
    started_message = False
    text_value = ""
    active_tool: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = []
    completed = False

    def start_response() -> Iterator[bytes]:
        nonlocal started_response
        if started_response:
            return
        started_response = True
        response_payload = {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "model": model,
            "status": "in_progress",
            "output": [],
            "output_text": "",
        }
        if isinstance(accio, dict) and accio:
            response_payload["accio"] = dict(accio)
        yield _build_responses_event(
            "response.created",
            {"type": "response.created", "response": response_payload},
        )
        yield _build_responses_event(
            "response.in_progress",
            {"type": "response.in_progress", "response": response_payload},
        )

    def ensure_message_item() -> Iterator[bytes]:
        nonlocal started_message
        if started_message:
            return
        for event_bytes in start_response():
            yield event_bytes
        started_message = True
        yield _build_responses_event(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "response_id": response_id,
                "output_index": 0,
                "item": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
            },
        )

    try:
        for event_name, event_payload in iter_anthropic_sse_events(response, model):
            update_usage_summary(summary, event_name, event_payload)

            if event_name == "content_block_start":
                block = event_payload.get("content_block")
                if not isinstance(block, dict):
                    continue
                if str(block.get("type") or "") != "tool_use":
                    continue
                for event_bytes in start_response():
                    yield event_bytes
                active_tool = {
                    "id": str(block.get("id") or uuid.uuid4().hex),
                    "name": str(block.get("name") or ""),
                    "arguments": "",
                }
                tool_index = len(tool_calls) + (1 if started_message else 0)
                active_tool["output_index"] = tool_index
                yield _build_responses_event(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "response_id": response_id,
                        "output_index": tool_index,
                        "item": {
                            "id": active_tool["id"],
                            "type": "tool_call",
                            "name": active_tool["name"],
                            "arguments": "",
                            "status": "in_progress",
                        },
                    },
                )
                continue

            if event_name == "content_block_delta":
                delta = event_payload.get("delta")
                if not isinstance(delta, dict):
                    continue
                if delta.get("text") is not None:
                    for event_bytes in ensure_message_item():
                        yield event_bytes
                    text_part = str(delta.get("text") or "")
                    text_value += text_part
                    yield _build_responses_event(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "response_id": response_id,
                            "item_id": message_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": text_part,
                        },
                    )
                    continue

                if (
                    delta.get("partial_json") is not None
                    and active_tool is not None
                ):
                    active_tool["arguments"] = str(active_tool.get("arguments") or "") + str(
                        delta.get("partial_json") or ""
                    )
                continue

            if event_name == "content_block_stop" and active_tool is not None:
                tool_call = {
                    "id": str(active_tool.get("id") or uuid.uuid4().hex),
                    "name": str(active_tool.get("name") or ""),
                    "input": _parse_json_dict(active_tool.get("arguments")),
                }
                tool_calls.append(tool_call)
                yield _build_responses_event(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "response_id": response_id,
                        "output_index": int(active_tool.get("output_index") or 0),
                        "item": {
                            "id": tool_call["id"],
                            "type": "tool_call",
                            "name": tool_call["name"],
                            "arguments": str(active_tool.get("arguments") or "{}"),
                            "status": "completed",
                        },
                    },
                )
                active_tool = None

        for event_bytes in start_response():
            yield event_bytes

        if (text_value or not tool_calls) and not started_message:
            for event_bytes in ensure_message_item():
                yield event_bytes

        if active_tool is not None:
            tool_call = {
                "id": str(active_tool.get("id") or uuid.uuid4().hex),
                "name": str(active_tool.get("name") or ""),
                "input": _parse_json_dict(active_tool.get("arguments")),
            }
            tool_calls.append(tool_call)
            yield _build_responses_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "response_id": response_id,
                    "output_index": int(active_tool.get("output_index") or 0),
                    "item": {
                        "id": tool_call["id"],
                        "type": "tool_call",
                        "name": tool_call["name"],
                        "arguments": str(active_tool.get("arguments") or "{}"),
                        "status": "completed",
                    },
                },
            )
            active_tool = None

        if started_message:
            yield _build_responses_event(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "response_id": response_id,
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": text_value,
                },
            )
            yield _build_responses_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "response_id": response_id,
                    "output_index": 0,
                    "item": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": text_value,
                                "annotations": [],
                            }
                        ],
                    },
                },
            )

        final_payload = _build_stream_payload(
            text=text_value,
            tool_calls=tool_calls,
            summary=summary,
        )
        response_payload = build_openai_responses_response(
            final_payload,
            model=model,
            response_id=response_id,
            message_id=message_id,
            created_at=created_at,
            status="completed",
            accio=accio,
        )
        yield _build_responses_event(
            "response.completed",
            {"type": "response.completed", "response": response_payload},
        )
        completed = True
    finally:
        response.close()
        if completed and on_complete is not None:
            on_complete(summary)


def _build_chunk(
    completion_id: str,
    created: int,
    model: str,
    choices: list[dict[str, Any]],
) -> bytes:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": choices,
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def iter_openai_chat_sse_bytes(
    response: requests.Response,
    model: str,
    on_complete: Callable[[dict[str, Any]], None] | None = None,
) -> Iterator[bytes]:
    summary = {
        "model": model,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
        "stop_reason": "end_turn",
        "content_blocks": 0,
        "text_chars": 0,
        "thinking_chars": 0,
        "tool_use_blocks": 0,
        "tool_json_chars": 0,
    }
    completion_id = f"chatcmpl_{uuid.uuid4().hex}"
    created = int(time.time())
    emitted_role = False
    finish_emitted = False
    active_tool_index = -1
    completed = False

    try:
        for event_name, event_payload in iter_anthropic_sse_events(response, model):
            update_usage_summary(summary, event_name, event_payload)

            if not emitted_role:
                emitted_role = True
                yield _build_chunk(
                    completion_id,
                    created,
                    model,
                    [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                )

            if event_name == "content_block_start":
                block = event_payload.get("content_block")
                if not isinstance(block, dict):
                    continue
                if str(block.get("type") or "") != "tool_use":
                    continue
                active_tool_index += 1
                yield _build_chunk(
                    completion_id,
                    created,
                    model,
                    [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": active_tool_index,
                                        "id": str(block.get("id") or uuid.uuid4().hex),
                                        "type": "function",
                                        "function": {
                                            "name": str(block.get("name") or ""),
                                            "arguments": "",
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                )
                continue

            if event_name == "content_block_delta":
                delta = event_payload.get("delta")
                if not isinstance(delta, dict):
                    continue
                if delta.get("text") is not None:
                    yield _build_chunk(
                        completion_id,
                        created,
                        model,
                        [
                            {
                                "index": 0,
                                "delta": {"content": str(delta.get("text") or "")},
                                "finish_reason": None,
                            }
                        ],
                    )
                    continue

                if delta.get("partial_json") is not None and active_tool_index >= 0:
                    yield _build_chunk(
                        completion_id,
                        created,
                        model,
                        [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": active_tool_index,
                                            "function": {
                                                "arguments": str(delta.get("partial_json") or "")
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    )
                continue

            if event_name != "message_delta":
                continue

            delta = event_payload.get("delta")
            if not isinstance(delta, dict) or not delta.get("stop_reason"):
                continue
            finish_emitted = True
            yield _build_chunk(
                completion_id,
                created,
                model,
                [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": _map_finish_reason(
                            str(delta.get("stop_reason") or ""),
                            int(summary.get("tool_use_blocks") or 0) > 0,
                        ),
                    }
                ],
            )

        if not emitted_role:
            emitted_role = True
            yield _build_chunk(
                completion_id,
                created,
                model,
                [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            )

        if not finish_emitted:
            yield _build_chunk(
                completion_id,
                created,
                model,
                [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": _map_finish_reason(
                            str(summary.get("stop_reason") or "end_turn"),
                            int(summary.get("tool_use_blocks") or 0) > 0,
                        ),
                    }
                ],
            )

        completed = True
        yield b"data: [DONE]\n\n"
    finally:
        response.close()
        if completed and on_complete is not None:
            on_complete(summary)
