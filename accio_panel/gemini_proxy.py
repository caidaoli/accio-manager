from __future__ import annotations

import json
import uuid
from typing import Any
from typing import Callable
from typing import Iterator

import requests


SUPPORTED_GEMINI_TEXT_MODELS = (
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3-pro-preview",
)
SUPPORTED_GEMINI_IMAGE_MODELS = (
    "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview",
)
SUPPORTED_GEMINI_MODELS = (
    *SUPPORTED_GEMINI_TEXT_MODELS,
    *SUPPORTED_GEMINI_IMAGE_MODELS,
)
SUPPORTED_GEMINI_MODELS_SET = set(SUPPORTED_GEMINI_MODELS)
DEFAULT_GEMINI_MODEL = SUPPORTED_GEMINI_TEXT_MODELS[0]

GEMINI_MODEL_METADATA = {
    "gemini-3-flash-preview": {
        "display_name": "Gemini 3 Flash",
        "input_limit": 1_000_000,
        "output_limit": 16_384,
    },
    "gemini-3.1-pro-preview": {
        "display_name": "Gemini 3.1 Pro",
        "input_limit": 1_000_000,
        "output_limit": 16_384,
    },
    "gemini-3-pro-preview": {
        "display_name": "Gemini 3 Pro",
        "input_limit": 1_000_000,
        "output_limit": 16_384,
    },
    "gemini-3-pro-image-preview": {
        "display_name": "Gemini 3 Pro Image Preview",
        "input_limit": 65_536,
        "output_limit": 8_192,
    },
    "gemini-3.1-flash-image-preview": {
        "display_name": "Gemini 3.1 Flash Image Preview",
        "input_limit": 131_072,
        "output_limit": 8_192,
    },
}


def gemini_error_payload(
    status: int,
    message: str,
    *,
    error_status: str = "INVALID_ARGUMENT",
) -> dict[str, Any]:
    return {
        "error": {
            "code": status,
            "message": message,
            "status": error_status,
        }
    }


def normalize_gemini_model_name(model_name: Any) -> str:
    normalized = str(model_name or "").strip()
    if normalized.lower().startswith("models/"):
        return normalized[7:].strip()
    return normalized


def _supported_generation_methods() -> list[str]:
    return ["generateContent", "streamGenerateContent"]


def build_gemini_model_payload(model_name: str) -> dict[str, Any] | None:
    normalized_name = normalize_gemini_model_name(model_name)
    if not normalized_name:
        return None
    metadata = GEMINI_MODEL_METADATA.get(normalized_name, {})
    if not metadata and normalized_name not in SUPPORTED_GEMINI_MODELS_SET:
        return None
    return {
        "name": f"models/{normalized_name}",
        "baseModelId": normalized_name,
        "displayName": metadata.get("display_name", normalized_name),
        "description": "Accio Gemini 兼容代理模型",
        "inputTokenLimit": metadata.get("input_limit", 1_000_000),
        "outputTokenLimit": metadata.get("output_limit", 16_384),
        "supportedGenerationMethods": _supported_generation_methods(),
    }


def build_gemini_models_payload() -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for model_name in SUPPORTED_GEMINI_MODELS:
        model_payload = build_gemini_model_payload(model_name)
        if model_payload:
            models.append(model_payload)
    return {"models": models}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _stringify_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value or {}, ensure_ascii=False)


def _parse_json_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return dict(parsed) if isinstance(parsed, dict) else None
    return None


def _parse_json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _normalize_role(value: Any) -> str:
    normalized = str(value or "user").strip().lower()
    if normalized in {"assistant", "model"}:
        return "model"
    if normalized == "tool":
        return "tool"
    return "user"


def _extract_system_instruction(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""
    parts = value.get("parts")
    if not isinstance(parts, list):
        return ""
    text_parts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if text is not None:
            text_parts.append(str(text))
    return "\n".join(part for part in text_parts if part)


def _normalize_part(part: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}

    if part.get("text") is not None:
        normalized["text"] = str(part.get("text") or "")

    if part.get("thought") is not None:
        normalized["thought"] = bool(part.get("thought"))

    thought_signature = part.get("thoughtSignature", part.get("thought_signature"))
    if thought_signature:
        normalized["thoughtSignature"] = str(thought_signature)

    inline_data = part.get("inlineData", part.get("inline_data"))
    if isinstance(inline_data, dict):
        normalized["inline_data"] = {
            "mime_type": str(
                inline_data.get("mimeType", inline_data.get("mime_type")) or ""
            ),
            "data": str(inline_data.get("data") or ""),
        }

    file_data = part.get("fileData", part.get("file_data"))
    if isinstance(file_data, dict):
        normalized["file_data"] = {
            "file_uri": str(
                file_data.get("fileUri", file_data.get("file_uri")) or ""
            ),
            "mime_type": str(
                file_data.get("mimeType", file_data.get("mime_type")) or ""
            ),
        }

    function_call = part.get("functionCall", part.get("function_call"))
    if isinstance(function_call, dict):
        args_value = function_call.get("argsJson")
        if args_value is None:
            args_value = function_call.get("args", {})
        normalized["functionCall"] = {
            "id": str(
                function_call.get("id")
                or function_call.get("callId")
                or function_call.get("name")
                or uuid.uuid4().hex
            ),
            "name": str(function_call.get("name") or ""),
            "argsJson": _stringify_json(args_value),
        }

    function_response = part.get("functionResponse", part.get("function_response"))
    if isinstance(function_response, dict):
        response_value = function_response.get("responseJson")
        if response_value is None:
            response_value = function_response.get("response", {})
        normalized["functionResponse"] = {
            "id": str(
                function_response.get("id")
                or function_response.get("callId")
                or function_response.get("name")
                or uuid.uuid4().hex
            ),
            "name": str(function_response.get("name") or ""),
            "responseJson": _stringify_json(response_value),
        }

    return normalized


def _normalize_contents(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    contents: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        raw_parts = item.get("parts")
        if not isinstance(raw_parts, list):
            raw_parts = []
        parts = [
            normalized_part
            for normalized_part in (
                _normalize_part(part) for part in raw_parts if isinstance(part, dict)
            )
            if normalized_part
        ]
        contents.append(
            {
                "role": _normalize_role(item.get("role")),
                "parts": parts or [{"text": ""}],
            }
        )
    return contents


def _normalize_tools(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized_tools: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue

        if item.get("name"):
            parameters_value = item.get("parameters_json", item.get("parametersJson"))
            normalized_tools.append(
                {
                    "name": str(item.get("name") or ""),
                    "description": str(item.get("description") or ""),
                    "parametersJson": _stringify_json(parameters_value),
                    "parameters_json": _stringify_json(parameters_value),
                }
            )
            continue

        function_declarations = item.get(
            "functionDeclarations",
            item.get("function_declarations"),
        )
        if not isinstance(function_declarations, list):
            continue
        for declaration in function_declarations:
            if not isinstance(declaration, dict) or not declaration.get("name"):
                continue
            schema = declaration.get("parameters", declaration.get("parameters_json"))
            normalized_tools.append(
                {
                    "name": str(declaration.get("name") or ""),
                    "description": str(declaration.get("description") or ""),
                    "parametersJson": _stringify_json(schema),
                    "parameters_json": _stringify_json(schema),
                }
            )
    return normalized_tools


def _normalize_properties(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    normalized = dict(value)
    for key in ("tool_config", "generation_config"):
        if normalized.get(key) is not None:
            normalized[key] = _stringify_json(normalized.get(key))
    return normalized


def build_accio_request_from_gemini(
    body: dict[str, Any],
    *,
    model: str,
    token: str,
    utdid: str,
    version: str,
) -> dict[str, Any]:
    generation_config = body.get("generationConfig", body.get("generation_config"))
    if not isinstance(generation_config, dict):
        generation_config = {}

    request_body: dict[str, Any] = {
        "utdid": utdid,
        "version": version,
        "token": token,
        "empid": str(body.get("empid") or ""),
        "tenant": str(body.get("tenant") or ""),
        "iai_tag": str(body.get("iai_tag", body.get("iaiTag")) or ""),
        "stream": True,
        "model": model,
        "request_id": str(
            body.get("request_id")
            or body.get("requestId")
            or f"req-{uuid.uuid4()}"
        ),
        "message_id": str(body.get("message_id", body.get("messageId")) or ""),
        "incremental": True,
        "max_output_tokens": _as_int(
            generation_config.get(
                "maxOutputTokens",
                body.get("max_output_tokens", body.get("maxOutputTokens", 8192)),
            ),
            8192,
        ),
        "contents": _normalize_contents(body.get("contents")),
        "include_thoughts": False,
        "stop_sequences": [],
        "properties": {},
    }

    system_instruction = body.get("system_instruction")
    if system_instruction is None:
        system_instruction = body.get("systemInstruction")
    normalized_system_instruction = _extract_system_instruction(system_instruction)
    if isinstance(system_instruction, str) and system_instruction.strip():
        normalized_system_instruction = system_instruction.strip()
    if normalized_system_instruction:
        request_body["system_instruction"] = normalized_system_instruction

    numeric_mappings = {
        "temperature": "temperature",
        "topP": "top_p",
        "topK": "top_k",
    }
    for source_key, target_key in numeric_mappings.items():
        source_value = generation_config.get(source_key, body.get(source_key))
        if source_value is not None:
            request_body[target_key] = source_value

    stop_sequences = generation_config.get(
        "stopSequences",
        body.get("stop_sequences", body.get("stopSequences")),
    )
    if isinstance(stop_sequences, str):
        normalized_stop_sequences = [stop_sequences] if stop_sequences.strip() else []
    elif isinstance(stop_sequences, list):
        normalized_stop_sequences = [
            str(item or "").strip()
            for item in stop_sequences
            if str(item or "").strip()
        ]
    else:
        normalized_stop_sequences = []
    request_body["stop_sequences"] = normalized_stop_sequences

    if generation_config.get("candidateCount") is not None:
        request_body["candidate_count"] = generation_config.get("candidateCount")
    if body.get("timeout") is not None:
        request_body["timeout"] = body.get("timeout")

    tools = _normalize_tools(body.get("tools"))
    if tools:
        request_body["tools"] = tools

    properties = _normalize_properties(body.get("properties"))
    if properties:
        request_body["properties"].update(properties)

    if generation_config:
        properties_payload = request_body.setdefault("properties", {})
        properties_payload["generation_config"] = _stringify_json(generation_config)

    tool_config = body.get("toolConfig", body.get("tool_config"))
    if tool_config is not None:
        properties_payload = request_body.setdefault("properties", {})
        properties_payload["tool_config"] = _stringify_json(tool_config)

    for passthrough_key in ("message_id", "session_key", "conversation_id"):
        if body.get(passthrough_key) is not None:
            request_body[passthrough_key] = body.get(passthrough_key)

    return request_body


def _normalize_response_part(part: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}

    if part.get("text") is not None:
        normalized["text"] = str(part.get("text") or "")

    if part.get("thought") is not None:
        normalized["thought"] = bool(part.get("thought"))

    thought_signature = part.get("thoughtSignature", part.get("thought_signature"))
    if thought_signature:
        normalized["thoughtSignature"] = str(thought_signature)

    inline_data = part.get("inlineData", part.get("inline_data"))
    if isinstance(inline_data, dict):
        normalized["inlineData"] = {
            "mimeType": str(
                inline_data.get("mimeType", inline_data.get("mime_type")) or ""
            ),
            "data": str(inline_data.get("data") or ""),
        }

    file_data = part.get("fileData", part.get("file_data"))
    if isinstance(file_data, dict):
        normalized["fileData"] = {
            "fileUri": str(
                file_data.get("fileUri", file_data.get("file_uri")) or ""
            ),
            "mimeType": str(
                file_data.get("mimeType", file_data.get("mime_type")) or ""
            ),
        }

    function_call = part.get("functionCall", part.get("function_call"))
    if isinstance(function_call, dict):
        args_value = function_call.get("args")
        if args_value is None:
            args_value = _parse_json_value(function_call.get("argsJson"))
        normalized["functionCall"] = {
            "id": str(
                function_call.get("id")
                or function_call.get("callId")
                or function_call.get("name")
                or uuid.uuid4().hex
            ),
            "name": str(function_call.get("name") or ""),
            "args": args_value if args_value is not None else {},
        }

    function_response = part.get("functionResponse", part.get("function_response"))
    if isinstance(function_response, dict):
        response_value = function_response.get("response")
        if response_value is None:
            response_value = _parse_json_value(function_response.get("responseJson"))
        normalized["functionResponse"] = {
            "id": str(
                function_response.get("id")
                or function_response.get("callId")
                or function_response.get("name")
                or uuid.uuid4().hex
            ),
            "name": str(function_response.get("name") or ""),
            "response": response_value if response_value is not None else {},
        }

    return normalized


def _normalize_candidate(candidate: Any, index: int, fallback_finish_reason: str) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        return {
            "index": index,
            "content": {"role": "model", "parts": []},
            "finishReason": fallback_finish_reason,
        }

    content = candidate.get("content")
    role = "model"
    parts: list[dict[str, Any]] = []
    if isinstance(content, dict):
        role = "model" if _normalize_role(content.get("role")) != "tool" else "tool"
        raw_parts = content.get("parts")
        if isinstance(raw_parts, list):
            parts = [
                normalized_part
                for normalized_part in (
                    _normalize_response_part(part)
                    for part in raw_parts
                    if isinstance(part, dict)
                )
                if normalized_part
            ]

    finish_reason = str(
        candidate.get("finishReason", candidate.get("finish_reason"))
        or fallback_finish_reason
        or "STOP"
    )
    return {
        "index": _as_int(candidate.get("index"), index),
        "content": {
            "role": role,
            "parts": parts,
        },
        "finishReason": finish_reason,
    }


def _normalize_token_details(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        modality = str(item.get("modality") or "").strip()
        if not modality:
            continue
        normalized.append(
            {
                "modality": modality,
                "tokenCount": _as_int(item.get("tokenCount", item.get("token_count")), 0),
            }
        )
    return normalized


def _normalize_usage_metadata(value: Any) -> dict[str, Any]:
    usage = value if isinstance(value, dict) else {}
    prompt_tokens = _as_int(
        usage.get("promptTokenCount", usage.get("prompt_token_count")),
        0,
    )
    candidate_tokens = _as_int(
        usage.get("candidatesTokenCount", usage.get("candidates_token_count")),
        0,
    )
    total_tokens = _as_int(
        usage.get("totalTokenCount", usage.get("total_token_count")),
        prompt_tokens + candidate_tokens,
    )
    normalized = {
        "promptTokenCount": prompt_tokens,
        "candidatesTokenCount": candidate_tokens,
        "totalTokenCount": total_tokens,
    }

    prompt_details = _normalize_token_details(
        usage.get("promptTokensDetails", usage.get("prompt_tokens_details"))
    )
    if prompt_details:
        normalized["promptTokensDetails"] = prompt_details

    candidate_details = _normalize_token_details(
        usage.get("candidatesTokensDetails", usage.get("candidates_tokens_details"))
    )
    if candidate_details:
        normalized["candidatesTokensDetails"] = candidate_details

    thoughts_tokens = usage.get("thoughtsTokenCount", usage.get("thoughts_token_count"))
    if thoughts_tokens is not None:
        normalized["thoughtsTokenCount"] = _as_int(thoughts_tokens, 0)

    return normalized


def normalize_gemini_response_payload(
    payload: dict[str, Any],
    *,
    model: str,
    fallback_usage: dict[str, Any] | None = None,
    fallback_finish_reason: str | None = None,
) -> dict[str, Any]:
    candidates_value = payload.get("candidates")
    candidates: list[dict[str, Any]] = []
    if isinstance(candidates_value, list):
        candidates = [
            _normalize_candidate(candidate, index, fallback_finish_reason or "STOP")
            for index, candidate in enumerate(candidates_value)
        ]

    usage_source = payload.get("usageMetadata", payload.get("usage_metadata"))
    if usage_source is None:
        usage_source = fallback_usage or {}

    normalized: dict[str, Any] = {
        "candidates": candidates,
        "usageMetadata": _normalize_usage_metadata(usage_source),
        "modelVersion": str(
            payload.get("modelVersion", payload.get("model_version")) or model
        ),
    }

    prompt_feedback = payload.get("promptFeedback", payload.get("prompt_feedback"))
    if isinstance(prompt_feedback, dict):
        normalized["promptFeedback"] = prompt_feedback

    return normalized


def decode_gemini_generate_content_response(
    response: requests.Response,
    model: str,
) -> dict[str, Any]:
    raw_payload: dict[str, Any] | None = None
    wrapper_usage: dict[str, Any] | None = None
    wrapper_finish_reason: str | None = None

    try:
        for raw_line in response.iter_lines(decode_unicode=True):
            line = (raw_line or "").strip()
            if not line or line.startswith(":"):
                continue

            json_text = line[5:].strip() if line.startswith("data:") else line
            if json_text == "[DONE]":
                continue

            event_payload = _parse_json_dict(json_text)
            if not event_payload:
                continue

            usage_candidate = event_payload.get(
                "usageMetadata",
                event_payload.get("usage_metadata"),
            )
            if isinstance(usage_candidate, dict):
                wrapper_usage = usage_candidate

            finish_candidate = event_payload.get(
                "finishReason",
                event_payload.get("finish_reason"),
            )
            if finish_candidate:
                wrapper_finish_reason = str(finish_candidate)

            raw_candidate = event_payload.get("raw_response_json")
            if raw_candidate is not None:
                parsed_raw = _parse_json_dict(raw_candidate)
                if parsed_raw:
                    raw_payload = parsed_raw
                    continue

            if isinstance(event_payload.get("candidates"), list):
                raw_payload = event_payload
    finally:
        response.close()

    if raw_payload is None:
        raise ValueError("上游未返回有效的 Gemini 响应。")

    return normalize_gemini_response_payload(
        raw_payload,
        model=model,
        fallback_usage=wrapper_usage,
        fallback_finish_reason=wrapper_finish_reason,
    )


def build_gemini_generate_content_response(
    payload: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    return normalize_gemini_response_payload(payload, model=model)


def iter_gemini_generate_content_sse_bytes(
    response: requests.Response,
    model: str,
    on_complete: Callable[[dict[str, Any]], None] | None = None,
) -> Iterator[bytes]:
    payload = decode_gemini_generate_content_response(response, model)
    summary = summarize_gemini_response(payload)
    summary["usage"] = extract_gemini_usage(payload)
    summary["stop_reason"] = extract_gemini_finish_reason(payload)
    try:
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
    finally:
        if on_complete is not None:
            on_complete(summary)


def extract_gemini_usage(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usageMetadata")
    if not isinstance(usage, dict):
        usage = {}
    return {
        "input_tokens": _as_int(usage.get("promptTokenCount"), 0),
        "output_tokens": _as_int(usage.get("candidatesTokenCount"), 0),
        "total_tokens": _as_int(usage.get("totalTokenCount"), 0),
        "thought_tokens": _as_int(usage.get("thoughtsTokenCount"), 0),
    }


def extract_gemini_finish_reason(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            finish_reason = candidate.get(
                "finishReason",
                candidate.get("finish_reason"),
            )
            if finish_reason:
                return str(finish_reason)
    finish_reason = payload.get("finishReason", payload.get("finish_reason"))
    if finish_reason:
        return str(finish_reason)
    return "STOP"


def summarize_gemini_response(payload: dict[str, Any]) -> dict[str, int | bool]:
    text_chars = 0
    tool_use_blocks = 0
    image_blocks = 0

    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                if part.get("text") is not None:
                    text_chars += len(str(part.get("text") or ""))
                if isinstance(part.get("functionCall"), dict):
                    tool_use_blocks += 1
                if isinstance(part.get("inlineData"), dict) or isinstance(
                    part.get("fileData"), dict
                ):
                    image_blocks += 1

    return {
        "text_chars": text_chars,
        "tool_use_blocks": tool_use_blocks,
        "image_blocks": image_blocks,
        "empty_response": text_chars <= 0 and tool_use_blocks <= 0 and image_blocks <= 0,
    }
