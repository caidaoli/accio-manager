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


def _is_image_generation_model(model_name: Any) -> bool:
    return "image-preview" in normalize_gemini_model_name(model_name).lower()


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

    if part.get("thought") is True:
        normalized["thought"] = True

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
            parameters_value = item.get(
                "parameters_json",
                item.get("parametersJson", item.get("input_schema")),
            )
            normalized_tools.append(
                {
                    "name": str(item.get("name") or ""),
                    "description": str(item.get("description") or ""),
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
                    "parameters_json": _stringify_json(schema),
                }
            )
    return normalized_tools


def build_generate_content_request(
    body: dict[str, Any],
    *,
    token: str,
    model: str | None = None,
) -> dict[str, Any]:
    generation_config = body.get("generationConfig", body.get("generation_config"))
    if not isinstance(generation_config, dict):
        generation_config = {}

    normalized_model = str(model or body.get("model") or "").strip()
    request_body: dict[str, Any] = {
        "token": token,
        "model": normalized_model,
        "request_id": str(
            body.get("request_id")
            or body.get("requestId")
            or f"req-{uuid.uuid4()}"
        ),
        "message_id": str(
            body.get("message_id")
            or body.get("messageId")
            or f"msg-{uuid.uuid4().hex}"
        ),
        "max_output_tokens": _as_int(
            generation_config.get(
                "maxOutputTokens",
                body.get("max_output_tokens", body.get("maxOutputTokens", 8192)),
            ),
            8192,
        ),
        "contents": _normalize_contents(body.get("contents")),
    }

    system_instruction = body.get("system_instruction")
    if system_instruction is None:
        system_instruction = body.get("systemInstruction")
    normalized_system_instruction = _extract_system_instruction(system_instruction)
    if isinstance(system_instruction, str) and system_instruction.strip():
        normalized_system_instruction = system_instruction.strip()
    if normalized_system_instruction:
        request_body["system_instruction"] = normalized_system_instruction.replace(
            "Claude", "Accio"
        )

    tools = _normalize_tools(body.get("tools"))
    if tools:
        request_body["tools"] = tools

    for passthrough_key in (
        "message_id",
        "session_key",
        "conversation_id",
        "conversation_name",
    ):
        value = body.get(passthrough_key)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        request_body[passthrough_key] = value

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
        mime_type = str(
            inline_data.get("mimeType", inline_data.get("mime_type")) or ""
        )
        data = str(inline_data.get("data") or "")
        normalized["inlineData"] = {
            "mimeType": mime_type,
            "data": data,
        }
        normalized["inline_data"] = {
            "mime_type": mime_type,
            "data": data,
        }

    file_data = part.get("fileData", part.get("file_data"))
    if isinstance(file_data, dict):
        file_uri = str(file_data.get("fileUri", file_data.get("file_uri")) or "")
        mime_type = str(
            file_data.get("mimeType", file_data.get("mime_type")) or ""
        )
        normalized["fileData"] = {
            "fileUri": file_uri,
            "mimeType": mime_type,
        }
        normalized["file_data"] = {
            "file_uri": file_uri,
            "mime_type": mime_type,
        }

    function_call = part.get("functionCall", part.get("function_call"))
    if isinstance(function_call, dict):
        args_value = function_call.get("args")
        if args_value is None:
            args_value = _parse_json_value(function_call.get("argsJson"))
        call_id = str(
            function_call.get("id")
            or function_call.get("callId")
            or function_call.get("name")
            or uuid.uuid4().hex
        )
        call_name = str(function_call.get("name") or "")
        call_args = args_value if args_value is not None else {}
        normalized["functionCall"] = {
            "id": call_id,
            "name": call_name,
            "args": call_args,
        }
        normalized["function_call"] = {
            "id": call_id,
            "name": call_name,
            "args": call_args,
        }

    function_response = part.get("functionResponse", part.get("function_response"))
    if isinstance(function_response, dict):
        response_value = function_response.get("response")
        if response_value is None:
            response_value = _parse_json_value(function_response.get("responseJson"))
        call_id = str(
            function_response.get("id")
            or function_response.get("callId")
            or function_response.get("name")
            or uuid.uuid4().hex
        )
        call_name = str(function_response.get("name") or "")
        call_response = response_value if response_value is not None else {}
        normalized["functionResponse"] = {
            "id": call_id,
            "name": call_name,
            "response": call_response,
        }
        normalized["function_response"] = {
            "id": call_id,
            "name": call_name,
            "response": call_response,
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
        "finish_reason": finish_reason,
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

    normalized_usage = _normalize_usage_metadata(usage_source)
    normalized_model_version = str(
        payload.get("modelVersion", payload.get("model_version")) or model
    )
    normalized: dict[str, Any] = {
        "candidates": candidates,
        "usageMetadata": normalized_usage,
        "usage_metadata": normalized_usage,
        "modelVersion": normalized_model_version,
        "model_version": normalized_model_version,
    }

    prompt_feedback = payload.get("promptFeedback", payload.get("prompt_feedback"))
    if isinstance(prompt_feedback, dict):
        normalized["promptFeedback"] = prompt_feedback
        normalized["prompt_feedback"] = prompt_feedback

    return normalized


def _merge_gemini_parts(
    existing_parts: list[dict[str, Any]],
    incoming_parts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = [dict(part) for part in existing_parts]
    for index, incoming_part in enumerate(incoming_parts):
        incoming_has_structured = any(
            isinstance(incoming_part.get(key), dict)
            for key in ("inlineData", "fileData", "functionCall", "functionResponse")
        )
        if index >= len(merged):
            merged.append(dict(incoming_part))
            continue

        current_part = dict(merged[index])
        current_has_structured = any(
            isinstance(current_part.get(key), dict)
            for key in ("inlineData", "fileData", "functionCall", "functionResponse")
        )
        incoming_text = incoming_part.get("text")
        current_text = current_part.get("text")

        if incoming_has_structured and not current_has_structured and current_text is not None:
            merged.append(dict(incoming_part))
            continue

        if current_has_structured and not incoming_has_structured and incoming_text is not None:
            merged.append(dict(incoming_part))
            continue

        if incoming_text is not None and not incoming_has_structured and not current_has_structured:
            current_part["text"] = f"{current_text or ''}{incoming_text}"

        for key in (
            "inlineData",
            "fileData",
            "functionCall",
            "functionResponse",
            "thought",
            "thoughtSignature",
        ):
            if incoming_part.get(key) is not None:
                current_part[key] = incoming_part.get(key)
        merged[index] = current_part
    return merged


def _merge_gemini_candidates(
    existing_candidates: list[dict[str, Any]],
    incoming_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = [dict(candidate) for candidate in existing_candidates]
    for index, incoming_candidate in enumerate(incoming_candidates):
        if index >= len(merged):
            merged.append(dict(incoming_candidate))
            continue

        current_candidate = dict(merged[index])
        incoming_content = incoming_candidate.get("content")
        current_content = current_candidate.get("content")
        if isinstance(incoming_content, dict):
            current_content = dict(current_content) if isinstance(current_content, dict) else {}
            incoming_parts = incoming_content.get("parts")
            current_parts = current_content.get("parts")
            if isinstance(incoming_parts, list):
                current_content["parts"] = _merge_gemini_parts(
                    current_parts if isinstance(current_parts, list) else [],
                    [part for part in incoming_parts if isinstance(part, dict)],
                )
            if incoming_content.get("role") is not None:
                current_content["role"] = incoming_content.get("role")
            current_candidate["content"] = current_content

        if incoming_candidate.get("finishReason") is not None:
            current_candidate["finishReason"] = incoming_candidate.get("finishReason")
        merged[index] = current_candidate
    return merged


def _merge_gemini_response_payload(
    base_payload: dict[str, Any] | None,
    incoming_payload: dict[str, Any],
    *,
    model: str,
    fallback_usage: dict[str, Any] | None = None,
    fallback_finish_reason: str | None = None,
) -> dict[str, Any]:
    normalized_incoming = normalize_gemini_response_payload(
        incoming_payload,
        model=model,
        fallback_usage=fallback_usage,
        fallback_finish_reason=fallback_finish_reason,
    )
    if not isinstance(base_payload, dict) or not base_payload:
        return normalized_incoming

    merged = dict(base_payload)
    merged["candidates"] = _merge_gemini_candidates(
        base_payload.get("candidates")
        if isinstance(base_payload.get("candidates"), list)
        else [],
        normalized_incoming.get("candidates")
        if isinstance(normalized_incoming.get("candidates"), list)
        else [],
    )

    incoming_usage = normalized_incoming.get("usageMetadata")
    if isinstance(incoming_usage, dict) and incoming_usage:
        merged["usageMetadata"] = incoming_usage
        merged["usage_metadata"] = incoming_usage

    incoming_feedback = normalized_incoming.get("promptFeedback")
    if isinstance(incoming_feedback, dict) and incoming_feedback:
        merged["promptFeedback"] = incoming_feedback
        merged["prompt_feedback"] = incoming_feedback

    incoming_model_version = normalized_incoming.get("modelVersion")
    if incoming_model_version:
        merged["modelVersion"] = incoming_model_version
        merged["model_version"] = incoming_model_version
    return merged


def iter_gemini_generate_content_payloads(
    response: requests.Response,
    model: str,
) -> Iterator[dict[str, Any]]:
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
            _raise_upstream_turn_error_if_present(event_payload)

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
            source_payload: dict[str, Any] | None = None
            if raw_candidate is not None:
                source_payload = _parse_json_dict(raw_candidate)
            elif isinstance(event_payload.get("candidates"), list):
                source_payload = event_payload

            if not source_payload:
                continue

            yield normalize_gemini_response_payload(
                source_payload,
                model=model,
                fallback_usage=wrapper_usage,
                fallback_finish_reason=wrapper_finish_reason,
            )
    finally:
        response.close()


def decode_gemini_generate_content_response(
    response: requests.Response,
    model: str,
) -> dict[str, Any]:
    merged_payload: dict[str, Any] | None = None
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
            _raise_upstream_turn_error_if_present(event_payload)

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
                    merged_payload = _merge_gemini_response_payload(
                        merged_payload,
                        parsed_raw,
                        model=model,
                        fallback_usage=wrapper_usage,
                        fallback_finish_reason=wrapper_finish_reason,
                    )
                    continue

            if isinstance(event_payload.get("candidates"), list):
                merged_payload = _merge_gemini_response_payload(
                    merged_payload,
                    event_payload,
                    model=model,
                    fallback_usage=wrapper_usage,
                    fallback_finish_reason=wrapper_finish_reason,
                )
    finally:
        response.close()

    if merged_payload is None:
        raise ValueError("上游未返回有效的 Gemini 响应。")

    return merged_payload


def _raise_upstream_turn_error_if_present(payload: dict[str, Any]) -> None:
    if not payload.get("turn_complete"):
        return
    error_code = str(payload.get("error_code") or "").strip()
    error_message = str(payload.get("error_message") or "").strip()
    if not error_code and not error_message:
        return
    from .anthropic_proxy import UpstreamTurnError

    raise UpstreamTurnError(
        error_code=error_code,
        error_message=error_message,
        payload=payload,
    )


def build_gemini_generate_content_response(
    payload: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    return normalize_gemini_response_payload(payload, model=model)


def _estimate_base64_bytes(data: Any) -> int:
    text = str(data or "").strip()
    if not text:
        return 0
    padding = len(text) - len(text.rstrip("="))
    return max(0, (len(text) * 3) // 4 - padding)


def _collect_gemini_image_details(payload: dict[str, Any]) -> dict[str, Any]:
    image_blocks = 0
    image_data_chars = 0
    image_data_bytes = 0
    image_mime_types: list[str] = []
    image_sources: list[str] = []

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

                inline_data = part.get("inlineData")
                if isinstance(inline_data, dict):
                    image_blocks += 1
                    mime_type = str(
                        inline_data.get("mimeType", inline_data.get("mime_type")) or ""
                    ).strip()
                    if mime_type and mime_type not in image_mime_types:
                        image_mime_types.append(mime_type)
                    if "inlineData" not in image_sources:
                        image_sources.append("inlineData")
                    data = str(inline_data.get("data") or "")
                    image_data_chars += len(data)
                    image_data_bytes += _estimate_base64_bytes(data)
                    continue

                file_data = part.get("fileData")
                if isinstance(file_data, dict):
                    image_blocks += 1
                    mime_type = str(
                        file_data.get("mimeType", file_data.get("mime_type")) or ""
                    ).strip()
                    if mime_type and mime_type not in image_mime_types:
                        image_mime_types.append(mime_type)
                    if "fileData" not in image_sources:
                        image_sources.append("fileData")
                    data = str(file_data.get("data") or "")
                    if data:
                        image_data_chars += len(data)
                        image_data_bytes += _estimate_base64_bytes(data)

    return {
        "has_image_data": image_blocks > 0,
        "image_blocks": image_blocks,
        "image_mime_types": image_mime_types,
        "image_sources": image_sources,
        "image_data_chars": image_data_chars,
        "image_data_bytes": image_data_bytes,
    }


def iter_gemini_generate_content_sse_bytes(
    response: requests.Response,
    model: str,
    on_complete: Callable[[dict[str, Any]], None] | None = None,
) -> Iterator[bytes]:
    latest_payload: dict[str, Any] | None = None
    last_chunk_had_image = False
    summary = {
        "text_chars": 0,
        "tool_use_blocks": 0,
        "image_blocks": 0,
        "has_image_data": False,
        "image_mime_types": [],
        "image_sources": [],
        "image_data_chars": 0,
        "image_data_bytes": 0,
        "empty_response": True,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "thought_tokens": 0,
        },
        "stop_reason": "STOP",
    }
    try:
        for payload in iter_gemini_generate_content_payloads(response, model):
            latest_payload = _merge_gemini_response_payload(
                latest_payload,
                payload,
                model=model,
            )
            chunk_summary = summarize_gemini_response(payload)
            usage = extract_gemini_usage(payload)
            summary["text_chars"] = int(summary["text_chars"]) + int(chunk_summary["text_chars"])
            summary["tool_use_blocks"] = int(summary["tool_use_blocks"]) + int(chunk_summary["tool_use_blocks"])
            summary["image_blocks"] = max(
                int(summary["image_blocks"]),
                int(chunk_summary["image_blocks"]),
            )
            summary["has_image_data"] = bool(summary["has_image_data"]) or bool(
                chunk_summary["has_image_data"]
            )
            summary["image_data_chars"] = max(
                int(summary["image_data_chars"]),
                int(chunk_summary["image_data_chars"]),
            )
            summary["image_data_bytes"] = max(
                int(summary["image_data_bytes"]),
                int(chunk_summary["image_data_bytes"]),
            )
            current_mime_types = [
                str(value).strip()
                for value in summary.get("image_mime_types", [])
                if str(value).strip()
            ]
            for mime_type in chunk_summary.get("image_mime_types", []):
                normalized_mime = str(mime_type).strip()
                if normalized_mime and normalized_mime not in current_mime_types:
                    current_mime_types.append(normalized_mime)
            summary["image_mime_types"] = current_mime_types
            current_sources = [
                str(value).strip()
                for value in summary.get("image_sources", [])
                if str(value).strip()
            ]
            for source in chunk_summary.get("image_sources", []):
                normalized_source = str(source).strip()
                if normalized_source and normalized_source not in current_sources:
                    current_sources.append(normalized_source)
            summary["image_sources"] = current_sources
            summary["empty_response"] = False
            summary["usage"] = usage
            summary["stop_reason"] = extract_gemini_finish_reason(payload)
            last_chunk_had_image = bool(chunk_summary["has_image_data"])
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

        if (
            _is_image_generation_model(model)
            and isinstance(latest_payload, dict)
            and latest_payload
        ):
            final_summary = summarize_gemini_response(latest_payload)
            if bool(final_summary["has_image_data"]) and not last_chunk_had_image:
                yield (
                    f"data: {json.dumps(latest_payload, ensure_ascii=False)}\n\n".encode(
                        "utf-8"
                    )
                )
    finally:
        if latest_payload is not None:
            final_summary = summarize_gemini_response(latest_payload)
            summary["text_chars"] = max(
                int(summary["text_chars"]),
                int(final_summary["text_chars"]),
            )
            summary["tool_use_blocks"] = max(
                int(summary["tool_use_blocks"]),
                int(final_summary["tool_use_blocks"]),
            )
            summary["image_blocks"] = max(
                int(summary["image_blocks"]),
                int(final_summary["image_blocks"]),
            )
            summary["has_image_data"] = bool(summary["has_image_data"]) or bool(
                final_summary["has_image_data"]
            )
            summary["image_mime_types"] = list(final_summary.get("image_mime_types", []))
            summary["image_sources"] = list(final_summary.get("image_sources", []))
            summary["image_data_chars"] = max(
                int(summary["image_data_chars"]),
                int(final_summary["image_data_chars"]),
            )
            summary["image_data_bytes"] = max(
                int(summary["image_data_bytes"]),
                int(final_summary["image_data_bytes"]),
            )
            summary["empty_response"] = bool(final_summary["empty_response"])
            summary["usage"] = extract_gemini_usage(latest_payload)
            summary["stop_reason"] = extract_gemini_finish_reason(latest_payload)
        if on_complete is not None:
            on_complete(summary)


def extract_gemini_usage(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usageMetadata")
    if not isinstance(usage, dict):
        usage = {}
    candidate_tokens = _as_int(usage.get("candidatesTokenCount"), 0)
    thought_tokens = _as_int(usage.get("thoughtsTokenCount"), 0)
    return {
        "input_tokens": _as_int(usage.get("promptTokenCount"), 0),
        "output_tokens": candidate_tokens + thought_tokens,
        "total_tokens": _as_int(usage.get("totalTokenCount"), 0),
        "thought_tokens": thought_tokens,
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


def summarize_gemini_response(payload: dict[str, Any]) -> dict[str, Any]:
    text_chars = 0
    tool_use_blocks = 0

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

    image_details = _collect_gemini_image_details(payload)
    image_blocks = int(image_details["image_blocks"])

    return {
        "text_chars": text_chars,
        "tool_use_blocks": tool_use_blocks,
        "image_blocks": image_blocks,
        "has_image_data": bool(image_details["has_image_data"]),
        "image_mime_types": list(image_details["image_mime_types"]),
        "image_sources": list(image_details["image_sources"]),
        "image_data_chars": int(image_details["image_data_chars"]),
        "image_data_bytes": int(image_details["image_data_bytes"]),
        "empty_response": text_chars <= 0 and tool_use_blocks <= 0 and image_blocks <= 0,
    }
