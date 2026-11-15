"""LiteLLM in-process adapters; demo mode deliberately produces INFERRED examples."""
import json
import math
import logging
import asyncio
import os
import re
from pathlib import Path
from urllib.parse import urlsplit
from time import perf_counter
from .schema import COLUMNS, EFFECT, FUSION

logger = logging.getLogger(__name__)


class OpenKiloQuotaWaitTimeout(TimeoutError):
    """OpenKilo stayed rate-limited for the configured wait window."""


PROMPTS = Path(__file__).resolve().parent.parent / "prompts"

def load_prompt(name):
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8")

def _openkilo_api_key():
    key = os.getenv("OPENKILO_API_KEY", "").strip()
    if not key and (path := os.getenv("OPENKILO_API_KEY_FILE")):
        key = Path(path).read_text(encoding="utf-8").strip()
    return key if key and key not in {"unused", "replace_me"} else None

def normalize_kilo_model_id(value):
    if not isinstance(value, str):
        return None
    model = value.strip()
    if model.startswith("openai/"):
        model = model[len("openai/"):]
    if (not model or model == "kilo-auto/free" or len(model) > 200
            or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.:-]+", model)):
        return None
    return model

def extract_json(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Model response contains no text")
    text = text.strip()
    if text.startswith("```"):
        _, separator, body = text.partition("\n")
        if not separator:
            raise ValueError("Model response contains an invalid code block")
        text = body.strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            result, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result
    raise ValueError("Model response contains no valid JSON object")

async def _call_model_once(role: str, payload: dict, prompt_name: str):
    public_payload = dict(payload)
    public_payload.pop("_kilo_model", None)
    if "task" in payload:
        public_payload["task"] = {
            key: value for key, value in payload["task"].items()
            if key not in {"local_source_path", "zip_member"}
        }
    from litellm import acompletion
    if prompt_name in {"planner", "agent_trace_repair", "agent_trace_supervisor_feedback", "agent_trace_claim_feedback", "agent_trace_navigation_review", "agent_case_supervisor"}:
        prefix = "ORCHESTRATOR"
        model = os.getenv("ORCHESTRATOR_MODEL", "openai/conductor")
    else:
        prefix = "OPENKILO"
        model = os.getenv("OPENKILO_MODEL")
    key = _openkilo_api_key() if prefix == "OPENKILO" else os.getenv(prefix + "_API_KEY")
    base = os.getenv(prefix + "_BASE_URL")
    pinned_model = normalize_kilo_model_id(payload.get("_kilo_model")) if prefix == "OPENKILO" else None
    if pinned_model:
        model = "openai/" + pinned_model
    if not model:
        model_var = "ORCHESTRATOR_MODEL" if prefix == "ORCHESTRATOR" else prefix + "_MODEL"
        raise RuntimeError(f"Configure {model_var} for live mode")
    if prompt_name in {"planner", "agent_trace_repair", "agent_trace_supervisor_feedback", "agent_trace_claim_feedback", "agent_trace_navigation_review", "agent_case_supervisor"} and not base:
        raise RuntimeError("Configure ORCHESTRATOR_BASE_URL for the hosted Nemotron endpoint")
    if prefix == "OPENKILO" and not base:
        raise RuntimeError("Configure OPENKILO_BASE_URL; no endpoint is assumed")
    timeout = float(os.getenv(prefix + "_TIMEOUT_SECONDS", "300" if role == "worker" else "90"))
    retries = int(os.getenv(prefix + "_MAX_RETRIES", "2"))
    if not math.isfinite(timeout) or timeout <= 0 or retries < 0:
        raise ValueError(f"{prefix} requires a positive finite timeout and nonnegative retries")
    case_fields = ", ".join(field for field in COLUMNS if field != "validation_status")
    prompt = load_prompt(prompt_name) + "\nReturn case objects with these fields: " + case_fields
    prompt += "\nAllowed expected_kernel_effect values: " + ", ".join(sorted(EFFECT))
    prompt += "\nAllowed fusion_type values: " + ", ".join(sorted(FUSION))
    if prompt_name in {"planner", "agent_trace_worker", "agent_trace_validator", "agent_trace_repair", "agent_trace_supervisor_feedback", "agent_trace_claim_feedback", "agent_trace_navigation_review", "agent_case_supervisor", "agent_case_repro", "agent_case_fix"}:
        prompt = load_prompt(prompt_name)
    kwargs = dict(model=model, messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(public_payload, ensure_ascii=False)},
    ], timeout=timeout, max_retries=retries, temperature=0)
    if prefix == "ORCHESTRATOR":
        kwargs["response_format"] = {"type": "json_object"}
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    if key and (prefix == "OPENKILO" or role != "worker"):
        kwargs["api_key"] = key
    elif prefix == "ORCHESTRATOR":
        kwargs["api_key"] = "local-server"
    if base:
        kwargs["api_base"] = base
    if output_limit := os.getenv(prefix + "_MAX_OUTPUT_TOKENS", "512" if prefix == "ORCHESTRATOR" else ""):
        max_tokens = int(output_limit)
        if max_tokens < 1:
            raise ValueError(f"{prefix}_MAX_OUTPUT_TOKENS must be positive")
        kwargs["max_tokens"] = max_tokens
    started = perf_counter()
    try:
        if prefix == "OPENKILO" and "api_key" not in kwargs:
            import httpx
            from openai import AsyncOpenAI

            async def remove_authorization(request):
                request.headers.pop("Authorization", None)

            async with httpx.AsyncClient(event_hooks={"request": [remove_authorization]}) as transport:
                client = AsyncOpenAI(
                    base_url=base, api_key="anonymous", http_client=transport,
                    timeout=timeout, max_retries=0,
                )
                response = await acompletion(**kwargs, client=client)
        else:
            response = await acompletion(**kwargs)
    except Exception as exc:
        error_text = _exception_text(exc)
        status = getattr(exc, "status_code", None)
        if any(marker in error_text for marker in ("connecterror", "connection error", "timed out", "timeout")):
            category = "connection_or_timeout"
        elif status is not None:
            category = f"http_{status}"
        else:
            category = "provider_or_response_error"
        logger.warning(
            "provider=%s stage=%s model=%s endpoint_host=%s outcome=error "
            "category=%s exception=%s latency_seconds=%.3f",
            prefix, prompt_name, model, urlsplit(base).hostname or "unconfigured",
            category, type(exc).__name__, perf_counter() - started,
        )
        raise
    usage = getattr(response, "usage", None)
    routed_model = normalize_kilo_model_id(getattr(response, "model", None)) if prefix == "OPENKILO" else None
    logger.info(
        "provider=%s stage=%s routed_model=%s latency_seconds=%.3f prompt_tokens=%s completion_tokens=%s",
        prefix, prompt_name, routed_model or "unreported",
        perf_counter() - started,
        getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None),
    )
    if not response.choices:
        raise ValueError("Model response contains no choices")
    try:
        result = extract_json(response.choices[0].message.content)
    except ValueError as exc:
        if routed_model:
            exc.kilo_model_id = routed_model
        raise
    if routed_model:
        result["_kilo_model"] = routed_model
    return result


def _exception_text(exc: BaseException) -> str:
    """Return useful nested provider error text without exposing credentials."""
    parts = [str(exc), type(exc).__name__]
    for nested in (getattr(exc, "__cause__", None), getattr(exc, "__context__", None)):
        if nested is not None and nested is not exc:
            parts.extend((str(nested), type(nested).__name__))
    response = getattr(exc, "response", None)
    if response is not None:
        parts.append(str(getattr(response, "text", "")))
    return " ".join(part for part in parts if part).lower()


def _is_quota_or_rate_limit(exc: BaseException) -> bool:
    """Match explicit rate/quota exhaustion only; do not mask auth/config errors."""
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if response is not None:
        status = status or getattr(response, "status_code", None)
    text = _exception_text(exc)
    markers = (
        "rate_limit_exceeded", "insufficient_quota", "quota exceeded",
        "quota has been exceeded", "rate limit", "too many requests",
        "promotion_model_limit_reached",
        "requests per minute", "tokens per minute", "tokens per day",
        "tpm limit", "rpm limit", "tpd limit", "http error 429",
        "status code: 429", "status_code=429",
    )
    return status == 429 or any(marker in text for marker in markers)


def _is_upstream_provider_rate_limit(exc: BaseException) -> bool:
    text = _exception_text(exc)
    return any(marker in text for marker in (
        "tpm_rate_limit_exceeded", "limit_source=upstream_provider",
        "upstream_provider_shared_pool",
    ))


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


async def _call_openkilo_with_quota_wait(payload: dict, prompt_name: str):
    """Retry OpenKilo after quota exhaustion, with bounded exponential backoff."""
    wait_limit = _positive_int_env("LLM_QUOTA_WAIT_SECONDS", 1800)
    delay = _positive_int_env("LLM_QUOTA_RETRY_INITIAL_SECONDS", 15)
    max_delay = _positive_int_env("LLM_QUOTA_RETRY_MAX_SECONDS", 300)
    started = perf_counter()
    request_payload = dict(payload)
    retried_via_auto_free = False

    while True:
        try:
            return await _call_model_once("worker", request_payload, prompt_name)
        except Exception as exc:
            if (request_payload.get("_kilo_model") and not retried_via_auto_free
                    and _is_upstream_provider_rate_limit(exc)):
                request_payload.pop("_kilo_model", None)
                retried_via_auto_free = True
                logger.warning("Pinned Kilo model hit an upstream limit; retrying through Auto Free")
                continue
            if not _is_quota_or_rate_limit(exc):
                raise
            elapsed = int(perf_counter() - started)
            remaining = wait_limit - elapsed
            if remaining <= 0:
                raise OpenKiloQuotaWaitTimeout(
                    f"OpenKilo remained rate-limited for {wait_limit} seconds"
                ) from exc
            sleep_for = min(delay, max_delay, remaining)
            logger.warning(
                "OpenKilo request was rate-limited; retrying in %s seconds "
                "(waited %s/%s seconds, stage=%s)",
                sleep_for, elapsed, wait_limit, prompt_name,
            )
            await asyncio.sleep(sleep_for)
            delay = min(delay * 2, max_delay)


def _validate_provider_response(prompt_name: str, payload: dict, result: dict):
    if not isinstance(result, dict):
        raise ValueError('provider response must be a JSON object')
    if prompt_name == 'planner':
        focus = result.get('focus')
        if not isinstance(focus, str) or not focus.strip():
            raise ValueError('planner response focus must be a nonempty string')
    elif prompt_name in {'validator', 'repair'}:
        if not isinstance(result.get('cases'), list) or not all(
                isinstance(case, dict) for case in result['cases']):
            raise ValueError(f'{prompt_name} response cases must be list[object]')
        if not isinstance(result.get('issues'), list):
            raise ValueError(f'{prompt_name} response issues must be a list')
    elif prompt_name == 'agent_trace_navigation_review':
        if result.get('next') is not None and not isinstance(result.get('next'), dict):
            raise ValueError('trace navigation review next must be an object or null')
    elif prompt_name == "agent_trace_supervisor_feedback":
        if not isinstance(result.get("correction"), str) or not result["correction"].strip():
            raise ValueError("trace supervisor feedback correction must be a nonempty string")
    elif prompt_name == "agent_trace_claim_feedback":
        if not isinstance(result.get("correction"), str) or not result["correction"].strip():
            raise ValueError("trace claim feedback correction must be a nonempty string")
    elif prompt_name == "agent_case_supervisor":
        if not isinstance(result.get("correction"), str) or not result["correction"].strip():
            raise ValueError("case supervisor correction must be a nonempty string")
    elif prompt_name == 'agent_trace_repair':
        if not isinstance(result.get('claims'), list):
            raise ValueError('trace repair response claims must be a list')
        if result.get('next') is not None and not isinstance(result.get('next'), dict):
            raise ValueError('trace repair response next must be an object or null')
        if result.get('terminal') not in {'source_end', 'unresolved', 'cuda_launch_candidate', 'binary_candidate'}:
            raise ValueError('trace repair response terminal is invalid')
    elif prompt_name == 'agent_trace_validator':
        approved = result.get('approved_indices')
        claims = payload.get('claims', [])
        if (not isinstance(approved, list)
                or any(type(index) is not int or not 0 <= index < len(claims)
                       for index in approved)):
            raise ValueError('trace validator response has invalid approved_indices')
        rejected = result.get('rejected_indices')
        reasons = result.get('rejection_reasons')
        if rejected is not None and (
                not isinstance(rejected, list)
                or any(type(index) is not int or not 0 <= index < len(claims)
                       for index in rejected)
                or set(approved) & set(rejected)
                or set(approved) | set(rejected) != set(range(len(claims)))):
            raise ValueError('trace validator response has invalid rejected_indices')
        if reasons is not None and (
                not isinstance(reasons, list)
                or any(not isinstance(item, dict)
                       or type(item.get('index')) is not int
                       or item['index'] not in (rejected or [])
                       or not isinstance(item.get('reason'), str)
                       or not item['reason'].strip()
                       for item in reasons)
                or {item['index'] for item in reasons} != set(rejected or [])):
            raise ValueError('trace validator response has invalid rejection_reasons')
    return result


async def call_model(role: str, payload: dict, prompt_name: str):
    """
    Nemotron plans and supervises traces; OpenKilo researches and validates.
    If OpenKilo is quota-limited, wait with exponential
    backoff up to LLM_QUOTA_WAIT_SECONDS (default 30 minutes).
    """
    if prompt_name in {"planner", "agent_trace_repair", "agent_trace_supervisor_feedback", "agent_trace_claim_feedback", "agent_trace_navigation_review", "agent_case_supervisor"}:
        result = await _call_model_once(role, payload, prompt_name)
        try:
            return _validate_provider_response(prompt_name, payload, result)
        except ValueError as exc:
            if prompt_name == "planner":
                logger.warning("Nemotron planner returned an invalid response; retrying once: %s", exc)
                retry_payload = dict(payload)
                retry_payload["format_correction"] = (
                    "The previous response was invalid. Return exactly one JSON object "
                    "with a nonempty string field named focus."
                )
            elif prompt_name == "agent_trace_supervisor_feedback":
                logger.warning("Nemotron supervisor feedback failed schema validation; retrying once: %s", exc)
                retry_payload = dict(payload)
                retry_payload["format_correction"] = (
                    f"The previous response failed local schema validation: {exc}. "
                    "Return exactly one JSON object with a nonempty string field named correction."
                )
            elif prompt_name == "agent_trace_claim_feedback":
                logger.warning("Nemotron claim feedback failed schema validation; retrying once: %s", exc)
                retry_payload = dict(payload)
                retry_payload["format_correction"] = (
                    f"The previous response failed local schema validation: {exc}. "
                    "Return exactly one JSON object with a nonempty string field named correction."
                )
            elif prompt_name == "agent_case_supervisor":
                logger.warning("Nemotron case feedback failed schema validation; retrying once: %s", exc)
                retry_payload = dict(payload)
                retry_payload["format_correction"] = (
                    f"The previous response failed local schema validation: {exc}. "
                    "Return exactly one JSON object with a nonempty string field named correction."
                )
            elif prompt_name == "agent_trace_repair":
                logger.warning("Nemotron repair response failed schema validation; retrying once: %s", exc)
                retry_payload = dict(payload)
                retry_payload["format_correction"] = (
                    f"The previous response failed local schema validation: {exc}. "
                    "Return exactly one JSON object with claims as a list of objects, "
                    "next as an object or null, and terminal as one of "
                    "source_end, unresolved, cuda_launch_candidate, or binary_candidate. "
                    "Do not add markdown or change the evidence rules."
                )
            else:
                raise
            result = await _call_model_once(role, retry_payload, prompt_name)
            return _validate_provider_response(prompt_name, retry_payload, result)

    if role == "worker":
        return await _call_openkilo_with_quota_wait(payload, prompt_name)

    result = await _call_openkilo_with_quota_wait(payload, prompt_name)
    try:
        return _validate_provider_response(prompt_name, payload, result)
    except ValueError as exc:
        if prompt_name != "agent_trace_validator":
            raise
        logger.warning("OpenKilo trace validator returned an invalid response; retrying once: %s", exc)
        retry_payload = dict(payload)
        retry_payload["format_correction"] = (
            f"The previous approved_indices response failed local validation: {exc}. "
            f"Return only integer indices from 0 through {len(payload.get('claims', [])) - 1}; "
            "use an empty list when no claim is supported."
        )
        result = await _call_openkilo_with_quota_wait(retry_payload, prompt_name)
        return _validate_provider_response(prompt_name, retry_payload, result)

def demo_worker(task):
    """Example only, NOT a verified source-derived optimization."""
    from .schema import normalize_case
    snippet = task.get("context", "").lower()
    if not any(s in snippet for s in ("fuse", "fusion", "epilogue", "rewrite", "kernel", "dispatch", "batch_norm")):
        return []
    case = normalize_case({
        "framework": "UNKNOWN", "framework_version": task.get("revision", "UNKNOWN"),
        "library": "UNKNOWN", "library_version": task.get("revision", "UNKNOWN"),
        "backend": "UNKNOWN", "hardware": "UNKNOWN", "optimization_stage": "UNKNOWN",
        "optimization_name": "demo_candidate_requires_review", "layer_combination": "UNKNOWN",
        "operator_pattern": "UNKNOWN", "topology_constraints": "UNKNOWN",
        "layer_hyperparameters": "UNKNOWN", "tensor_constraints": "UNKNOWN", "compiler_flags": "UNKNOWN",
        "applicability_predicate": "UNKNOWN", "optimization_outcome": "possible_source_optimization",
        "fusion_type": "other", "fused_operations": "UNKNOWN", "remaining_operations": "UNKNOWN",
        "expected_kernel_effect": "possible_change", "alternative_outcome": "UNKNOWN",
        "source_repository": task["repository"], "source_file": task["source_file"],
        "source_function": task["source_function"], "source_lines": task["source_lines"],
        "evidence_status": "INFERRED", "validation_status": "PENDING",
    })
    if task.get("repository") == "synthetic-demo" and task.get("source_function") == "example_fuse_conv_bn":
        case.update({
            "case_id": "DEMO_CONV_BN_FOLD",
            "framework": "UNKNOWN",
            "optimization_name": "illustrative_conv_bn_folding",
            "layer_combination": "Conv -> BatchNorm",
            "operator_pattern": "conv -> batch_norm",
            "topology_constraints": "sequential input (illustrative)",
            "layer_hyperparameters": json.dumps({"training": False}),
            "applicability_predicate": "not training and conv.out_channels == bn.num_features",
            "optimization_outcome": "fold_bn_into_conv (illustrative)",
            "alternative_outcome": "separate_conv_and_bn",
            "fused_operations": json.dumps(["conv", "batch_norm"]),
            "remaining_operations": json.dumps([]),
        })
    elif task.get("repository") == "synthetic-demo" and task.get("source_function") == "example_epilogue":
        case.update({
            "case_id": "DEMO_MATMUL_EPILOGUE",
            "optimization_name": "illustrative_matmul_epilogue",
            "layer_combination": "MatMul -> Bias -> ReLU",
            "operator_pattern": "matmul -> bias -> relu",
            "topology_constraints": "sequential input (illustrative)",
            "layer_hyperparameters": json.dumps({"enable_epilogue": True}),
            "applicability_predicate": "enable_epilogue and bias is not None",
            "optimization_outcome": "matmul_bias_relu_epilogue (illustrative)",
            "alternative_outcome": "separate_kernels",
            "fusion_type": "epilogue",
            "fused_operations": json.dumps(["matmul", "bias", "relu"]),
            "remaining_operations": json.dumps([]),
        })
    return [case]
