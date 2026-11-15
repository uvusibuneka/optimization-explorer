"""Exact schema v1.0. Unknown values are explicit; examples are not source evidence."""
import hashlib
import json
from typing import Any

COLUMNS = "case_id,framework,framework_version,library,library_version,backend,hardware,optimization_stage,optimization_name,layer_combination,operator_pattern,topology_constraints,layer_hyperparameters,tensor_constraints,compiler_flags,applicability_predicate,optimization_outcome,fusion_type,fused_operations,remaining_operations,expected_kernel_effect,alternative_outcome,source_repository,source_file,source_function,source_lines,evidence_status,validation_status".split(",")
FUSION = {"vertical", "horizontal", "epilogue", "reduction", "none", "other"}
EFFECT = {"kernel_merge", "kernel_split", "kernel_replacement", "dependency_change", "kernel_selection", "possible_change"}
EVIDENCE = {"SOURCE_CONFIRMED", "INFERRED", "UNKNOWN"}
VALIDATION = {"PENDING", "VALIDATED", "REJECTED", "NEEDS_REVIEW"}
JSON_FIELDS = {"layer_hyperparameters", "tensor_constraints", "compiler_flags", "fused_operations", "remaining_operations"}

def normalize_case(raw: dict[str, Any]) -> dict[str, Any]:
    case = {k: raw.get(k, "UNKNOWN") for k in COLUMNS}
    for k in JSON_FIELDS:
        if isinstance(case[k], (dict, list)):
            case[k] = json.dumps(case[k], ensure_ascii=False, sort_keys=True)
    if not case["case_id"] or case["case_id"] == "UNKNOWN":
        identity = {k: case[k] for k in COLUMNS if k not in {"case_id", "validation_status"}}
        case["case_id"] = "CASE_" + hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
    return case

def validate_shape(case: dict[str, Any]) -> list[str]:
    errors = []
    for k in COLUMNS:
        if k not in case:
            errors.append(f"missing column {k}")
        elif not isinstance(case[k], str):
            errors.append(f"{k}: expected string (JSON-encoded for structured fields)")
    for k, allowed in [("fusion_type", FUSION), ("expected_kernel_effect", EFFECT), ("evidence_status", EVIDENCE), ("validation_status", VALIDATION)]:
        value = case.get(k)
        if not isinstance(value, str) or value not in allowed:
            errors.append(f"{k}: invalid value {value!r}")
    for k in JSON_FIELDS:
        val = case.get(k)
        if val != "UNKNOWN" and isinstance(val, str):
            try:
                json.loads(val)
            except ValueError:
                errors.append(f"{k}: expected JSON or UNKNOWN")
    return errors
