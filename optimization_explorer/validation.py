"""Deterministic validation cannot establish semantic truth; status is conservative."""
import hashlib
import re
from zipfile import BadZipFile, ZipFile
from pathlib import Path
from .indexer import MAX_SOURCE_BYTES
from .schema import validate_shape

def check_case(case, task):
    errors = validate_shape(case)
    if local_path := task.get("local_source_path"):
        try:
            if member := task.get("zip_member"):
                with ZipFile(local_path) as archive, archive.open(member) as stream:
                    data = stream.read(MAX_SOURCE_BYTES + 1)
            else:
                with Path(local_path).open("rb") as stream:
                    data = stream.read(MAX_SOURCE_BYTES + 1)
            if len(data) > MAX_SOURCE_BYTES or hashlib.sha256(data).hexdigest() != task.get("source_sha256"):
                errors.append("source content changed since indexing")
        except (OSError, BadZipFile, KeyError, RuntimeError, NotImplementedError):
            errors.append("indexed source file cannot be read")
    if case.get("source_repository") != task["repository"]:
        errors.append("source_repository mismatch")
    if case.get("source_file") != task["source_file"]:
        errors.append("source_file mismatch")
    if case.get("source_function") not in (task["source_function"], "UNKNOWN"):
        errors.append("source_function not in indexed context")
    lines = case.get("source_lines", "")
    if lines != "UNKNOWN":
        m = re.fullmatch(r"([0-9]{1,12})(?:-([0-9]{1,12}))?", lines) if isinstance(lines, str) else None
        if not m:
            errors.append("source_lines must be N or N-M")
        else:
            lo, hi = int(m[1]), int(m[2] or m[1])
            indexed = task.get("source_lines")
            bounds = re.fullmatch(r"([0-9]{1,12})-([0-9]{1,12})", indexed) if isinstance(indexed, str) else None
            if not bounds or not 1 <= int(bounds[1]) <= int(bounds[2]):
                errors.append("indexed source line range is invalid")
            elif lo < 1 or lo > hi or lo < int(bounds[1]) or hi > int(bounds[2]):
                errors.append("source_lines outside indexed snippet")
    elif task.get("evidence_kind") != "binary":
        errors.append("source_lines unknown")
    for field in ("applicability_predicate", "operator_pattern", "optimization_outcome"):
        if field == "operator_pattern" and case.get("fusion_type") == "none" and case.get(field) == "UNKNOWN":
            continue
        if case.get(field) in ("UNKNOWN", "", None):
            errors.append(f"{field} unknown")
    if task.get("evidence_kind") == "binary" and case.get("evidence_status") == "SOURCE_CONFIRMED":
        errors.append("binary metadata cannot establish source-level confirmation")
    if case.get("evidence_status") == "SOURCE_CONFIRMED":
        errors.append("SOURCE_CONFIRMED requires independent source-level semantic review")
    return errors
