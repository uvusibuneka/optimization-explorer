"""Bounded, read-only binary retrieval and auditable research-stage tracing.

The symbol catalogue is candidate evidence only, never proof of a runtime dispatch.
"""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

_STOP = {"torch", "cuda", "kernel", "fused", "fusion", "source", "function", "python", "device", "float", "tensor", "unknown", "forward", "backward", "impl", "native", "operator"}


def _root():
    configured = os.getenv("OE_DECOMPILED_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent.parent / "sources" / "decompiled" / "sm75"


def binary_candidates(task, max_files=4, max_symbols=8):
    """Rank relevant symbol *names* without reading whole SASS or asserting execution."""
    tokens = set(re.findall(r"[a-z][a-z0-9_]{3,}", str(task.get("source_function", "")).lower()))
    tokens |= set(re.findall(r"[a-z][a-z0-9_]{3,}", str(task.get("source_file", "")).lower().split("/")[-1]))
    tokens = {t for t in tokens if t not in _STOP and len(t) >= 5}
    if not tokens:
        return []
    directory = _root() / "functions"
    if not directory.is_dir():
        return []
    matches = []
    for path in sorted(directory.glob("*.functions.txt")):
        if not path.is_file() or path.stat().st_size > 50_000_000:
            continue
        found = []
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if any(token in line.lower() for token in tokens):
                    found.append(line.strip()[:400])
                    if len(found) >= max_symbols:
                        break
        if found:
            matches.append({"architecture": "sm_75", "symbol_file": str(path), "symbols": found,
                            "evidence_status": "symbol_name_candidate_only", "runtime_verified": False})
            if len(matches) >= max_files:
                break
    return matches


def trace_event(event, **fields):
    path = os.getenv("OE_TRACE_PATH")
    if not path:
        return
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode()
    fd = os.open(str(dest), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
