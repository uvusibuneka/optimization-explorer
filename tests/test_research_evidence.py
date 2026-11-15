import json
from optimization_explorer.research_evidence import binary_candidates, trace_event


def test_binary_candidate_not_runtime(tmp_path, monkeypatch):
    root = tmp_path / "sm75"
    folder = root / "functions"
    folder.mkdir(parents=True)
    (folder / "libexample.functions.txt").write_text("Function : selective_scan_fwd_kernel\nFunction : unrelated\n")
    monkeypatch.setenv("OE_DECOMPILED_DIR", str(root))
    rows = binary_candidates({"source_function": "selective_scan_fwd_kernel", "source_file": "mamba.py"})
    assert len(rows) == 1
    assert rows[0]["runtime_verified"] is False
    assert rows[0]["evidence_status"] == "symbol_name_candidate_only"


def test_trace_jsonl(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("OE_TRACE_PATH", str(path))
    trace_event("worker_started", task_id="abc")
    assert json.loads(path.read_text())["event"] == "worker_started"
