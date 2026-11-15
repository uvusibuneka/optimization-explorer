# Existing sm_75 graph correlation

Run after `tools/sm75_evidence.py` has imported the artifacts:

```bash
./.venv/bin/python tools/sm75_trace.py --db results/optimization_cases.sqlite --root ..
./.venv/bin/python -m unittest discover -s tests -p test_sm75_trace.py -v
```

This processes the existing `vLLM selective_scan_fwd` graph only. It validates graph endpoints and matches static CUDA launch sites to `sm75-kernels.jsonl` by exact mangled kernel symbol and library, with exact launch address and architecture evidence. The result is **static binary correlation**, not proof of runtime execution or selection predicates. Source-supported edges are retained as source assertions without re-verifying the referenced source files. Hypothesis edges remain hypotheses. No framework-wide lowering or runtime fusion is inferred. No optimization cases or atlas tables are changed.

The derived `sm75_trace_nodes`, `sm75_trace_edges`, and `sm75_trace_metadata` tables are replaced atomically on a successful run. With `--root`, source artifact SHA-256 values are checked against the imported artifacts before any derived data is changed. `runtime_verified` remains 0 unless a future separate runtime tracing implementation supplies execution evidence.
