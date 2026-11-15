# Import previously collected sm_75 evidence

From the repository root run:

```bash
./.venv/bin/python tools/sm75_evidence.py \
  --root . --db results/multilayer-live.sqlite
./.venv/bin/python tools/sm75_trace.py \
  --db results/multilayer-live.sqlite --root .
./.venv/bin/python -m unittest discover -s tests -p test_sm75_evidence.py -v
```

This **does not run cuobjdump**, rebuild binaries, scan sources, or create optimization cases.
It stores original JSON/JSONL/TSV/text payloads, demangled .so function
dumps, and Linear run-specific trace, source-provenance, launch, SASS, branch,
symbol, and cubin metadata with source paths and SHA-256 hashes. Cubin payloads
are not copied into SQLite; their size and SHA-256 are recorded. Raw `.so` and cubin files are indexed by size and SHA-256 only in the separate `sm75_artifacts` and `sm75_evidence` tables. The local pilot inventory includes 445 cubins (1,003,839,016 bytes); each file is streamed and stored as path, size, and hash. Re-running updates the
imported evidence idempotently. The existing case schema and `atlas_*` tables
are not changed. Missing artifacts are reported; `--strict` rejects them.

Important: graph edges are retained as supplied by the existing artifact; this
importer **does not validate their semantics** or assert layer-to-kernel proof.
The current local manifest layout imports the v2 sm_75 selective-scan graph and v5 launch-site map. An `ok` summary status does not establish complete kernel coverage; `timeout` does not mean an empty library. TensorFlow/PyTorch source scanning remains the
responsibility of `layer_atlas.py` after their source trees are present in
`sources`. This is an evidence import, **not yet a verified end-to-end
layer → lowering → dispatch → kernel trace engine**.
