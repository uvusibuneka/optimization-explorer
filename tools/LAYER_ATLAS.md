# Layer-first source atlas (single patch)

By default it scans Keras source, TensorFlow operator/kernel/compiler roots,
PyTorch module/compiler/CUDA roots, cuDNN Frontend, CUTLASS headers/examples,
and existing sm_75 decompilation artifacts. This includes C++ and CUDA
implementation files plus .so function-symbol dumps and host-dispatch assembly.
The JAX tree, tests, generated/vendor trees, and unrelated LLVM paths are
excluded. Binary dumps remain unlinked evidence. Pass --repositories to select
a subset of the six configured roots.

Run from the application root:

```bash
./.venv/bin/python tools/layer_atlas.py --sources sources --db results/optimization_cases.sqlite
./.venv/bin/python -m unittest discover -s tests -p test_layer_atlas.py -v
```

This scanner discovers *candidate* layer/operation definitions across supported source repository
folders, extracts Python constructor parameters, Python AST nested-call compositions,
source branch predicates, C++ fusion-site candidates, and `sm_75` text artifacts.
It excludes the JAX source tree and test files. It has **no hardcoded allowlist of layers**. `--max-files 0` means no file-count cap.

The atlas is stored in separate `atlas_*` tables; it does **not** populate `tasks`,
`optimization_cases`, or fabricate fusion cases. Existing 28-column case schema is unchanged.
`atlas_coverage` explicitly marks unproven lowering/fusion/kernel/sm75 connections as `unknown`.

## Coverage limitations (important)

- The selected path roots focus the scan on framework dispatch and GPU/native
  implementations. Other repository paths and source files are outside its scope.
- A source inventory cannot establish complete semantic coverage of every operation.
  Names not matching the broad lexical detector can be missed.
- Nested Python calls establish syntax, **not** compiler fusion. C++ matching lines are
  candidate sites, **not** validated fusion rules.
- All possible operation combinations form an unbounded space. This tool records observed
  compositions rather than inventing every Cartesian-product combination.
- It does not infer cross-repository edges from similar names, and does not prove that an
  `sm_75` artifact belongs to a specific framework layer.
- Predicate extraction does not solve branch constraints or generate concrete shape/dtype
  boundary cases; those require semantic analysis and execution evidence.
- The atlas refresh replaces only its own `atlas_*` rows. No preexisting cases are removed.

A genuinely complete evidence-backed layer → lowering → backend → binary and fusion
analysis requires additional traced call graph and compiler/dispatch semantics. The output
must not be represented as complete optimization-case coverage.
