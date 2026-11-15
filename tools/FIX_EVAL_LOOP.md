# Per-case Python repro and fix/eval loop

The trace orchestrator records source-backed cases. Run this separate process to
generate one Python repro per case, execute it, and give failures to a dedicated
OpenKilo worker for up to five fixes. Each script and run result is kept under
`results/case-repros/<case_id>/`; SQLite stores iteration hashes, exit codes,
stdout/stderr, status, and the last error in `case_eval_state` and
`case_eval_runs`. A passed script prints a final JSON record with `reached=true`
after invoking the real framework operation. Unfixed cases remain `failed` with
their error retained for investigation.

Generate-only mode writes one AST-valid repro per recorded case without executing it
or changing either eval table. It stores hashes and status in the separate
case_repro_scripts table and defaults to results/generated-repros/:

    ./.venv/bin/python tools/fixevalloop.py --db results/multilayer-live.sqlite --sources sources --generate-only --concurrency 1

Existing repros are reused only when their eval state and run both say passed,
the recorded SHA matches the local file, and the script passes AST validation.

generated_ast_valid means only that the script passed the AST/import/API policy;
it does not prove that the repro reaches the operation. Runtime validation remains
the separate fix/eval phase.

Start the fix/eval loop in one terminal, then run DFS separately:

    ./.venv/bin/python tools/fixevalloop.py --db results/optimization_cases.sqlite --sources sources --watch --concurrency 1
    ./.venv/bin/python tools/trace_orchestrator.py --db results/optimization_cases.sqlite --sources sources --execute --limit 12 --concurrency 2

Both are independent processes and both use the existing OpenKilo gateway; fix/eval
requests use the dedicated `agent_case_repro` and `agent_case_fix` prompts. Set
`OPENKILO_BASE_URL`, `OPENKILO_MODEL`, and `OPENKILO_API_KEY` in the environment
when the local gateway defaults are not appropriate. Increase fix/eval concurrency
only when the gateway and GPU can serve that many repro jobs.

A repro is parsed and restricted to framework/Python utility imports, runs in an
isolated interpreter with a timeout and bounded captured output, and cannot use
filesystem, network, subprocess, or shell APIs. Native-only cases that cannot honestly
be triggered from Python must remain failed with an explanation; they are not marked
passed from source inspection alone. Retry failed cases with `--retry`.
