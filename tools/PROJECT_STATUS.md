# OptimizationExplorer project status

Last updated: 2026-11-15 12:12 MSK

Acceptance target: Promp.pdf requires reproducible evidence-backed paths from canonical PyTorch, TensorFlow, and Keras layers through runtime/backend dispatch to real CUDA launches and, where available, binary and sm_75 evidence. Tests and source-only traces do not satisfy runtime acceptance.

### Verified bootstrap checkpoint (2026-11-15 08:53 UTC)

- The initial source-backed seed completed with exit 0 and reported 198 candidates, 0 inserted, and 0 excluded. It ran without reprogen active, so pruning could proceed safely. Supervisor PID 1102289 now has trace PID 1111663 and reprogen PID 1111431 running concurrently. Both model endpoints are healthy; latest memory use is 55.84%; no trace errors were recorded in the latest 10-minute window. The prior crash loop was caused by re-arming seeding on every trace restart while reprogen held a running generation row.
- Fixed the supervisor to serialize the initial case seed before starting reprogen, retry a failed seed/discovery startup, and avoid re-seeding after successful bootstrap. Verified the post-seed concurrent state live. The dedicated regression covers the startup role gate; eval remains disabled.
- Live case inventory: 207 cases (Keras 90, PyTorch 52, TensorFlow 65). At the latest supervisor polls, the global repro registry moved from 71 to 72 `generated_ast_valid`, its retryable failed count moved from 5 to 4, 5 rows are `reused_eval_passed`, and 1 generation is running. **The generate-only worker does not execute generated scripts.**
- Eval state is unchanged at 25 passed / 6 pending. Root inventory is 173 Keras, 167 PyTorch, and 177 TensorFlow. Existing runtime/CUDA acceptance gaps in the table below remain open.
- Verification: full suite **161 tests passed**, `py_compile` passed, and `git diff --check` passed. The live restart verified seed completion, both concurrent workers, and one successful reprogen cycle. The trace backlog remains 536 pending / 32 unresolved after this short observation, so sustained DFS throughput is still unproven.

## Current audited stages

| Stage | Status | Current evidence / remaining gate |
|---|---|---|
| A. Audit existing project and evidence | PARTIAL | Current WSL checkout and 26-page Promp.pdf reviewed. The historical linear-dfs.sqlite cited below is absent from this checkout; its Nsight/SASS/cubin artifacts remain and hashes match their JSON manifest. The manifest's CASE_f79c37dd145897712434 row is not present in the live DB. |
| B. Stabilize infrastructure and recovery | PARTIAL | Supervisor PID 1102289 is active with trace PID 1133901 and generate-only repro PID 1111431. At 09:12 UTC both endpoints were healthy, memory was 55.2% used, and no trace errors occurred in the preceding 10 minutes. Long-term availability remains open. |
| C. Discover canonical framework layers | COMPLETE (current inventory) | 537 stored roots include 20 suppressed aliases; 517 unique active roots remain: Keras 173, PyTorch 167, TensorFlow 177. A source scan added 89 roots, including public exported Keras LSTM and Conv2D classes whose intermediate bases are imported. Discovery is reproducible but not an exhaustive semantic registry. |
| D. Orchestrate durable DFS | PARTIAL | Current queue: 5 completed, 532 pending, 1 running, 0 failed, 36 unresolved, and 20 suppressed aliases; 744 trace steps are persisted. The live runner records source and validation evidence and preserves retry state. End-to-end runtime acceptance remains open. |
| E. Trace Python/native/compiler/CUDA | PARTIAL | Existing evidence records an actual torch.nn.Linear run on Quadro RTX 6000 (sm_75) and a CUTLASS launch. The installed Python source hash differs from the checked-out source hash; current WSL has no PyTorch or nvidia-smi, so a fresh local CUDA run is unavailable. |
| F. Correlate launches with binary/sm_75 | PARTIAL | Imported the local vLLM selective_scan_fwd graph and kernel map. sm75_trace records 80 exact static binary correlations, 125 hypotheses, and 5 source assertions. Indexed all 445 pilot cubins (1,003,839,016 bytes) by path, size, and SHA-256. Runtime-verified edges remain 0; the fresh Linear profile still lacks the loaded-module hash. |
| G. Analyze optimization branches and fusion | PARTIAL | 207 active cases remain after source-hash and optimization-control review; 183 cases are archived with payload/provenance. The gate retains recognized flags and compared hyperparameters, plus selected shape dispatches that choose a recognized optimized implementation. Cases remain INFERRED/NEEDS_REVIEW. Controlled alternate configurations and verified fusion comparisons remain open. |
| H. Validate and bounded autonomous run | NOT COMPLETE | Evaluator execution remains disabled per user instruction. Eval state is 25 passed / 6 pending; the latest global repro registry has 98 generated_ast_valid, 5 reused_eval_passed, and 1 running. Among active cases, 72 have generated_ast_valid scripts, 3 reuse eval-passed scripts, and 1 generation is running. Newly generated scripts are not executed in this phase. |

### Historical snapshot (2026-11-15 07:46 UTC)

- The database has 517 unique active roots: Keras 173, PyTorch 167, TensorFlow 177 (537 stored roots including 20 suppressed aliases). Current DB query: 5 completed trace tasks, 544 pending, 1 running, 0 failed, 24 unresolved, and 20 suppressed aliases; 708 trace steps are persisted. These are source-trace metrics, not runtime acceptance.
- Of 206 active source cases, 35 have generated AST-valid scripts and 3 reuse hash-verified passing scripts; 168 remain without an accepted repro artifact. The global repro registry has 64 generated_ast_valid and 5 reused_eval_passed rows, including archived-case history. Generation-only does not execute scripts. Eval remains disabled; historical tables stand at 25 passed / 6 pending and 25 passed / 30 failed runs.
- `2f63aea` adds a third bounded Nemotron correction round for AST/policy-rejected repro scripts. `49ceccd` resolves TensorFlow module-alias and package-re-export base classes; live task `a7dab64062069c5c25699607` advanced to `tensorflow/python/keras/layers/pooling.py:393`. `4910d9d` resolves same-file calls to local function definitions; live Keras task `ce71f2d24124d8602d0ce788` advanced from line 3084 to `_layer_normalization` at 3094.
- Checkpoint `4ff0422` adds one deterministic source-validation correction retry for Nemotron navigation reviews; the existing local source candidate remains the fallback. The exported-root scan now recognizes public Keras/TF decorators and found LSTM/Conv2D roots; LSTM optimization branches were seeded with source evidence only. A previously failed TensorFlow navigation task advanced from line 4788 to line 4811 after a backup-first retry.
- Checkpoints `740ee78` and `868cb31` exclude training-only, backend-identity, and tensor-type branches. Checkpoint `a864554` also excludes ordinary shape adaptation unless the source path selects an optimization dispatch, retaining actual `vectorized_map`, fast-path, and kernel/algorithm routes. The latest DB backup `multilayer-live.sqlite.before-optimization-case-prune.bd57747e.bak` predates archival of 82 additional cases; total archived cases are 183. One static-shape-to-vectorized-map case was added, and two PyTorch MHA fast-path dimension cases remain active.
- Latest full test run: **157 passed**. All 38 accepted repro artifacts attached to active cases pass path containment, symlink, SHA-256, and current AST/import-policy checks; scripts were not executed. Supervisor PID 831265 remains active; the latest event reported healthy endpoints, memory at 49.5%, and zero recent task errors. The evaluator stays disabled and runtime acceptance is incomplete.
- Latest source-trace gaps: the Keras normalization path still needs to proceed through backend dispatch, and the BatchNorm / activation roots need to reach optimization-dependent branches and backend implementations. Python/native/CUDA runtime correlation and fresh sm_75 verification remain open; eval-loop execution remains deferred.

### Nemotron claim-review supervision (2026-11-15 08:05 UTC)

- OpenKilo claim validation now reports approved/rejected indices and a reason per rejected claim. Nemotron sends rejected claims and those reasons back to the worker for up to three correction rounds; each replacement must pass deterministic source/navigation checks and OpenKilo claim validation again. Every verdict and rejection reason is persisted in the trace step. Nemotron cannot override validator decisions or promote source evidence to runtime evidence.
- Regression scenario verifies rejection → Nemotron feedback → corrected worker response → independent revalidation, with the rejected and accepted verdicts retained. Full suite: **163 passed, 9 subtests passed**; `py_compile` and `git diff --check` pass.
- Live validation completed in task `999660951a510e0b71f7ee40`, source `keras/src/layers/pooling/global_average_pooling2d.py:12`: Kilo rejected a claim that added `super().__init__` arguments not present in the cited source. Nemotron gave one correction; the worker removed that claim, and the second independent review approved the remaining six claims. The two reviews and rejection reason are persisted. The task itself remains unresolved because the next source transition was not grounded; claim repair and navigation completion are separate outcomes.
- Supervisor PID 831265 remains in single-worker trace/reprogen alternation with fix/eval execution disabled. Latest observed queue: 5 completed, 543 pending, 1 running, 25 unresolved, 20 suppressed; repro registry: 71 generated-AST-valid, 5 reused passing, 161 cases without an accepted artifact. Evaluation remains at 25 passed / 6 pending and 25 passed / 30 historical runs.

## Historical verification and runtime snapshot (2026-11-14 15:58 UTC)

- py_compile for tools/trace_orchestrator.py and tools/fixevalloop.py: passed.
- .venv/bin/python -m pytest -q: **87 passed, 9 subtests passed**.
- Isolated DB: results/multilayer-live.sqlite; not the main research DB.
- 9 roots, 3 per framework.
- DFS: 13 failed, 3 unresolved; mostly gateway/LiteLLM connection failures, plus one timeout.
- Repro/eval: 3 passed, 6 failed. The passed Linear repro used CPU; inspect device before counting runtime evidence.
- Both trace and fix/eval watch processes were terminated after the snapshot; confirm no detached workers before restart.

## Next checkpoints

1. Continue monitoring the live supervisor, retry recovery, queue movement, memory, and provider failures; do not stop or replace a supervisor that already owns the lock.
2. Continue the supervisor's single-worker alternation between bounded DFS batches and serial repro generation. Current backlog is 168 cases without accepted repro artifacts; generated-only artifacts remain separate from eval evidence.
3. Recover the missing Linear case descriptor only from a verified source/artifact record; refresh the runtime-to-module-hash link on the remote GPU host when that deferred phase is resumed.
4. Keep the source candidates INFERRED/NEEDS_REVIEW until independent review and controlled runtime/configuration evidence meet the Promp.pdf gates.

### Nemotron worker correction loop (2026-11-15 06:17 MSK)

- A deterministic local validation failure now triggers up to three Nemotron supervisory feedback rounds. Each round passes the rejected worker JSON, exact validation error, and visible-source correction hints to Nemotron; OpenKilo receives Nemotron's concise instruction plus the prior response and must return a corrected full trace. The same local source/target validation and OpenKilo claim validator still gate persistence. If three worker corrections fail, the existing bounded direct Nemotron recovery remains available. Transport errors retain the existing recovery/retry path.
- Live validation on the single-worker trace process: after restarting only child PID 610909 through supervisor recovery, PID 642739 produced a persisted step with supervisor_feedback_attempts=2, supervisor_model=openai/conductor, and repair_model=openai/conductor; the task advanced to another pending DFS location. This proves two feedback cycles and final locally validated persistence. No second worker was started and fix/eval stayed disabled.
- Supervisor endpoints were healthy, memory use about 54%, with one worker transport error in the recent window. Queue at the observation: 4 completed, 439 pending, 1 running, 40 unresolved, 20 suppressed. The next work remains broad DFS and reproducible case-script generation.
- Verification after this change: full suite **137 passed, 9 subtests passed**; Python compileall and git diff --check passed.

### Provider request diagnostics (2026-11-15 06:27 MSK)

- Added INFO completion logs and WARNING failure logs containing provider, stage, model, endpoint hostname, outcome/category, exception class, and latency. Request bodies, URL paths, and credentials are not logged. Successful and failed completion calls can now be attributed to the configured endpoint rather than only to a generic task-level connection error.
- The active child PID 658633 loaded this change after supervisor lease recovery. Recent live calls completed through OpenKilo at 26.886 seconds for worker and 10.451 seconds for validator, both routed to nvidia/nemotron-3-ultra-550b-a55b:free. A later real correction chain logged worker (15.387s) -> Nemotron feedback (1.613s) -> worker (18.430s); the corrected step persisted with supervisor_feedback_attempts=1 and an OpenKilo validator model. The DB now contains five steps with supervisor feedback, including a previously observed two-round repair. No completion failure has recurred since restart, so the previous connection error's underlying cause remains unproven.
- Queue at this observation: 4 completed, 439 pending, 1 running, 40 unresolved, 20 suppressed. Evaluation remains disabled.

### Retrying recoverable validation failures (2026-11-15 06:40 MSK)

- Audited unresolved task rows against their exact stored errors. Eighteen had deterministic worker/repair quote, line-bound, next-location, or malformed-schema failures; twenty-two had no stored error and were left untouched. Added a backup-first retry command that only requeues this explicit validation-error allowlist and preserves trace_task_errors plus prior trace_results.
- First recovery requeued 18 tasks using results/multilayer-live.sqlite.before-validation-retry.f9fd8bbb.bak. A nineteenth error-backed task had an invalid OpenKilo approved_indices response; OpenKilo trace-validator calls now get one bounded format correction retry, and their failures use the separate claim_validator stage. That task was requeued after results/multilayer-live.sqlite.before-validation-retry.19fbb254.bak.
- After restarting the single child through supervisor lease recovery, PID 676852 advanced trace_steps from 522 to 530 and the queue from 23 to 22 unresolved. The invalid-approved-indices task is pending for retry but has not yet been observed completing. The 18 first-batch validation retries are also still pending at their original source locations; no retry success is claimed for them yet. Twenty-two tasks without response-validation errors remain unresolved pending source review.
- Verification: full suite **140 passed, 9 subtests passed**; compileall and git diff --check passed. Fix/eval remains disabled.

### Live long-request supervisor correction (2026-11-15 06:48 MSK)

- One OpenKilo worker completion took 315.925 seconds and returned successfully. Nemotron produced correction feedback in 1.665 seconds; OpenKilo returned the corrected response in 56.152 seconds. The resulting trace step persisted with supervisor_feedback_attempts=1 and passed the configured OpenKilo validator; a subsequent step also persisted. This verifies that a slow completion exceeding five minutes was allowed to finish and that Nemotron feedback was applied to the subordinate worker.
- The single trace child PID 676852 remains healthy. At the latest sample the queue was 4 completed, 457 pending, 1 running, 22 unresolved, 20 suppressed; 530 trace steps are stored. Seven persisted steps contain Nemotron feedback. Evaluation remains disabled.

### Prioritized validation recovery (2026-11-15 07:12 MSK)

- The 19 unresolved tasks with recorded, allowlisted validation errors were already requeued but remained behind the ordinary retry backlog. Added `retry_priority`, with backup-first migration of the live SQLite database at `results/multilayer-live.sqlite.before-retry-priority.bak`. Retry priority is consumed after one validated persisted step, retained across transient errors, and cleared on terminal failure.
- Checkpoint `7c170ea` contains the scheduler and tests. Full suite: **140 passed, 9 subtests passed**; `git diff --check` passed.
- The existing supervisor restarted only the trace child after its lease cleared. New child PID 719683 selected a `retry_priority=1` task first; it persisted a trace step, advanced its depth, and reset priority to 0. The sole OpenKilo worker remains active; fix/eval is still disabled.

### Serial DFS and repro-generation operation (2026-11-15 07:36 MSK)

- Added `--generate-repros` to the existing supervisor. It alternates a bounded two-task DFS process with a one-case `--generate-only` process, never starts both Kilo workers together, and leaves fix/eval disabled. Repro selection skips accepted artifacts and stale claims; failed generation rows are not retried forever. The supervisor reports the generation backlog and recovers interrupted generation claims.
- Added a systemd drop-in at `~/.config/systemd/user/oe-agent-supervisor.service.d/generate-repros.conf`; the existing credential drop-in remains loaded. Service PID 758052 runs the new command. In the live log, the trace child exited before `reprogen` PID 760939 started; the generator then exited successfully and trace restarted as PID 761858.
- Two new files were registered `generated_ast_valid`; both parse as Python and their SHA-256 values match the registry. Registry totals: 5 reused passing artifacts, 2 generated AST-valid artifacts, 250 cases still without an accepted artifact. Generation did not execute scripts. Existing eval tables stayed at 25 passed / 6 pending states and 25 passed / 30 failed historical runs.
- The active queue remains 4 completed, 457 pending, 1 running, 22 unresolved, and 20 suppressed. Endpoints were healthy and memory remained near 53% used in the observed snapshots. Full suite: **143 passed, 9 subtests passed**; Python compilation and diff checks passed. Repro scripts still need runtime/path validation in the explicitly deferred eval phase.

### First steady-state repro batches (2026-11-15 07:40 MSK)

- The alternating supervisor completed a second serial repro batch and returned to DFS. Three generated scripts parse as Python and all three hashes match `case_repro_scripts`; registry state is 3 generated AST-valid plus 5 reused passing, leaving 249 cases without artifacts. No scripts were executed and eval tables remain unchanged.
- Supervisor PID 758052 and one trace child PID 766129 are active; endpoints are healthy, memory is about 53% used, and the queue remains 4 completed / 457 pending / 1 running / 22 unresolved / 20 suppressed. The generator is scheduled again after the next DFS batch.

### Nemotron supervision for rejected repro scripts (2026-11-15 08:04 MSK)

- Fixed prompt routing so `agent_case_repro` and `agent_case_fix` receive their dedicated prompt without the unrelated optimization-case schema suffix.
- Generation-only repro flow now sends local AST/import-policy validation errors and the rejected script to Nemotron, then gives its bounded correction to OpenKilo. It allows at most two supervisor/worker repair rounds; accepted artifacts still require local validation, and scripts are not executed.
- Regression tests reproduced the rejected `import sys` error, verified the correction payload and configured Nemotron endpoint, verified OpenKilo received Nemotron's correction, and asserted the script was not executed. Targeted tests: **38 passed**; full suite: **149 passed, 9 subtests passed**; `git diff --check` passed.
- The live one-worker supervisor completed a generation-only batch after the change: `TRACE_034b4bbe2c56967091dd` was stored as `generated_ast_valid`; eval tables stayed at their historical 25 passed / 6 pending states. That case did not trigger the Nemotron repair branch, so an actual live repair cycle remains unobserved. Supervisor, endpoints, and memory checks remained healthy (49.46% used); the evaluator remains disabled.

### Live Nemotron repro correction and policy revalidation (2026-11-15 08:25 MSK)

- A live sequential probe sent a rejected `import sys` script and the exact validator error to Nemotron. Nemotron returned an actionable instruction; OpenKilo then produced a replacement accepted by the then-current `validate_script`. The script was not executed. An initial one-round probe showed the worker can still return a prohibited import, which is why the production flow allows a second bounded supervisor/worker round; subsequent artifact review tightened the validator further against `exit()` and `quit()`.
- The actual DDP repro `TRACE_01d55197dd9fe4937811` was retried and stored as `generated_ast_valid`; it reports the distributed-initialization blocker and now exits via `raise RuntimeError`. No claim is made that the DDP source path was reached.
- Review of that stored script found it had used the interactive `exit(1)` helper despite the prompt. The validator now rejects `exit()` and `quit()`. A backup was made at `results/multilayer-live.sqlite.before-repro-policy-revalidation-20261115-0821.bak`; the stale status was requeued, and both failed rows then regenerated as `generated_ast_valid` using one sequential worker.
- Revalidated all **16** accepted artifacts after the policy change: all file hashes match the registry, all pass the current AST/import/builtin policy, with zero stale files. After another live one-worker batch, registry totals are 12 `generated_ast_valid`, 5 `reused_eval_passed`, 243 pending, and no failed/running repro rows. Eval tables remain unchanged at 25 passed / 6 pending states and 25 passed / 30 historical runs; eval remains disabled.
- Supervisor restarted and is active with one DFS child. Latest observed memory use was about 53%; both configured endpoints reported healthy. Verification after the policy change: **149 passed, 9 subtests passed**, `git diff --check` passed.

### Reopening unresolved source navigation (2026-11-15 08:39 MSK)

- Audited all 22 unresolved tasks against current source. Seventeen have a concrete next source location that passes local `validate_step`: same-function calls, imported base classes, or later class methods. Five have no locally verifiable continuation and remain untouched.
- Added `--retry-unresolved-navigation`. It requeues only those validated candidates, preserves prior trace steps and error history, removes only the current visited marker, increments the trace sequence, sets retry priority, and creates a database backup first: `results/multilayer-live.sqlite.before-navigation-retry.0dfb4f06.bak`.
- The live scheduler selected retry-priority task `b0434ccba8672df946fb716e` under the existing single-worker configuration. It appended a validated step, advanced from `tensorflow/python/ops/nn_ops.py:3955` to `:3990`, and cleared retry priority. The unresolved count fell from 22 to 5; all 17 requeued tasks left the unresolved state. The supervisor remains active, alternates one DFS/reprogen worker, endpoints are healthy, and the recent error count is zero. Evaluation remains disabled.
- Verification: trace orchestrator tests **44 passed**; full suite **150 passed, 9 subtests passed**; `git diff --check` passed.

### Supervising empty trace claims (2026-11-15 08:50 MSK)

- Empty `claims` with `terminal='unresolved'` previously bypassed Nemotron feedback even when the current AST exposed a source continuation. The trace validator now treats that combination as a correctable response error only when a same-function call, imported base class, or next class method is locally resolvable. Responses without such a continuation remain unresolved.
- The three no-claim tasks with exact local continuations were requeued using backup `results/multilayer-live.sqlite.before-navigation-retry.a9f7547e.bak`. They are pending at retry priority; their new supervised trace attempts have not completed yet.
- Added an end-to-end test that supplies an empty worker response, verifies Nemotron feedback includes the empty-claims error, and verifies the corrected source claim advances the task. Trace tests: **45 passed**; full suite: **151 passed, 9 subtests passed**; `git diff --check` passed. Eval remains disabled.

### Autonomous queue snapshot (2026-11-15 08:52 MSK)

- The systemd supervisor is active and alternating one worker. Latest sample: both endpoints healthy, zero worker errors in ten minutes, memory use 52.94%.
- Repro registry: 18 `generated_ast_valid`, 5 `reused_eval_passed`, 233 pending; one generation-only worker is live. Trace state: 619 persisted steps, 477 pending, 2 unresolved; 15 retries have priority. The three no-claim tasks are queued for the newly committed Nemotron feedback path and have not completed yet.
- Eval remains disabled; historical tables remain 25 passed / 6 pending states and 25 passed / 30 runs.

## Evidence rule

Keep static source candidates, compiler evidence, runtime traces, CUDA launches, binary symbols, SASS, LLM hypotheses, and unresolved gaps distinct. A test, repro file, source call chain, or unrelated prior artifact is not proof of a CUDA launch for a current case.

### Gateway and tunnel diagnostic (2026-11-14)

- Local gateway `/health` returned HTTP 200. A structured worker request through `model=worker` returned HTTP 200 and JSON from the configured Kilo Auto Free route. A plain-text prompt returned 502 because the gateway requires a JSON object in the model response; the worker prompt must request the structured format.
- `optimization_explorer/llm.py` catches Groq validator/repair exceptions broadly and retries through OpenKilo, including connection and response-validation failures. Keep this fallback behavior.
- The reverse SSH tunnel logged banner timeouts and `remote port forwarding failed for listen port 18765`. During the same period, SSH reported `Connection closed by UNKNOWN port 65535`; a later SSH attempt succeeded. The GPU-side endpoint probe returned connection reset. The systemd unit can report active while autossh is retrying, so `active` alone does not prove the route works.
- The local Kilo worker path has a successful minimal probe, but the GPU-to-WSL route is intermittent/unverified. Resume DFS only after a successful GPU-side probe to `127.0.0.1:18765`, then verify Groq failure fallback end to end.


### Runtime evidence and gateway configuration update (2026-11-14)

- Fresh Nsight evidence is recorded at `results/linear-e2e-20261114/runtime-evidence.json` and linked to the true `module.bias is not None` evidence row for `CASE_f79c37dd145897712434` in `results/linear-sm75.8hW8WW/linear-dfs.sqlite`. The manifest records the fresh launch, source/profile hashes, matching SASS symbol, SASS hash, cubin hash, and sm_75 ELF architecture. It explicitly limits the binary claim to symbol/configuration correlation because the cubin capture is from an existing capture, not the fresh Nsight session.
- A live Groq transport failure followed by a valid OpenKilo validator response was exercised successfully.
- The local worker gateway accepts `openai/worker`; it returned 502 for `openai/kilo-auto/free` because that alias is not recognized by its HTTP handler. `tools/run_jlab.sh` and CLI preflight were changed to use local worker/validator gateways and require a Groq key only for non-local Groq endpoints.
- Targeted verification after these changes: `bash -n tools/run_jlab.sh`; `pytest tests/test_cli_live_configuration.py tests/test_llm_fallback.py -q` — 10 passed. Full-suite verification and a bounded cross-framework DFS/fix-eval run remain next.


### Autonomous supervisor verification (2026-11-14)

- tools/agent_supervisor.py starts the trace DFS and independent fix/eval watcher against the same isolated SQLite DB, monitors planner/OpenKilo/Groq health, host available memory, queue states and recent worker errors, writes JSONL snapshots, and restarts exited workers with bounded exponential backoff. Memory pressure and DB errors block new starts/restarts; Groq health does not block OpenKilo fallback work.
- A 120-second smoke run sampled every 5 seconds. Planner, worker and validator endpoints remained HTTP healthy; memory stayed at 12.07–12.17% used with at least 474 GiB available; recent DFS errors remained zero. Eval passed count increased from 7 to 10 while trace advanced steps.
- Smoke shutdown exposed stale claims. The supervisor now returns a stopped worker's leased DFS tasks and running eval claims to pending; fix/eval cancellation also terminates a child repro process and releases its claim. Recovery tests cover both roles.
- Verification after all changes: 94 passed, 9 subtests passed; bash syntax and Python compilation passed.
- Current long-run monitoring and multi-framework completion remain outstanding.


### Live supervisor run (started 2026-11-14 15:36 UTC)

- Detached supervisor PID is recorded in results/agent-supervisor-live.pid; trace and fix/eval roles share results/multilayer-live.sqlite. Monitoring is continuing with a 10-second sample interval.
- In 35 snapshots through 15:42 UTC, all planner/worker/validator health checks passed; host memory use ranged from 9.93% to 10.21%; no memory-pressure sample occurred.
- Controlled trace-worker termination was detected as exit -15; the supervisor recovered two leases and launched a replacement PID. Controlled fix/eval termination recovered its active claim and launched a replacement.
- One TensorFlow Conv1D repro was initially rejected because the worker prompt import allowlist contradicted the validator. The prompt now permits tensorflow/keras/numpy imports, with the same AST restrictions. Current isolated DB eval state is 12 passed and 3 running, with no final failed cases; 94 tests and 9 subtests pass.
- The live observation window is not complete. Keep monitoring to 20 minutes, then summarize queue movement, provider/task errors, restart events, and memory trend.


### Expanded discovery and framework-aware eval (2026-11-14 15:58 UTC)

- Framework repros now select their installed interpreter: PyTorch uses the app environment; TensorFlow and Keras use `/opt/conda/envs/base-tf/bin/python`. TensorFlow subprocesses set `TF_FORCE_GPU_ALLOW_GROWTH=true` to avoid reserving all GPU memory. Targeted tests and full suite: **95 passed, 9 subtests passed**.
- Commit `233f551` records framework-aware eval. A real TensorFlow 2.21 Conv1D smoke ran on Quadro RTX 6000 GPUs (sm_75) and returned `[2,5,4]`.
- Discovery scanned framework trees and inserted 426 roots: Keras 87, PyTorch 170, TensorFlow 178. The live trace queue contains 444 tasks (including five unresolved from earlier cases). This is source inventory, not 426 confirmed optimization cases.
- Supervisor remains live with trace concurrency 1, eval concurrency 2, and rediscovery enabled on trace-worker starts. It is monitoring the isolated live DB. At 15:58 UTC endpoints were healthy, host memory use about 10.2%, and no memory pressure was present.
- Provider capacity is currently the throughput blocker: worker and trace logs show OpenKilo quota/rate-limit responses and backoff retries (up to the configured 30-minute wait); earlier Groq TPD was also nearly exhausted. At this checkpoint eval has 15 passed, three running/waiting, zero final failed; trace inventory has 443 pending, three running, five unresolved. Do not treat a healthy HTTP endpoint as model availability.
- Full 20-minute autonomous observation after discovery restart is still in progress; use later snapshots to confirm work resumes after quota clears.


### Rate-limit attribution correction (2026-11-14 16:16 UTC)

- Kilo documents a 200 free-model requests/hour/IP limit, and notes that free upstream providers may impose additional limits. Therefore Auto Free can be rate-limited under parallel use; it is not accurate to describe that as practically impossible. See https://kilo.ai/docs/getting-started/rate-limits-and-costs.
- The local OpenKilo gateway invokes OpenCode and turns all CLI failures into HTTP 502, preserving only the exception string. At 19:04 local time it logged three upstream HTTP 429 failures; later OpenKilo calls returned HTTP 200. The response body was not retained, so whether those 429s came from Kilo's IP ceiling or a partner's shared pool is unverified. LLM retry logs now say “request was rate-limited” instead of asserting quota exhaustion.
- The separate Groq cause is explicit in provider error text: model `openai/gpt-oss-120b` reached 200,000 TPD. This causes trace validation to fall back to OpenKilo as required.
- After 20 minutes with the expanded discovery run, eval remained at 18 passed; the live queue was 445 pending, one running, five unresolved. Memory was about 10.3%, endpoints returned 200, and no DB errors occurred. The system is operational but has not yet demonstrated trace throughput over this window.


### Direct Kilo gateway and upstream attribution (2026-11-14 16:24 UTC)

- Replaced the worker gateway's OpenCode subprocess route with direct unauthenticated `https://api.kilo.ai/api/gateway/chat/completions`, retaining the `kilo-auto/free` model and local OpenAI-compatible endpoint. It forwards upstream status/body and `Retry-After`, instead of converting provider errors to an opaque 502.
- Verified a direct Kilo request returned HTTP 200 with `baidu/ernie-5.5-flash-preview:free`, then tested the same local gateway through the GPU reverse tunnel. The gateway journal captured a real 429 body: provider Baidu, `tpm_rate_limit_exceeded`, `limit_source=upstream_provider_shared_pool`; the next attempt 36 seconds later returned HTTP 200. This is an upstream shared-pool TPM limit, not evidence that the Kilo free-model per-IP request ceiling was reached.
- Kilo documents 200 free-model requests/hour/IP; upstream partner limits are separate. Current supervisor workers use the new gateway, with retry/backoff unchanged. Full suite after gateway and log changes: **98 passed, 9 subtests passed**.
- Throughput still requires verification: at 16:23 UTC the DB showed 18 eval passes and the trace root still running, with 445 pending and five unresolved. Groq TPD remains the recurring validator limit; every Groq validation failure still falls back to OpenKilo.


### Current Kilo endpoint correction (2026-11-15)

- The active WSL supervisor had been pointed at the legacy `api.kilo.ai/api/gateway` URL. An unauthenticated request there returned HTTP 429 `PROMOTION_MODEL_LIMIT_REACHED`.
- Kilo's current docs specify `https://ai-gateway.kilo.ai/api/v1`; an unauthenticated request to `kilo-auto/free` succeeded, resolved to `baidu/ernie-5.5-flash-preview:free`, and returned valid JSON through `optimization_explorer.llm._call_model_once`.
- The service defaults and health probe now use the current endpoint. See Kilo's [gateway authentication](https://kilo.ai/docs/gateway/authentication), [models and providers](https://kilo.ai/docs/gateway/models-and-providers), and [rate limits](https://kilo.ai/docs/getting-started/rate-limits-and-costs). The official anonymous limit is 200 free-model requests/hour/IP; upstream provider caps remain separate.
- `results/linear-e2e-20261114/runtime-evidence.json`, Nsight `.nsys-rep`/SQLite, SASS, and cubin files remain present and hash-verified. The separate `results/linear-sm75.8hW8WW/linear-dfs.sqlite` referenced by an older status note is missing; the E2E case row cannot currently be verified in the main DB.


### Trace validation and live rate-limit checkpoint (2026-11-15 22:42 UTC)

- Worker responses that fail deterministic validation receive one self-correction attempt with the prior error; if the next worker response fails validation again, it is stored as `unresolved` with the full reason in `trace_results`. Transport and quota errors retain their wait/retry behavior. Regression tests cover exact quotes and invalid next locations; full suite after this change: **111 passed, 9 subtests passed**.
- Restarted the local trace-only supervisor; trace concurrency remains 1 and fix/eval remains disabled. The new worker is alive and planner/worker health checks returned healthy.
- During the subsequent observation, OpenKilo returned rate-limit responses for worker calls and the client backed off 15, 30, then 60 seconds. The queue remained at 1 completed, 438 pending, 1 running, 12 unresolved; progress is not yet demonstrated. Current logs do not retain the provider response body, so the exact limiter (Kilo IP ceiling vs selected upstream) remains unverified.


### Kilo anonymous cap and authenticated credential handoff (2026-11-15 22:47 UTC)

- A direct, unauthenticated request to the current `https://ai-gateway.kilo.ai/api/v1/chat/completions` returned HTTP 429 `PROMOTION_MODEL_LIMIT_REACHED` with the message “Sign up for free to continue.” This is the current anonymous promotion gate, distinct from the previously misconfigured legacy route. Kilo documents anonymous free requests as IP-limited and requires an API key for programmatic clients outside its CLI/extension; exact relationship between this promotion error and the documented hourly ceiling is not established.
- Checked local OpenCode auth metadata without printing credentials: the saved `openkilo` entry is the six-character placeholder `unused`, not an account key. No Kilo extension session was found in the checked local config paths.
- `optimization_explorer/llm.py` now accepts `OPENKILO_API_KEY` or `OPENKILO_API_KEY_FILE` and sends a Bearer key for worker requests when configured; anonymous mode still strips Authorization. The service has a systemd `LoadCredential` drop-in pointing to `/home/kacha/.config/optimization-explorer/kilo-gateway-api-key` (mode 600). The key file is empty until the user finishes Kilo signup and supplies a Gateway API key.
- Targeted LLM tests: **15 passed**. Full suite: **110 passed, 9 subtests passed**. Authenticated Auto Free inference and renewed queue movement remain unverified pending that key.

- The user has registered for Kilo and is locating the Gateway API key. Do not ask them to paste it into chat; the local credential path is prepared for secure entry.


### Cross-model stability and unresolved audit (2026-11-15 02:11 MSK)

- A direct authenticated `kilo-auto/free` request succeeded with HTTP 200 and routed to `nvidia/nemotron-3-ultra-550b-a55b:free`. The active service has the credential via systemd `LoadCredential`; the key is not present in agent logs.
- The trace DB had 16 unresolved tasks at the audit checkpoint. Recent error-backed examples were multiline or overlong source quotes and invalid/out-of-range `next` locations. Some other unresolved results correctly record that the source did not establish a next edge. There was only one worker invocation per trace step; the earlier apparent duplicate was a truncated-output misread.
- `context()` truncated prompt lines to 300 bytes while quote validation checked the full source line. Validation now accepts quotes only from the visible excerpt, with explicit truncation markers. The worker prompt requires one-line exact quotes.
- Auto Free's routed model is captured and pinned per framework layer in `trace_model_assignments`, so DFS siblings and retries use the same provider model. The assigned model is recorded in trace-step evidence and survives restarts. An explicit upstream-provider limit retries through Auto Free once and updates the layer assignment to the newly routed model. Gateway-wide rate limits continue through the existing bounded wait.
- Targeted stability tests: 46 passed. Full suite: **115 passed, 9 subtests passed**; Python compilation and `git diff --check` passed. Restart the one-worker trace-only supervisor and verify actual persisted `worker_model` IDs and queue movement before calling the model-stability change validated.


### Nemotron worker repair supervisor (2026-11-15)

- Trace DFS now sends deterministic worker-response evidence/quote/navigation failures to the existing Nemotron gateway for one bounded repair attempt. Nemotron receives the exact source payload, rejected worker JSON, and concrete validation error; the corrected response must pass the same local `validate_step`, then the Kilo trace validator. No new service was started.
- The trace step records the repair model. Repair failures remain in the durable bounded retry/unresolved path.
- Live synthetic repair through WSL gateway `127.0.0.1:18327/v1`, model `openai/conductor`, returned a corrected exact source quote. The Windows loopback is not the service path; the existing WSL SSH forward is healthy.
- Verification: live synthetic Nemotron repair returned a corrected exact quote accepted by `validate_step`; an end-to-end DFS smoke persisted the repaired quote and `repair_model=openai/conductor`. Full suite passes **117 tests, 9 subtests**; `py_compile` and `git diff --check` pass. Checkpoint `50c0d96` contains the implementation. Restarted the existing supervisor at 02:34 MSK after checking its lease recovery; new trace worker PID `331858` is active, endpoint probes pass, memory is 50–54%, and the DFS queue is draining. Fix/eval remains disabled.


### Nemotron repair correction hint follow-up (2026-11-15)

- The first live repaired response still failed exact-quote validation: Nemotron returned `False,` for source line 1762. The deterministic validator already had the allowed visible source line, but only embedded it in an error string.
- Added `TraceValidationError.correction_hints` carrying the exact visible line and added an instruction to copy it byte-for-byte or drop the claim. Replayed the same real source/error through the existing Nemotron endpoint: it returned the exact line and local `validate_step` passed. Targeted suite: **48 passed**; full suite and activation pending.


### Iterative Nemotron repair and duplicate-root suppression (2026-11-15)

- The live repair error showed Nemotron could repeat an invalid short quote (False,) even with an exact-source hint. Trace repair now gives Nemotron up to three bounded rounds, passing each deterministic validation failure and current exact-source hints back to it. Every response still passes the same local validator before Kilo reviews claims. The trace step records the successful repair-round count.
- Regression coverage reproduces the first failed repair and verifies Nemotron receives the new error, then returns a source-exact quote. The project suite now passes **112 tests**; Python compilation and git diff --check pass.
- The live SQLite inventory contained seven duplicate natural-key roots. Migration created backup results/multilayer-live.sqlite.before-layer-dedup.bak, marked seven rows as aliases, moved their pending tasks to unresolved with canonical IDs, and preserved all existing trace and case rows. The partial unique index and queue filters prevent reprocessing aliases; discovery also reuses existing natural-key layers.
- Checkpoint: 94cd500. The previous worker stopped cleanly and returned its lease. Supervisor PID 379628 now owns the lock and trace worker PID 379629 is active. A duplicate launch was rejected by the lock, so the current owner was left untouched. Nemotron and Kilo endpoint checks are healthy, memory usage is about 55%, and fresh trace rows include the new repair_attempts field. Fix/eval remains disabled as requested.


### Live Nemotron second-round repair verification (2026-11-15 00:10 UTC)

- Replayed the exact prior ConvTranspose2d failure: worker quote False, at source line 1906 pointed to a comment, while the exact visible line was the explanatory comment about bias being hardcoded to False. A live second-round request to Nemotron returned that exact line; the unchanged local validator accepted it.
- The active supervisor reports both inference endpoints healthy, no memory pressure, one trace worker, and queue state completed=1, pending=402, running=1, unresolved=48. The most recent recorded errors are from before the restart; the new worker has added trace steps.
- Current WSL lacks nvidia-smi and PyTorch in the project venv, so a fresh local CUDA/Nsight run cannot be produced here. Existing Quadro RTX 6000/Nsight evidence remains on disk, but the older side database that contained its case row is missing; the runtime artifact is not yet linked as a reviewed optimization case in the live DB. The binary relation remains symbol/configuration correlation, not a captured module-hash match or performance validation.


### Supervisor live watch (2026-11-15 00:16 UTC)

- Across a 20-second poll, the active supervisor and trace worker stayed alive, endpoint checks remained healthy, and memory stayed near 55% with no pressure. The worker added two trace steps and forked two pending DFS paths (pending rose 402 to 404); no task errors were recorded in the latest 10 minutes.
- A trace task remains in flight at attempt 2 with its lease renewing. The supervisor does not expose per-request stage or duration, so an in-flight call cannot yet be distinguished from a slow model response using its event log alone.


### Semantic optimization-case deduplication (2026-11-15 00:34 UTC)

- The live DB contained eight pairs of rows with identical optimization-case fields and different IDs, generated from duplicate roots. A transactional migration wrote backup results/multilayer-live.sqlite.before-case-dedup.bak, retained each removed row as JSON in trace_case_aliases, and removed duplicate rows from the active optimization_cases table.
- The eight old trace_case_evidence rows remain unchanged; matching provenance was also copied to each canonical case ID. All 265 original case payloads were verified byte-for-value as either active canonical rows or alias payloads. The active case table now has 257 unique semantic rows and 257 semantic-key entries.
- New persistence code uses the semantic key before inserting a case and records further equivalent IDs as aliases. Regression tests cover the migration, provenance retention, and subsequent alias writes. Full suite: **114 tests passed**; py_compile and git diff --check passed.
- The current worker was already running when this code was added, and another supervisor owns the shared lock. I left that worker undisturbed; the migration protects existing data, and semantic dedup applies to new workers once that supervisor replaces the child.


### Repair-format retry and current child version (2026-11-15 00:37 UTC)

- The live worker recorded a malformed Nemotron repair response at 00:32 UTC: claims was not a list. The repair loop now catches deterministic schema/type errors from the Nemotron call itself and feeds that error into the next bounded correction round. Network failures remain on the ordinary retry path.
- Regression coverage simulates the malformed repair response followed by a valid exact quote. Full suite: **115 tests passed**; Python compilation and git diff --check passed.
- The current trace child PID 379629 was started before this latest code and a different process owns supervisor PID 379628's shared lock. A second supervisor launch was rejected by that lock. I left the live owner untouched and asked whether to restart only its child; the answer is pending.


### Thirty-three-minute supervisor observation (2026-11-15 00:41 UTC)

- From 00:08:25 to 00:41:13 UTC the supervisor wrote 183 snapshots. Its trace child PID 379629 remained stable with zero worker-exit/restart events. Memory use ranged 51.52-58.30%, free memory stayed above 2.41 GiB, and no memory-pressure condition occurred.
- Nemotron and Kilo health probes succeeded in 182/183 snapshots. One Nemotron probe failed at 00:40:08 UTC; it recovered on the next 10-second sample and remained healthy during four subsequent polls. No DB error was reported.
- The queue moved from 1 completed / 403 pending / 48 unresolved to 1 completed / 411 pending / 1 running / 52 unresolved. DFS created work, but no task reached completed in this window; this is a throughput and unresolved-case problem, not evidence of a stopped supervisor. The most recent task error remains the 00:32 repair-response schema failure.


### Nemotron unresolved-navigation review (2026-11-15)

- When a worker returns validated claims but no grounded next step (`terminal=unresolved`), the trace now asks the configured Nemotron validator to review one source transition. Suggestions pass the same exact-visible-quote, eligible-file, shared-symbol, and source-hash checks as worker navigation; unvisited optimization branch barriers and deterministic next-class-method traversal run first and retain priority. Nemotron cannot promote terminal states, runtime execution, or binary claims.
- Regression tests verify a valid cross-file helper transition is recovered, a fabricated quote is rejected, and an unvisited optimization branch barrier prevents review from bypassing it. Full suite: **118 tests passed**; `py_compile` and `git diff --check` passed. Checkpoint `f8db2b3` records the implementation.
- The live child PID 379629 predates this change and cannot use it until restarted. At 00:53 UTC the durable queue was completed=1, pending=410, running=1, unresolved=53; latest repair error was another overlong source quote at 00:48 UTC. The supervisor lock owner remains PID 379628. Restart of the child was requested earlier and awaits user direction; no process was stopped.


### Reload and short live check (2026-11-15 01:03 UTC)

- Sent SIGTERM only to the previously authorized trace child PID 379629. The supervisor PID 379628 recovered one leased task and launched fresh child PID 456160 from the committed repair retries, navigation review, and source-quote prompt. Fix/eval remained disabled.
- Over the next three minutes, both processes stayed alive, endpoints remained healthy, memory stayed below 62% used with no pressure, and the supervisor recorded no recent task errors. The queue changed from pending=410 to pending=411 while trace steps continued; completed stayed at 1 and unresolved at 53. This demonstrates DFS branch creation and continued processing, not completed-root throughput.


### Nemotron route correction and activation (2026-11-15 01:10 UTC)

- Audit found that the first navigation-review implementation had no prompt file and was routed by default to OpenKilo. Added an explicit Nemotron-only route, JSON shape validation, and dedicated supervisor prompt. The default generic Kilo repair/worker paths are unchanged.
- A live request through call_model to the configured local Nemotron gateway returned valid JSON with next=null. This proves route and response formatting, but not that the model can recover a particular source edge. Full suite: **120 tests passed**; Python compilation and diff checks passed. Checkpoint 6ee82df contains the route and prompt.
- Sent SIGTERM to child PID 456160; supervisor PID 379628 recovered its current lease and started child PID 471451 with the Nemotron-only route. The queue then stood at completed=1, pending=412, running=1, unresolved=53, with no new task errors.
- Trace evidence now records navigation_review_attempted separately from model ID so future audit can distinguish an actual review call from an unattempted step.

APPENDTEST


### Source-candidate repair and imported base tracing (2026-11-15 01:46 UTC)

- Live review had returned next=null even when the current function contained an exact same-file call. Added a deterministic same-function call candidate, passed it to Nemotron, and use the exact candidate as a fallback only after quote validation. Regression coverage proves the task advances without converting source inference into runtime evidence.
- The next live blocker was Keras Conv2D calling super().__init__ with imported BaseConv. Added import-resolved base-class candidates with an exact current class-declaration claim, exact target-class quote, shared-symbol validation, and target-file SHA256. Tests cover the local resolver and Nemotron-null fallback.
- Checkpoint bee3e51 contains imported-base tracing; earlier 012ae94 contains same-file call recovery. Full suite: **123 tests passed**; Python compilation and git diff checks pass.
- Seventeen retry tasks were created from claim-bearing unresolved records after backup results/multilayer-live.sqlite.before-nav-retry-20261115-042655.bak. Parent evidence and trace history remain intact. A second backup, results/multilayer-live.sqlite.before-base-retry-20261115-044004.bak, precedes one focused retry of the still-unresolved Keras Conv2D case.
- Supervisor PID 379628 and child PID 517554 are healthy. The child loaded the imported-base fix; the focused Conv2D retry remains pending under fair scheduling. Live queue at 01:46 UTC: completed=2, pending=430, running=1, unresolved=58. Memory stayed below 63 percent without pressure; provider endpoints remained healthy. One database-lock error occurred during the earlier multi-row retry enqueue and recovered to pending. The latest model validation error is still a long exact-quote mismatch.

### Duplicate-root status accounting (2026-11-15 02:06 UTC)

- Audit found 20 duplicate layer aliases were labeled unresolved, inflating the failure count. Changed deduplication to state suppressed and migrated the 20 existing rows after creating results/multilayer-live.sqlite.before-suppressed-duplicate-migration-20261115.bak. No task or evidence rows were deleted.
- Queue now reports completed=3, pending=431, running=1, unresolved=38, suppressed=20. The active trace lease remains in place; no worker restart was needed.
- Full suite: 124 tests passed. py_compile and git diff --check passed. Fix/eval remains disabled.

### Local sm_75 artifact import and static correlation (2026-11-15 02:18 UTC)

- Fixed the importer paths to match the current local artifact layout. It now indexes four external shared objects by SHA-256 and size without copying binary payloads into SQLite, imports the demangled .so function dumps, 11,849 kernel-index records, the v2 sm_75 execution graph, v5 launch-site map, SASS, and the recorded Linear run artifacts.
- Created results/multilayer-live.sqlite.before-sm75-import-20261115-0216.bak before writing the evidence tables. sm75_trace produced 211 nodes and 210 edges: 80 static_binary_correlated, 125 hypotheses, 4 source-only assertions, and 1 source-plus-binary assertion. runtime_verified remains 0.
- Local shared-object hashes match the recorded external manifest. The older selective-scan JSONL, exact Linear source provenance, CUPTI launch log, branch review/counters, and binary inventory files are absent and remain reported as missing; no substitutes were fabricated.
- Full suite: 125 tests passed. Fix/eval remains disabled. The live trace worker retained its lease during the import.

### Full pilot cubin inventory (2026-11-15 02:24 UTC)

- Extended the artifact importer to stream and hash every cubin under the local Linear pilot capture. Imported 445 records totaling 1,003,839,016 bytes; SQLite stores only paths, sizes, per-file SHA-256 values, and a deterministic aggregate hash. No cubin payloads were copied into the database.
- Backup: results/multilayer-live.sqlite.before-cubin-inventory-20261115-0223.bak. The aggregate inventory SHA-256 is a90865d2becb6f13984bd0c85f911a06eb8b6b26c90698a8e7292feef41c858f. The sm_75 graph still has 80 static correlations and 0 runtime-verified edges.
- Full suite: 126 tests passed before the local import. Fix/eval remains disabled; the trace worker stayed live and retained its lease.


### Nemotron recovery after OpenKilo worker failures (2026-11-15)

- Extended the existing Nemotron supervisor path: when an OpenKilo trace-worker request throws, Nemotron can derive a response from the identical bounded source excerpt. It includes the worker error for diagnosis, but the error is explicitly not source evidence. Every response must pass deterministic local quote/navigation validation and the independent Kilo claim review; failed recovery remains on the normal retry path.
- Added regression coverage for transport failure recovery. Full suite: **132 passed, 9 subtests passed**; Python compilation and git diff checks pass. No live queue or evaluator was started by this verification.
- The existing trace child predates this change; activate it by having the current supervisor replace only that child after checkpointing. The supervisor remains trace-only and fix/eval stays disabled.


### Generate-only repro preparation (2026-11-15)

- Added fixevalloop.py --generate-only: reads the 257 current source cases serially, asks one OpenKilo worker for each repro, applies the existing AST/import/API policy, writes scripts under results/generated-repros, and stores hashes/status separately in case_repro_scripts. It does not call execute_script or create/update case_eval_state and case_eval_runs.
- The targeted test confirms an AST-valid script is saved and eval tables remain absent. generated_ast_valid intentionally is not runtime or path-reached evidence.
- Live generation is not started in this checkpoint because the DFS OpenKilo worker is already active at concurrency 1; running both would add a second simultaneous worker against the same free gateway. The existing eval loop remains disabled. Current DB has 257 cases; historical eval tables contain 31 states and 55 run records.


### Nemotron schema retry and retry-queue recovery (2026-11-15 02:51 UTC)

- The live repair error was a malformed Nemotron schema response (claims was not a list). Commit d61a2f7 adds one same-provider retry with the exact local schema error and an explicit output shape. Commit 408acc9 schedules pending retries before untried work, highest attempt count first, so deep DFS backlog cannot starve their recovery.
- Full suite: **135 passed, 9 subtests passed**; Python compilation and diff checks pass. Trace child PID 610909 runs under supervisor PID 542991 with trace concurrency 1 and eval disabled.
- Across a 30-second observation, pending attempt-3 tasks fell from 135 to 133 while attempt-0 tasks rose from 275 to 277; queue stayed healthy with no DB error, provider endpoints healthy, and memory at 55.29% used. This is evidence that retry tasks re-entered tracing; it is not runtime/CUDA verification.
- The current source-case database has 257 cases. Generate-only repros are implemented but not yet generated because that would add a second concurrent OpenKilo worker. Runtime eval remains intentionally deferred.


### Reuse verified repro artifacts without a second worker (2026-11-15)

- Added hash-checked reuse of prior passed repros in generate-only mode. Five existing files had matching SHA-256, a passed eval-state row, a zero-exit passed run, and valid AST/import policy. They were copied to results/generated-repros and registered as reused_eval_passed. No model request or script execution occurred.
- The live eval tables remain unchanged at 25 passed/6 pending and 25 passed/30 failed. The repro registry contains five reused_eval_passed artifacts; 252 of 257 cases still need script generation. The trace worker remains the only OpenKilo worker.


### Linear runtime artifact re-audit (2026-11-15 02:58 UTC)

- Recomputed and matched all four hashes in runtime-evidence.json: clean.nsys-rep, clean.sqlite, linear.sass, and module-28.cubin. The Nsight SQLite contains four CUDA launches; one CUTLASS GEMM launch matches the manifest symbol and grid/block configuration. Nsight device metadata says Quadro RTX 6000, compute capability 7.5; the SASS explicitly declares sm_75.
- This verifies the capture artifacts and symbol/configuration correlation. The fresh profile still has no loaded-module hash, and neither the capture nor the current source proves that bias presence changes fusion/performance.
- The referenced CASE_f79c37dd145897712434 descriptor is absent from the live DB and git history; only its ID, predicate, true value, runtime metadata, and evidence links survive in runtime-evidence.json. Do not reconstruct a full optimization_cases row or attach it to another current candidate without the missing source-case provenance.


### Live retry backlog observation (2026-11-15 03:03 UTC)

- Over the observation interval, pending attempt-3 tasks fell from 122 to 120 and attempt-0 tasks rose from 288 to 290; this indicates two retried traces advanced instead of remaining on the retry queue. The queue still has 440 pending, one running, 39 unresolved, and 20 suppressed tasks.
- One OpenKilo worker remains active. Eval is disabled. Provider endpoint health was green; this does not establish upstream model availability on every future call.
