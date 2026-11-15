# Trace-first orchestration

The trace queue has roots for Keras, TensorFlow, and PyTorch. It discovers
public Layer subclasses in Keras/TensorFlow, public nn.Module subclasses in
PyTorch, and explicitly recognized functional entry points. Test files,
private names, and unsupported source paths are excluded. The inventory is
source-backed but is not an exhaustive semantic layer registry.

Each worker advances one source step. Same-file navigation requires an exact
target quote; cross-file navigation also requires a shared symbol grounded at
both ends. The DFS pauses at an optimization branch before allowing the agent to
skip past it. A Python branch qualifies only when its condition depends on a
recognized optimization flag or a compared hyperparameter and the source paths
differ. Generic shape, rank, or dimension checks also need source-path evidence that they select a recognized optimized, vectorized, fast-path, or kernel/algorithm dispatch; input-shape adaptation alone is excluded. Native dimension checks use the same dispatch requirement. The true/false source paths are saved as an idempotent row in
optimization_cases with INFERRED evidence and NEEDS_REVIEW; this records a
candidate and never claims a runtime or kernel effect. Cases with identical
fields apart from case_id share one active optimization_cases row; each removed
ID and its full payload are retained in trace_case_aliases. Provenance remains
under the original IDs and is also copied to the canonical case for unified
review.

## SASS branch gate

SASS branches are forked only when the edge contains an optimization_dependency
record with:
- dimension equal to optimization_flag or hyperparameter;
- a nonempty setting name;
- observations for at least two distinct setting values; and
- opposite taken/fallthrough outcomes across those values.

A single-run profile, even with nonzero branch counters, cannot meet this
criterion. The comparison record must be tied to the branch edge and its
disassembly hash. Accepted branch observations still have aggregate-run scope;
they do not prove co-occurrence with every decision in a DFS path.

## Live providers

The active local supervisor uses:
- Planner: existing Nemotron gateway at `http://127.0.0.1:18327/v1`.
- Worker and trace validator: Kilo Auto Free through the current Kilo AI
  Gateway at `https://ai-gateway.kilo.ai/api/v1`, model
  `openai/kilo-auto/free`. Anonymous free-model requests omit `Authorization`.

The older `https://api.kilo.ai/api/gateway` route is not used by the active
supervisor. Kilo documents anonymous access for eligible free models and a
200 requests/hour/IP gateway limit; the selected upstream can have separate
limits. The planner does not fall back. If a worker response fails deterministic
quote/navigation/schema validation, Nemotron gets the exact task excerpt, the
worker response, and the validator error for up to three repair rounds. Each
rejected correction and its latest validation error are sent back to Nemotron,
including malformed repair-response schemas. The accepted response goes through
the same local validator and then the independent Kilo claim validator.
If the OpenKilo worker request fails, Nemotron can independently produce a
response from the same bounded source excerpt. That response uses the same local
validator and independent Kilo claim validator; it cannot override validation
or certify runtime evidence. If Nemotron cannot recover a valid response, the
task uses the ordinary bounded retry path. Responses that remain invalid after three rounds stay in the
bounded task retry path and are eventually stored unresolved. The chosen
Nemotron model and repair-round count are recorded in the trace step. OpenKilo rate limits use exponential retry for up to 30 minutes by
default. Worker request timeout defaults to 300 seconds.

The independent OpenKilo claim validator returns approved and rejected claim
indices with a source-based reason for every rejection. Nemotron can send those
reasons to the worker for up to three correction rounds. Each replacement trace
passes deterministic source/navigation validation and the independent claim
validator again. The persisted step retains every validator verdict and reason;
Nemotron cannot change an approval or claim runtime evidence.

## Queue commands

Run from app/. Inspect the queue with:
    ./.venv/bin/python tools/trace_orchestrator.py --db results/optimization_cases.sqlite --sources sources

Execute a bounded run with:
    ./.venv/bin/python tools/trace_orchestrator.py --db results/optimization_cases.sqlite --sources sources --execute --limit 12 --concurrency 4

Back up and prune invalid roots with:
    ./.venv/bin/python tools/trace_orchestrator.py --db results/optimization_cases.sqlite --sources sources --prune-invalid-roots

Back up and suppress pending tasks whose branch decisions fail the current optimization gate:
    ./.venv/bin/python tools/trace_orchestrator.py --db results/optimization_cases.sqlite --sources sources --prune-ineligible-branches

The limit is the maximum number of agent steps. Concurrency must be 1 through
4. The framework tree is scanned only when the DB has no layer inventory; pass
--discover-layers to explicitly look for newly added public layers. Expired
leases are recovered when the CLI starts. prune-invalid-roots backs
up SQLite and deletes only roots excluded by current framework, source-path,
or source-file rules, including their task and step rows. It refuses to delete
roots that still have live leases.

reset-old-queue backs up SQLite before removing the old symbol queue and trace
research tables. It preserves optimization cases, atlas tables, and imported
sm75 evidence. Use it only for an intentional queue reset.
rebuild-layer-queue replaces only a queue that has not started.

## Evidence boundaries

Each discovered source branch creates one candidate case row containing the
predicate, both source-path ranges, source revision hash, and explicit unknowns.
Run with `--seed-branch-cases` to seed every currently inventoried layer root;
the command makes a SQLite backup before adding rows and is idempotent. The live
supervisor enables it at startup. Rows remain INFERRED/NEEDS_REVIEW until an
independent review upgrades them.
The trace step records quoted source claims and exact navigation evidence.
SASS navigation can use hash-linked disassembly and branch-counter artifacts.
runtime_verified and sm75_verified remain false unless a separate verifier
has established those facts. This tool does not infer full path co-occurrence,
cross-library lowering, fusion, or performance from source names or counters.

## Autonomous recovery

Run with --watch in its own process to keep draining branch forks and new
pending trace work. Failed model/validation steps retain their error and stage
in trace_task_errors, then retry with exponential backoff up to five attempts
by default. The last error is included in the next planner/worker payload so it
can correct the failed response. Pending retries are scheduled before untried work, with the most-attempted
retries first, so a large DFS backlog cannot starve failed corrections. After the retry
limit the task stays failed
with its error; --retry-failed resets only those task retry counters for
another bounded cycle. Example:

    ./.venv/bin/python tools/trace_orchestrator.py --db results/optimization_cases.sqlite --sources sources --execute --watch --concurrency 2 --max-attempts 5

With `--trace-only --generate-repros`, the supervisor runs one DFS worker and
one persistent generate-only repro worker concurrently. On startup, if
`--seed-branch-cases` is enabled, it waits for the trace worker's seed pass to
exit successfully before starting repro generation. This protects case pruning
from a concurrent repro claim. After seeding, the roles claim work from separate
SQLite queues; the repro worker writes AST/policy-validated scripts without
executing them or touching eval tables. Each role has its own process and is
recovered independently if it exits. This mode uses at most one Kilo request
from each role at a time. It does not start the deferred eval loop.

Run the generator alone with:

    ./.venv/bin/python tools/fixevalloop.py --db results/optimization_cases.sqlite --sources sources --generate-only --watch --limit 1 --concurrency 1
