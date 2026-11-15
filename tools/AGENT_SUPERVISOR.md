# Autonomous trace research supervisor

Run from the GPU-host app directory:

    ./.venv/bin/python tools/agent_supervisor.py --db results/multilayer-live.sqlite --sources sources --out results/multilayer-repros --interval 10 --trace-concurrency 2 --eval-concurrency 3

The supervisor starts the trace DFS and separate fix/eval worker against the same
SQLite database. DFS writes cases there; fix/eval watches that table and generates,
executes, and repairs each case's Python repro. Worker logs and timestamped
resource/queue snapshots are written under results/agent-supervisor/.

Each snapshot records host available memory and used percentage, planner/worker/
validator endpoint health, DFS/eval queue states, recent DFS errors, worker PIDs,
and worker exits. Memory pressure or a database error prevents launching or
restarting workers. A missing planner or OpenKilo endpoint also blocks startup.
The Groq validator endpoint is monitored but does not gate work: its request
failures are handled by the Groq-to-OpenKilo fallback in llm.py.

The worker processes keep their own bounded task retries and repro repair loop.
The supervisor restarts crashed workers with exponential backoff capped at 120
seconds. Stop it with SIGINT/SIGTERM; it relays SIGINT to both workers and waits
for them to exit. A bounded run for supervisor verification can use
--duration-seconds 120. --min-available-gib and --max-used-percent set
the resource startup/restart thresholds.
