# CONDUCTOR integration (Nemotron-Orchestrator-8B Q8_0)

The existing `conductor`/`conductor_bounded` graph modes request a JSON plan
(`provider`: `kilo` or `groq`, `focus`: nonempty string). They do NOT call
OpenAI `tools` in the graph; tool calling was tested separately at the model API.

On the remote machine, when the llama-server listens on 127.0.0.1:18327:

```bash
export ORCHESTRATOR_MODEL='openai/conductor'
export ORCHESTRATOR_BASE_URL='http://127.0.0.1:18327/v1'
export ORCHESTRATOR_API_KEY='local-server'
export ORCHESTRATOR_MAX_OUTPUT_TOKENS=512
```

Configure the **existing** worker/validator endpoints too: OPENKILO_MODEL,
OPENKILO_BASE_URL, GROQ_MODEL, GROQ_API_KEY, optionally GROQ_BASE_URL.
No real model API calls occur in `--demo` mode. Do not put credentials in git.

```bash
python -m pytest -q
python -m optimization_explorer --db results/live.sqlite index examples --repository synthetic-demo --revision example-v1 --max-files 2 --max-hits 1
python -m optimization_explorer --db results/live.sqlite run --mode conductor_bounded --limit 1 --concurrency 1 --max-repair 0 --log-model-metrics
python -m optimization_explorer --db results/live.sqlite stats
```

The live run needs valid OpenKilo and Groq endpoints; absence or failures are
recorded as unresolved tasks. Check output before treating it as verified research.
