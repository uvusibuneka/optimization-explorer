import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from .indexer import iter_directory_tasks, iter_zip_tasks
from .storage import Store

def missing_live_configuration(environ, mode):
    required = ["OPENKILO_MODEL", "OPENKILO_BASE_URL", "GROQ_MODEL"]
    local_validator = (
        environ.get("GROQ_BASE_URL") == "http://127.0.0.1:18766/v1"
        and environ.get("GROQ_MODEL", "").startswith("openai/")
    )
    if not local_validator:
        required.append("GROQ_API_KEY")
    if mode != "fixed":
        required.append("ORCHESTRATOR_BASE_URL")
    return [name for name in required
            if environ.get(name, "").strip() in ("", "replace_me")]


def main():
    p = argparse.ArgumentParser(prog="optimization-explorer", description="Static source optimization discovery via LangGraph + LiteLLM")
    p.add_argument("--db", default="results/optimization_cases.sqlite")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("index", help="Index a source directory or ZIP without executing source")
    s.add_argument("path")
    s.add_argument("--repository", required=True)
    s.add_argument("--revision", default="UNKNOWN")
    s.add_argument("--max-files", type=int, default=1000)
    s.add_argument("--max-hits", type=int, default=3)
    r = sub.add_parser("run", help="Process queued tasks; --demo avoids external APIs")
    r.add_argument("--demo", action="store_true")
    r.add_argument("--limit", type=int, default=10)
    r.add_argument("--concurrency", type=int, default=3)
    r.add_argument("--mode", choices=("fixed", "conductor", "conductor_bounded"), default="fixed")
    r.add_argument("--log-model-metrics", action="store_true", help="Log model latency and token usage to stderr")
    r.add_argument("--max-repair", type=int, default=2)
    e = sub.add_parser("export")
    e.add_argument("--out", default="results/optimization_cases.csv")
    ing = sub.add_parser("ingest", help="Index source repositories")
    ing.add_argument("--sources", default="sources")
    ing.add_argument("--max-files", type=int, default=0, help="0=unlimited")
    ing.add_argument("--max-hits", type=int, default=0, help="0=unlimited per file")
    inv = sub.add_parser("inventory", help="Show source repository availability")
    inv.add_argument("--sources", default="sources")
    sub.add_parser("stats")
    search = sub.add_parser("search", help="Find indexed source fragments")
    search.add_argument("repository")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=10)
    sub.add_parser("graph")
    args = p.parse_args()
    if args.cmd == "inventory":
        from .workspace import inventory
        print(json.dumps(inventory(args.sources),indent=2,ensure_ascii=False))
        return
    if args.cmd == "graph":
        from .graph import build_graph
        print(build_graph().get_graph(xray=True).draw_mermaid())
        return
    store = Store(args.db)
    if args.cmd == "ingest":
        from .workspace import ingest
        print(json.dumps(ingest(args.sources,store,args.max_files,args.max_hits),indent=2,ensure_ascii=False))
    elif args.cmd == "index":
        if args.max_files < 1 or args.max_hits < 1:
            p.error("--max-files and --max-hits must be positive")
        fn = iter_zip_tasks if args.path.lower().endswith(".zip") else iter_directory_tasks
        tasks = fn(args.path, args.repository, args.revision, args.max_files, args.max_hits)
        count = store.add_indexed_tasks(tasks)
        print(json.dumps({"indexed_tasks": count, "stats": store.stats()}, indent=2))
    elif args.cmd == "run":
        if args.limit < 1 or args.concurrency < 1 or args.max_repair < 0:
            p.error("limit/concurrency must be positive and max-repair nonnegative")
        store.conn.close()
        from .graph import run_graph
        if not args.demo:
            missing = missing_live_configuration(os.environ, args.mode)
            if missing:
                p.error("Missing live configuration: " + ", ".join(missing))
        if args.log_model_metrics:
            metrics = logging.getLogger("optimization_explorer.llm")
            metrics.setLevel(logging.INFO)
            metrics.addHandler(logging.StreamHandler())
            metrics.propagate = False
        print(json.dumps(asyncio.run(run_graph(args.db, args.demo, args.limit, args.max_repair, args.concurrency, mode=args.mode)), indent=2))
        return
    elif args.cmd == "export":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        print(f"Exported {store.export(args.out)} cases to {args.out}")
    elif args.cmd == "search":
        from .retrieval import retrieve
        if args.limit < 1:
            p.error("--limit must be positive")
        results = retrieve(args.query, store.source_fragments(args.repository), args.limit)
        print(json.dumps([{k: hit[k] for k in ("source_file", "source_function", "source_lines", "context")}
                          for hit in results], indent=2, ensure_ascii=False))
    elif args.cmd == "stats":
        print(json.dumps(store.stats(), indent=2))
    store.conn.close()

if __name__ == "__main__":
    main()
