"""Offline preflight for the OptimizationExplorer agent pipeline.

Does not make API calls, load secrets from files, or modify the repository.
"""
import argparse
import importlib.util
import os
from pathlib import Path


def check(root: Path, environ=None):
    env = os.environ if environ is None else environ
    results = []
    for relative in ("tools/first_cases.py", "results/candidate_snippets.jsonl"):
        exists = (root / relative).is_file()
        results.append(("OK" if exists else "MISSING", relative))
    for relative in ("decompiled", "binaries"):
        exists = (root / relative).is_dir()
        results.append(("OK" if exists else "MISSING", relative + "/"))
    for name in ("OPENKILO_API_KEY", "OPENKILO_BASE_URL", "OPENKILO_MODEL", "GROQ_API_KEY", "GROQ_MODEL"):
        results.append(("OK" if env.get(name, "").strip() else "UNSET", name))
    for name, module in (("litellm", "litellm"), ("langgraph", "langgraph")):
        results.append(("OK" if importlib.util.find_spec(module) else "MISSING", "Python: " + name))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    results = check(args.root.resolve())
    for status, label in results:
        print(f"{status:7} {label}")
    print("\nNo API calls made. Variable names are provisional until agent code is inspected.")
    return 0 if all(status == "OK" for status, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
