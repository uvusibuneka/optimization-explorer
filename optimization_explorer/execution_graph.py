"""Evidence-based enrichment of sm_75 execution graphs."""
import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


def demangle(symbol):
    return subprocess.check_output(
        ["c++filt", "-n", symbol], text=True
    ).strip()


def enrich(graph, launch_sites):
    dispatchers = {
        n["name"]: n for n in graph["nodes"]
        if n["kind"] == "cpp_cuda_dispatch"
    }
    kernels = {
        n["name"]: n for n in graph["nodes"]
        if n["kind"] == "cuda_kernel_specialization"
    }
    edges = {
        (e["source"], e["target"]): e
        for e in graph["edges"]
    }
    matched = set()

    for row in launch_sites:
        if row.get("status") != "binary_symbol_matched":
            continue

        name = demangle(row["dispatcher_mangled"])
        match = re.search(r"selective_scan_fwd_cuda<[^>]+>", name)
        if match is None:
            raise ValueError(f"Unknown dispatcher: {name}")

        dispatcher = dispatchers[match.group(0)]
        kernel = kernels[row["kernel_mangled"]]

        if kernel["architecture"] != "sm_75":
            raise ValueError("Non-sm_75 kernel")

        types = [
            kernel["template_parameters"][f"parameter_{i}"]
            for i in (8, 9, 10)
        ]
        if types != dispatcher["template_types"]:
            raise ValueError("Dispatcher/kernel type mismatch")

        key = (dispatcher["id"], kernel["id"])
        if key in matched:
            raise ValueError("Duplicate launch mapping")
        matched.add(key)

        edge = edges[key]
        if edge["relation"] != "may_launch":
            raise ValueError("Unexpected edge relation")
        if edge["status"] != "hypothesis":
            raise ValueError("Edge already promoted")

        edge["status"] = "binary_supported"
        edge["evidence"].append({
            "kind": "static_cuda_launch",
            "tool": "objdump + c++filt + cuobjdump",
            "dispatcher_address": row["dispatcher_address"],
            "call_address": row["call_address"],
            "handle_mangled": row["handle_mangled"],
            "kernel_mangled": row["kernel_mangled"],
            "architecture": "sm_75",
            "runtime_verified": False,
            "report": row["_report_path"],
        })
        edge["note"] = (
            "Static CUDA launch handle matched to sm_75 kernel. "
            "Runtime launch and branch conditions not verified."
        )

    return len(matched)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--launch-sites", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        parser.error("Output already exists")

    graph = json.loads(args.graph.read_text())
    rows = json.loads(args.launch_sites.read_text())
    for row in rows:
        row["_report_path"] = str(args.launch_sites)

    count = enrich(graph, rows)
    args.output.write_text(
        json.dumps(graph, indent=2, ensure_ascii=False) + "\n"
    )
    print("MATCHED:", count)
    print("STATUSES:", dict(Counter(
        edge["status"] for edge in graph["edges"]
    )))
    print("OUTPUT:", args.output)


if __name__ == "__main__":
    main()
