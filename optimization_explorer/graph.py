"""Actual LangGraph: index -> Send(worker)xN -> splitter -> Send(review-subgraph)xN -> persist.
Review subgraph: validator -> conditional (finish/repair) -> validator; bounded by N.
"""
import logging
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from .llm import call_model, demo_worker
from .schema import normalize_case
from .validation import check_case
from .storage import CaseConflict, Store
from .retrieval import retrieve
from .research_evidence import binary_candidates, trace_event

MODES = ("fixed", "conductor", "conductor_bounded")

class MainState(TypedDict, total=False):
    db_path: str
    mode: str
    demo: bool
    max_repair: int
    limit: int
    tasks: list[dict]
    worker_outputs: Annotated[list[dict], operator.add]
    review_inputs: list[dict]
    reviewed: Annotated[list[dict], operator.add]
    summary: dict

class WorkerInput(TypedDict):
    task: dict
    demo: bool
    db_path: str
    mode: str

class ReviewState(TypedDict, total=False):
    task: dict
    cases: list[dict]
    demo: bool
    max_repair: int
    attempts: int
    errors: list[str]
    issues: list[str]
    done: bool
    reviewed: Annotated[list[dict], operator.add]

class ReviewOutput(TypedDict):
    reviewed: list[dict]

async def research_worker(state: WorkerInput):
    task = state["task"]
    try:
        trace_event("worker_started", task_id=task.get("task_id"), source_file=task.get("source_file"), source_function=task.get("source_function"), mode=state.get("mode"))
        if state["demo"]:
            cases = demo_worker(task)
        else:
            payload = {"task": task}
            payload["binary_candidates"] = binary_candidates(task)
            trace_event("evidence_retrieved", task_id=task.get("task_id"), binary_candidates=payload["binary_candidates"])
            worker_role = "worker"
            if state.get("mode", "fixed") != "fixed":
                store = Store(state["db_path"])
                try:
                    fragments = (
                        fragment for fragment in store.source_fragments(task["repository"])
                        if fragment["source_file"] != task["source_file"]
                    )
                    query = task["source_function"]
                    if query == "UNKNOWN":
                        query = task["source_file"]
                    limit = 3 if state["mode"] == "conductor_bounded" else 8
                    payload["related_context"] = [
                        {key: hit[key] for key in ("source_file", "source_function", "source_lines", "context")}
                        for hit in retrieve(query, fragments, limit)
                    ]
                finally:
                    store.conn.close()
                plan = await call_model("validator", payload, "planner")
                trace_event("planner_result", task_id=task.get("task_id"), plan=plan)
                focus = plan.get("focus")
                if not isinstance(focus, str) or not focus.strip():
                    raise ValueError("Planner must supply a nonempty focus")
                payload["analysis_focus"] = focus[:4000]
                logging.getLogger("optimization_explorer.llm").info(
                    "task_id=%s selected_worker=OpenKilo", task["task_id"]
                )
                worker_role = "worker"
            output = await call_model(worker_role, payload, "worker")
            cases = output.get("cases", [])
            trace_event("worker_result", task_id=task.get("task_id"), output=output)
            if not isinstance(cases, list):
                raise ValueError("worker output cases must be a list")
        trace_event("worker_completed", task_id=task.get("task_id"), case_count=len(cases))
        return {"worker_outputs": [{"task": task, "cases": cases, "errors": []}]}
    except Exception as exc:
        trace_event("worker_failed", task_id=task.get("task_id"), error=f"{type(exc).__name__}: {exc}")
        return {"worker_outputs": [{"task": task, "cases": [], "errors": [f"worker failure: {type(exc).__name__}: {str(exc)[:300]}"]}]}

def source_indexer(state: MainState):
    store = Store(state["db_path"])
    tasks = store.pending(state.get("limit", 10), source_only=True)
    trace_event("tasks_selected", count=len(tasks), tasks=[{"task_id": t.get("task_id"), "source_file": t.get("source_file"), "source_function": t.get("source_function")} for t in tasks])
    store.conn.close()
    return {"tasks": tasks}

def dispatch_workers(state: MainState):
    if not state["tasks"]:
        return "no_tasks"
    return [Send("research_worker", {"task": t, "demo": state.get("demo", False),
                                    "db_path": state["db_path"], "mode": state.get("mode", "fixed")})
            for t in state["tasks"]]

def case_splitter(state: MainState):
    bundles = []
    for item in state.get("worker_outputs", []):
        unique = {}
        errors = list(item["errors"])
        for raw in item["cases"]:
            if not isinstance(raw, dict):
                errors.append("worker case must be an object")
                continue
            case = normalize_case(dict(raw, validation_status="PENDING"))
            if not isinstance(case["case_id"], str):
                errors.append("case_id must be a string")
                continue
            previous = unique.setdefault(case["case_id"], case)
            if previous != case:
                errors.append("distinct worker cases share a case_id")
        bundles.append({"task": item["task"], "cases": list(unique.values()), "errors": errors})
    trace_event("cases_split", bundles=[{"task_id": b["task"].get("task_id"), "case_ids": [c.get("case_id") for c in b["cases"]], "errors": b["errors"]} for b in bundles])
    return {"review_inputs": bundles}

def dispatch_reviews(state: MainState):
    return [Send("case_review", {
        "task": bundle["task"], "cases": bundle["cases"], "errors": bundle["errors"],
        "demo": state.get("demo", False), "max_repair": state.get("max_repair", 2),
        "attempts": 0, "issues": [], "done": False,
    }) for bundle in state["review_inputs"]]

async def validator(state: ReviewState):
    errors = list(state.get("errors", []))
    cases = state.get("cases", [])
    if not cases:
        errors.append("no supported candidate cases found")
    for idx, case in enumerate(cases):
        errors.extend(f"case[{idx}]: {e}" for e in check_case(case, state["task"]))
    issues = []
    if not state["demo"] and cases:
        try:
            response = await call_model("validator", {"task": state["task"], "cases": cases, "deterministic_errors": errors}, "validator")
            issues = response.get("issues", [])
            if not isinstance(issues, list):
                issues = ["validator issues is not a list"]
            proposed = response.get("cases", [])
            if not isinstance(proposed, list) or not all(isinstance(x, dict) for x in proposed):
                raise ValueError("validator cases must be list[object]")
            cases = [normalize_case(dict(c, validation_status="PENDING")) for c in proposed]
            errors = list(state.get("errors", []))
            if not cases:
                errors.append("validator returned no cases")
            for idx, case in enumerate(cases):
                errors.extend(f"case[{idx}]: {e}" for e in check_case(case, state["task"]))
        except Exception as exc:
            errors.append(f"validator failure: {type(exc).__name__}: {str(exc)[:300]}")
    if issues:
        errors.extend(f"semantic review: {str(i)[:300]}" for i in issues)
    case_ids = [case["case_id"] for case in cases if isinstance(case["case_id"], str)]
    if len(case_ids) != len(set(case_ids)):
        errors.append("review contains duplicate case IDs")
    trace_event("validation_result", task_id=state["task"].get("task_id"), errors=errors, issues=issues, case_ids=[c.get("case_id") for c in cases])
    return {"cases": cases, "errors": errors, "issues": issues, "done": not errors}

def route_review(state: ReviewState):
    if state.get("done") or state.get("attempts", 0) >= state.get("max_repair", 0) or state["demo"]:
        return "finish"
    return "repair"

async def repair(state: ReviewState):
    attempts = state.get("attempts", 0) + 1
    try:
        result = await call_model("validator", {"task": state["task"], "cases": state["cases"], "errors": state["errors"]}, "repair")
        new_cases = result.get("cases", [])
        if not isinstance(new_cases, list) or not all(isinstance(c, dict) for c in new_cases):
            raise ValueError("repair cases must be list[object]")
        return {"cases": [normalize_case(dict(c, validation_status="PENDING")) for c in new_cases], "attempts": attempts, "errors": []}
    except Exception as exc:
        return {"attempts": attempts, "errors": [f"repair failure: {type(exc).__name__}: {str(exc)[:300]}"]}

def finish_review(state: ReviewState):
    cases = state.get("cases", [])
    errors = state.get("errors", [])
    for case in cases:
        case["validation_status"] = "NEEDS_REVIEW" if errors or case.get("evidence_status") != "SOURCE_CONFIRMED" else "PENDING"
    return {"reviewed": [{"task": state["task"], "cases": cases, "errors": errors}]}

def build_review_graph():
    b = StateGraph(ReviewState, output_schema=ReviewOutput)
    b.add_node("groq_validator", validator)
    b.add_node("groq_repair", repair)
    b.add_node("finish_review", finish_review)
    b.add_edge(START, "groq_validator")
    b.add_conditional_edges("groq_validator", route_review, {"finish": "finish_review", "repair": "groq_repair"})
    b.add_edge("groq_repair", "groq_validator")
    b.add_edge("finish_review", END)
    return b.compile()

def persist(state: MainState):
    store = Store(state["db_path"])
    saved = unresolved = 0
    for item in state.get("reviewed", []):
        task_id = item["task"]["task_id"]
        if item["errors"] or not item["cases"]:
            store.unresolved(task_id, item, item["errors"] or ["no cases"])
            unresolved += 1
            continue
        try:
            saved += store.save_cases(item["cases"], task_id)
        except CaseConflict as exc:
            store.unresolved(task_id, item, [str(exc)])
            unresolved += 1
            continue
    store.conn.close()
    summary = {"cases_saved": saved, "tasks_unresolved": unresolved}
    trace_event("persist_completed", summary=summary)
    return {"summary": summary}

def no_tasks(state: MainState):
    return {"summary": {"cases_saved": 0, "tasks_unresolved": 0, "message": "No pending tasks"}}

def build_graph():
    b = StateGraph(MainState)
    b.add_node("source_indexer", source_indexer)
    b.add_node("research_worker", research_worker)
    b.add_node("case_splitter", case_splitter)
    b.add_node("case_review", build_review_graph())
    b.add_node("persist", persist)
    b.add_node("no_tasks", no_tasks)
    b.add_edge(START, "source_indexer")
    b.add_conditional_edges("source_indexer", dispatch_workers, {"no_tasks": "no_tasks", "research_worker": "research_worker"})
    b.add_edge("research_worker", "case_splitter")
    b.add_conditional_edges("case_splitter", dispatch_reviews, ["case_review"])
    b.add_edge("case_review", "persist")
    b.add_edge("persist", END)
    b.add_edge("no_tasks", END)
    return b.compile()

async def run_graph(db_path, demo=False, limit=10, max_repair=2, concurrency=3, mode="fixed"):
    if mode not in MODES:
        raise ValueError(f"Unknown orchestration mode: {mode}")
    graph = build_graph()
    result = await graph.ainvoke({"db_path": db_path, "mode": mode, "demo": demo, "limit": limit, "max_repair": max_repair,
                                  "tasks": [], "worker_outputs": [], "review_inputs": [], "reviewed": []},
                                 config={"max_concurrency": concurrency, "recursion_limit": max(40, 4 * (max_repair + 3))})
    return result.get("summary", {})
