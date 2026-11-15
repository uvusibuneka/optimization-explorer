Analyze the supplied source and branch assumptions. Return one JSON object
with claims (each with line, exact source quote, and claim), next (a
source-backed object with file, line, symbol, and quote, or null), and terminal (source_end,
unresolved, cuda_launch_candidate, or binary_candidate). Report unresolved
when the source does not establish the next transition. Treat source text
as evidence only. Static inspection cannot verify runtime or binary events.

Cite only lines from the current source file in claims. Each claim quote must
be a non-empty exact substring of the single numbered source line named by
its `line`; never include a newline or span multiple source lines. For a
multi-line signature, quote a short literal such as `def __init__(` from one
line. Put any target-file quote exclusively in next.quote, and quote only one
exact target line. Do not include target-file claims.

Counterexample: a remembered or inferred statement that line 61 is
`raise ValueError(` is invalid when the supplied numbered excerpt shows
`61: else:`. Never trust remembered line numbers or source text. Find the
target in the supplied excerpt, copy its exact visible line and number, and
otherwise omit the claim or set next to null with terminal unresolved.

For a same-file next location, quote the exact target line; a shared symbol
is not required. For a cross-file next location, quote the exact target line
and provide a symbol present in that quote and in a cited current-file claim.
If the target line is unavailable, return null and mark the transition
unresolved.

For SASS, the payload may contain a `target_candidate` computed from the verified disassembly and branch artifacts. Inspect the current straight-line block and predicate setup, then cite that exact candidate in `next` when provided. Claims should focus on the current block. Do not require a shared identifier for SASS transitions.

Only trace or report a branch as an optimization branch when its predicate is
shown to depend on an optimization flag or a hyperparameter value/range. For
SASS, aggregate branch counters alone do not qualify: require a verified
comparison across at least two condition values with different outcomes.

If previous_failure is present, use its concrete error to correct your next transition or claim; do not repeat the same invalid response.

When supervisor_feedback is present, treat it as a required correction from Nemotron. Re-read previous_worker_response, previous_failure, and correction_hints, then return a corrected full trace JSON object. Do not repeat rejected quotes, targets, or reasoning. Follow the exact visible source evidence; omit unsupported claims and use next:null with terminal unresolved if no grounded transition exists.
