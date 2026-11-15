Validate the worker's source-backed claims against the supplied numbered source.
Return ONLY JSON {"approved_indices":[0,2],"rejected_indices":[1],"rejection_reasons":[{"index":1,"reason":"The quote is exact, but the source does not support the claim that this selects a CUDA kernel."}]}. Include every claim index exactly once, either approved or rejected. Give a concise source-based reason for each rejected claim. Approve only claims supported by exact source quotes. Source quotes do not prove dynamic execution,
optimization applicability, cross-repository lowering or sm_75 runtime launch.

If format_correction is present, keep the same JSON schema. Use only indices within the supplied claims list, and use empty approved/rejected lists when there are no claims.
