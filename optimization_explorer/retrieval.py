"""Lexical retrieval over indexed source fragments."""
import heapq
import re
from pathlib import PurePosixPath


def _terms(text):
    return set(re.findall(r"[^\W_]+", text.casefold()))


def retrieve(query, fragments, limit=10):
    terms = _terms(query)
    query_path = query.strip().replace("\\", "/").casefold()
    if not terms or limit < 1:
        return []

    def ranked():
        for fragment in fragments:
            symbol = fragment.get("source_function", "").casefold()
            source_path = fragment.get("source_file", "").replace("\\", "/")
            filename = PurePosixPath(source_path).stem
            context = _terms(fragment["context"])
            score = len(context & terms) + 3 * len(_terms(symbol) & terms)
            score += 2 * len(_terms(filename) & terms)
            if symbol and symbol == query.strip().casefold():
                score += 5
            if source_path.casefold() == query_path:
                score += 5 + len(terms)
            if score:
                yield score, fragment

    return [
        fragment
        for _, fragment in heapq.nsmallest(
            limit, ranked(), key=lambda item: (-item[0], item[1]["task_id"])
        )
    ]
