"""Detection metrics over an explicitly labeled set of candidates."""

import argparse
import csv
import json
from pathlib import Path


def recall_at_k(relevant, ranked, k):
    """Fraction of gold targets retrieved in the first k ranks."""
    from itertools import islice

    if k < 1:
        raise ValueError("k must be positive")
    relevant = set(relevant)
    retrieved = set(islice(ranked, k))
    return len(relevant & retrieved) / len(relevant) if relevant else None


def detection_metrics(labels, predicted):
    """Score candidate IDs. Labels map each candidate ID to a boolean."""
    predicted = set(predicted)
    if not predicted <= labels.keys():
        raise ValueError("Predictions contain candidates outside the labeled set")
    positives = {candidate for candidate, expected in labels.items() if expected}
    negatives = labels.keys() - positives
    tp = len(positives & predicted)
    fp = len(negatives & predicted)
    fn = len(positives - predicted)
    tn = len(negatives - predicted)

    def rate(numerator, denominator):
        return numerator / denominator if denominator else None

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": rate(tp, tp + fp),
        "recall": rate(tp, tp + fn),
        "f1": rate(2 * tp, 2 * tp + fp + fn),
        "f2": rate(5 * tp, 5 * tp + fp + 4 * fn),
        "false_positive_rate": rate(fp, fp + tn),
        "false_negative_rate": rate(fn, fn + tp),
    }


def score_cases(candidates, cases):
    def key(row, predicate_field):
        return (row["source_file"], row["source_function"],
                " ".join(row[predicate_field].split()))

    by_source = {key(row, "predicate"): row["id"] for row in candidates}
    labels = {row["id"]: row["expected_optimization"] for row in candidates}
    if len(by_source) != len(candidates) or len(labels) != len(candidates):
        raise ValueError("Gold candidates must have unique IDs and source predicates")
    predicted = set()
    unmatched = []
    for case in cases:
        candidate = by_source.get(key(case, "applicability_predicate"))
        if candidate is None:
            unmatched.append(case["case_id"])
        elif case["fusion_type"] != "none":
            predicted.add(candidate)
    return {**detection_metrics(labels, predicted), "unmatched_case_ids": unmatched}


def main():
    parser = argparse.ArgumentParser(description="Score fusion detection on a labeled fixture")
    parser.add_argument("gold", type=Path)
    parser.add_argument("cases", type=Path)
    args = parser.parse_args()
    candidates = json.loads(args.gold.read_text(encoding="utf-8"))
    with args.cases.open(encoding="utf-8", newline="") as stream:
        print(json.dumps(score_cases(candidates, csv.DictReader(stream)), indent=2))


if __name__ == "__main__":
    main()
