"""Repeatable evidence retrieval and optional real-generation regression checks."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from talker.retrieval import CorpusRetriever


def evaluate_cases(retriever, cases, *, top_k=3, answer_fn=None):
    cases = list(cases)
    if not cases:
        raise ValueError("Evaluation suite is empty")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    # Validate the entire suite before running expensive generation. An
    # unanswerable question may still retrieve related, irrelevant passages.
    for case in cases:
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError("Each case requires a nonempty question")
        expected = case.get("expected_evidence", [])
        if case.get("unanswerable", False):
            if expected:
                raise ValueError("Unanswerable cases cannot require answer evidence")
            if not (case.get("answer_contains") or case.get("answer_any_of") or case.get("answer_patterns")):
                raise ValueError("Unanswerable cases require an explicit abstention assertion")
        elif not expected:
            raise ValueError("Answerable cases require expected_evidence")
        for target in expected:
            if not (target.get("source") or target.get("section") or target.get("contains")):
                raise ValueError("Evidence targets must constrain source, section, or content")
        for key in ("answer_contains", "answer_excludes", "answer_any_of", "answer_patterns", "answer_forbidden_patterns"):
            values = case.get(key, [])
            if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{key} must be a list of nonempty strings")
        for pattern in case.get("answer_patterns", []) + case.get("answer_forbidden_patterns", []):
            re.compile(pattern)
    results = []
    for case in cases:
        question = case["question"]
        expected = case.get("expected_evidence", [])
        passages = retriever.search(question, limit=top_k)
        evidence_checks = [
            any(
                (not target.get("source") or passage.source == target["source"])
                and (not target.get("section") or passage.section == target["section"])
                and all(value in passage.text for value in target.get("contains", []))
                for passage in passages
            )
            for target in expected
        ]
        result = {
            "id": case.get("id", question), "question": question,
            "split": case.get("split", "unspecified"),
            "unanswerable": case.get("unanswerable", False),
            "evidence_passed": all(evidence_checks) if expected else None,
            "evidence_checks": evidence_checks,
            "retrieved": [
                {"source": item.source, "section": item.section, "text": item.text}
                for item in passages
            ],
        }
        if answer_fn is not None:
            answer = answer_fn([{"role": "user", "content": question}])
            result["generated_answer"] = answer
            checks_provided = any(case.get(key) for key in (
                "answer_contains", "answer_excludes", "answer_any_of",
                "answer_patterns", "answer_forbidden_patterns"))
            result["answer_string_checks_passed"] = (
                all(value in answer for value in case.get("answer_contains", []))
                and all(value not in answer for value in case.get("answer_excludes", []))
                and (not case.get("answer_any_of") or any(value in answer for value in case["answer_any_of"]))
                and all(re.search(pattern, answer) for pattern in case.get("answer_patterns", []))
                and all(not re.search(pattern, answer) for pattern in case.get("answer_forbidden_patterns", []))
            ) if checks_provided else None
        results.append(result)
    count = len(results)
    def scores(items):
        return {
            "cases": len(items),
            "evidence_scored": sum(item["evidence_passed"] is not None for item in items),
            "evidence_passed": sum(item["evidence_passed"] is True for item in items),
            "answer_scored": sum(item.get("answer_string_checks_passed") is not None for item in items),
            "answer_passed": sum(item.get("answer_string_checks_passed") is True for item in items),
            "abstention_scored": sum(item["unanswerable"] and item.get("answer_string_checks_passed") is not None for item in items),
            "abstention_passed": sum(item["unanswerable"] and item.get("answer_string_checks_passed") is True for item in items),
        }
    return {
        "mode": "retrieval+generation" if answer_fn else "retrieval-only",
        "cases": count, "top_k": top_k,
        **scores(results),
        "by_split": {split: scores([item for item in results if item["split"] == split])
                     for split in sorted({item["split"] for item in results})},
        "note": "Evidence recall and answer string checks do not establish semantic correctness; review the saved passages and answers.",
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--corpus-id")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    root = args.output_dir.resolve()
    model = None
    if args.generate:
        from talker.model import TalkerModel
        model = TalkerModel(root)
        model.load()
        retriever = model.retriever
        if args.corpus_id and args.corpus_id != retriever.corpus_id:
            parser.error("--corpus-id must match the loaded adapter")
    else:
        corpus_id = args.corpus_id
        if corpus_id is None:
            corpora = [path.name for path in (root / "_corpora").iterdir() if path.is_dir()]
            if len(corpora) != 1:
                parser.error("Specify --corpus-id when output contains multiple corpora")
            corpus_id = corpora[0]
        retriever = CorpusRetriever(root, root / "_talker" / "evaluation_fts.sqlite3", corpus_id=corpus_id)
    report = evaluate_cases(retriever, cases, top_k=args.top_k,
                            answer_fn=model.answer if model else None)
    if model:
        report["model"] = model.info()
    report["corpus_id"] = retriever.corpus_id
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
