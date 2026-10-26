"""Fictional held-out-style fixtures for scope, row associations, and JSON types."""
from __future__ import annotations

import argparse
import json
import random
from datetime import date
from pathlib import Path


def build_fixtures(*, seed: int = 20261030, count: int = 96) -> list[dict]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        kind = index % 5
        code = f"PX_{rng.randrange(1000, 9999)}"
        if kind == 0:
            first = f"Source {rng.choice(['A', 'C', 'E', 'G'])}"
            second = f"Source {rng.choice(['B', 'D', 'F', 'H'])}"
            p1, p2 = rng.choice(["UP", "DOWN"]), rng.choice(["UP", "DOWN"])
            if p1 == p2:
                p2 = "DOWN" if p1 == "UP" else "UP"
            context = (
                f"[{first}]\nProgram = {p1}\nProbe = {code}\n"
                f"[{second}]\nProgram = {p2}\nProbe = {code}\n"
                "No section takes precedence."
            )
            expected = {first: {"Program": p1, "Probe": code},
                        second: {"Program": p2, "Probe": code}}
            question = (
                "请按每个来源分别列出 Program 和 Probe。使用以来源为键的嵌套严格 JSON；保留不同来源的值，只输出 JSON。"
                if index % 2 else
                "Report Program and Probe for every source in a strict JSON object, nested by source. "
                "Keep differing scopes separate; output JSON only."
            )
        elif kind == 1:
            locations = rng.sample(
                ["left rail", "right rail", "corner hook", "center bracket", "side tab",
                 "lower clip", "upper tab", "hinge area", "front seam", "rear seam"], 8
            )
            criteria = []
            for location in locations:
                criteria.append({
                    "location": location,
                    "count": str(rng.randrange(3, 12)),
                    "max_height": f"0.{rng.randrange(1, 10):02d}",
                })
            context = "Inspection criteria:\n" + "\n".join(
                f"{item['location']}: {item['count']} points; maximum height {item['max_height']} mm."
                for item in criteria
            )
            expected = {"criteria": criteria}
            question = (
                "Give the inspection-height details. Include every location, count, and maximum height, "
                "keeping each row's values together. Return strict JSON only."
                if index % 2 else
                "List every location with its count and maximum height in a strict JSON object "
                "using an array of records under criteria. Keep each row's values together. Output JSON only."
            )
        elif kind == 2:
            name = f"K{rng.randrange(10000, 99999)}"
            value_a, value_b = f"{rng.randrange(10, 200)}.{rng.randrange(100):02d} mm", f"{rng.randrange(10, 200)}.{rng.randrange(100):02d} mm"
            context = f"Step 1: {name} = {value_a}\nStep 2: {name} = {value_b}\nNo step takes precedence."
            expected = {"Step 1": {name: value_a}, "Step 2": {name: value_b}, "unique": False}
            question = (
                f"What is {name}'s value? Return a strict JSON object with Step 1 and Step 2 "
                "as nested objects and unique as a Boolean. Do not merge the steps. Output JSON only."
            )
        elif kind == 3:
            drawing = f"DRW-{rng.randrange(100000, 999999)}"
            barcode = f"PPPYWWD{rng.randrange(1000, 9999)}ERXBLLSMMC"
            serial = f"SN-{rng.randrange(100000, 999999)}"
            revision = rng.choice(["A", "B", "C", "D"])
            context = (
                f"Drawing Number: {drawing}\nBarcode Template: {barcode}\n"
                f"Serial Number: {serial}\nRevision: {revision}"
            )
            expected = {"Barcode Template": barcode}
            question = (
                "What is the Barcode Template? Return only a strict JSON object with that exact key. "
                "Do not substitute the nearby drawing number, serial number, or revision."
            )
        else:
            year = rng.randrange(2018, 2026)
            history = []
            for revision_index, revision in enumerate(("V01.01", "V01.02", "V01.03", "V02.01")):
                month = rng.randrange(1, 13)
                day = rng.randrange(1, 29)
                when = date(year, month, day).isoformat()
                change = rng.choice([
                    "Update measurement program",
                    "Change probe assignment",
                    "Add feature inspection",
                    "Revise nominal value",
                    "Move procedure to another station",
                ])
                history.append({"revision": revision, "date": when, "change": change})
            context = "Revision history:\n" + "\n".join(
                f"{item['revision']} | {item['date']} | {item['change']}" for item in history
            )
            expected = {"history": history}
            question = (
                "Give the complete revision history, preserving each version, date, and change together. "
                "Return only JSON with an array under history."
            )
        answer = json.dumps(expected, ensure_ascii=False)
        rows.append({
            "mode": "instruction", "question": question, "context": context,
            "text": answer, "answer": answer,
            "source_path": f"fictional-robust-{index + 1:04d}.txt",
            "fixture_kind": kind, "fictional": True,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--count", type=int, default=96)
    args = parser.parse_args()
    rows = build_fixtures(seed=args.seed, count=args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(json.dumps({"records": len(rows), "seed": args.seed, "fictional": True,
                      "by_kind": {str(kind): sum(row["fixture_kind"] == kind for row in rows)
                                  for kind in range(5)}}))


if __name__ == "__main__":
    main()
