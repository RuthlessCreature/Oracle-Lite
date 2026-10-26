"""Held-out document structures for corpus-independent Talker generalization.

These fictional cases deliberately use prose and document layouts that differ
from the synthetic templates used by the instruction fine-tuning fixtures.
They test evidence association, explicit missingness, and unresolved conflicts
with corpus retrieval disabled.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .grounding_stress import score_json_mapping


def build_cases(*, seed: int = 20300930, count: int = 60) -> list[dict]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    cases = []
    for index in range(count):
        family = index % 6
        language = "zh" if index % 2 else "en"
        serial = f"QX-{rng.randrange(100000, 999999)}"
        if family == 0:
            left, right = rng.sample(["North", "South", "East", "West", "Upper", "Lower"], 2)
            pressure_a = f"{rng.randrange(10, 99)}.{rng.randrange(100):02d} kPa"
            pressure_b = f"{rng.randrange(10, 99)}.{rng.randrange(100):02d} kPa"
            meter_a, meter_b = f"M-{rng.randrange(1000, 9999)}", f"M-{rng.randrange(1000, 9999)}"
            context = (
                f"At the {left} station, the operator wrote in the shift diary that line pressure "
                f"settled at {pressure_a}; the installed meter carried tag {meter_a}. "
                f"A separate note from the {right} station records line pressure at {pressure_b} "
                f"with meter tag {meter_b}. The notebook contains no instruction to treat either "
                f"station as the default. Ticket {serial} is an administrative identifier."
            )
            expected = {
                left: {"line_pressure": pressure_a, "meter_tag": meter_a},
                right: {"line_pressure": pressure_b, "meter_tag": meter_b},
            }
            question = (
                "请从值班日志中按站点提取 line pressure 和 meter tag，保留每个站点自己的数值。"
                "只输出严格 JSON，不要把工单号当作读数。"
                if language == "zh" else
                "From the shift diary, return each station's line pressure and meter tag in a strict "
                "JSON object keyed by station. Keep the readings paired with their station. "
                "Do not treat the ticket number as a reading."
            )
        elif family == 1:
            name = f"{rng.choice(['Aster', 'Birch', 'Cedar', 'Delta'])} {rng.choice(['Lee', 'Morgan', 'Rao', 'Kim'])}"
            old = f"{rng.randrange(20, 200)}.{rng.randrange(100):02d} V"
            new = f"{rng.randrange(20, 200)}.{rng.randrange(100):02d} V"
            date_a = f"202{rng.randrange(1, 6)}-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}"
            date_b = f"202{rng.randrange(1, 6)}-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}"
            while date_b == date_a:
                date_b = f"202{rng.randrange(1, 6)}-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}"
            field = rng.choice(["charge ceiling", "relay threshold", "bus target"])
            context = (
                f"Subject: {field} review\n\n"
                f"Message received {date_a} from {name}: In my bench copy, the {field} reads {old}.\n\n"
                f"Forwarded message received {date_b} from {serial}: The value shown in my attachment "
                f"for the {field} is {new}.\n\n"
                "The mail thread does not identify a released copy or say which attachment supersedes the other."
            )
            expected = {
                f"message_{date_a}": {field: old},
                f"message_{date_b}": {field: new},
                "unique_released_value": False,
            }
            question = (
                f"邮件中 {field} 的值是什么？请分别保留每封邮件的日期和值，并说明能否确定唯一已发布值。"
                "严格输出 JSON。"
                if language == "zh" else
                f"What value for the {field} appears in each email? Preserve the date/value pairing and "
                "say whether the thread establishes one uniquely released value. Return strict JSON."
            )
        elif family == 2:
            locations = rng.sample(["Harbor", "Summit", "Grove", "Meadow", "Ridge", "Quarry"], 2)
            present = f"202{rng.randrange(1, 7)}-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}"
            context = (
                f"The {locations[0]} bench card says the next calibration visit is booked for {present}. "
                f"The {locations[1]} card describes a replacement sensor, asset {serial}, but gives no "
                "calibration date. The excerpt ends there; it does not say that the two benches share a schedule."
            )
            expected = {
                locations[0]: {"next_calibration": present},
                locations[1]: {"next_calibration": None},
            }
            question = (
                "按 bench 汇总 next calibration；没有写明的日期用 JSON null 表示，不要沿用其他 bench 的日期。"
                "只输出 JSON。"
                if language == "zh" else
                "For each bench, report next_calibration. Use JSON null where that bench's card omits a "
                "date; do not copy another bench's date. Output JSON only."
            )
        elif family == 3:
            assets = rng.sample(["circulation pump", "feed pump", "return pump", "bypass pump", "coolant pump"], 3)
            values = [f"{rng.randrange(1, 99)}.{rng.randrange(100):02d} L/min" for _ in assets]
            identifiers = [f"{rng.randrange(100000, 999999)}" for _ in assets]
            context = "Field notes, copied in visit order:\n" + "\n".join(
                f"I tagged the {asset} as unit {identifier}; its flow reading during the visit was {value}."
                for asset, identifier, value in zip(assets, identifiers, values)
            ) + f"\nThe work order was {serial}; it is not an equipment tag."
            expected = {
                asset: {"unit_tag": identifier, "flow": value}
                for asset, identifier, value in zip(assets, identifiers, values)
            }
            question = (
                "把每台泵的 unit tag 和 flow 配对列出。工单号不是设备编号；按设备名称组织为严格 JSON。"
                if language == "zh" else
                "Pair each pump's unit tag with its flow reading. Organize the result as strict JSON keyed "
                "by equipment name; the work order is not a unit tag."
            )
        elif family == 4:
            component = rng.choice(["hinge arm", "latch plate", "guide shoe", "carrier tab"])
            nominal_a = f"{rng.randrange(1, 90)}.{rng.randrange(100):02d} mm"
            nominal_b = f"{rng.randrange(1, 90)}.{rng.randrange(100):02d} mm"
            tol_a = f"±0.{rng.randrange(1, 20):02d} mm"
            tol_b = f"+0.{rng.randrange(1, 20):02d}/-0.{rng.randrange(1, 20):02d} mm"
            context = (
                f"On the {component} master sample, the nominal span is {nominal_a}; the allowed "
                f"deviation is {tol_a}. A later paragraph in a different attachment describes the "
                f"{component} nominal span as {nominal_b}, with acceptance band {tol_b}. No revision "
                f"link between the attachments is included. Drawing ref {serial} appears in the footer."
            )
            expected = {
                "master_sample": {"nominal": nominal_a, "tolerance": tol_a},
                "other_attachment": {"nominal": nominal_b, "tolerance": tol_b},
            }
            question = (
                f"请分别提取两个附件中的 {component} nominal 和 tolerance，不能混淆名义尺寸与公差。"
                "以附件为键输出 JSON。"
                if language == "zh" else
                f"Extract the {component}'s nominal and tolerance from each attachment separately. "
                "Keep nominal dimensions distinct from tolerances and return JSON keyed by attachment."
            )
        else:
            names = rng.sample(["Orchid", "Pine", "Quartz", "Sable", "Tundra"], 3)
            value = f"{rng.randrange(5, 80)}.{rng.randrange(100):02d} bar"
            alternatives = [f"{rng.randrange(5, 80)}.{rng.randrange(100):02d} bar" for _ in range(2)]
            context = (
                f"A signed bench slip from {names[0]} records {value} for the inlet ceiling. "
                f"A maintenance margin in the {names[1]} notebook records {alternatives[0]}. "
                f"The handover sheet from {names[2]} also records {alternatives[1]}. "
                "These are separate observations; the packet states no precedence rule, approval status, "
                "or voting rule."
            )
            expected = {
                names[0]: value,
                names[1]: alternatives[0],
                names[2]: alternatives[1],
                "unique_authoritative_value": False,
            }
            question = (
                "三个记录对 inlet ceiling 分别写了什么？是否能据此确定唯一权威值？请保留来源并输出严格 JSON。"
                if language == "zh" else
                "What does each record say for the inlet ceiling, and do these records establish one "
                "authoritative value? Preserve each source and return strict JSON."
            )

        answer = json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
        cases.append({
            "id": f"novel-{index + 1:03d}",
            "mode": "instruction",
            "family": family,
            "language": language,
            "question": question,
            "context": context,
            "expected": expected,
            "answer": answer,
            "text": answer,
            "fictional": True,
            "template_split": "held_out_surface_form",
        })
    return cases


def run(output_dir: Path, report_dir: Path, *, seed: int = 20300930, count: int = 60) -> dict:
    cases = build_cases(seed=seed, count=count)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "cases.jsonl").write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )

    import torch

    if not torch.cuda.is_available():
        report = {
            "suite": "unseen-document-templates-v1",
            "status": "not_run",
            "reason": "CUDA device unavailable; model quality was not measured",
            "seed": seed,
            "cases": len(cases),
            "expected_families": 6,
            "model": None,
        }
        (report_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report

    from talker.model import TalkerModel

    model = TalkerModel(output_dir)
    model.load()
    model.retriever.context = lambda *args, **kwargs: ""
    model.retriever.curated_context = lambda *args, **kwargs: None
    model.retriever.visual_context = lambda *args, **kwargs: []
    results = []
    with (report_dir / "answers.jsonl").open("w", encoding="utf-8") as stream:
        for case in cases:
            started = time.monotonic()
            answer = model.answer([{
                "role": "user",
                "content": case["question"],
                "attachments": [{
                    "filename": "Fictional novel-template source",
                    "parsed_text": case["context"],
                    "images": [],
                }],
            }])
            score = score_json_mapping(answer, case["expected"])
            result = {
                "id": case["id"], "family": case["family"],
                "language": case["language"], "expected": case["expected"],
                "answer": answer, **score,
                "seconds": round(time.monotonic() - started, 2),
            }
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps({"id": result["id"], "json_only": score["json_only"],
                              "mapping_exact": score["mapping_exact"]}, ensure_ascii=False), flush=True)

    report = {
        "suite": "unseen-document-templates-v1",
        "seed": seed,
        "cases": len(results),
        "json_only_passed": sum(result["json_only"] for result in results),
        "mapping_exact_passed": sum(result["mapping_exact"] for result in results),
        "fields_matched": sum(result["fields_matched"] for result in results),
        "expected_fields": sum(result["expected_fields"] for result in results),
        "by_family": {
            str(family): {
                "cases": sum(result["family"] == family for result in results),
                "json_only_passed": sum(result["family"] == family and result["json_only"] for result in results),
                "mapping_exact_passed": sum(result["family"] == family and result["mapping_exact"] for result in results),
            }
            for family in range(6)
        },
        "model": model.info(),
        "note": (
            "Randomized fictional facts in narrative surface forms held out from current training fixtures; "
            "task skills overlap intentionally and corpus retrieval is disabled. This does not prove universal accuracy."
        ),
    }
    (report_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20300930)
    parser.add_argument("--count", type=int, default=60)
    args = parser.parse_args()
    report = run(args.output_dir, args.report_dir, seed=args.seed, count=args.count)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report.get("status") != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
