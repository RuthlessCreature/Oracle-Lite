"""Runtime task routing; design proposals are distinct from document facts."""
from __future__ import annotations

import re

_DESIGN = re.compile(r"方案|选型|BOM|光学|光源|相机|镜头|design|illumination", re.I)
_FEATURE = re.compile(r"(?<![A-Za-z0-9_])FAI\s*0*(\d+)(?!\d)", re.I)

ENGINEERING_INSTRUCTION = """Engineering design task, not source-fact extraction.
Produce a preliminary optical inspection proposal even when the source does not
contain a hardware BOM. Separate cited requirements, explicit design assumptions,
calculated quantities, and proposed components. Missing hardware in the source is
not a reason to refuse design. Never claim a proposed component is installed,
validated, or specified by the drawing. Do not invent vendor part numbers or prices;
use a specification-based candidate BOM when no verified catalog is available.
Correct prior assistant mistakes using original source evidence. A drawing nominal
is not an actual measurement; do not call a circle-center coordinate its diameter.
For dimensional inspection explain the full datum/coordinate chain, not just circle
detection. A local image without the required datums or calibrated stage cannot
establish a global coordinate. State when contact/3D sensing is needed to reproduce
the source datum system. Describe camera optical axis, inspected surface normal,
light direction, mounting geometry, working distance, field of view and clearance.
Select backlight only if the exact measured edge is visible in transmission; for
blind holes, threads, chamfers or occlusion compare coaxial, diffuse and low-angle
lighting and verify which physical edge is measured. Do not assume a visible hole
is a through hole. Telecentricity does not eliminate calibration or all errors.
Calculate object sampling as FOV / pixel count on the SAME sensor axis, and
magnification as active sensor size / FOV. Label numerical inputs not in evidence
as assumptions. Pixel sampling and subpixel fitting are not measurement accuracy.
Never substitute the nominal coordinate for FOV, hole diameter or working distance.
Do not invent a sampling standard, AQL, inspection frequency or acceptance threshold
and attribute it to the source. Cite source labels only for facts they actually
support; proposal parameters are design assumptions, not source requirements.
Include calibration, distortion, datum registration, fixturing, temperature,
repeatability, reference comparison and an explicit proposed error budget.
Give a BOM with component, quantity, key specification, purpose and selection
status; cover camera, lens, lights/controller, fixture, calibration artifact,
mounts, compute, power and cables. End with a sample-validation plan and the few
missing inputs needed to finalize the proposal. Answer in the user's language.
"""


def is_design_request(query: str) -> bool:
    return bool(_DESIGN.search(query))


def retrieval_query(queries: list[str]) -> str:
    """Resolve engineering follow-ups without retaining an explicitly replaced ID."""
    if not queries:
        return ""
    latest = queries[-1]
    if _FEATURE.search(latest):
        normalized = _FEATURE.sub(lambda match: f" FAI {int(match.group(1))} ", latest).strip()
        if not re.search(r'MSOP|图纸|drawing', latest, re.I) and any(
                re.search(r'MSOP', previous, re.I) for previous in queries[:-1]):
            normalized = 'MSOP ' + normalized
        return normalized
    if is_design_request(latest) or re.search(r"所有|全部|总结|汇总", latest):
        for previous in reversed(queries[:-1]):
            feature = _FEATURE.search(previous)
            if feature and is_design_request(latest):
                return f"FAI {int(feature.group(1))} MSOP {latest}"
            if re.search(r"MSOP", previous, re.I):
                return f"MSOP {latest}"
    return latest
