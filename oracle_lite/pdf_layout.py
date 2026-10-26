from __future__ import annotations

import re


_CODE_LABEL = re.compile(r"^[A-Z]{2,15}$")
_NUMBER = re.compile(r"^\d{1,8}$")
_NUMERIC_FRAGMENT = re.compile(
    r"(?:(?:\d+|[Oo]?)[.,]\d+|[±+−-]\s*\d|\d+\s*(?:mm|cm|in|°|%))", re.I,
)
_CONTINUATION_WORDS = {"and", "of", "to", "with", "±"}


def _is_short_continuation(text: str) -> bool:
    words = text.split()
    if not words or len(words) > 3 or not _NUMERIC_FRAGMENT.search(text):
        return False
    if words[0].casefold() in _CONTINUATION_WORDS:
        return True
    return bool(re.search(r"(?:mm|cm|in|°|%|±)\s*$", text, re.I))


def _clean_block_text(value: str) -> str:
    return "\n".join(" ".join(line.split()) for line in value.splitlines() if line.strip())


def extract_pdf_positioned_blocks(page) -> list[dict]:
    """Return PDF text blocks with normalized positions and conservative grouping.

    The ordinary reading-order text remains the searchable source. These blocks
    retain the page geometry needed to interpret diagrams and flattened tables.
    Short numeric continuations are joined only when one nearby preceding block
    is clearly the nearest geometric match; opaque identifier balloons remain
    separately marked so callers can omit them from unrelated list questions.
    """
    rect = page.rect
    width, height = float(rect.width), float(rect.height)
    if width <= 0 or height <= 0:
        return []

    raw = []
    for block in page.get_text("blocks"):
        if len(block) < 5:
            continue
        x0, y0, x1, y1, source_text = block[:5]
        text = _clean_block_text(source_text)
        if not text:
            continue
        raw.append({
            "_bbox": (float(x0), float(y0), float(x1), float(y1)),
            "text": text,
        })
    raw.sort(key=lambda item: (item["_bbox"][1], item["_bbox"][0]))

    # PDF drawing labels such as "FAI" and "54" may be separate text blocks.
    # Keep such pairs recognizable as identifiers, not as a numeric continuation
    # of a nearby measurement or callout.
    combined = []
    consumed = set()
    index = 0
    while index < len(raw):
        if index in consumed:
            index += 1
            continue
        item = raw[index]
        text = item["text"].strip()
        if _CODE_LABEL.fullmatch(text):
            x0, y0, x1, y1 = item["_bbox"]
            candidates = []
            for other_index in range(index + 1, len(raw)):
                next_item = raw[other_index]
                number_match = re.match(r"^(\d{1,8})(?:\s|$)", next_item["text"].strip())
                if not number_match or len(next_item["text"].split()) > 3:
                    continue
                nx0, ny0, nx1, ny1 = next_item["_bbox"]
                dx = abs((x0 + x1 - nx0 - nx1) / 2)
                dy = max(0.0, ny0 - y1)
                if dx <= width * 0.02 and dy <= height * 0.02:
                    candidates.append((dx / width + dy / height, other_index))
            candidates.sort()
            if candidates and (len(candidates) == 1 or
                    candidates[1][0] > candidates[0][0] * 1.5 + 0.002):
                other_index = candidates[0][1]
                next_item = raw[other_index]
                nx0, ny0, nx1, ny1 = next_item["_bbox"]
                item = {
                    "_bbox": (min(x0, nx0), min(y0, ny0), max(x1, nx1), max(y1, ny1)),
                    "text": f"{text} {next_item['text'].strip()}",
                    "kind": "identifier",
                }
                consumed.add(other_index)
        combined.append(item)
        index += 1

    positioned = []
    for item in combined:
        x0, y0, x1, y1 = item["_bbox"]
        text = item["text"]
        if ((x1 - x0) / width > 0.75 and (y1 - y0) / height > 0.25):
            item["kind"] = "unpositioned"
            positioned.append(item)
            continue
        if (item.get("kind") == "identifier"
                or re.fullmatch(r"[A-Z]{2,}[-_/.]?\d+[A-Z0-9._/-]*", text, re.I)
                or (re.search(r"\b[A-Z]{2,}\s*\d+\b", text, re.I)
                    and not _NUMERIC_FRAGMENT.search(text))):
            item["kind"] = "identifier"
            positioned.append(item)
            continue

        # Rejoin text blocks split across adjacent lines when their horizontal
        # spans overlap strongly. This handles wrapped labels without joining
        # neighboring table columns or unrelated callouts.
        merged = False
        starts_new_record = bool(re.match(r"^\d{1,4}\s+\S+", text))
        starts_continuation = bool(re.match(r"^(?:OF|TO|AND)\b", text, re.I))
        for previous in (() if starts_new_record or starts_continuation else reversed(positioned)):
            if previous.get("kind") in {"identifier", "unpositioned"} or len(previous["text"]) < 12:
                continue
            px0, py0, px1, py1 = previous["_bbox"]
            vertical_gap = max(0.0, y0 - py1, py0 - y1)
            overlap = max(0.0, min(x1, px1) - max(x0, px0))
            smaller_width = min(max(1.0, x1 - x0), max(1.0, px1 - px0))
            if ((x1 - x0) <= width * 0.35 and (px1 - px0) <= width * 0.35
                    and vertical_gap <= height * 0.01
                    and overlap / smaller_width >= 0.8):
                previous["text"] += "\n" + text
                previous["_bbox"] = (min(x0, px0), min(y0, py0), max(x1, px1), max(y1, py1))
                merged = True
                break
        if merged:
            continue

        # Some PDF producers put the last value of one note at the start of the
        # next block. Preserve it with the preceding block only if the short
        # numeric fragment has a unique close predecessor in both page axes.
        lines = text.splitlines()
        prefix = ""
        if len(lines) > 1 and _is_short_continuation(lines[0]):
            prefix = lines[0]
            text = "\n".join(lines[1:]).strip()
        elif len(lines) == 1 and _is_short_continuation(lines[0]):
            prefix = lines[0]
            text = ""

        if prefix:
            candidates = []
            for previous in positioned:
                if previous.get("kind") in {"identifier", "unpositioned"} or len(previous["text"]) < 12:
                    continue
                px0, py0, px1, py1 = previous["_bbox"]
                vertical_gap = max(0.0, y0 - py1, py0 - y1)
                horizontal_gap = max(0.0, x0 - px1, px0 - x1)
                if vertical_gap <= height * 0.05 and horizontal_gap <= width * 0.15:
                    candidates.append((vertical_gap / height + horizontal_gap / width, previous))
            candidates.sort(key=lambda pair: pair[0])
            unique = (candidates and (len(candidates) == 1 or
                      candidates[1][0] > candidates[0][0] * 1.5 + 0.002))
            if unique:
                candidates[0][1]["text"] += "\n" + prefix
            else:
                text = (prefix + ("\n" + text if text else "")).strip()
        if text:
            positioned.append({"_bbox": item["_bbox"], "text": text, "kind": "text"})

    result = []
    for item in positioned:
        x0, y0, x1, y1 = item["_bbox"]
        result.append({
            "x": round(((x0 + x1) / 2) / width * 100, 1),
            "y": round(((y0 + y1) / 2) / height * 100, 1),
            "kind": item.get("kind", "text"),
            "text": item["text"],
        })
    return result


def needs_positioned_pdf_text(page) -> bool:
    """Identify dense, visually arranged PDF pages that need a block view."""
    blocks = page.get_text("blocks")
    if len(blocks) < 40:
        return False
    try:
        return len(page.get_drawings()) >= 300
    except Exception:
        return False


def extract_pdf_text(page) -> str:
    """Preserve geometric rows and visible column gaps without guessing fields.

    PDF object insertion order often places all labels before all values.
    Word coordinates provide a source-grounded reading order instead.
    """
    words = sorted(page.get_text("words"), key=lambda word: (word[3], word[0]))
    if not words:
        return page.get_text("text").strip()
    rows = []
    for word in words:
        baseline = word[3]
        if rows:
            anchor, height, row_words = rows[-1]
            tolerance = max(1.0, min(4.0, height * 0.35))
        if not rows or baseline - anchor > tolerance:
            rows.append((baseline, word[3] - word[1], [word]))
        else:
            row_words.append(word)

    lines = []
    for _baseline, _height, row_words in rows:
        line = ""
        previous_end = None
        for word in sorted(row_words, key=lambda value: value[0]):
            if previous_end is not None:
                gap = word[0] - previous_end
                line += "    " if gap > max(12.0, (word[3] - word[1]) * 1.5) else " "
            line += word[4]
            previous_end = word[2]
        lines.append(line)
    return "\n".join(lines).strip()
