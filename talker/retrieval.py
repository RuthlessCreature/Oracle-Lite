from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from oracle_lite.canonical import CanonicalDocument


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*|[\u3400-\u9fff]+", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u3400-\u9fff]+")
_TOKENIZER_VERSION = "bilingual-fts-v2"
_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "about", "according", "can", "do", "does",
    "for", "from", "give", "how", "in", "is", "it", "list", "of", "on",
    "or", "please", "provide", "tell", "that", "the", "these", "this", "those",
    "to", "use", "used", "using", "what", "which", "with", "与", "的", "了",
    "吗", "呢", "和", "在", "有", "是", "这", "请", "中", "么", "么？",
}
_BILINGUAL_EXPANSIONS = {
    "公差": "tolerance",
    "角度": "angles",
    "尺寸": "dimension length width height diameter",
    "dimensions": "dimension length width height diameter",
    "图纸": "drawing",
    "投影": "projection",
    "单位": "millimeters",
    "测量": "measurement measured measure",
    "夹具": "fixture",
    "焊接": "weld welding",
    "拉力": "pull test",
    "拉片": "weld tab",
    "焊点": "weld spot",
    "焊缝": "weld seam",
    "高度": "height",
    "最大": "max maximum",
    "明细": "details detail",
    "详情": "details detail",
    "探针": "probe",
    "基准": "datum",
    "版本": "version",
    "日期": "date",
    "编码": "barcode code",
    "条码": "barcode code",
    "默认": "default",
    "视图": "view",
    "方法": "method",
    "程序": "program",
    "项目": "item feature",
    "公制": "metric",
    "updated": "update date",
    "history": "update description",
}
_IDENTIFIER_PHRASE_RE = re.compile(
    r"\b([a-z][a-z0-9._/-]{1,})\s+([a-z]|\d{1,})(?![a-z0-9._/-])",
    re.IGNORECASE,
)
_COMPREHENSIVE_QUERY_RE = re.compile(
    r"\b(?:all|every|each|list|details?|complete|full|enumerate|how many|version history)\b"
    r"|全部|所有|列出|明细|详情|详细|多少|哪些|历史", re.I,
)
_IDENTIFIER_QUERY_RE = re.compile(
    r"\b(?:id|identifier|barcode|serial|revision|version|fai|feature number)\b"
    r"|编号|条码|序列号|版本|修订", re.I,
)
_CALLOUT_QUERY_RE = re.compile(
    r"\b(?:weld(?:ing)?|spots?|seams?|heights?|tolerances?|limits?|criteria)\b"
    r"|焊点|焊缝|焊接|高度|公差|限值|标准", re.I,
)


def query_requests_complete_list(query: str) -> bool:
    """Detect explicit exhaustive intent without treating every plural as exhaustive."""
    return bool(_COMPREHENSIVE_QUERY_RE.search(query))
_COMPACT_IDENTIFIER_RE = re.compile(
    r"\b([a-z]{2,})[-_]?([0-9]{1,})\b", re.IGNORECASE
)
PASSAGE_CHARS = 1_200
PASSAGE_OVERLAP = 160


@dataclass(slots=True, frozen=True)
class RetrievedPassage:
    source: str
    section: str
    text: str
    content_hash: str = ""

    def render(self, index: int) -> str:
        label = self.source
        if self.section:
            label += f", {self.section}"
        return f"[Source {index}: {label}]\n{self.text}"


def _terms(text: str) -> list[str]:
    terms: list[str] = []
    for match in _TOKEN_RE.findall(text.lower()):
        if _CJK_RE.fullmatch(match):
            if len(match) == 1:
                terms.append(match)
                continue
            terms.extend(match[i : i + 2] for i in range(len(match) - 1))
            if len(match) > 3:
                terms.extend(match[i : i + 3] for i in range(len(match) - 2))
        else:
            terms.append(match)
            if any(separator in match for separator in "._/-"):
                terms.extend(part for part in re.split(r"[._/-]+", match) if part)
    return terms


def _identifiers(text: str) -> set[str]:
    """Extract generic mixed letter/number labels such as REV A or FAI7."""
    identifiers = {
        f"{match.group(1).lower()}:{match.group(2).lower()}"
        for match in _IDENTIFIER_PHRASE_RE.finditer(text)
    }
    identifiers.update(
        f"{match.group(1).lower()}:{match.group(2)}"
        for match in _COMPACT_IDENTIFIER_RE.finditer(text)
    )
    return identifiers


def _query_identifiers(text: str) -> set[str]:
    expanded = text.lower()
    for source_term, expansion in _BILINGUAL_EXPANSIONS.items():
        if source_term in expanded:
            expanded = expanded.replace(source_term, expansion)
    return _identifiers(expanded)


def _passages(text: str):
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not text:
        return
    if len(text) <= PASSAGE_CHARS:
        yield text
        return
    start = 0
    while start < len(text):
        end = min(len(text), start + PASSAGE_CHARS)
        if end < len(text):
            boundary = text.rfind("\n", start + PASSAGE_CHARS // 2, end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            yield piece
        if end >= len(text):
            break
        start = max(start + 1, end - PASSAGE_OVERLAP)


class CorpusRetriever:
    """Small local FTS index over the canonical corpus under an Oracle output root."""

    def __init__(
        self,
        output_dir: str | Path,
        index_path: str | Path,
        *,
        corpus_id: str | None = None,
    ):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.index_path = Path(index_path).expanduser().resolve()
        self.corpus_id = corpus_id
        self._lock = threading.RLock()
        self._synced = False
        self._sync_key = None
        self._qa_cache_key: tuple[tuple[str, int, int], ...] | None = None
        self._qa_cache: list[dict[str, str]] = []

    def select_corpus(self, corpus_id: str | None) -> None:
        if corpus_id == self.corpus_id:
            return
        self.corpus_id = corpus_id
        self._synced = False
        self._sync_key = None
        self._qa_cache_key = None

    def _active_qa_paths(self) -> list[Path]:
        paths: set[Path] = set()
        for registry_path in self.output_dir.glob("_corpora/*/_state/registry.sqlite3"):
            if self.corpus_id and registry_path.parent.parent.name != self.corpus_id:
                continue
            uri = registry_path.resolve().as_uri() + "?mode=ro"
            try:
                with sqlite3.connect(uri, uri=True) as conn:
                    rows = conn.execute(
                        "SELECT path FROM source_files WHERE status='active'"
                    ).fetchall()
            except (sqlite3.Error, OSError):
                continue
            for (value,) in rows:
                path = Path(str(value))
                if path.name == "qa_pairs.jsonl" and path.is_file():
                    paths.add(path.resolve())
        return sorted(paths)

    def _curated_qa(self) -> list[dict[str, str]]:
        paths = self._active_qa_paths()
        key = tuple(
            (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            for path in paths
        )
        if key == self._qa_cache_key:
            return self._qa_cache
        rows: list[dict[str, str]] = []
        for path in paths:
            try:
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        if not isinstance(item, dict):
                            raise ValueError("Reviewed QA requires a JSON object")
                        if all(isinstance(item.get(key), str) and item[key].strip()
                               for key in ("question", "answer", "context")):
                            rows.append({
                                "question": str(item["question"]),
                                "answer": str(item["answer"]),
                                "context": str(item["context"]),
                                "source_path": str(item.get("source_path") or path.name),
                            })
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        self._qa_cache_key = key
        self._qa_cache = rows
        return rows

    def curated_context(self, query: str, *, context_query: str = "") -> str | None:
        """Retrieve reviewed QA as evidence, never as an automatic user answer."""
        query_lower = query.lower()
        query_terms = set(_terms(query))
        query_identifiers = _query_identifiers(query)
        for source_term, expansion in _BILINGUAL_EXPANSIONS.items():
            if source_term in query_lower:
                query_terms.update(_terms(expansion))
        query_terms = {
            term for term in query_terms
            if term not in _QUERY_STOPWORDS and (len(term) > 1 or term.isdigit())
        }
        if context_query and query_identifiers and len(query_terms) < 6:
            query_terms.update(
                term for term in _terms(context_query.lower())
                if term not in _QUERY_STOPWORDS and (len(term) > 1 or term.isdigit())
            )
        if not query_terms:
            return None

        ranked: list[tuple[float, dict[str, str]]] = []
        for item in self._curated_qa():
            qa_terms = set(_terms(item["question"]))
            qa_identifiers = _identifiers(item["question"])
            for source_term, expansion in _BILINGUAL_EXPANSIONS.items():
                if source_term in item["question"].lower():
                    qa_terms.update(_terms(expansion))
            qa_terms = {
                term for term in qa_terms
                if term not in _QUERY_STOPWORDS and (len(term) > 1 or term.isdigit())
            }
            shared = query_terms & qa_terms
            if len(shared) < 3:
                continue
            if query_identifiers and not query_identifiers.issubset(qa_identifiers):
                continue
            coverage = len(shared) / max(1, min(len(query_terms), len(qa_terms)))
            if query_identifiers and len(shared) >= 2 and coverage >= 0.15:
                ranked.append((coverage, item))
            elif len(shared) >= 3 and coverage >= 0.45:
                ranked.append((coverage, item))

        ranked.sort(key=lambda pair: pair[0], reverse=True)
        if not ranked or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
            return None
        item = ranked[0][1]
        answer = (
            "Reviewed QA evidence (verify that it addresses the current question):\n"
            f"Question: {item['question']}\nSupporting excerpt: {item['context']}\n"
            f"Reviewed answer: {item['answer']}"
        )
        sources = [part.strip() for part in ranked[0][1]["source_path"].split(";") if part.strip()]
        if sources:
            citations = "\n".join(
                f"[Reviewed QA Source {index}: {source}]"
                for index, source in enumerate(sources, start=1)
            )
            answer = f"{answer}\n\n{citations}"
        return answer

    def _connect(self) -> sqlite3.Connection:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.index_path)
        conn.row_factory = sqlite3.Row
        conn.execute(
            """CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(
                   content_hash UNINDEXED,
                   source_path UNINDEXED,
                   section UNINDEXED,
                   body UNINDEXED,
                   search_text,
                   tokenize='unicode61'
               )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS indexed_docs(
                   content_hash TEXT PRIMARY KEY,
                   canonical_path TEXT NOT NULL,
                   fingerprint TEXT NOT NULL
               )"""
        )
        return conn

    def _active_canonical_docs(self) -> list[tuple[str, str, Path]]:
        docs: dict[str, tuple[str, str, Path]] = {}
        has_registry = False
        for registry_path in self.output_dir.glob("_corpora/*/_state/registry.sqlite3"):
            if self.corpus_id and registry_path.parent.parent.name != self.corpus_id:
                continue
            uri = registry_path.resolve().as_uri() + "?mode=ro"
            try:
                with sqlite3.connect(uri, uri=True) as conn:
                    rows = conn.execute(
                        """SELECT sf.content_hash, MIN(sf.path), da.canonical_path
                             FROM source_files sf
                             JOIN derived_artifacts da ON da.id=(
                                 SELECT candidate.id FROM derived_artifacts candidate
                                  WHERE candidate.content_hash=sf.content_hash
                                    AND candidate.status='ready'
                                    AND candidate.canonical_path IS NOT NULL
                                  ORDER BY candidate.created_at DESC, candidate.id DESC
                                  LIMIT 1
                             )
                            WHERE sf.status='active'
                              AND da.status='ready'
                              AND da.canonical_path IS NOT NULL
                            GROUP BY sf.content_hash"""
                    ).fetchall()
                    has_registry = True
            except (sqlite3.Error, OSError):
                continue
            for content_hash, source_path, canonical_path in rows:
                if Path(str(source_path)).name == "qa_pairs.jsonl":
                    continue
                path = Path(canonical_path)
                if path.exists():
                    docs[str(content_hash)] = (str(content_hash), str(source_path), path)

        if has_registry:
            return list(docs.values())

        # Older outputs may predate the registry. Fall back to canonical headers.
        for canonical_dir in self.output_dir.glob("_corpora/*/canonical"):
            if self.corpus_id and canonical_dir.parent.name != self.corpus_id:
                continue
            for path in canonical_dir.rglob("*.json"):
                try:
                    header = CanonicalDocument.read_header(path)
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                content_hash = str(header.get("content_hash") or path.stem)
                source_path = str(header.get("source_path") or path.name)
                docs[content_hash] = (content_hash, source_path, path)
        return list(docs.values())

    @staticmethod
    def _fingerprint(path: Path) -> str:
        header_stat = path.stat()
        sidecar = CanonicalDocument._sidecar_path(path)
        try:
            sidecar_stat = sidecar.stat()
            sidecar_value = f"{sidecar_stat.st_mtime_ns}:{sidecar_stat.st_size}"
        except OSError:
            sidecar_value = "missing"
        return (
            f"{_TOKENIZER_VERSION}:{path.resolve()}:{header_stat.st_mtime_ns}:"
            f"{header_stat.st_size}:{sidecar_value}"
        )

    def _sync(self) -> None:
        with self._lock:
            docs = self._active_canonical_docs()
            sync_key = tuple(sorted(
                (content_hash, source_path, str(path), self._fingerprint(path))
                for content_hash, source_path, path in docs
            ))
            if sync_key == self._sync_key:
                return
            conn = self._connect()
            try:
                active_hashes = {doc[0] for doc in docs}
                indexed = {
                    row["content_hash"]: row["fingerprint"]
                    for row in conn.execute("SELECT content_hash,fingerprint FROM indexed_docs")
                }
                stale = set(indexed) - active_hashes
                for content_hash in stale:
                    conn.execute("DELETE FROM passages WHERE content_hash=?", (content_hash,))
                    conn.execute("DELETE FROM indexed_docs WHERE content_hash=?", (content_hash,))

                for content_hash, source_path, canonical_path in docs:
                    try:
                        fingerprint = self._fingerprint(canonical_path)
                    except OSError:
                        continue
                    if indexed.get(content_hash) == fingerprint:
                        continue

                    conn.execute("DELETE FROM passages WHERE content_hash=?", (content_hash,))
                    try:
                        header = CanonicalDocument.read_header(canonical_path)
                        segments = list(CanonicalDocument.iter_segments(canonical_path))
                    except (OSError, ValueError, json.JSONDecodeError):
                        continue

                    rows = []
                    if segments:
                        for segment_index, segment in enumerate(segments, start=1):
                            metadata = segment.metadata or {}
                            page = metadata.get("page")
                            section = f"page {page}" if page else f"segment {segment_index}"
                            for body in _passages(segment.text):
                                rows.append((section, body))
                    else:
                        for body in _passages(str(header.get("text") or "")):
                            rows.append(("", body))

                    display_path = str(header.get("source_path") or source_path)
                    conn.executemany(
                        """INSERT INTO passages(content_hash,source_path,section,body,search_text)
                           VALUES(?,?,?,?,?)""",
                        [
                            (content_hash, display_path, section, body, " ".join(_terms(body)))
                            for section, body in rows
                        ],
                    )
                    conn.execute(
                        """INSERT INTO indexed_docs(content_hash,canonical_path,fingerprint)
                           VALUES(?,?,?) ON CONFLICT(content_hash) DO UPDATE SET
                           canonical_path=excluded.canonical_path,
                           fingerprint=excluded.fingerprint""",
                        (content_hash, str(canonical_path.resolve()), fingerprint),
                    )
                conn.commit()
            finally:
                conn.close()
            self._synced = True
            self._sync_key = sync_key

    def search(self, query: str, *, limit: int = 4) -> list[RetrievedPassage]:
        query_lower = query.lower()
        raw_terms = list(dict.fromkeys(_terms(query)))
        for source_term, expansion in _BILINGUAL_EXPANSIONS.items():
            if source_term in query_lower:
                raw_terms.extend(_terms(expansion))
        raw_terms = [
            term for term in dict.fromkeys(raw_terms)
            if term not in _QUERY_STOPWORDS and (len(term) > 1 or any(c.isdigit() for c in term))
        ]
        # Preserve single-character suffixes only when they form a technical
        # identifier phrase (for example, “Revision B” or “Datum A”).
        expanded_query = query.lower()
        for source_term, expansion in _BILINGUAL_EXPANSIONS.items():
            if source_term in expanded_query:
                expanded_query = expanded_query.replace(source_term, expansion)
        query_identifiers = _identifiers(expanded_query)
        for match in _IDENTIFIER_PHRASE_RE.finditer(expanded_query):
            raw_terms.append(match.group(2).lower())
        # Put part numbers, FAI identifiers, and measurements before generic
        # wording so long natural-language questions cannot truncate them.
        terms = sorted(
            raw_terms,
            key=lambda term: (
                0 if any(c.isdigit() for c in term) else 1,
                -len(term),
            ),
        )[:96]
        if not terms:
            return []
        self._sync()
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT content_hash,source_path,section,body,search_text,bm25(passages) AS rank
                         FROM passages
                        WHERE passages MATCH ?
                        ORDER BY rank
                        LIMIT ?""",
                    (match, max(limit * 32, 128)),
                ).fetchall()
            finally:
                conn.close()

        ranked: list[tuple[float, float, sqlite3.Row]] = []
        query_terms = set(terms)
        for row in rows:
            body_terms = set(str(row["search_text"]).split())
            body_text = str(row["body"])
            if query_identifiers and not query_identifiers.issubset(_identifiers(body_text)):
                continue
            matched = query_terms & body_terms
            if not matched:
                continue
            weighted_score = sum(
                8.0 if any(c.isdigit() for c in term)
                else 1.0 + min(len(term), 8) / 8.0
                for term in matched
            )
            source_terms = set(_terms(Path(str(row["source_path"])).name))
            weighted_score += 0.35 * len(query_terms & source_terms)
            ranked.append((weighted_score, float(row["rank"]), row))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        results: list[RetrievedPassage] = []
        seen: set[tuple[str, str, str]] = set()
        for _score, _rank, row in ranked:
            key = (str(row["source_path"]), str(row["section"]), str(row["body"]))
            if key in seen:
                continue
            seen.add(key)
            results.append(
                RetrievedPassage(
                    source=Path(str(row["source_path"])).name,
                    section=str(row["section"]),
                    text=str(row["body"]),
                    content_hash=str(row["content_hash"]),
                )
            )
            if len(results) >= limit:
                break
        return results

    def visual_context(self, query: str, *, limit: int = 2) -> list[dict]:
        """Attach the actual pages behind retrieved text, preserving provenance.

        Text extraction alone cannot establish spatial drawing relationships.
        Use canonical page assets rather than interpreting nearby text as labels.
        """
        if limit <= 0:
            return []
        passages = self.search(query, limit=3)
        docs = self._active_canonical_docs()
        attachments = []
        seen = set()
        terms = _terms(query)
        for word, expansion in _BILINGUAL_EXPANSIONS.items():
            if word in query.lower():
                terms.extend(_terms(expansion))
        for passage in passages:
            for content_hash, source_path, canonical_path in docs:
                if passage.content_hash and content_hash != passage.content_hash:
                    continue
                header = CanonicalDocument.read_header(canonical_path)
                source = Path(str(header.get("source_path") or source_path)).name
                if source != passage.source:
                    continue
                for index, segment in enumerate(CanonicalDocument.iter_segments(canonical_path), start=1):
                    page = (segment.metadata or {}).get("page")
                    section = f"page {page}" if page else f"segment {index}"
                    if section != passage.section:
                        continue
                    for asset in segment.images:
                        if asset in seen or not Path(asset).is_file():
                            continue
                        seen.add(asset)
                        label = f"{source}, {section}"
                        attachments.append({
                            "filename": label,
                            "parsed_text": f"Attached source image {len(attachments) + 1}: {label}",
                            "images": [asset],
                        })
                        if len(attachments) >= limit:
                            return attachments
                    if page and attachments and len(attachments) < limit:
                        from .visual_evidence import pdf_detail_view
                        detail = pdf_detail_view(
                            Path(str(header.get('source_path') or source_path)), content_hash,
                            int(page), terms, self.output_dir / '_talker' / 'page-crops',
                        )
                        if detail is not None and str(detail) not in seen:
                            seen.add(str(detail))
                            attachments.append({
                                'filename': f'{source}, {section}, detail view',
                                'parsed_text': f'Attached source image {len(attachments)+1}: enlarged original region containing small query-matching text from {source}, {section}.',
                                'images': [str(detail)],
                            })
                            if len(attachments) >= limit:
                                return attachments
        return attachments

    def _complete_segment(
        self, passage: RetrievedPassage, *, positioned: bool = False,
        include_identifiers: bool = False, query: str = "",
        filter_callouts: bool = False,
    ) -> str:
        for content_hash, source_path, canonical_path in self._active_canonical_docs():
            if passage.content_hash and content_hash != passage.content_hash:
                continue
            header = CanonicalDocument.read_header(canonical_path)
            if Path(str(header.get("source_path") or source_path)).name != passage.source:
                continue
            for index, segment in enumerate(CanonicalDocument.iter_segments(canonical_path), start=1):
                page = (segment.metadata or {}).get("page")
                section = f"page {page}" if page else f"segment {index}"
                if section == passage.section:
                    blocks = (segment.metadata or {}).get("positioned_blocks")
                    if positioned and blocks:
                        visible = [
                            block for block in blocks
                            if block.get("kind") != "unpositioned"
                            and (include_identifiers or block.get("kind") != "identifier")
                        ]
                        if filter_callouts:
                            callout_fields = {
                                "spot", "spots", "seam", "seams", "height", "heights",
                                "tolerance", "tolerances", "limit", "limits", "criteria",
                            }
                            matching = [
                                block for block in visible
                                if callout_fields.intersection(_terms(block.get("text", "")))
                            ]
                            if matching:
                                visible = matching
                        if filter_callouts:
                            lines = [
                                "Directly printed measurement callouts; each bullet preserves one callout's fields together.",
                            ]
                            lines.extend(f"- {block.get('text', '')}" for block in visible)
                        else:
                            lines = [
                                "Positioned PDF text blocks; x/y are approximate page percentages. "
                                "Use block boundaries and positions to keep nearby values with their labels. "
                                "Do not repeat coordinates unless requested.",
                            ]
                            lines.extend(
                                f"[x={block.get('x')}%, y={block.get('y')}%] {block.get('text', '')}"
                                for block in visible
                            )
                        return "\n".join(lines).strip()
                    return segment.text.strip()
        return passage.text

    def _has_positioned_blocks(self, passage: RetrievedPassage) -> bool:
        for content_hash, source_path, canonical_path in self._active_canonical_docs():
            if passage.content_hash and content_hash != passage.content_hash:
                continue
            header = CanonicalDocument.read_header(canonical_path)
            if Path(str(header.get("source_path") or source_path)).name != passage.source:
                continue
            for index, segment in enumerate(CanonicalDocument.iter_segments(canonical_path), start=1):
                page = (segment.metadata or {}).get("page")
                section = f"page {page}" if page else f"segment {index}"
                if section == passage.section:
                    return bool((segment.metadata or {}).get("positioned_blocks"))
        return False

    def measurement_inventory(self, query: str, *, max_chars: int) -> str:
        """Scan all active MSOP segments; heading inventory is not a full row export."""
        if not (query_requests_complete_list(query)
                and re.search(r"MSOP", query, re.I)):
            return ""
        entries = []
        pages_scanned = 0
        for content_hash, source_path, canonical_path in self._active_canonical_docs():
            if not re.search(r"MSOP", Path(source_path).name, re.I):
                continue
            # Respect explicit document/part-number scope when present.
            part_numbers = re.findall(r"\b\d{3}-\d{4,}\b", query)
            if part_numbers and not all(number in Path(source_path).name for number in part_numbers):
                continue
            for index, segment in enumerate(CanonicalDocument.iter_segments(canonical_path), 1):
                pages_scanned += 1
                heading = next((line.strip() for line in segment.text.splitlines()
                                if re.search(r"MSOP\s*:\s*FAI\s*\d", line, re.I)), None)
                if not heading:
                    continue
                # Keep the heading verbatim. Conflicting notes/ranges remain
                # separate evidence; never infer missing sequential IDs.
                notes = [line.strip() for line in segment.text.splitlines()
                         if re.search(r"can't|can’t|cannot|不能", line, re.I)]
                page = (segment.metadata or {}).get("page")
                section = f"page {page}" if page else f"segment {index}"
                entries.append(RetrievedPassage(Path(source_path).name, section,
                                               heading + ("\n" + "\n".join(notes) if notes else ""),
                                               content_hash))
        if not entries:
            return ""
        prefix = (f"MSOP heading inventory: scanned {pages_scanned} active document segments. "
                  "Only explicitly printed FAI headings and conflicting limitation notes are listed; "
                  "this is not every submeasurement, datum setup or method. Retrieve individual pages "
                  "for those details. Do not infer IDs from missing numbers or silently resolve "
                  "a heading/note conflict.\n")
        blocks = []
        used = len(prefix) + 160  # Reserve a truthful budget-coverage notice.
        for item in entries:
            block = item.render(len(blocks) + 1)
            if used + len(block) + 2 > max_chars:
                continue
            blocks.append(block)
            used += len(block) + 2
        if not blocks:
            return ""
        notice = (f"\nCoverage: {len(blocks)} of {len(entries)} detected heading pages supplied. "
                  "This does not establish completeness of all measurement rows.")
        return prefix + "\n\n".join(blocks) + notice

    def context(self, query: str, *, limit: int = 3, max_chars: int = 4_000) -> str:
        inventory = self.measurement_inventory(query, max_chars=max_chars)
        if inventory:
            return inventory
        passages = self.search(query, limit=limit)
        positioned = query_requests_complete_list(query)
        include_identifiers = bool(_IDENTIFIER_QUERY_RE.search(query))
        filter_callouts = positioned and bool(_CALLOUT_QUERY_RE.search(query))
        if filter_callouts:
            layout_passages = [item for item in passages if self._has_positioned_blocks(item)]
            if layout_passages:
                passages = layout_passages
        rendered = []
        used = 0
        seen = set()
        for item in passages:
            key = (item.content_hash, item.source, item.section)
            if key in seen:
                continue
            complete = self._complete_segment(
                item, positioned=positioned, include_identifiers=include_identifiers,
                query=query, filter_callouts=filter_callouts,
            )
            expanded = RetrievedPassage(item.source, item.section, complete, item.content_hash)
            expanded_block = expanded.render(len(rendered) + 1)
            block = item.render(len(rendered) + 1)
            # Keep a logical page/table whole when it fits. A ranked text window
            # may omit another requested field elsewhere on the same page.
            if used + len(expanded_block) + (2 if rendered else 0) <= max_chars:
                block = expanded_block
                seen.add(key)
            required = len(block) + (2 if rendered else 0)
            if used + required > max_chars:
                continue
            rendered.append(block)
            used += required
        result = "\n\n".join(rendered)
        if passages and re.search(r"FAI\s*\d", query, re.I):
            dependencies = self.alignment_context(passages[0], max_chars=max_chars - len(result) - 2,
                                                  start_index=len(rendered) + 1)
            if dependencies:
                result += "\n\n" + dependencies
        if result and query_requests_complete_list(query):
            notice = "\n[Coverage: ranked excerpts only; this is not a complete document review.]"
            if len(result) + len(notice) <= max_chars:
                result += notice
        return result

    def alignment_context(self, passage: RetrievedPassage, *, max_chars: int, start_index: int = 1) -> str:
        """Follow explicit alignment/datum references within the same MSOP."""
        if 'msop' not in passage.source.lower() or max_chars < 200:
            return ''
        reference_re = re.compile(r'\bALI_[A-Z0-9_]+\b|\bDATUM\s*[ABCD](?:_[A-Z]+)?', re.I)

        def references(text):
            values = set()
            for match in reference_re.finditer(text):
                value = match.group().upper()
                if value.startswith('DATUM'):
                    value = 'DATUM ' + re.search(r'DATUM\s*([ABCD])', value).group(1)
                values.add(value)
            return values

        for content_hash, source_path, canonical_path in self._active_canonical_docs():
            if content_hash != passage.content_hash:
                continue
            segments = list(CanonicalDocument.iter_segments(canonical_path))
            main_text = self._complete_segment(passage)
            wanted = references(main_text)
            program = re.search(r'Program\s+(UP|DOWN)\b', main_text, re.I)
            picked = set()
            blocks = []
            # Fixed-point traversal is bounded by the number of document pages.
            for _ in range(len(segments)):
                added = False
                for index, segment in enumerate(segments, 1):
                    if index in picked:
                        continue
                    heading = re.search(r'MSOP\s*:\s*(ALI_[A-Z0-9_]+|DATUM\s+[ABCD])\b', segment.text, re.I)
                    if not heading or heading.group(1).upper() not in wanted:
                        continue
                    candidate_program = re.search(r'Program\s+(UP|DOWN)\b', segment.text, re.I)
                    if program and candidate_program and program.group(1).upper() != candidate_program.group(1).upper():
                        continue
                    method = re.split(r'Location Definitions', segment.text, maxsplit=1, flags=re.I)[0].strip()
                    unit_label = re.search(r'X,\s*Y,\s*Z\s*\(mm\)', segment.text, re.I)
                    if unit_label:
                        method += '\nCoordinate unit label: ' + unit_label.group()
                    page = (segment.metadata or {}).get('page')
                    label = f'page {page}' if page else f'segment {index}'
                    blocks.append(f'[Source {start_index + len(blocks)}: {passage.source}, {label}]\n{method}')
                    wanted.update(references(method))
                    picked.add(index)
                    added = True
                if not added:
                    break
            output = []
            used = 0
            for block in blocks:
                if used + len(block) + 2 <= max_chars:
                    output.append(block)
                    used += len(block) + 2
            if len(output) < len(blocks) and used + 75 <= max_chars:
                output.append('[Alignment coverage incomplete: budget omitted some dependency pages.]')
            return '\n\n'.join(output)
        return ''
