# FAI overlay refinement

The point refinement path is drawing-agnostic. It consumes the active MSOP, decoded CAD triangle mesh, the local-model/retrieval candidate face set, and a coordinate-frame calibration artifact. Drawing-specific facts such as axis signs and translations belong in that calibration artifact; they are input data, not code branches.

For each FAI, point identity includes the point name, measurement program and coordinate frame. Calibration artifacts are applicable only to their verified program/frame scope and source hashes. Reused UP/DOWN frame names do not authorize reuse of an unscoped transform.

Equal source Z describes a sampling section in the MSOP coordinate frame; it does not imply one CAD face or equal CAD Z. With a valid rigid registration, the matcher intersects the transformed source section with CAD triangles and finds points on that intersection. A contour may traverse several CAD faces, and an inclined source section may produce different CAD Z values. Without a valid registration, the section is reported as awaiting registration and must not authorize replacement of existing markers. Explicit plane/flatness constructions require separate physical-surface validation; CAD face seams alone do not define separate physical surfaces.

The MSOP measurement parser extracts explicit characteristic indicators. Examples include `⏥` for flatness, `⏤` for straightness, `⊥` for perpendicularity, `∥` for parallelism, `⌖` for position, `ΔX`/`ΔY`/`ΔZ` for axis-parallel distances, and `X`/`Y`/`Z` for coordinate values. The Viewer shows the symbol, feature, and nominal/tolerance on the 3D pin; the complete method, probe, program, and source page remain available in the detail panel. A symbol is only shown when the MSOP text supports it.

Scanned, image-only tables use a separate `supplemental_fai_data` record on the active MSOP document. The local multimodal model can transcribe rows, then a visual check verifies the values and stores the source page, measurement symbol, feature code, and limits. These records do not masquerade as searchable native PDF text. Conflicts between pages are surfaced on each affected FAI; for example, the D52 pages 103 and 104 disagree about FAI 61-79.

## Recheck a project

```bash
.venv/bin/python scripts/mine_msop_cad_candidates.py \
  --project PROJECT_ID \
  --calibration PATH_TO_FRAME_CALIBRATION.json \
  --dry-run --recheck-pending --recheck-model-pending
```

Review the report before writing. When the point correspondence and face-constrained projection improve the location, rerun without `--dry-run`. Only pending automatic annotations can be refreshed. Manual bindings and reviewed annotations are not replaced; an update preserves the candidate origin and local-model confidence.

## Current D52 audit (2026-10-12)

The earlier 2026-10-09 common-face placement is historical display output, not proof of correct mapping. In particular, MSOP page 94 describes FAI 13 as a line profile with five equal-source-Z points and changing XY normals. Those points describe a contour and need not share a single CAD face. FAI 13 belongs to the DOWN program; no validated DOWN-to-CAD rigid registration is available. Its existing markers remain pending, and this iteration does not claim their placement is correct. The engineering report retains the source section and explicitly flags the missing registration.

Regression checks cover sections crossing multiple CAD faces, tilted sections with varying CAD Z, missing intersections, and rejection of unregistered equal-Z mappings. Measurement indicators include line profile `⌒` and surface profile `⌓` when supported by the MSOP.

## Local-model-only FAI completion (2026-10-15)

The automatic marking path treats geometric registration as a set of hypotheses, never as a placement decision. A local model must authorize a transform for each program/frame before coordinate-derived candidates can be emitted. The Viewer checkpoints each successful local-model comparison into the registration receipt; a retry with the same CAD, MSOP, model identity, and decision policy resumes at the next comparison, preserving both completed frames and partial pairwise tournaments. Transient inference failures are left retryable.

Only an accepted `local_model_candidate` annotation counts toward automatic coverage or is rendered as an FAI pin. Legacy manual bindings and hand-authored annotations remain provenance only: they cannot suppress candidate generation, block a fresh local-model decision, appear as automatic pins, or count as covered FAI. If the model abstains, the item remains visibly unmarked.

When an MSOP explicitly declares a point/feature count but only some or none of its XYZ rows can be recovered, the whole FAI is sent to the local multimodal model with the original method page, any referenced coordinate-definition page (including pages found from a partial point list), drawing page when available, every recovered XYZ row, and the CAD six-view atlas. The same full-group route is offered when XYZ exists but no complete transform has local-model authorization or the complete transformed group fails geometric gates. In that case, rejected transform positions are summarized only as diagnostic reasons and are explicitly barred from reuse. The prompt requires the exact declared marker count. The model must choose every view and location; a missing, extra, duplicate, or non-projectable marker rejects the entire FAI. Projection converts only the model-selected view pixels to mesh hits and does not select or repair locations.

The two active 3-suite projects currently have 82 FAI for 613-13599 Rev.06 and 68 FAI for 613-16076 Rev.08. A fresh source parse finds XYZ rows for 71 and 67 FAI respectively; 613-13599 FAI 48 has one of two declared points, while 613-16076 FAI 96 declares two vision lines without XYZ. Ten more 613-13599 entries have no XYZ, including non-measurement/feature-only records that still require a local-model decision. All existing annotations on both projects are stale and are excluded from coverage. The saved registration receipts contain no local-model-selected frames, and no local inference service is currently listening, so these projects have not yet been re-marked by a real model run. Unit tests verify the decision authority, exact-count fallback, checkpoint/resume behavior, and all-or-nothing anchor validation; they do not stand in for live local inference or project coverage.

The generic frame-candidate minimum is now five MSOP points: four are used to generate/refine a hypothesis and at least one remains held out. It still produces proposals only; the local model must select the full transform, and per-FAI point and section gates still apply. Temporary candidate-only runs over the two real project meshes show a concrete gain: 613-16076 `UP / ALI_ABD_A` previously had six source points and no rigid hypothesis; it now has 10 complete alternatives, with the best proposal's maximum surface residual at 0.0094 mm. Those source labels are referenced by 12 FAI items. This is evidence that the group can be presented for model review, not that the transform is correct or those 12 items are marked. The candidate runs were written only under `/tmp`; project receipts and annotations were not changed.

## All-FAI candidate coverage audit (2026-10-15)

The active project store was re-read through `ViewerStore` and `build_candidates(..., persist=False)` against the current-code, temporary five-point registration hypotheses. Candidate coverage is 82/82 for 613-13599 Rev.06 (46 same-source stale seeds plus 36 full-group local-vision candidates) and 68/68 for 613-16076 Rev.08 (52 seeds plus 16 local-vision candidates). These temporary runs changed no project annotations. Every FAI is now routable to a fresh local-model decision; this is candidate coverage only, not accepted marker coverage.

The source-page audit found that OCR recovers XYZ for 72/82 and 67/68 FAIs. FAI 48 has one of two named points, FAI 47/50 are explicit vision circle/line features, FAI 54-60 share an image-only page with grouped weld-location artwork, and FAI 96 names two vision lines. Page 102 also contains image-only `6X WCA MEASUREMENT LOCATIONS` artwork while its extracted method text says there is no chassis carrier in this operation. Those image-only instructions must be resolved by the local visual model against the full method page; the deterministic parser must not invent coordinates or treat hidden raster text as verified text. No local inference endpoint was listening during this audit, and the existing 59/52 marks remain stale.

The Viewer now counts and renders only current `local_model_candidate` results with `auto_marked` status and non-empty anchors. Legacy manual positions do not block model reprocessing, appear as active pins, or increase automatic coverage.
