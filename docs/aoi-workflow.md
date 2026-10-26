> 2026-10-10更新：当前入口为“方案设计与设备库”。观察方向由CAD关联面自动生成，并执行标记点光路遮挡检查；下文初版±Z与早期覆盖数据为历史记录，不能作为当前方案依据。当前有效结果请以Viewer中的任务编号和过期状态为准。

# 通用 AOI 方案工作流

Viewer 入口：项目工具栏 → **方案工作流 · LIBRARY**。六个页面串联测量任务、可测性、规格库、本地模型决策、三维布局/BOM、验证与交付。

## 决策与工具边界

本地 Viewer 模型（当前 Strata native / Qwen3.8-Flash-Next IQ2_XS）收到本项目方法证据、设计输入和工程工具校核后的候选，返回 `selected_head_ids` 与理由。确定性工具执行源字段提取、规格/光学计算和有限候选集的最少光学头求解。模型选择合法候选后，场景与 BOM 从同一配置生成。

模型未加载、输出截断、候选 ID 不合法、证据超过上下文预算都会失败并保留原因。模型可以弃权，工具推荐不会自动冒充模型决策。单项 AOI 文字方案也必须经过 Viewer 本地模型推理。没有使用云端模型生成方案。

当前模型通道仅支持文本。没有宣称它直接分析 PDF 页面图像或进行原生 CAD 几何计算。已有标记保留各自来源和待复核状态，不能重新标为 Strata 生成。

## 数据与复用

`oracle_viewer/workflow.py` 不包含项目编号或 FAI 编号特例。项目只提供 MSOP、CAD、标记和设计输入。所有显式 FAI 都进入任务表，保留原方法全文、测头、坐标系、设置、名义值/公差、基准依赖、来源页及位置来源/置信度。

初版几何规划器支持视觉圆、CAD XY 平面 ±Z 视向、暂定背光。特征范围由显式输入的锚点外扩半径暂代，尚非精确物理轮廓。其他方法/视向显示需要通道适配；没有位置不能产生场景。默认自动纳入全部FAI，支持手动选择。小候选集使用精确集合覆盖，大项目采用有界空间候选和贪心覆盖，不宣称全局工站最少。模型证据超过预算会明确报错，避免静默截断。

有限候选集通过精确动态规划 set cover 求解最少光学头作为模型工具建议。相同硬件/方向的大规模候选可去除被更大覆盖支配的项。此优化没有证明整机工站数最少；工站、相机、平台运动和多曝光仍需额外约束。

输入指纹覆盖 MSOP/CAD 元数据、标记、库与算法版本；有模型身份记录的运行还检查后端/型号变化。资料更新后草案显示过期。更新采用原子写入，运行过程并发受锁保护，服务重启会解除中断运行的锁定状态。

## 规格 LIBRARY

结构化目录：`oracle_viewer/catalog/hardware.json`；下载记录：`manifest.json`；原文件：`assets/`。

第一批 7 个型号：Basler acA2440-35um；Opto Engineering TC23024 / TC23036 / TC23048；LT2BC048036-R-CO 背光；LTRN024RD 环光；LTIC1CH-D1-4 控制器。7 份规格/性能 PDF、7 个厂商网页已落盘，记录 URL、下载时间、文件长度与 SHA256。未知价格、库存和供应商 CAD 不生成默认值。

重新下载：

```bash
.venv/bin/python scripts/sync_hardware_library.py
```

脚本直接访问公开厂商资料，验证 HTTP 状态与 PDF/HTML 类型，失败记录到 manifest。目录扩展需审核厂商字段，按相同单位和来源结构录入。现有库是首批工程种子库，不是穷尽市场的产品库。

官方来源：

- [Basler acA2440-35um](https://docs.baslerweb.com/aca2440-35um)
- [TC23024](https://www.opto-e.com/en/products/tc-series/TC23024)
- [TC23036](https://www.opto-e.com/en/products/tc-series/TC23036)
- [TC23048](https://www.opto-e.com/en/products/tc-series/TC23048)
- [LT2BC048036-R-CO](https://www.opto-e.com/en/products/lt2bc-series/LT2BC048036-R-CO)
- [LTRN024RD](https://www.opto-e.com/en/products/ltrnst-series/LTRN024RD)
- [LTIC1CH-D1-4](https://www.opto-e.com/en/products/ltic-series/LTIC1CH-D1-4)

## 613-13599 Rev06 实际试点

自动模式纳入全部82项；手动模式可选择FAI编号。最新完整流程验证中，本地Strata选择两个主布局覆盖6项（23、24、31、33、80、81），另选照明对比布局。76项仍未覆盖，具体原因与原MSOP方法保留在任务表。完整范围进入规划不代表全部方法已经适配。当前二维通道、±Z视向与输入特征包络无法替代接触/3D测量。

物理工具排除了30个干涉或安装间隙不足的候选。相机/面板包络与工件三角面采用分离轴检测；圆柱镜头、环光与设备互相干涉采用表面/顶点采样；用户可提供夹具禁入包络和最小安装间隙。采样可能漏检，精确供应商BREP、支架/螺钉、线缆弯曲、装卸和运动轨迹仍缺失。安装状态保持未验证。布局代表独立曝光姿态，尚未完成多头同时安装校核。

三维工具栏提供照明模拟开关、CAMERA PREVIEW和物理校核说明。模拟使用光源形态布置的4个近似点光源及假想材质，未模拟真实阴影、透射、BRDF、光谱响应或传感器。相机预览使用物方远心正交视场，隐藏设备包络并按FOV比例留黑边；未模拟光学PSF、畸变、景深或噪声。因此只能查看视场和概念照明，不可作为照明定版或计量证据。

完整交付 JSON：`GET /api/viewer/projects/{id}/workflow/export`；候选 BOM CSV 加 `?format=csv`。验证项涵盖原方法等效、基准、拟合/滤波、光学、机械、标定、不确定度/GRR和节拍。初始均为 pending，采购就绪和计量验收保持 false。

## 验证命令

```bash
.venv/bin/python -m pytest -q tests/test_viewer_workflow.py tests/test_talker.py tests/test_oracle_viewer.py tests/test_viewer_annotations.py tests/test_engineering.py tests/test_engineering_plan.py
npx -y -p node@22 node scripts/check_viewer_render.mjs
npx -y -p node@22 node scripts/check_viewer_workflow.mjs
```

软件检查通过不代表 AOI 测量通过。硬件打样、原基准复现、平台坐标转换、与海克斯康参考测量的偏差和 GRR 尚需实际证据。

### Local lighting decisions and automatic scene display

The v2 candidate compiler enumerates transmitted panel, front ring and oblique ring trial poses from the same manufacturer library. It does not select illumination quality. Ring distance uses the published distance range; 35°/60° tilts are trial design assumptions and remain subject to physical rejection. Unknown surface properties, transmission, shadowing, illumination coverage and mechanical clearance remain explicit trial requirements. No project or FAI number determines the lighting type.

The local Strata decision chooses primary `selected_head_ids` and optional `alternate_head_ids`. Same-coverage lighting comparison groups are supplied to prevent confusing a backlight candidate with the only possible merged head. Both ID sets must reference checked candidates, with no overlap or duplicates. User `lighting_evidence` notes are treated as unverified evidence; this text-only model does not inspect test images. Compare edge contrast, saturation/glare, continuity, fitting repeatability and reference bias before deciding the lighting geometry. Procurement and metrology acceptance remain false.

A run submitted from the current browser automatically opens its first model-selected 3D head when it completes. Failed, stale and abstained runs do not render a guessed replacement. The 3D toolbar switches between primary and model-selected comparison layouts. The library can be opened without automatically displaying an old draft. Primary BOM and individual comparison BOMs are separate. Hardware meshes remain manufacturer dimension envelopes, not vendor CAD or a validated optical simulation.

Regression: `scripts/check_multilight.mjs` submits an actual local Strata run and verifies automatic display and all model-selected lighting scenes. `scripts/check_firefox_viewer.py` cycles the layouts using installed Firefox and checks disposal, idle rendering and drawing-buffer limits.

当前端到端验证：`npx -y node@22 scripts/check_full_workflow.mjs`；报告位于 `/home/cyg/Oracle/reports/oracle-viewer/workflow/full-workflow-test.json`。

### v4: automatic scope is all marked FAI; progress and staged local inference

Automatic mode selects every current marked FAI, including original contact/vector-point and depth methods. Unmarked source items remain in the MSOP index outside the automatic proposal denominator. Stale marks remain a source problem and cannot silently become valid acquisition positions. Manual mode still selects explicit items.

The acquisition compiler operates on every marked anchor (`FAI:anchor_index`), tiles features larger than the field or depth range and merges nearby points. An FAI has acquisition coverage only when **all** its anchor IDs are covered by selected layouts. A partial view containing one point cannot count as complete FAI coverage. The renderer shows only points assigned to the current layout. These layouts describe camera acquisition positions; they do not implement 3D metrology.

Local Strata proposes source-constrained measurement routes for all selected FAI in adaptive source batches. Entire source methods and dependency pages are retained in the snapshots; oversized batches are split rather than truncated. Routes distinguish 2D, 3D surface, confocal height, hybrid or required method review. Layout selection is a separate inference stage containing checked candidates, point coverage, optical checks and compact route references; method descriptions and original methods are not redundantly copied into that stage. Original method outputs and their input hashes remain available in the export. Failed runs can reuse validated method outputs when the source task snapshot, design and local model identity match exactly.

Submission returns immediately; geometry checks and inference run in the background. `run.progress` reports phase, actual completed/total work, description and update time. Phases include queued, geometry, physical, method_routes, layout_model, complete/failed. The workflow displays these stages, item counts and elapsed time. Inference progress is indeterminate while a batch is being generated; validated batch counts advance only after parsing and exact ID validation. Partial method routes appear as they finish.

Actual native Strata verification for 613-13599 Rev06: **52/52 marked FAI proposals; 310/310 anchors assigned acquisition layouts**. Routes: 42 `3d_surface`, 9 `2d_vision`, 1 `confocal_height`. 3D/confocal catalog selection, method equivalence, datum registration, visibility and complete installation checks remain unvalidated; all metrology acceptance flags remain false. Proposal/acquisition coverage does not mean actual measured coverage. Candidate layouts are exposure poses, not a verified number of simultaneously installed cameras or stations.

Verification: `scripts/check_marked_workflow.mjs`; report `/home/cyg/Oracle/reports/oracle-viewer/workflow/marked-workflow-test.json`. It checks all marked IDs, every anchor, genuine local-model route provenance, visible progress and selected 3D layouts.

## Whole-project alternatives and expanded manufacturer library (2026-10-10)

The library now contains 92 model records: 11 area cameras, 14 lenses, 33
lights, 2 light controllers, 31 chromatic-confocal heads, and 1 laser profiler.
Sources include HIKROBOT, HuaRay/iRAYPLE, OPT, Hypersen and the original
Basler/Opto Engineering records. Public manufacturer parameter tables are
retained where PDF download requires vendor registration. HIKROBOT's two PDF
sources can be read through the web reader, but direct download returned WAF
HTML; their local TXT files are explicitly curated field extracts, not original
PDF downloads. OPT public parameter-table snapshots carry `tls_verified=false`
because its public site's certificate was expired during collection. No account
or gated PDF download was bypassed. Each local asset has a SHA256.

A source record is distinct from a geometry-ready record. Missing lens length,
front diameter, lighting dimensions and placement limits prevent automatic
geometry use. Color/rolling-shutter cameras need a separate metrology adapter.
`optical_pair` reports these exclusions rather than inventing specifications.
The local model chooses a bounded valid optical pair and trial lights before
candidate geometry generation, keeping all marked FAI in scope while avoiding
an expanded-library Cartesian-product context overflow.

After original-method proposals and pose selection, native Strata produces
three whole-project alternatives: shared equipment/station priority, throughput
priority, and domestic equipment priority. These are model decisions, not
prewritten station assignments. Every selected pose must belong to exactly one
station, and every marked FAI must have exactly one measurement station.
Depth/confocal channels remain distinct from 2D acquisition coverage; neither
station assignment nor a sensor selection proves Hexagon method equivalence.

A physical resource is identified by station and manufacturer model. Its
quantity is one for sequential use within that station, regardless of the number
of exposure poses. The same model used in two stations requires two instances.
Sharing assumes a controlled movement/reconfiguration sequence and is not a
proof of collision-free travel. Lighting controllers are checked for physical
channel count and available electrical ratings; insufficient or unknown
channel/wiring/current data remains a visible gap. Confocal heads do not imply
a bundled compatible controller or fiber kit; those missing items remain in
integration BOM rows with unknown quantity and specification.

Viewer workflow step 05 compares complete alternatives, shows station FAI
assignments, motion, datum registration, acquisition/lighting sequence, resource
instances and per-alternative BOM. Scene inspection loads one acquisition pose
at a time. Depth-sensor geometries and inter-station installed assemblies are
not rendered as validated layouts. JSON and CSV exports accept `scheme_id`.
The run retains all native-model attempts, input hashes and stage progress.

Collection: `scripts/expand_hardware_library.py` and
`scripts/expand_confocal_library.py`. Refresh uses
`scripts/sync_hardware_library.py`, retaining existing hash-verified originals
on download failure. A partial catalogue entry must be completed from its
manufacturer before it can become a geometry-ready configuration.

Whole-project decisions can reference immutable source `head_groups` and
`method_groups`. Tools expand these references; the local model still chooses
which station receives each group. Overlapping groups, duplicate/missing IDs,
unknown models and truncated output are rejected. Multi-pose sharing requires
`move_optics` or `move_part`; the relative camera-position span is reported as a
necessary framing range, not as an accepted stage travel specification. Duplicate
alternative configurations are rejected even when their names differ. A final
local-model narrative audit uses the compiled resource instances and motion
modes to correct claims that conflict with the actual BOM. Both the original
and audited model text are retained.

For cooperating stations, source-group memberships describe acquisition work;
`method_group_owners` is the model-selected unique final-result owner. The
compiler expands this to a per-FAI owner map. A duplicated acquisition membership
without an explicit final owner is rejected. Raw points from all participating
stations must be registered to the actual part datum and fitted once under the
original MSOP; averaging separately fitted station results is not permitted.
`scan_region_group_ids` references provisional marker search areas for depth
sensors, not camera exposure assemblies. These references do not instantiate
additional cameras/lenses/lights, nor prove a viable depth-sensor pose. Complete
native camera decisions may be reused only when their regenerated tool-evidence
hash matches exactly; physics is still checked against current geometry.

### Whole assembly renderer and merged illumination

The whole-scheme tab now offers **在 Renderer 渲染整机总方案**. A separate native
Strata stage allocates the source exposures to multiple physical camera units,
chooses their home poses and assigns light ownership. Each camera owns an exact,
non-overlapping subset of the source exposure IDs; each exposure has one light
owner. Its own expanded BOM replaces the original camera/lens/light instance counts.
The existing minimum-equipment alternatives remain available for comparison.

Spatial groups are tool proposals, independent of project IDs. The model chooses
groups and home poses, while tools validate membership, optical compatibility,
station ownership, exact complete exposure coverage and home-envelope intersections
and clearance. Detected collisions are fed back to native inference; unresolved
warnings remain explicit. Home-envelope screening does not validate parallel motion,
fixtures, cables, depth-sensor installation or metrology.

Only compatible, same-direction transmitted panel illumination can be combined by
the current area adapter. The union includes complete source FOV footprints and a
margin; rings, oblique or reflective lights cannot be relabeled as a backlight.
A larger emitter without verified supplier dimensions is rendered as a translucent
orange **requirement surface**, with no fabricated body thickness or catalog model.
Its required area appears in the expanded BOM. Electrical capacity, body dimensions,
uniformity, collimation, edge identity and transmission must still be validated.

The renderer draws one envelope per physical camera/lens/light instance, plus
lightweight FOV outlines and relative target references. It does not duplicate CAD
or add GPU lights for every exposure. Camera selection permits the existing geometric
preview; the FAI-label selector remains effective in whole-assembly mode. Unknown
3D/confocal sensor installation is shown as search-region references rather than an
invented body pose. No claim of supplier-detail rendering or imaging simulation is made.

Assembly endpoints:
- `POST /api/viewer/projects/{project}/workflow/assembly/{shared|throughput|domestic}`
- `GET` the same route for progress, full native evidence, assembly scene and expanded BOM.
- Add `?format=csv` for expanded BOM export.

Saved assemblies are tied to the parent workflow and assembly engine digest; input
changes mark them stale. Interrupted jobs are unlocked on server startup. Each
whole-scheme card offers regeneration and its own JSON/CSV exports.

### Step 05: actions and paths PowerPoint

Step 05 provides **导出动作与路径 PPT** for the selected whole scheme. It snapshots
the current assembly and workflow, captures actual Viewer overview/top/camera
frames, and asks local native Strata to explain the acquisition actions. Progress
is shown while the model, renderer and presentation builder run; completion
downloads an editable PPTX and leaves a download link in the panel.

The deck includes the whole assembly, all camera-group exposure positions,
illumination ownership/merge requirements, original FAI method evidence, expanded
BOM, and remaining mechanical validation work. Position-chart connections show
source ordering, not validated collision-free machine trajectories. Missing axis,
speed, trigger, fixture or metrology facts remain explicitly pending.

Presentation jobs require the exact current assembly ID. Changed source evidence
marks jobs stale; interrupted jobs are failed on startup rather than left running.
Job files are stored under each project's `assembly-presentations` directory.
The `presentation` POST/GET and `presentation/{job}/download` routes extend the
assembly endpoint. The artifact-tool runtime and presentation skill location can
be configured with `ORACLE_PRESENTATION_RUNTIME` and
`ORACLE_PRESENTATION_SKILL_DIR`. Preview rendering uses software Vulkan when
available to avoid competing with local model inference for GPU memory.

### Main plan and independently rendered subplans

Step 05 lists each source acquisition pose separately, with its parent scheme,
station, FAI IDs, anchor count and hardware. A child pose can be rendered on its
own. It retains a link back to the current assembly; restoring the main plan
checks its identity and stale state. Source changes also prevent opening an old
child snapshot as if it belonged to the current run.

The assembly renderer selector distinguishes the full main plan, individual
camera groups and depth search regions. A camera-group view hides other cameras
and keeps only its related shared lights and assigned marker atoms. It filters
existing lightweight objects rather than duplicating CAD. A depth view shows
search references, not fabricated installed sensor poses. Child-pose trial lights
are distinct from the main assembly's merged-light requirement and expanded BOM.

## 2026-10-10：工程界面文案与自动观察方向

界面使用“测量要求、设备规格库、设备选型与拍摄安排、整机布置、安装与干涉检查”等工程术语。日常状态显示任务范围与方案/拍摄覆盖；本地模型身份、任务编号、版本、原始字段和错误信息保留在展开详情中。MSOP原文、FAI编号、GD&T符号及导出数据不改写。词汇集中在`static/copy.js`，新增状态和规格字段应提供工程名称。

观察方向不再作为用户输入。生产提交统一设置`normal_axis=auto`，旧客户端提交的±Z值也被忽略。`visibility.py`从标记的当前CAD关联面取得局部方向，评估两侧观察，并用三角网格BVH光线检查排除被工件和夹具禁入范围遮挡的点。方向、相机平面和深度范围确定后才合并拍摄；背光还需通过反向透射路径检查。标记点缺少面关联、明显偏离关联面或无可用光路时保留为未覆盖，02步骤显示逐点原因。

三维视场、相机外形、相机预览和共用背光面积使用局部表面坐标系。正反方向的背光不能合成同一块光源。整机共用相机若跨观察方向，必须由本地模型明确选择转动安装或翻面夹具，并记录旋转与安装验证待完成；只有平移不能覆盖背面点。

这些检查仍基于显示网格，针对标记点中心的几何光路。完整被测边缘、透明材料、实物夹具、线缆、机械行程、照明效果和计量方法等效需另外验证。通过光路检查不代表安装或计量验收。源面存在近似或标记位置不可靠时，仍需人工对照CAD与MSOP。

为了保持全项目推理上下文有界，工具向本地模型提供完整可用点集的一组布置建议及同覆盖范围的照明对比，完整候选仍保留在记录中。工具不会直接成为最终方案，最终选择、测量方式和工站分配继续由本地Strata生成。

613-13599 Rev.06 后续验证结果见本节末的 2026-10-12 记录；上面的 31/21 坐标数和旧的 68 姿态总装均为历史版本，不作为当前数据。

MSOP 的 `Location Definitions` 坐标现在进入测量通道规划和拍摄点位规划。系统按点名、坐标系、来源页保留 XYZ 与可识别的 IJK；项目专用校准必须同时匹配图号、Revision、CAD 哈希与 MSOP 哈希。当前613-13599校准只注册了各坐标系的 XY 变换，**没有注册 MSOP Z 到 CAD Z**。因此工具将每个点的原始 XY 投影到该 FAI 已关联的 CAD 面，并保留原始 Z 作为测量点组证据；同一坐标系中同 Z 的构造点会联合核对是否落在同一 CAD 面。只有全部源点、CAD标记和一对一位置匹配都完整时，规划才用该点表替代旧显示锚点；否则保留原锚点，逐点提供已成功/未成功映射状态给本地测量方法规划。

当前坐标统计见下方 2026-10-12 记录。FAI 53 原页未标明坐标系时，只有项目内多个校准假设对全点集产生明确且充分的几何支持差异，才采用支持最高的假设；否则不映射。点面映射仍是CAD显示几何证据，不代表计量坐标配准或验收。

总装格式重试会给出全部旋转配置错误及缺失拍摄组，携带上一轮完整的本地模型配置供修正，避免修正一个位置时漏掉其余位置。只有输入任务和模型身份相同才引用此前失败配置作为修正输入；引用来源与原始输出保留，配置仍必须重新通过本地模型生成和完整校验。

### 2026-10-12：MSOP XYZ 点位与总装复核

当前运行编号 `58392cb18f6d4e3c80c4d6a06e14ff85`（Strata native / `qwen3.8-flash-next-iq2_xs`）。52/52项均生成测量方法方案，68个拍摄姿态、3套工站方案；其中26/52项的全部现有标记点进入拍摄布置，另26项保留具体不可见或设备约束原因。测量方法已生成不等于AOI测量能力验收通过。

点位映射限值收紧为 CAD 平面 XY 残差不超过0.35 mm。52项共提取368个显式 MSOP 点，275个点满足当前几何邻近限值；25项全点可一对一对应，27项为部分对应，后者继续使用原CAD标记位置，并将完整点表交给测量方法规划。0.35 mm 是本项目显示几何映射门限，不是计量精度或FAI公差。超限点不会被拉到最近CAD面。FAI 7 的33点和FAI 13 的5点均完整映射；FAI 23 与24分别有3/6和4/6点通过邻近筛选，故保留原标记。MSOP Z 未注册到 CAD Z，不能直接转换。

总装组按相机、镜头和照明类别形成4个拍摄组，保留组内全部68个原始姿态；视向不同仍强制显式旋转/翻面。Strata选择4组相机与4组移动光源后，几何工具仅从各自已分配的拍摄姿态中选择初始停靠位姿，并筛除停靠包络相交；本次结果0处停靠包络碰撞。运动轨迹、夹具、线缆、照明效果和安装验收仍待实物验证。Viewer 页面已检查 XYZ 点位表、来源与坐标系说明、侧向相机预览、自动观察方向、主方案/子方案切换及整机回切。

注意：Strata 当前加载的是量化 Qwen3.8 GGUF 推理模型；现有 Oracle LoRA 训练链路面向 Qwen3.5，不能套用到该 GGUF。此轮改进使用本地点位证据、分组规则和几何校验，不等于更新了模型权重。


## 三件套与 Station / Setup 通用流程（2026-10-13）

方案入口“编制全项目方案（本地模型）”处理项目全部 MSOP FAI，包含尚未定位的项。流程读取项目自己的二维图纸、三维 CAD 与 MSOP 文档身份、测量方法、关联基准、显式 XYZ/IJK 点组以及当前 CAD 面关联，不按项目编号写分支。613-13599 与 613-16076 用于交叉验证。

本地 Strata 分批读取完整 MSOP 方法和关联页，给出二维、三维表面、共焦、混合或需进一步核对的方法路由。上下文超限时缩小批次；不截断原方法，也不接受截断的方案。完整通道分组可由模型引用，工具无损展开 FAI 编号，避免长编号清单漏项。设备候选、照明与工站分组也由本地模型提出；工具检查范围、规格匹配、唯一编号和资源归属。工具校验不是另一套模型决策。

一个 station 可包含多个 setup。每个 setup 保留测量方式、FAI、设备实例、装夹状态、运动及采集顺序。二维与三维通道不能因在同一工站而混作同一测量方式。每个 FAI 有一个结果归属；跨 setup 的依赖需要保留 MSOP 基准和实际工件配准。相同型号默认是独立设备；明确指向同一实例才计为共用，不能跨工站共用一台设备。

Viewer 六步：

1. 测量要求：公差、测量指标、方法和关联页。
2. 测量区域与测量方式：本地模型逐项路由、方法等效性缺口及显式点位证据。
3. 相机、镜头与照明库：厂家参数、来源与本地资料。
4. 设备选型与拍摄安排：按 setup 分列设备候选和测量区域。
5. Station / Setup 与 BOM：工站、子 setup、动作顺序、设备实例、CT，分别查看三维测量区域并导出完整记录。
6. 验证与交付：配准、夹具、遮挡、碰撞、安装、误差预算及节拍的未完成条件。

“生成已定位项的安装候选”保留原有设备外形/光路初筛，不能把它的二维拍摄覆盖等同于全项目测量方法覆盖。不同工站的设备按独立场景检查，不能将相同局部位置当成跨工站设备相撞。

CT 输入为各 setup 的完整动作时间，加每站上下料、定位等开销；缺一个时间便不输出总时长。当前是串行候选估算，整线流转、并行资源、稳定等待及实际控制时序仍待验证。像素间距只做候选规格初筛，逐 FAI 的系统误差预算尚未完成。

当前工站输出属于候选工程方案。它还没有自动生成并验证全部设备安装位姿、夹具 CAD 和整线动作；Viewer 的工站/Setup 按钮显示已定位 FAI 区域，不能当作设备安装渲染。未定位项保留在方案清单，安装及计量验证覆盖明确为零。

### 第 05 步：导出完整方案 PPT

完整且未过期的全项目方案提供“导出完整方案 PPT”。导出保留六个节点的输入、处理动作、输出和核对条件；总方案、各 Station 与 Setup 分别保留当前 CAD 测量区域视图。设备实例 BOM、候选规格及厂家资料来源、装夹要求、运动与采集顺序、节拍缺口、逐项测量要求、原始 MSOP 点表和当前 CAD 标记坐标均进入文件。尚未定位的 FAI 继续列入附录，不能从方案范围中消失。

节点说明由本地 Strata 分别生成，工具负责结构检查、排版、数值保留和截图。节点回复截断、格式错误或内容不全时重新请求；成功节点按完整输入、方案和模型身份缓存。六个节点与所有源 FAI 完整后才开放下载。页面显示生成进度，刷新或重新打开项目仍可查看任务状态和下载结果。方案或资料发生变化后，旧文件标为过期，须重新导出。

PPT 中的表格可以编辑；完整原方法、未简写的坐标及 CAD 面身份保留在备注。原始 MSOP XYZ/IJK 与当前 CAD 坐标分别列示，未建立的坐标变换不补造。设备型号相同不自动计为共用，只有同一工站中明确共享的设备实例合并数量。CAD 截图展示测量位置范围，不表示硬件已经安装、动作路径已经避障或计量验收已经通过。
