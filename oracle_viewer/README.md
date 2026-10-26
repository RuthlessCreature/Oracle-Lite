# Oracle Viewer v0.3 — FAI 叠加与 MSOP 对照

本地工程查看与测量方法关联应用。与 TALKER 同进程并共用项目资料；当前部署的 TALKER 和 Viewer 均连接本机 Strata 原生模型端点。自动配准、FAI 标记和方案生成使用该本地模型。

入口：`http://127.0.0.1:7870/viewer`。当前机器使用已配置 Strata 环境的用户服务启动：

```bash
cd /home/cyg/Oracle/Oracle-Lite
systemctl --user start oracle-viewer.service
```

### Strata 原生模型模式

先在 Strata 仓库启动本机已有的原生 Qwen3.8 模型，再启动 Oracle Viewer：

```bash
cd /home/cyg/Strata && scripts/run-oracle-viewer-model.sh
cd /home/cyg/Oracle/Oracle-Lite && talker/start-strata-viewer.sh
```

Viewer 的方案生成和 FAI 候选复核调用 Strata 原生 API，TALKER 继续使用独立模型端点。页面顶部会显示 Viewer 当前模型和后端。现有 Qwen3.5 FAI LoRA 与 Strata 的 Qwen3.8 引擎架构不兼容，因此 Strata 模式不会加载这份 LoRA，并会在复核记录中明确说明。此前保存的标记保留其原有来源，不会被标成 Strata 生成。该 Strata 模型当前只接收文本。

## 已实现

- 项目创建，图号/Revision 标签，PDF MSOP/2D 图纸与 STEP/STP、X_T/X_B 原件导入。
- 从有文本层的 MSOP 的明确标题建立 FAI 索引；保留重复方法页，展开明确编号范围，提示标题/正文范围冲突。不能据此宣称所有子测量项已核验。
- 原 PDF 页面预览及下载；每项保留测量系统、尺寸系统、测头、程序、原文和基准/坐标系依赖链。结构字段缺失时不补猜值。
- STEP 通过隔离的 FreeCAD 进程读取，B-Rep 有效性检查；逐面生成显示网格及按源文件 SHA256 确定作用域的面 ID。
- X_T/X_B 在独立 `.venv-cad` 中解码；保留源面、装配变换及重复实例，再生成显示网格。真实 `613-13599-06_CYIA.x_t` 的 5,099 个源面已全部生成网格，107 个装配实例对应 5,221 个实例面。当前显示网格采用 0.01 mm 线性与 0.12 rad 角度偏差，含 1,207,916 个三角形；15 条短直边在源顶点容差内用端点弦线供显示拓扑使用，306 个退化三角形被剔除。它仍是预览网格，不是计量用 B-Rep。
- 若 X_T/X_B 所需的精确 Parasolid schema 未安装，Viewer 先尝试解析器提供的精确版本内置 profile，再尝试外部 schema。profile 覆盖不全时保留原始 CAD 文件并显示未覆盖节点；安装匹配 schema 后可对原件重新执行 CAD rebuild，不以相邻 schema 代替。
- 本地 Three.js WebGL 3D Renderer：CAD 着色材质、RoomEnvironment 环境光、旋转/缩放、俯视/正视、透明、Z 剖切、面选择（Shift 多选）。按装配实例合并为 107 次绘制调用，三角形仍可回溯源面；相机按模型包围球动态调整近/远裁剪面，默认只显示当前 FAI 标记以免遮住零件，可切换显示全部。
- FAI 显示标记支持 X_T 预览网格：点选真实面上位置、Shift 多选、编号引线、拖动布局、点击定位、全部/当前/隐藏显示和 MSOP 并排对照。保存时核对 CAD/MSOP 版本并校验锚点在所选显示面上。
- 通用候选挖掘：`oracle_viewer/candidate_mining.py` 与 `scripts/mine_msop_cad_candidates.py` 不绑定图号或 FAI 编号；它们索引 MSOP 命名点、展开跨页/缩写点位、按项目坐标校准假设投影到 CAD 网格并记录逐点残差。项目校准保存在独立 JSON 中，不能把一个图纸的坐标变换套到其他图纸。几何支持率不足 65% 时 Viewer 本地模型会跳过该候选。
- 本地模型自动配准与标记：打开有有效 CAD 和 MSOP 的项目时自动启动；几何代码只生成完整点组的刚体配准假设，由本地模型逐坐标系选择或弃答，再按每个 FAI 的完整 MSOP 点组、CAD 面及原文证据逐项决定 `AUTO_MARK` 或弃答。写入的有效状态为 `origin=local_model_candidate`、`status=auto_marked`；不等待人工确认。项目、源文件、配准算法或模型指纹变化后旧标记失效并重新判断。
- MSOP 有明确 XYZ 时，保留程序/坐标系作用域及全组点位，以模型选中的统一刚体变换生成显示候选；逐点核对 CAD 距离和源截面偏差，点位组不能拆散或只留易匹配点。注册候选未获模型授权时，视觉回退由本地多模态模型读取原始 MSOP、2D 图纸、XYZ 与 CAD 六视图，逐点选择点名和视图位置；不同程序/坐标系的 XYZ 点分别拟合刚体变换。对应坐标系通过拟合和表面门限时，marker 从模型选定的源点组计算几何中心后投影到 CAD 面；未获得可用变换的 marker 保留模型直接选中的视线交点。
- 同名 XYZ 跨页或跨程序/坐标系有冲突时，原始坐标版本作为明确选项送入本地模型；冲突行不会默认为第一行、不会平均，也不会与已确认点组混在一起。无冲突点仍可提供整帧变换候选；具体 FAI 的视觉回退要求模型逐个选择冲突点的原文页、程序、坐标系和 XYZ，否则不落标。
- 每个模型标记保留本地模型身份、原文证据和模型自评置信度（低/中/高）。置信度未经校准，不是统计正确率；`auto_marked` 表示自动生成的 CAD 显示位置，不等于计量验收或 AOI 能力确认。
- 位置覆盖只统计有锚点的本地模型 `auto_marked` 项；人工标记、待复核建议、过期记录和几何候选都不算完成覆盖。现场状态（2026-10-15）：613-13599 Rev.06 为 0/82（59 项旧标记 stale），613-16076 Rev.08 为 0/68（52 项旧标记 stale）；两份配准收据都没有 `local_model_registration` 模型决定，遗留标记不能算作当前覆盖。
- CAD/MSOP 变更自动使旧定位失效，保留事件记录；旧方案按输入快照标为 stale。
- 项目助手后台任务及结果保存、Markdown 方案下载、JSON 关联清单导出。
- Viewer 打开 TALKER 时携带项目与 FAI；显式询问其他 FAI 覆盖入口选项，回答不混用其他项目的检索资料。结果同步保存在 Viewer 项目任务中。
- 明确字段足够的圆测量方案使用既有原文核对/计算规则，其引擎名称明确展示；其他请求调用共享本地模型，最多 1024 新 token，长度不足有 truncated 标记。所有输出为待核对草案。

## 当前边界

- 当前 X_T 解码已通过，源拓扑检查通过；严格 Parasolid → OCCT B-Rep/计量转换仍未通过。显示导入器使用原始 CHART 曲线插值、等半径管面和源边界三角化；111 个 OCCT 面未通过严格有效性检查，3 个面采用明确记录的显示重建。所有源面均显示，不代表所有边界的计量精度已证明。`metrology_ready=false`，不输出可信计量体积、不启用 FAI 几何绑定。
- STEP 导入支持实体/面，但不保留完整产品树/PMI；面 ID 仅在完全相同源文件内稳定，不支持跨版本自动转移。
- 当前 FAI 标记是显示位置及原方法页关联，支持一项多个锚点；尚未完成 ALI_ABD / BD_ORIGN 实际基准配准，不代表 AOI 方法等效或计量验收。来源缺少完整 XYZ、坐标框架无法唯一确定、模型没有作出有效选择或候选未通过完整点组门限时，系统显示未标记及原因，不以几何近邻或人工确认冒充自动结果。
- 二维图纸只供查看与核对，尚未实现 2D 到参数化 3D 的重建或扫描 PDF OCR。
- 图纸重建试验生成的简化外包络属于明确标注的预览几何，不能代表完整零件、原生 X_T 或 FAI 计量几何。
- 依赖链按明确引用与 UP/DOWN 程序范围提取；算法设置、采点顺序、局部坐标变换仍需逐项人工核对。
- 不执行实际 AOI 测量，不宣称与海克斯康程序等效；尚未连接 PC-DMIS 程序、光学打样、GRR/误差验收或 AOI 相机/光源/工装的完整场景渲染。
- 不提供多进程写入/部署：原子 JSON + 进程内锁，仅支持一个 Uvicorn worker。当前是本地/LAN 工程开发应用，沿用 TALKER 的访问方式。

## 运行依赖与数据

- Python 环境复用 TALKER（FastAPI、PyMuPDF 等）。
- STEP 工作者默认 `/usr/bin/python3` 和 `/usr/lib/freecad-python3/lib`；需要本机 FreeCAD Python 模块。当前机器为 FreeCAD 0.19。
- X_T 工作者：运行 `scripts/setup_viewer_cad.sh`；隔离 Rust/Python 环境，固定 parasolid-kit 源版本 `2bc959687c8ca87145c4e0594828c3fa11aab912`，本地补丁见 `scripts/parasolid-kit-oracle.patch`，涵盖 SWEPT_SURF、NURBS 有理控制点、偏置法向、锥面参数及容差内端点。`sch_13006.sch_txt` 原样取自 MIT ps-parser；SHA256 `0dd291ea706fc306f16a78140e05e595e75c85ab63e4077e68941b127fcfdcc3`。没有更名 schema 或改写 X_T 文件头。
- X_T 当前源单位显式采用米、输出毫米，适用于当前 NX 文件；其他源单位需先配置/核验。不会根据外形大小自动猜单位。
- 标准 `SCH_*_20000` 已增加精确外部 catalog：`../third_party/parasolid-schemas/sch_20000.sch_txt`，内部 ID `20000`，SHA256 `f66857fb80ce2c85669f0fbd0ba57244cf58b3266da1fae41821185fb66dc730`。原始字节来自 Debian Gmsh 历史源码提交 `c49de80e94234f72219b0e0c9fdcbacbd3ffdf5c`；来源和验证记录见同目录 `sch_20000-provenance.json`，原发布版权说明单独保留，不能套用 `13006` 文件的 MIT 说明。安装脚本按固定提交下载并验证哈希，不更名其他版本，也不改写 X_T 头。
- 大模型显示网格超过 150 万三角形时，保持 0.01 mm 线性细分参数，依次调整角度细分参数并清理旧三角化后重新生成。保留全部源面和实例，输出 `meshing_attempts`、最终细分参数和源 schema/节点/边/顶点计数；仍超预算则明确失败。显示参数不是计量准确度保证。
- 前端 Three.js 0.170.0 / RoomEnvironment / OrbitControls 来自 npm 官方发行包，MIT 许可见 `static/vendor/THREE-LICENSE.txt`。运行无需 CDN，需支持 import maps/WebGL 的现代浏览器。
- 项目记录、原件和网格保存到 `talker/state/viewer/<project-id>/`，原件按 UUID 存储，原名/哈希保留。上传限制 128 MiB，STEP 几何转换 180 秒 / Parasolid 显示转换 600 秒，网格上限 150 万三角形。

## 验证

```bash
.venv/bin/python -m pytest tests/test_viewer_annotations.py tests/test_oracle_viewer.py tests/test_talker.py tests/test_engineering.py tests/test_engineering_plan.py tests/test_optical_calculations.py -q
```

覆盖负名义尺寸与编号范围消歧、重复页/冲突、基准依赖程序范围、项目隔离、定位有效性及资料更新失效、任务输入快照/重启恢复、PDF API、真实 STEP 尺寸/体积/面映射、TALKER/Viewer 共用状态与模型。

### 严格源边审计

独立于显示导入器运行，失败项保留源边 ID、源节点编号、曲线类型及源容差。不会丢弃失败的边后宣布转换通过，部分审计也不会标成完整。

```bash
.venv-cad/bin/python oracle_viewer/parasolid_audit.py \
  ../3suite/613-13599-06_CYIA.x_t \
  ../reports/oracle-viewer/strict/source-edge-audit.json --source-unit m
.venv-cad/bin/python -m pytest tests/test_viewer_cad_geometry.py -q
```

单位必须显式指定。`--max-edges N` 可用于局部排查；报告每 100 条边更新，返回码 2 表示存在失败项或未审完全部边。源边成功构造仍需验证曲面裁剪、壳、实体、法向和基准配准，报告始终不将这一步单独认定为计量合格。

局部补丁已补充近乎平坦锥面的稳定投影、闭合交线均衡加密、源直线端点容差，以及经单侧导数/非有理控制网验证的 G1 偏置接缝。没有用放宽容差来掩盖真实折角。其他有理/周期 C0 接缝仍需单独验证。

DOM 回归脚本 `scripts/check_viewer_dom.mjs` 需 Node 22+、jsdom 26.1.0，默认模块目录 `/tmp/oracle-viewer-ui-test`（通过 `VIEWER_JSDOM_DIR` 指定）。该测试覆盖列表/选项/原页链接/状态控件，并模拟无 WebGL 分支；不代替真实浏览器的视觉和 WebGL 验收。独立浏览器测试 `scripts/check_viewer_render.mjs` 使用 Chromium 154 / SwiftShader 实际绘制当前 X_T，验证完整面数、实例数、107 次绘制调用、旋转、缩放、面选择、透明、剖切、原文切换和预览标记点选。截图与 JSON 报告见 `../reports/oracle-viewer/render/`；已视觉检查截图。软件 WebGL 验证不代表所有客户端 GPU/浏览器均已验收。

FAI 浏览器回归：`npx -y -p node@22 node scripts/check_viewer_fai_overlay.mjs`，使用临时项目验证原页联动、对照视图、显示过滤、引线拖动与刷新持久化、点选保存/复核/移除。报告位于 `../reports/oracle-viewer/fai-overlay/`。

下一阶段按 FAI 优先：扩展特征候选匹配 → 完整测量方法与基准结构化 → AOI 可测性与方案布局；严格 B-Rep 精度验证保留为计量前置条件。

### Features 坐标表格式

支持按原始 `Features X Y Z` 表格读取完整 XYZ，并保留 `Output system` 标识、原始行与页码；`Features X Y` 只保留两轴证据，不补 Z。无 Program 字段的文档保持程序为空，仅在文档没有任何显式 Program 且表格明确给出输出坐标系时构建该输出坐标系的配准候选。通用 `Measure N points` 与整表点数严格对齐后，可将点名清单交给本地模型核对；代码不决定配准或 FAI 标记位置。
