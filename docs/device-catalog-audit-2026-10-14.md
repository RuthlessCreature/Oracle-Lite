# AOI 器件库下载与几何核查

更新：2026-10-14。器件库版本：`2026-10-14.48`。

## 当前数量

- 目录有 2,353 条器件记录；2,255 条记录关联 CAD 文件（包括 STEP/STP/IGES、DXF 等），其中 104 条来自 HIKROBOT 型号标注的设备记录，101 条关联实体3D格式；按原始可计量CAD的型号状态，92 条通过、9 条无效、3 条没有可用实体3D。另有 4 条完整 OPT 型号代码跨供应商匹配、2 条相同几何签名的跨颜色SKU共用模型，以及 2 条精确基础型号代码但颜色未指定的OPT-RID家族预览；来源、颜色、OEM与验证级别均有标注。
- 本轮以 `hardware.json` 的 `cad_files` 和 `assets/` 实物做了 SHA-256 复核：2,353 条记录中 2,255 条有关联 CAD 文件、98 条没有；2,507 个被引用文件名对应 2,462 份不同文件内容。STEP/STP/IGES/Parasolid/SAT 等实体CAD共 2,398 个文件名、2,364 份不同内容。内容哈希去重不等于独立产品型号或几何等价型号数；型号标签仍须与实体外形、接口和修订核对。
- 清单有 2,748 条资产关联、2,616 个不同文件名；同一文件可被多个商品记录引用，因此关联数不能等同于独立模型数。
- `oracle_viewer/catalog/assets/` 有 2,620 个本地文件。HIKROBOT全目录CAD批次导入后，按CAD成员名修正了13个被旧导入器截短的家族型号记录，将47份重复提取文件迁到精确SKU记录并去重；原始ZIP保留。导入批次新增71个型号记录；处理了107个结构图纸下载项，其中35个与此前原始包SHA-256相同，已复用归档，避免重复占用约439 MiB。另新增6份逐SKU命名的机身预览，来源B-Rep保留在库中；这些预览可渲染但被标为 `cad_preview_only_files`，不参与FAI面绑定。其余72个下载包/直接CAD文件共约315 MiB，保存在 `source_archives/hikrobot/mvisionpro/all-cad/`；仅下载到PDF资料的型号没有被记为有3D。
- 其中 Shuangyi 经销商公开产品页来源有 1,404 条记录，1,403 条带实体 CAD；8 条标为停产。另从双翌官方 FLC 产品页补齐了 7 个侧部开孔面光源 IGES 模型，并通过 FreeCAD B-Rep 检查。供货现状和准确订货型号仍需供应商确认。
- 大恒图像制造商技术图纸库新增 167 个以官方 CAD 文件名区分的模型，177 个 STEP 文件关联。38 个原始 ZIP（约 324 MB）保存在 `oracle_viewer/catalog/source_archives/daheng/`，抽出的实体 CAD 保存在 `assets/`。
- GU Optics 上海公开目录新增 567 个商品记录。541 个 SKU 的商品页直链下载到 STEP，去重后为 488 个本地 STEP 文件，占用约 3.02 GB（按 SKU 合计下载数据约 3.27 GB）；颜色变体或重复导出会共用完全相同的文件。另逐一复查了 26 个失败链接：25 个返回标题为 404 的 HTML 页面，1 个返回 HTTP 200 空内容。 本轮又对这些型号探测 `.step/.stp/.STEP/.STP` 四种扩展名，未发现任何有效 CAD。目录将它们保留为“无可用 3D 文件”记录，未用相似型号替代。该站点按经销商公开目录登记，OEM/模型源头未核实，装配尺寸还需供应商确认。
- GU 器件记录保留原始中文规格字段，同时归一化光源类型、颜色、功率、电压和明确标出的有效发光尺寸，供 Viewer 检索及后续布局筛选。仅当来源明确同时给出电压和功率时才计算电流；手册列出的机身外形尺寸单独存为 `manufacturer_outer_dimensions_mm`，不与 STEP 包围盒混为一谈。待几何审计后再逐项比较。
- 已存档 GU Optics 官方《全品类手册 9.0》电子目录（PDF 第 173–176 页含主要机器视觉光源表），493 条公开 SKU 记录能对应到手册系列行；用表中颜色分组功率、电压和外形尺寸补齐缺失字段，并保留手册页码和原始行。一个型号的详情页功率与 9.0 手册相差 0.2 W，保留详情页值并标记采购前复核。其余 74 个 SKU（64 个远心平行光、4 个近红外点光源和 6 个控制器）不在这版手册对应表中，未据旧目录推断规格。另发现 661407 标题写 24V、参数字段写 5V，已标记为选型前必须向供方确认。
- Mech-Mind 官方下载中心的 7 个 V3/V3S 相机 STEP 已收录；其中 6 个由原始 CAD 中分离出机身和 FOV 几何，原始文件仍保留供预览。分离结果用于方案初排，需用机械图纸复核实体边界；Laser L V3 暂无可分离的封闭机身实体。
- iRAYPLE 日文经销商产品页为 5 个相机型号提供了型号对应的 CAD 下载入口（A5200MG000E、A5501MG20E、A7500CG20E、A7500MG20E、A7500CU75E）。前四个 GigE 下载包逐字节相同，均含已登记的同一 29×29×42 GigE STEP，证明这些产品页共用该族模型，但不能证明接口/连接器细节对每个 SKU 都相同。A7500CU75E USB3 下载包包含独立 STEP，FreeCAD B-Rep 检查通过，包围尺寸约 30.76×30.90×46.70 mm；与型号图纸的 29 mm 机身及 12.20 + 34.6 mm 轴向尺寸关系相符。原始 RAR 已保存在 `oracle_viewer/catalog/source_archives/irayple/`，没有把产品页中的相近型号代替精确型号。

## 大恒图像几何检查

用 OpenCascade 对 167 个去重 STEP 文件读取并计算 CAD 坐标系包围尺寸：167 个均能读取并生成包围尺寸。进一步用 `BRepCheck_Analyzer` 检查拓扑：105 个模型通过，62 个未通过。未通过模型原文件保留，器件记录标为不可用于 Viewer FAI 绑定或碰撞结论；不能把“文件下载成功”当成“几何可用于工程方案”。详细逐文件结果分别在 `oracle_viewer/catalog/daheng_geometry_audit.json` 和 `oracle_viewer/catalog/daheng_brep_validation.json`。

至少一个通过拓扑检查的大恒 HN-6M 镜头 STEP 已用 Viewer 的 `geometry_worker.py` 生成面网格：14 个实体、706 个面、143,714 个三角形。抽测的无效模型会被同一 worker 拒绝，符合当前 FAI 面绑定的保护规则。

## 本机 F3D 报错抽检

本机 `f3d` 是 1.2.1，构建信息显示 `OpenCASCADE module: OFF`，因此该二进制没有 STEP 读取模块；“不支持格式”不能单独证明下载文件损坏。对用户提到的文件用 FreeCAD/OpenCascade worker 抽测：`MECH-EYE-DEEP-GL-right-angle-V2504.step` 可读且 B-Rep 有效，但包围盒约 3.70×3.61×2.86 m，打开后应先检查其中是否包含大范围辅助几何；Basler 原始 family STEP 的 B-Rep 无效，已另存的 `Basler-ace-USB3-C-Mount-family-v02-shape-fixed.step` 有效；`MV-DP2240-01P-vendor-beda0c.stp` 当前仍未通过 B-Rep 检查。`MV-CS050-10GMGC-family.dxf` 是 AutoCAD 2004 DXF 二维图，应通过 CAD 软件的 DXF 导入打开，而不是作为三维实体交给 F3D。

## HIKROBOT 三维设备 CAD 批次

- 从公开 HIKROBOT 分销商文档库下载并留存 35 个轮廓仪/三维视觉产品 CAD ZIP 包（约 461 MB），包内型号标签识别出 38 个具体型号/版本记录，抽取 91 个 STEP/STP、IGES/IGS、DXF 文件（约 481 MB）。大型 OBJ 网格留在原始 ZIP 中；清单没有把 OBJ 网格伪称为可编辑实体。
- 对 56 个原始 STEP/IGES 文件使用与 Viewer 相同的 FreeCAD 读取器核验：52 个 B-Rep 有效、4 个无效；38 个型号记录中，36 个至少有一个有效的原始实体文件，另 2 个原始文件集没有有效 B-Rep。不同 CAD 内核对部分 IGES 的拓扑和包围盒判定曾出现分歧，目录采用 FreeCAD `Part.read + Shape.isValid` 作为应用侧结果；交叉读取记录不覆盖此结果。
- 其中 9 个型号的源 CAD 含有测量范围/辅助几何，导致全文件包围盒不能代表设备外壳。通过单独筛取源文件中的有效封闭实体，生成 9 个明确标为“实体外壳预览”的派生 STEP；全部再次通过 FreeCAD B-Rep 检查和 Viewer 网格生成。派生件不覆盖原始文件，也不用于计量。例：DP2470-03P V2.0 从约 198×488×1,026 mm 收敛到约 198×59×127 mm；MV-RS-Ultra-XL 从约 3,607×2,800×2,600 mm 收敛到约 108×526×82 mm。最终机身包络和安装基准仍需对照对应版本厂商图纸复核。
- `MV-DL2125-04H-R` 的原始完整 STEP 未通过 B-Rep，但由其中 25 个有效闭合实体生成了可浏览预览；`MV-RS-Pro-XL` 原始文件仍未获得有效实体预览。此批器件 `layout_adapter_ready` 保持关闭，避免把扫描范围或未校核机身尺寸直接送进物理布局。

## 未纳入范围与剩余缺口

- 本轮从已下载的 Shuangyi 通用环形光源系列 STEP 中找回 4 个本地遗漏关联：RI15000-W、RI15060-W、RI12030-W、RI7060-W 的 STEP 内部 FILE_NAME 与 PRODUCT 两处均分别写明完整 OPT 型号 RI15000-W、RI15060-W、RI12030-W、RI7060-W，4 个文件均通过 B-Rep 检查。原始 Shuangyi 下载包的归档文件名为 PL-RI 系列，现按 STEP 内部完整型号代码跨来源关联到 OPT 记录；OEM 等价性和图纸修订仍需供应商确认。可做型号代码对应的预览，最终机械安装/碰撞审批前须复核供应商图纸。另有 RI5000-B、RI7000-B 两个跨颜色机身候选，均未标为精确 SKU。
- 当前 2,353 条器件记录中，98 条没有任何 CAD 关联：OPT 35、Hypersen 31、GU Optics 24、Opto Engineering 6、HIKROBOT 1、Shuangyi 1。另有两条OPT记录可显示来源型号代码分别为RID100/RID150、与目标OPT-RID100-R/OPT-RID150-B基础型号代码相符的Shuangyi IGES家族预览；颜色、OEM等价性及目标修订未确认，不能当作精确SKU模型或最终净空依据。RID100与RID150来源分别是`SHY-LIGHT-7ad3cf2f0271`和`SHY-LIGHT-f8ceac616c5b`，审计见`oracle_viewer/catalog/cross_vendor_color_family_model_audit.json`。RI5000-B、RI7000-B对应STEP内部型号分别为RI5000-W、RI7000-W，仍仅为跨颜色机械家族候选，精确SKU几何未确认。OPT的RI15000-W、RI15060-W、RI12030-W、RI7060-W现有STEP内完整型号代码证据，但文件取自Shuangyi产品列表；OEM与修订等价性尚待供应商确认。GU-660706-W复用GU-660706-B、GU-661200-B复用GU-661200-W的本地STEP；每组所有已有颜色款的有效B-Rep都通过OpenCascade完整几何签名比对，实体/面/边/顶点及坐标（1e-5 mm精度）一致，记录在`oracle_viewer/catalog/color_variant_geometry_audit.json`。这两个绑定只证明几何相同，不证明目标颜色外观、OEM或修订相同；目标SKU自身的下载链接仍不可用。其余缺口保持未匹配状态，不借用近似型号模型。
- OPT 官方下载中心列出 CAD/Technical Drawing 附件；本轮实际请求 OPT-HP08-110-F6.5 图纸时，下载端点明确要求登录，因此未收录文件。 Opto Engineering 官方产品页列出 TC23024 等型号的 STEP 布局文件；公开下载链接打开的是索取资料表，要求填写姓名、邮箱、公司、电话、国家并勾选隐私同意，未提交表单、未收录模型。
- Hypersen 的 HPS-CFL 探头系列和 HPS-HSC/HSL 高速相机在 TraceParts 有型号明确的 CAD 条目，其中 HPS-CFL007、HPS-CFL036 和五款 HPS-HSC/HSL 相机页面均列出 STEP 格式。网页可读取，但本次没有取得可归档的 STEP 文件，也未把网页预览当作本地 CAD 入库；这些型号仍按未下载处理。未用规格尺寸重建替代模型。另从公开 Longvista 归档下载到 HPS-CF1000/2000/4000 控制器的 3 个原厂标签 STEP；三者通过 OpenCascade B-Rep 有效性检查。STEP AABB 分别约为 395.05×171.28×283.59、173.27×124.35×260.43、285.05×133.10×264.59 mm；同包 2D 图标注的三向尺寸分别为 394×278×132、259×158.4×123.7、284×259×132 mm（轴向对应关系未假设）。CAD 包络更大，库中按 CAD 包络保守记录，安装前须核对连接器/附件和图纸版本。原始 RAR 归档保存在 `oracle_viewer/catalog/source_archives/hypersen/`。
- Shuangyi 有一条 OT-SR022X210-CO-5M-DH 的 3D 压缩包链接仍返回 404。其公开 2D PDF 已保存；没有用相近型号替代其 3D 模型。
- HIKROBOT 的 MV-DP3020-01H：官方资料目前给出的是外形尺寸图；型号页经销商挂出的 CAD 压缩包实际命名为 MV-DP3020-01P，后缀不同，未作为 01H 的模型使用。另一经销商的“结构图纸”链接解析到 Synology 分享中的 `結構圖紙_MV-DP3020-01H.zip`，但其 Download 按钮处于登录禁用状态；确认文件名对应后仍未取回归档内容，故保持未关联实体 CAD。
- 本轮对 Basler ace USB3 C-Mount 家族 STEP 做了独立 ShapeFix 副本：原始供应商文件不改写；OpenCascade 修复后 B-Rep 有效，FreeCAD Viewer `geometry_worker.py` 生成 1,315 面、389,310 三角形网格。包围盒、体积和表面积在数值精度内不变。该模型仍是家族外壳，不代表具体 SKU 的细节，也不能作为计量模型。HIK DP2240 模型尝试相同修复后仍未通过 B-Rep 检查，未作修复登记。
- FreeCAD/Viewer 同源历史审计（rev .36）覆盖 2,215 个 STEP/STP/IGES 文件：2,057 个通过 B-Rep，156 个未通过，2 个读取失败；rev .37 的 3 个 Hypersen 控制器另行通过 B-Rep 检查。这是旧批次快照，不代表之后新增文件的状态。当前目录关联 2,398 个 STEP/STP/IGES/Parasolid/SAT 实体CAD文件名，按 SHA-256 去重为 2,364 份不同文件内容。rev .48 的全目录审计覆盖 2,377 个 STEP/STP/IGES文件：2,190 个原始文件通过、170 个 B-Rep 无效、2 个读取错误、6 个超过单文件35秒限时而保持未确认；另有9个经保形/实体筛选检查的派生STEP有效。限时用于避免复杂单文件阻塞整批任务。HIKROBOT现有165个实体CAD文件中，133个原始文件通过、18个无效、6个未确认、8个派生预览/修复文件通过；型号级原始计量CAD状态为92个有效、9个无效、3个不可用。新增机身预览只供布局观察，清单用 `cad_preview_only_files` 明确禁止FAI面绑定和计量用途。IGES可直接由Viewer几何worker读入并生成网格；`geometry_worker.py` 现按实际扩展名标注 IGS/IGES/STEP。对于 127 条 GU SKU，B-Rep 有效但 CAD 包围尺寸与手册外形尺寸不一致；库里保留两组尺寸，自动净空筛选使用逐轴较大的包络并要求定版前复核。
- GU 这批 488 个不同 STEP 文件中，482 个通过 B-Rep、4 个未通过、2 个读取失败；按商品 SKU 统计则为 535 个有效、4 个无效、2 个读取失败、26 个源站没有返回 STEP。67 个背光/平行背光 SKU 已具备有效 CAD、有效发光尺寸、电气参数和可用于布局初筛的保守机身包络；最终安装方向、完整结构和光学效果仍须工程核验。
- 所有目录尺寸均来自 CAD 包围盒或来源规格，仅供布局预筛；厂商修订、安装孔、线缆空间和实际可供货状态仍须核实。

## 来源

- 大恒图像官方技术图纸下载页：<https://en.daheng-imaging.com/list-61-1.html>
- Mech-Mind 官方相机资料下载页：<https://downloads.mech-mind.com.cn/?tab=tab-camera-materials>
- GU Optics 上海机器视觉光源公开目录：<https://www.gu-optics.com/Machine-Vision-Light-Sources.html>
- GU Optics 官方《全品类手册 9.0》本地归档：`oracle_viewer/catalog/source_documents/gu-optics/gu-optics-electronic-catalog-9.0.pdf`。
- Shuangyi 大恒相机公开产品资料示例：<https://www.shuangyi-tech.com/product_287.html>
- Shuangyi YL-RID 系列产品页及RID100/RID150 3D链接：<https://www.shuangyi-tech.com/product_153.html>；OPT-RID100-R及OPT-RID150-B型号页：<https://en.optmv.com/content/details17_3184.html>、<https://en.optmv.com/content/details17_3191.html>。本地关联标为基础型号家族预览，不视作OPT原厂或颜色SKU确认。
- Shuangyi 海康相机公开产品资料示例：<https://www.shuangyi-tech.com/product_256.html>
- Shuangyi FLC 侧部开孔面光源官方 2D/3D 下载页：<https://www.shuangyi-tech.com/product_171.html>
- iRAYPLE A7500CU75E 型号页及其经销商 CAD 下载页：<https://linx.jp/irayple/a7500cu75e/>；<https://linx.jp/download/A7500CU75E_cad/>。
- Hypersen CF 控制器外形与 3D 文件公开下载页：<https://www.longvista.ch/industrial-automation/chromatic-confocal-sensor/HPS-CFL043/>；原始归档：`oracle_viewer/catalog/source_archives/hypersen/HPS-CF-Series-3D-drawing_System-Diagram.rar`。
- Hypersen TraceParts 型号CAD目录：HPS-CFL007 <https://www.traceparts.com/en/product/hypersen-chromatic-confocal-sensors-hpscfl007?CatalogPath=TRACEPARTS%3ATP09013002&Product=90-26092023-020349>；HPS-CFL036 <https://www.traceparts.com/en/product/hypersen-chromatic-confocal-sensors-hpscfl036?CatalogPath=HYPERSEN_974816128%3AHYPERSEN.020&Product=90-26092023-020367>；高速相机系列 <https://www.traceparts.com/en/search/hypersen-high-speed-industrial-camera?CatalogPath=HYPERSEN_974816128%3AHYPERSEN.030>。目录列出 STEP 格式；本地文件尚未取得，故未计为已下载 CAD。
- HIKROBOT 分销商公开资料与 CAD 下载库：<https://mvisionpro.com/hikrobot/documents/>；本轮逐项下载可用的结构图纸包并按型号登记，原始包保存在 `oracle_viewer/catalog/source_archives/hikrobot/mvisionpro/`，不以邻近型号替代。
- Opto Engineering TC23024 官方产品页（列有 STEP 布局文件）：<https://www.opto-e.cn/产品/tc-series/tc23024>。
