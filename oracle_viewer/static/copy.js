// Presentation vocabulary only. Source identifiers and model decisions remain unchanged.
const terms={
 oblique_ring:'斜射环光',move_optics:'移动相机或测头',move_part:'移动工件',fixed:'固定安装',specification_missing:'规格待补充',required_emitting_area_only_no_verified_supplier_body:'仅确定发光面积，型号待选',unverified_trial:'待试拍验证','2d_vision':'二维视觉',hybrid_metrology:'组合测量',method_review_required:'需核对测量方法',proposed_unvalidated:'已编制，待验证',integration:'配套部件',mono:'黑白',global:'全局快门',rolling:'卷帘快门',red:'红光',blue:'蓝光',white:'白光',object:'物方远心',front_ring:'正面环光',moving_light_envelope_installation_unverified:'共用移动光源，安装待验证',camera:'相机',lens:'镜头',light:'光源',controller:'光源控制器',confocal_sensor:'共焦传感器',surface_sensor:'轮廓传感器',
 pending:'待验证',pending_review:'位置待复核',reviewed:'位置已复核',missing:'未标记',located:'已标记',blocked:'需补充条件',conditional:'具备初步方案条件',
 conditional_candidate:'候选型号，待验证',candidate_available:'已有候选方案',local_model_candidate:'自动标记',geometry_candidate:'几何匹配',manual:'人工标记',
 '本地整机AOI复用方案':'设备共用方案','并行分站与国产照明试验方案':'并行工站方案','国产照明优先与分站复用方案':'国产照明优先方案','仅所选FAI、当前有限硬件和假设视向的候选光学头数量；非整机工站最少证明':'以上数量按当前测量范围、设备库和观察方向计算，尚未证明工站数已达到最少。','2d_visual_circle':'二维视觉测圆',needs_method_adapter:'需确认测量方式','2d':'二维视觉',confocal_height:'共焦高度测量','3d_surface':'三维表面测量',
 transmitted_backlight:'透射背光',oblique_panel_reflection:'面光斜射反射照明',oblique_bar_reflection:'条光斜射反射照明',backlight:'背光',coaxial:'同轴照明',coaxial_reflected:'同轴反射照明',diffuse:'漫射照明',diffuse_reflected:'漫射反射照明',low_angle_ring:'低角度环光',ring:'环形光',reflected_ring:'反射环光',
 edge:'被测边缘',datum:'基准与坐标系',fit:'采点与拟合',optics:'光学与照明',mechanics:'安装与干涉',calibration:'标定',uncertainty:'重复性与测量误差',cycle:'节拍与判定',
 low:'低',medium:'中',high:'高',
 price:'价格',availability:'供货情况',vendor_CAD:'厂家三维模型',body_dimensions:'外形尺寸',head_body_dimensions:'测头外形尺寸',installed_accuracy:'安装后的测量精度',compatible_controller_model:'配套控制器型号',connector_clearance:'接头安装空间',controller_channels:'控制器通道数',emission_geometry:'出光结构',light_distance:'照明距离',DOF_test_conditions:'景深测试条件',PDF_requires_vendor_registration:'规格书需向厂家申请',local_original_PDF_blocked_by_vendor_WAF:'厂家网站限制下载规格书',output_voltage_limits:'输出电压范围',per_channel_continuous_current:'单通道连续电流',
 '确认指定物理边、倒角/螺纹、孔口及透射光路':'确认实际测量边缘，区分倒角、螺纹和孔口；需要背光时检查透射光路',
 '按 MSOP 实际工件采点建立基准与最终坐标系；不能用夹具名义零点':'按MSOP采集工件基准点并建立测量坐标系，夹具位置不能代替工件实际基准',
 '复现采点密度、拟合与 Outlier 参数语义；与海克斯康逐项对比':'按MSOP复现采点密度、拟合和离群点过滤（Outlier）设置，并与海克斯康结果逐项比较',
 '核验镜头像圈、ROI 步进、MTF/波长、工作距、景深、照明与遮挡':'核对镜头像圈和图像区域（ROI）设置，验证清晰度（MTF）、波长、工作距离、景深、照明和遮挡',
 '完整装配/夹具碰撞、装夹变形、线缆与光路检查':'补全设备和夹具模型，检查干涉、装夹变形、线缆布置和光路',
 '可追溯尺度/畸变、平台与视觉坐标转换、温度补偿':'使用可追溯标准件标定尺度和畸变，核对平台与相机坐标转换及温度补偿',
 '重复采集/重复装夹/位置高度温度变化、参考偏差、GRR 与不确定度':'验证重复拍摄、重复装夹和不同位置、高度、温度下的误差，并评估参考偏差、GRR与测量不确定度',
 '验证曝光/触发同步、节拍、无效测量与判定规则':'确认相机与光源触发同步、测量节拍满足要求，并验证测量失败和超差时的处理规则',
 '尚未完成方法等效、基准配准与计量验证':'需核对AOI方法与MSOP是否一致，并完成基准标定和计量验证',
 '标记位置未复核':'请对照MSOP确认标记位置',
 'CAD 为显示几何，非已验收计量几何':'当前CAD用于查看位置，计量使用前还需验证',
 '原方法需要深度/接触或其他通道；单幅二维采集不能直接输出该测量结果':'该项需要高度、三维或接触测量，请确认相应设备和测量方式',
 '按 MSOP 明确标题收录；范围标题已展开，重复页保留。不代表子测量项全部核对完成。':'清单按MSOP中的FAI编号整理。请逐项核对测量内容；同一FAI可能包含多个测量点或子项。',
 '本地模型从扩展规格库筛选合法光学组合':'正在选择相机、镜头和光源',
 '准备标记点和原测量方法':'正在整理测量位置和MSOP要求',
 '已接收全部已标记FAI，等待后台规划':'任务已提交，等待开始',
 '校核设备干涉、工件碰撞和安装间隙':'正在检查设备干涉和安装间隙',
 '复用已完成且来源一致的本地模型方法方案':'正在复用已核对来源的测量方案',
 '本地模型合并采集布局；需覆盖每个标记点':'正在合并拍摄位置，检查测量点是否遗漏',
 '规划完成：3套整机方案、设备复用、工站与BOM已生成；计量和安装仍待验证':'方案已生成，可比较工站布置和设备清单。安装与测量效果还需验证。',
 '等待本地模型总装规划':'等待整机布置',
 '本地模型分配多相机、合并照明和总装实例':'正在分配相机、共用光源并整理设备数量',
 '本批测量通道路由已校验':'本批测量方案已整理','等待本地模型生成动作说明':'等待生成操作与拍摄说明','本地模型解释动作顺序与路径边界':'正在整理操作顺序和运动要求','制作可编辑PPT：设备、动作、逐组位置与共用光源':'正在制作PPT，整理设备、动作和共用光源说明','PPT已生成：可下载当前总方案动作与路径说明':'PPT已生成，可以下载','总装规划完成 · 安装与光学效果待验证':'整机布置已生成，安装和成像效果待验证'
};
export function wording(value){if(value===null||value===undefined)return '—';const text=String(value);if(terms[text])return terms[text];
 const prefixes={'本地模型修正总装布局 · ':'正在调整整机布置 · 第','本地模型整合整机方案 · ':'正在编制整机方案 · ','本地模型整机输出校验 · ':'正在检查整机方案 · ','本地模型规划测量通道 · ':'正在编制测量方案 · ','整机输出未接收，保留原始输出并重试 · ':'方案格式检查未通过，正在重试 · ','整机方案已生成并校验 · ':'整机方案已生成 · '};
 for(const [prefix,label]of Object.entries(prefixes)){if(text.startsWith(prefix))return label+text.slice(prefix.length).replace(/\bshared\b/g,'设备共用方案').replace(/\bthroughput\b/g,'并行工站方案').replace(/\bdomestic\b/g,'国产照明优先方案');}
 return text;}
export function errorMessage(value){const text=String(value||'');
 if(/FAI ownership mismatch|Every source FAI needs exactly one result-owning setup/.test(text))return '测量项归属核对未通过：存在漏项或重复归属。系统会交由本地模型修正；持续失败时可查看具体编号。';
 if(/Assembly validation failed|Every exposure|rotation for ALL|Different viewing sides|Home camera interference/i.test(text))return '整机布置未通过检查。请重新安排相机与共用光源；系统会核对拍摄位置分配、旋转方式和设备干涉。';
 if(/Server interrupted/i.test(text))return '方案编制因服务重启中断，请重新运行。';
 if(/context budget|context length|exceeds context|bounded stage budget|上下文/i.test(text))return '本次资料量超过推理容量，方案未生成。请保留任务记录，由维护人员检查分批处理设置后重试。';
 if(/method batch validation|routes must cover|Model output truncated|JSONDecodeError/i.test(text))return '本批方案未通过格式检查。已完成的结果会保留，请重试；持续失败时联系维护人员检查运行记录。';
 if(/Workflow already running/i.test(text))return '正在编制方案，请等待当前任务结束。';
 if(/Complete current workflow required/i.test(text))return '请先完成当前方案编制，再查看整机布置。';
 if(/Failed to fetch|NetworkError|fetch failed/i.test(text))return '连接中断。请检查网络和服务状态，再刷新页面。';
 if(/502|Bad Gateway/i.test(text))return '服务暂时无法连接。请稍后刷新；持续出现时联系维护人员检查服务。';
 if(/503|connection refused|model.*not.*loaded/i.test(text))return '推理服务尚未就绪。请联系维护人员检查本地服务后重试。';
 return wording(text);
}
const parameters={active_mm:'发光面积（mm）',angle_half_deg:'半角（°）',body_excludes:'外形尺寸未包含',body_height_mm:'高度（mm）',body_inner_mm:'内径（mm）',body_mm:'外形尺寸（mm）',body_outer_mm:'外径（mm）',cad_axis_aligned_envelope_mm:'CAD坐标系包围尺寸（X × Y × Z，mm）',channels:'通道数',collimation_film:'准直膜',color:'光色',connector:'接头',current_a:'电流（A）',distortion_max_pct:'最大畸变（%）',dof_mm:'景深（mm）',dof_reference_pixel_um:'景深参考像元（μm）',emission_deg:'出光角度（°）',emitting_diameter_mm:'发光直径（mm）',fps:'帧率（fps）',front_diameter_mm:'前端直径（mm）',full_height_px:'全幅高度（px）',full_width_px:'全幅宽度（px）',height_px:'图像高度（px）',image_circle_mm:'像圈直径（mm）',included_cable:'配套线缆',interface:'接口',laser_class:'激光等级',length_mm:'长度（mm）',light_distance_mm:'照明距离（mm）',linearity_pct_mr:'线性度（量程%）',magnification:'倍率',manufacturer_parameters:'厂家参数',max_duty_pct:'最大占空比（%）',max_pulse_ms:'最大脉冲时长（ms）',measuring_range_half_mm:'半量程（mm）',mode:'工作模式',mount:'安装接口',mtf30_lp_mm:'MTF30（lp/mm）',nonlinear_error_um:'非线性误差（μm）',output_current_a:'输出电流（A）',output_power_w:'输出功率（W）',output_voltage_v:'输出电压（V）',pixel_um:'像元尺寸（μm）',poe:'PoE供电',points_per_profile:'每条轮廓点数',power_description:'功率说明',power_supply_description:'供电说明',power_w:'功率（W）',profile_rate_hz:'轮廓频率（Hz）',rating_conditions:'额定条件',rayleigh_resolution_um:'瑞利分辨率（μm）',rayleigh_wavelength_nm:'参考波长（nm）',recommended_controller:'推荐控制器',reference_distance_mm:'参考距离（mm）',repeatability_conditions:'重复性测试条件',repeatability_um:'重复性（μm）',roi_max_rate_hz:'ROI最大频率（Hz）',sdk:'开发接口',sensor:'传感器',shutter:'快门',spot_um:'光斑（μm）',supply_voltage_v:'供电电压（V）',telecentric:'远心类型',telecentricity_max_deg:'最大远心度（°）',total_current_a:'总电流（A）',type:'类型',voltage_v:'电压（V）',wavelength_nm:'波长（nm）',weight_kg:'重量（kg）',width_px:'图像宽度（px）',working_distance_mm:'工作距离（mm）',working_f_number:'工作光圈',x_range_mm:'X向范围（mm）',z_range_mm:'Z向范围（mm）',z_resolution_um:'Z向分辨率（μm）'};
export const parameterLabel=key=>parameters[key]||wording(key);
export const parameterValue=value=>value===true?'是':value===false?'否':value===null?'待确认':Array.isArray(value)?value.join(' × '):typeof value==='object'?JSON.stringify(value):wording(value);
// Translate internal tokens in planning commentary, never in source documents or exports.
export function describe(value){let text=wording(value);const names={selected_catalog:'已选设备规格库',source_fai_groups:'全部FAI测量方式分组',tasks:'测量区域与要求',proposed_unvalidated:'待工程验证',selected_head_ids:'选中的拍摄布置',alternate_head_ids:'照明对比布置',selected_fai:'所选FAI',method_routes:'各FAI测量方案',transmitted_backlight:'透射背光',oblique_ring:'斜射环光',unverified_trial:'待试拍验证',conditional:'待验证',move_optics:'移动相机或测头',move_part:'移动工件',fixed:'固定安装','3d_surface':'三维表面测量',confocal_height:'共焦高度测量',光学头:'采集单元',工具推荐:'规格和布置检查',候选:'备选',缺口:'待补条件'};for(const [key,label]of Object.entries(names)){text=text.replaceAll(key,label);}return text;}
