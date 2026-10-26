"""Bounded source parsing and calculated draft for explicitly defined MSOP circles.

This is a transparent rule-based preliminary proposal, not a claim of learned
engineering competence or validated hardware. Unrecognized sources fall through.
"""
from __future__ import annotations

import re
from .optical_calculations import OpticalInputs


def circle_requirements(query: str, context: str) -> dict | None:
    requested = re.search(r'(?<![A-Za-z0-9_])FAI\s*0*(\d+)(?!\d)', query, re.I)
    if not requested:
        return None
    number = int(requested.group(1))
    blocks = re.split(r'(?=\[Source\s+\d+:)', context)
    matches = []
    for block in blocks:
        source = re.match(r'\[Source\s+\d+: ([^\]]+)\]', block)
        heading = re.search(rf'MSOP\s*:\s*FAI\s+{number}(?!\d)\s*(?:\(SPC\s+([^)]*)\))?\s+([-+]?\d+(?:\.\d+)?)\s+(\+\d+(?:\.\d+)?)/(-\d+(?:\.\d+)?)', block, re.I)
        feature = re.search(r'Measure\s+1\s+vision\s+circle\s+([A-Za-z0-9_]+)', block, re.I)
        axis = re.search(r'Dimension\s+([XY])\s+Value', block, re.I)
        alignment = re.search(r'Dimension System[ \t]+([A-Z0-9_]+)', block)
        program = re.search(r'Program[ \t]+([A-Z0-9_]+)', block)
        if not all((source, heading, feature, axis, alignment, program)):
            continue
        matches.append(dict(number=number, spc=heading.group(1), nominal=heading.group(2),
                            upper=heading.group(3), lower=heading.group(4), feature=feature.group(1),
                            axis=axis.group(1).upper(), alignment=alignment.group(1),
                            program=program.group(1), source=source.group(0),
                            filename=source.group(1).rsplit(', page ', 1)[0]))
    if not matches or any(item != matches[0] for item in matches[1:]):
        return None
    item = matches[0]
    origins = []
    for block in blocks:
        source = re.match(r'\[Source\s+\d+: ([^\]]+)\]', block)
        if not source or source.group(1).rsplit(', page ', 1)[0] != item['filename']:
            continue
        if not re.search(rf'MSOP\s*:\s*{re.escape(item["alignment"])}\b', block, re.I):
            continue
        program = re.search(r'Program[ \t]+([A-Z0-9_]+)', block)
        if not program or program.group(1) != item['program']:
            continue
        origin = re.search(rf'Set\s+([A-Z0-9_]+)\s+as\s+{item["axis"]}\s+Zero', block, re.I)
        if origin:
            origins.append((origin.group(1), source.group(0)))
    if not origins or len({origin for origin, _ in origins}) != 1:
        return None
    item['origin'], item['origin_source'] = origins[0]
    item['unit_mm_confirmed'] = any(
        item['filename'] in block and re.search(r'X,\s*Y,\s*Z\s*\(mm\)', block, re.I)
        for block in blocks)
    return item


def source_checked_circle_plan(query: str, context: str) -> str | None:
    if re.search(r'JSON|只输出|只提取', query, re.I):
        return None
    if re.search(r'(?:FOV|视场|视野|像元|pixels|px)[^\n]{0,12}\d', query, re.I):
        return None  # Do not replace explicitly supplied design inputs.
    item = circle_requirements(query, context)
    if item is None:
        return None
    # Use explicitly stated *example design inputs*, never a source coordinate.
    inputs = OpticalInputs(20.0, 2448, 3.45)
    values = inputs.calculate()
    width = values['sensor_width_mm']
    sampling = values['sampling_um_per_px']
    magnification = values['magnification']
    unit = 'mm' if item['unit_mm_confirmed'] else '（单位需确认；以下设计暂按mm）'
    budget = (float(item['upper']) - float(item['lower'])) * 1000 * 0.1
    return f'''这是由原文明确字段和计算规则生成的**初步规格方案**，尚未完成实物打样、型号选定或计量验收。

**测量要求**

- FAI {item['number']} / SPC {item['spc'] or '未标注'}：特征 `{item['feature']}`，输出 {item['axis']} 向坐标；名义值 {item['nominal']} {unit}，公差 {item['upper']}/{item['lower']}。名义值不是孔径，也不是实测值；当前引用的方法页未给出实测结果。{item['source']}
- 程序 `{item['program']}`，尺寸坐标系 `{item['alignment']}`；最终 {item['axis']} 零点为 `{item['origin']}`，应按原文对应的基准构建方法建立。不能沿用中间坐标系的零点或以夹具名义位置代替每件实际基准。{item['origin_source']}

**相机、镜头与光源几何关系**

以目标孔口面能正视为设计假设：相机位于被测面上方，光轴**平行于孔口面的法向、垂直于孔口面**；物方远心镜头与相机共轴。不能仅依据零件图上的圆就断言它是无遮挡通孔。

若实物确认指定孔边可透射，背光放在工件下方、镜头对面，治具对光路开窗；比较均匀与准直背光。若盲孔、螺纹或夹具遮挡透射光路，改为上方同轴/漫射光，并用可独立调光的低角度环光作对比，确认所提边缘是指定孔口而非螺纹或倒角。可暂按150 mm预留工作距空间，这是机械预留假设；实际工作距必须按最终镜头规范、景深与干涉检查确定。

局部图像只得到局部圆心。报告全局 {item['axis']} 坐标还需要标定后的计量平台和实际工件基准配准；若二维视觉无法复现基准平面/侧面，保留接触测头或3D测量通道。标定板自身不能代替工件基准建立。

**示例设计输入与代码核算**

以下是用于初选的假设，不是图纸指定的硬件：横向FOV=20 mm，横向2448 px，像元3.45 μm。实物孔大小与安装空间不符合时须重新选输入。

- 物方采样 = 20 / 2448 × 1000 = **{sampling:.4f} μm/px**。
- 传感器有效宽度 = 2448 × 3.45 / 1000 = **{width:.4f} mm**。
- 所需倍率 = {width:.4f} / 20 = **{magnification:.4f} 倍**。

采样与亚像素输出位数不等于测量准确度。作为待客户确认的开发目标，可暂取总公差带的10%，即绝对误差上界≤{budget:.2f} μm；这不是MSOP验收规定，也不是已达成结果。预算必须覆盖光学校准、边缘定位、平台/坐标转换、基准构建/装夹与热漂移，并由实际测试确认。

**候选规格 BOM**

| 部件 | 数量 | 规格与用途 | 状态 |
|---|---:|---|---|
| 单色全局快门工业相机 | 1台 | 横向2448 px、像元3.45 μm为示例输入；接口与触发匹配工控机 | 待型号核验 |
| 物方远心镜头 | 1支 | 约{magnification:.4f}倍起算，覆盖实际传感器；工作距、景深、畸变、MTF核对实物 | 待目录选型 |
| 背光或上方照明试验组件 | 1组 | 透射通畅时背光；受遮挡时同轴/漫射并比较低角度环光 | 打样选择，非全部同时采购 |
| 多路光源控制器 | 1台 | 独立调光、触发同步、电流稳定 | 待配套 |
| 校准计量平台与配准软件 | 1套 | 覆盖所有基准与目标位置；将局部圆心转换到实际工件坐标系 | 可复用已合格平台 |
| 基准测头/3D测量及标定组件 | 1套 | 二维视觉无法复现实际基准时必须配置 | 待基准可测性确认 |
| 专用治具、微调支架、遮光组件 | 1套 | 光路避让、防变形、稳定光轴与重复装夹 | 待机械设计 |
| 可追溯标定件与参考件 | 1套 | 畸变、尺度、平台/视觉配准与参考对比 | 待计量选型 |
| 工控机与视觉测量软件 | 1套 | 标定、指定物理边拟合、坐标转换、SPC与图像留存 | 待接口确认 |
| 电源、触发线、数据线与接地 | 1套 | 匹配所选设备和线长 | 待配套 |

**验证与定版**

先确认孔口物理边、背面遮挡和基准可测性，再做照明对比。验证尺度/畸变标定、坐标转换、重复采集、重复装夹、不同位置/高度/温度，以及与可追溯参考测量的偏差；按约定方案评估GRR与总误差。未取得有效基准或目标边时输出“测量无效”，不能用默认值判合格。

定版需要实物正反面照片、孔口/螺纹/倒角尺寸、CAD与夹具空间、可复用平台资料、节拍、预算及客户认可的计量指标。当前不编造具体型号、价格、抽样标准或AQL。
'''
