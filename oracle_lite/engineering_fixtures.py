"""Fictional engineering-method supervision with independently computed labels.

No corpus facts, manufacturer part numbers, or validated production claims.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build_cases(directory: Path, *, seed: int, count: int) -> list[dict]:
    from PIL import Image, ImageDraw, ImageFont
    rng = random.Random(seed)
    directory.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('DejaVuSans.ttf', 20)
    rows = []
    for index in range(count):
        family = index % 8
        pixels = rng.choice([2048, 2448, 2592, 3072, 4096])
        fov = rng.choice([16, 20, 24, 30, 36, 40])
        pitch = rng.choice([2.4, 3.45, 4.5])
        sampling = fov * 1000 / pixels
        sensor = pixels * pitch / 1000
        magnification = sensor / fov
        nominal = rng.randint(3000, 8000) / 100
        measured = nominal + rng.choice([-0.03, 0.02, 0.04])
        identity = f'FIX-{seed}-{index}'
        source = identity + '.txt'
        images = []
        expected = None
        prefix = f'FICTIONAL METHOD EXERCISE {identity}; not production evidence.\n'
        if family == 0:
            context = prefix + f'Design inputs: horizontal FOV={fov} mm; horizontal pixels={pixels}; pixel pitch={pitch} um. Nominal Y coordinate={nominal:.2f} mm. No accuracy test exists.'
            question = '计算横向采样、传感器有效宽度和镜头倍率。名义Y坐标能否作视场？能否宣称测量准确度？只输出JSON：sampling_um_per_px、sensor_width_mm、magnification均保留4位小数；coordinate_is_fov与accuracy_verified为布尔值。'
            expected = dict(sampling_um_per_px=round(sampling, 4), sensor_width_mm=round(sensor, 4),
                            magnification=round(magnification, 4), coordinate_is_fov=False, accuracy_verified=False)
            answer = json.dumps(expected, ensure_ascii=False)
        elif family == 1:
            old = f'ABC_{rng.randrange(100,999)}'
            final = f'ABD_{rng.randrange(100,999)}'
            context = prefix + f'Program UP. Datum A is a measured plane. Datum B is the middle line of left/right measured lines. Datum D is the middle line of top/bottom measured lines. Intermediate alignment {old}: BC_ORIGN sets Y zero. Final alignment {final}: BD_ORIGN, the intersection of B and D, replaces Y zero. Feature circle C1 uses {final}, output Y coordinate. Nominal={nominal:.2f} mm; tolerance=+0.12/-0.12 mm; actual={measured:.2f} mm. Program DOWN uses OTHER_ORIGN and is out of scope.'
            question = '只输出JSON，提取UP程序目标圆的alignment、y_origin、nominal_mm、actual_mm、tolerance_mm；名义值和实际值保持2位小数字符串；说明local_circle_alone_sufficient布尔值。'
            expected = dict(alignment=final, y_origin='BD_ORIGN', nominal_mm=f'{nominal:.2f}', actual_mm=f'{measured:.2f}',
                            tolerance_mm='+0.12/-0.12', local_circle_alone_sufficient=False)
            answer = json.dumps(expected, ensure_ascii=False)
        elif family in [2, 6]:
            through = family == 2
            condition = 'The specified edge is a clear through-hole silhouette; the fixture has an open window below it.' if through else 'The specified edge is the top rim of a blind threaded hole; the fixture blocks the lower optical path.'
            context = prefix + f'Goal: report circle center Y in final alignment ABD. Nominal={nominal:.2f} mm, tolerance=+0.12/-0.12 mm. A measured plane, B middle line and D middle line establish the final origin BD_ORIGN. {condition} No hardware is installed. Design-assumption inputs: horizontal FOV {fov} mm, {pixels} horizontal pixels, {pitch} um pixel pitch. Working distance and cycle time unknown. No vendor catalog or sampling standard is provided.'
            question = '请出具初步光学视觉方案，含几何关系、基准链、计算、带数量的规格BOM和验证计划。允许工程建议；区分原文要求与设计假设，不编造型号或标准。'
            lighting = ('光源置于工件下方、相机对面，透过治具开窗形成待测孔的轮廓；先比较均匀背光与准直背光，确认未选到倒角或螺纹边。' if through else
                        '下方被遮挡且为盲螺纹孔，不采用透射背光作本特征量产方案。上方同轴/漫射光与低角度环光分路打样，依据孔口边缘稳定性和参考偏差选型；不能把螺纹边当成指定孔口。')
            answer = (f'来源要求：圆心在ABD坐标系的Y名义值{nominal:.2f} mm，公差±0.12 mm；最终Y零点是B与D交点BD_ORIGN。名义值不是实测值、孔径或视场。[Source 1: {source}]\n'
                      f'初步设计：相机和物方远心镜头位于目标上方，光轴与目标孔口面法向对齐，治具避让镜头和光路。{lighting} 工作距须按最终镜头规格及机械空间确定，不能凭空定版。\n'
                      f'设计输入为横向FOV {fov} mm、横向{pixels} px、像元{pitch} μm；采样={fov}/{pixels}={sampling:.4f} μm/px；有效宽度={sensor:.4f} mm；倍率={magnification:.4f}倍。这些是设计假设，采样不是测量准确度。\n'
                      '基准链：保留实际A平面和B/D中线测量，将标定后的圆心转换到ABD；局部孔图不能独立建立全局Y。可复用合格计量平台，否则必须增加校准XY平台及接触/3D基准测量通道。\n'
                      '候选BOM：单色全局快门相机1台（像元/分辨率匹配上述计算）；物方远心镜头1支（倍率、像场、工作距、景深待核对）；上述照明试验组件1组、多路触发控制器1台；计量平台1套；基准测头及配准模块1套（无法视觉复现实际基准时所需）；开窗定位治具1套；可追溯标定件1套；支架遮光组件1套；工控机与测量软件1套；电源、触发线和数据线1套。均为规格候选，未核验具体型号/价格。\n'
                      '验证：确认指定物理边，做灯光对比、畸变与坐标转换标定、参考量具对比、重复装夹、不同位置/高度/温度测试。准确度与误差预算、GRR门槛需约定并实测；不能只凭像素或亚像素拟合判定达标。缺少实物图、空间、节拍及平台资料，先据此打样，不宣称量产验收通过。')
        elif family == 3:
            label = rng.choice(['CIR_M2', 'CIR_K7', 'CIR_P4'])
            neighbor = rng.choice(['CIR_J8', 'CIR_L3', 'CIR_R9'])
            diameter = rng.choice([2.8, 3.2, 4.5])
            image = Image.new('RGB', (800, 500), 'white')
            draw = ImageDraw.Draw(image)
            draw.text((15, 10), 'FICTIONAL DRAWING: dimension labels, not measured results', font=font, fill='black')
            draw.rectangle((140, 150, 590, 415), outline='black', width=3)
            draw.ellipse((245, 210, 285, 250), outline='black', width=3)
            draw.ellipse((425, 280, 465, 320), outline='black', width=3)
            draw.line((265, 230, 300, 140), fill='black', width=2)
            draw.text((100, 68), f'{label}: nominal Y = {nominal:.2f} +/-0.12 mm', font=font, fill='black')
            draw.text((100, 100), f'{label}: diameter {diameter:.2f} mm', font=font, fill='black')
            draw.line((445, 300, 645, 245), fill='black', width=2)
            draw.text((530, 215), f'{neighbor}: Y 18.80 mm', font=font, fill='black')
            draw.text((100, 450), 'Final alignment ABD; Y zero = BD_ORIGN', font=font, fill='black')
            target = directory / f'{identity}.png'
            image.save(target)
            images = [str(target.resolve())]
            source = target.name
            context = prefix + 'Read the attached drawing. Use leaders to associate labels. Numeric answers must come from the image. No measured result is supplied.'
            question = f'读图，只提取{label}；输出JSON：feature、nominal_y_mm、diameter_mm、tolerance_mm、y_origin、actual_y_mm。尺寸用2位小数字符串，未提供实测值用null。'
            expected = dict(feature=label, nominal_y_mm=f'{nominal:.2f}', diameter_mm=f'{diameter:.2f}',
                            tolerance_mm='+0.12/-0.12', y_origin='BD_ORIGN', actual_y_mm=None)
            answer = json.dumps(expected, ensure_ascii=False)
        elif family == 4:
            blind = rng.choice([True, False])
            condition = '盲螺纹孔，背面被夹具挡住' if blind else '通孔，背面开窗，指定孔边可透射且无遮挡'
            context = prefix + f'实物确认：{condition}。要测指定孔口中心，而不是螺纹或倒角。无高度测量资料。'
            question = '选择照明布局，说明相机和灯的位置、适用条件与验证，不能直接宣称某灯保证准确。'
            answer = ('相机从目标上方观察，光轴与孔口面法向对齐。来源确认盲孔且背面遮挡，透射背光不能形成目标孔边轮廓。建议上方同轴或漫射照明，低角度环光作对比试验；确认所提边缘是孔口而非螺纹/倒角。' if blind else
                      '相机从目标上方观察，光轴与孔口面法向对齐。来源确认通孔且光路无遮挡，建议光源置于工件下方、相机对面，背光轮廓成像；试验均匀/准直背光，治具开窗。')
            answer += f' 这属于初步设计；用实物图像、重复采集和参考测量验证边缘归属与稳定性。存在倾斜或高度任务时另评估接触/3D测量，二维图像不能凭空提供高度。[Source 1: {source}]'
        elif family == 5:
            errors = [rng.randint(2, 7) for _ in range(5)]
            total = sum(errors)
            target = rng.choice([20, 24, 30])
            context = prefix + f'设计目标：绝对误差上界≤{target} um（客户尚未验收）。光学校准、定位、平台配准、基准/装夹、热漂移的独立绝对上界分别为{errors} um。采用最坏情况相加，未完成实际验证。'
            question = '计算最坏情况预算，输出JSON：worst_case_um、target_um、budget_within_target、accuracy_verified。不要用RSS替代约定方法，不要把预算符合称作实际验证通过。'
            expected = dict(worst_case_um=total, target_um=target, budget_within_target=total <= target, accuracy_verified=False)
            answer = json.dumps(expected, ensure_ascii=False)
        else:
            claimed = nominal / pixels * 1000
            context = prefix + f'来源：名义Y={nominal:.2f} mm；实际Y未测；最终Y零点BD_ORIGN。设计输入FOV={fov} mm、横向{pixels}像素。错误草案声称：名义值是实测值，以名义Y作FOV得到{claimed:.4f} um/px；零点是BC_ORIGN；规定AQL1.0；相机和镜头可以独立测全局Y。来源无抽样标准。'
            question = '纠正草案里的名义值、坐标系、采样、抽样标准和硬件完整性。保持有用的初步设计，不拒绝做方案。'
            answer = (f'名义Y是{nominal:.2f} mm，实际值未测；最终Y零点为BD_ORIGN，不能沿用BC_ORIGN。[Source 1: {source}]\n'
                      f'采样必须使用设计FOV {fov} mm与同轴{pixels}像素，结果为{sampling:.4f} μm/px；名义坐标不是FOV，像素采样不是测量准确度。\n'
                      '来源没有抽样标准，删除AQL1.0的来源要求说法，采样/验收规则需另行确认。可给出设计假设下的初步方案：相机光轴对齐孔口法向，优先比较远心镜头与实际物理边适用的照明。规格BOM除相机1台、镜头1支、灯和控制器1组外，还需基准测量/校准平台1套、坐标配准软件1套、治具1套、标定件1套及支架、工控机和线缆电源各1套。局部圆检测须通过实际基准链转换到全局Y；硬件型号、精度和量产有效性均未验证。')
        row = dict(mode='instruction', question=question, context=context, answer=answer,
                   text=answer, source_path=source, images=images, fictional=True,
                   fixture_family=family, seed=seed, label_basis='deterministic calculation or explicit fictional source; design proposals unvalidated')
        if expected is not None:
            row['expected'] = expected
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--count', type=int, default=160)
    args = parser.parse_args()
    rows = build_cases(args.output.parent / (args.output.stem + '-images'), seed=args.seed, count=args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    print(json.dumps(dict(records=len(rows), seed=args.seed, fictional=True, output=str(args.output))))


if __name__ == '__main__':
    main()
