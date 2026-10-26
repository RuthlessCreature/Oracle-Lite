"""Write a bounded benchmark report; source reading is not FAI registration."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    train=json.loads(args.training.read_text());report=json.loads(args.evaluation.read_text())
    data=train['dataset_manifest'];metrics=report['metrics']
    if not report.get('complete'):
        raise ValueError('Full held-out evaluation has not completed')
    paired={}
    for row in report['items']:paired.setdefault(row['id'],{})[row['mode']]=row
    if any(set(row)!= {'base','adapter'} for row in paired.values()):
        raise ValueError('Each held-out page needs both base and candidate results')
    semantic_gains=[];regressions=[];format_gains=[];failures=[]
    for key,rows in paired.items():
        base,new=rows['base'],rows['adapter']
        if new['correct_points']>base['correct_points']:semantic_gains.append(key)
        if new['correct_points']<base['correct_points']:regressions.append(key)
        if (new['valid_json'] and not base['valid_json'] and
                new['correct_points']==base['correct_points'] and new['exact']==base['exact']):
            format_gains.append(key)
        if not new['exact']:failures.append(key)
    rows=['# 云端 Qwen 训练与独立项目验证', '',
          '本轮训练目标为 MSOP 图文坐标转录。XML 是由虚拟拍摄图与测量方案生成的最终产物，未作为本轮输入。', '',
          '## 数据与方法', '',
          f"- 训练项目：{', '.join(data['train_groups'])}，{data['samples']['train']} 页、{data['point_count']['train']} 个原文点。",
          f"- 验证项目：{', '.join(data['validation_groups'])}，{data['samples']['validation']} 页、{data['point_count']['validation']} 个原文点；按项目隔离，源文件指纹不交叉。",
          '- 输入包含 MSOP 页面图片和原生文本，原生文本包含被测坐标；因此这是图文辅助的结构化转录基准，不是纯图像 OCR、CAD 配准或假图到 XML 的基准。',
          '- 训练使用 NF4 QLoRA、rank 8、alpha 16、学习率 5e-5、1 epoch、batch 1、梯度累积 4。底座只读，训练独立适配器。',
          '- 对照使用同一套 vLLM BF16 推理实现，底座与候选接收相同图片、原文和提示，均要求完整输出。长表格预算提升至 8,192 tokens 后复测；最终结果无截断。候选注册不改变 Viewer 的默认模型。',
          '- 格式与内容分别计分：Markdown 代码块不满足裸 JSON 约束；解开单层代码块后，仍独立计算点名、坐标轴和数值正确性。', '',
          '## 完整对照结果', '',
          '| 指标 | 底座 | 新适配器 |','|---|---:|---:|']
    for key,name in [('pages','验证页数'),('exact_pages','全部坐标正确的页面'),('valid_json_pages','裸 JSON 合规页面'),
                     ('correct_points','正确坐标点'),('expected_points','应有坐标点'),('invented_points','额外点位/坐标轴')]:
        rows.append(f"| {name} | {metrics['base'][key]} | {metrics['adapter'][key]} |")
    rows += ['', f'坐标内容增加的页面：{len(semantic_gains)}；坐标内容回退页面：{len(regressions)}；仅格式合规改善的页面：{len(format_gains)}。', '',
             '## 结论与适用范围', '']
    if not semantic_gains and not regressions:
        rows.append('本轮没有观察到坐标内容准确率提升。若裸 JSON 合规率提升，其收益是减少输出清洗与结构解析失败，不能表述为 FAI 几何配准能力提升。')
    else:
        rows.append('坐标内容的改善或回退以逐页结果为准；本基准不支持推断 FAI 几何定位、光学方案或测量程序执行能力。')
    rows += ['训练后的适配器仍为候选版本。FAI 位置、完整点组、光路可见性、测量精度和假图到 XML 输出需要各自的独立验证。', '',
             '现有虚拟拍摄图需补齐拍摄时的相机位姿、CAD/方案指纹、FAI 与像素/空间坐标快照，才能构成后续配对监督数据。', '',
             '## 可追溯产物', '',
             f"- 底座配置 SHA-256：`{train['base_config_sha256']}`。",
             f"- 适配器 SHA-256：`{train['adapter_sha256']}`。",
             f"- 训练集 SHA-256：`{data['sha256']['train']}`。",
             f"- 验证集 SHA-256：`{data['sha256']['validation']}`。",
             f"- 训练耗时：{train['metrics']['train_runtime']:.1f} 秒；平均训练损失：{train['metrics']['train_loss']:.6f}。损失不是测量准确率。", '',
             '## 需要复查的验证页', '']
    rows += [f'- {key}' for key in failures] or ['无候选坐标内容错误页。']
    args.output.write_text('\n'.join(rows)+'\n')
    print(json.dumps({'report':str(args.output),'semantic_gain_pages':len(semantic_gains),
                      'regression_pages':len(regressions),'format_gain_pages':len(format_gains)},ensure_ascii=False))


if __name__=='__main__':main()
