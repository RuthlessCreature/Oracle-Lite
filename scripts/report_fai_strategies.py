"""Publish source-backed classification summaries; never select geometry."""
import argparse,json,sys
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from oracle_viewer.store import ViewerStore
from oracle_viewer.fai_strategy import strategy_state,FAMILIES,EVIDENCE,FAILURES,ROUTES
parser=argparse.ArgumentParser();parser.add_argument('projects',nargs='+');args=parser.parse_args()
store=ViewerStore(ROOT/'talker/state/viewer');records=[];unclassified=[];snapshots=[];groups=defaultdict(lambda:{'marked':0,'unmarked':0,'items':[]})
for pid in args.projects:
 project=store.get(pid);state=strategy_state(store,pid);items=store.items(project)
 classified={row['fai']:row for row in state.get('items',[])}
 snapshots.append({'project_id':pid,'drawing_no':project['drawing_no'],'status':state['status'],
    'classified':len(classified),'total':len(items),'source_stale':state.get('source_stale',False)})
 for item in items:
  row=classified.get(item['number'])
  if not row:unclassified.append({'drawing_no':project['drawing_no'],'fai':item['number']});continue
  annotation=item.get('annotation') or {}
  marked=annotation.get('status')=='auto_marked' and annotation.get('origin')=='local_model_candidate'
  record={**row,'project_id':pid,'drawing_no':project['drawing_no'],'revision':project['revision'],
    'current_marked':marked,'annotation_is_verified_ground_truth':False}
  records.append(record)
  group=groups[row['measurement_family']+'/'+row['evidence_class']]
  group['marked' if marked else 'unmarked']+=1;group['items'].append({'drawing_no':project['drawing_no'],'fai':item['number']})
report={'at':datetime.now(timezone.utc).isoformat(),'projects':snapshots,'classified':len(records),
 'unclassified':unclassified,'method_groups':dict(groups),'items':records,
 'classification_is_not_registration_accuracy':True,'manual_point_or_pose_selection':False}
out=ROOT/'reports/engineering-checks/20261016';out.mkdir(parents=True,exist_ok=True)
(out/'fai-method-classification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
lines=['# FAI 测量方法分类与分流记录','',f'生成时间：{report["at"]}',
 f'已确认分类 {len(records)} 项；尚未通过分类校验 {len(unclassified)} 项。',
 '分类由本地 Strata 生成，逐条核对类别与原文摘录。已有标记是处理样本，不作为正确真值。分类完成不代表几何定位完成。','',
 '## 全部测量项的分类','', '| 测量方法 | 证据条件 | 已有标记 | 未标记 |','|---|---|---:|---:|']
for key,group in sorted(groups.items(),key=lambda pair:(-pair[1]['unmarked'],pair[0])):
 family,evidence=key.split('/');lines.append(f'| {FAMILIES[family]} | {EVIDENCE[evidence]} | {group["marked"]} | {group["unmarked"]} |')
lines+=['','## 未标记项的原因与模型选择路线','','| 项目 | FAI | 证据条件 | 问题分类 | 下一步路线 | 原文页 |','|---|---:|---|---|---|---|']
for row in records:
 if row['current_marked']:continue
 lines.append(f'| {row["drawing_no"]} | {row["fai"]} | {EVIDENCE[row["evidence_class"]]} | {FAILURES[row["failure_class"]]} | {ROUTES[row["next_route"]]} | {", ".join(map(str,row["source_pages"]))} |')
lines+=['','## 新路线的验证边界','',
 '零部件路径先列出原始 CAD body、尺寸、曲面拓扑与装配位置，再由本地模型分批筛选、全局比较并核对图纸引线。所有 body 均进入候选核对，不能因图像容量而截去后续零件。',
 '零部件候选选择仍需完整点组和独立实体复核，不直接提高标记覆盖。多坐标系项保留每个原始坐标系，不允许把所有 XYZ 混成一个平面。',
 '明确不存在于当前装配阶段的实体进入适用性核对，保持未标记，不生成虚构点位。已接入完全相同源特征的共享配准候选；完整的基准构建关系图尚待实现，不能视作已验证的能力。','',
 '## 尚未完成的分类','']
lines += [f'- {row["drawing_no"]} / FAI {row["fai"]}' for row in unclassified] or ['无。']
(out/'FAI_方法分类与处理路线.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'classified':len(records),'unclassified':unclassified,'projects':snapshots},ensure_ascii=False))
