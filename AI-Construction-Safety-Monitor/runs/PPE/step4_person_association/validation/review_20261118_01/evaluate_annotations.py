"""Evaluate selected provisional boxes; this is not exhaustive scene recall."""
import csv, json
from pathlib import Path
from collections import Counter

HERE=Path(__file__).resolve().parent
rows=list(csv.DictReader((HERE/'person_annotations.csv').open(encoding='utf-8')))
data=json.loads((HERE/'inference/results.json').read_text(encoding='utf-8'))
def iou(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0])); y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-x*y
    return x*y/union if union else 0
details=[]; totals=Counter(); results=[]
for image in data['images']:
    name=Path(image['source']).name; gt=[r for r in rows if r['image_filename']==name]; pred=image['associations']
    candidates=[]
    for gi,g in enumerate(gt):
        box=[float(g[k]) for k in ('x1','y1','x2','y2')]
        for pi,p in enumerate(pred):
            score=iou(box,p['person_box'])
            if score>=.5: candidates.append((score,gi,pi))
    used_gt=set(); used_pred=set(); matches={}
    for score,gi,pi in sorted(candidates,reverse=True):
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi); used_pred.add(pi); matches[gi]=(pi,score)
    totals['selected_person_boxes']+=len(gt); totals['matched_selected_boxes']+=len(matches)
    for gi,g in enumerate(gt):
        row={'image':name,'gt_person_id':g['person_gt_id'],'pred_person_id':'','person_iou':'','helmet_gt':g['helmet_status'],'helmet_pred':'NOT_DETECTED','vest_gt':g['vest_status'],'vest_pred':'NOT_DETECTED'}
        if gi in matches:
            pi,score=matches[gi]; p=pred[pi]; row.update(pred_person_id=p['person_id'],person_iou=round(score,3),helmet_pred=p['helmet_status'],vest_pred=p['vest_status'])
            for kind in ('helmet','vest'):
                truth=g[kind+'_status']; decision=p[kind+'_status']; totals['matched_person_item_decisions']+=1
                if decision=='UNKNOWN': totals['unknown_decisions']+=1
                if truth in ('PRESENT','MISSING'):
                    totals['known_truth_item_decisions']+=1
                    if decision=='UNKNOWN': totals['known_truth_unknown']+=1
                    elif decision==truth: totals['status_correct']+=1
                    else: totals['status_incorrect']+=1
                    if truth=='PRESENT' and decision=='MISSING': totals['false_missing']+=1
                elif decision=='MISSING': totals['missing_on_unknown_truth']+=1
        details.append(row)
    results.append({'image':name,'selected_gt_people':len(gt),'matched_selected_people':len(matches),'predicted_people':len(pred),'note':'Selected GT boxes are non-exhaustive; unmatched predictions are not counted as false positives.'})
with (HERE/'matched_results.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=details[0].keys());w.writeheader();w.writerows(details)
out={'method':'Greedy IoU >= 0.5 on manually drawn assistant-provisional visible person boxes, then compare per-person statuses. Selected GT not exhaustive; human_verified=no for all. PPE-to-person physical assignment requires separate labelled PPE box ownership; status agreement is not assignment accuracy.', 'totals':dict(totals),'per_image':results,'physical_assignment_metrics':None,'reason_assignment_unavailable':'No independently annotated PPE-box ownership or human-verified labels','detection_confidence':.25,'status_confidence':.5,'iou_person_matching':.5}
(HERE/'metrics.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
