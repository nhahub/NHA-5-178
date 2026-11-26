"""Build visual review artifacts from existing saved predictions; no inference."""
from pathlib import Path
import csv, html, json, shutil
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[3]
BASE = PROJECT / 'runs/PPE/step4_person_association/validation/review_20261118_01'
DATA = PROJECT / 'dataset/processed/PPE/valid'
CLASSES = {0:'Helmet',1:'NoHelmet',2:'NoVest',3:'Vest'}
CASES = [('00016','A1'),('00025','B2'),('00025','B3'),('00783','C3'),('00031','D2')]
FONT = 'C:/Windows/Fonts/arial.ttf'
raw = json.loads((BASE/'inference/results.json').read_text(encoding='utf-8'))
annotations = list(csv.DictReader((BASE/'person_annotations.csv').open(encoding='utf-8')))
matches = list(csv.DictReader((BASE/'matched_results.csv').open(encoding='utf-8')))
review=[]; sections=[]

for prefix, case in CASES:
    a = next(x for x in annotations if x['person_gt_id']==case and x['image_filename'].startswith(prefix))
    match = next(x for x in matches if x['gt_person_id']==case and x['image']==a['image_filename'])
    entry = next(x for x in raw['images'] if Path(x['source']).name==a['image_filename'])
    target = int(match['pred_person_id'])
    folder = OUT/f'{prefix}_{case}'; folder.mkdir(exist_ok=True)
    source = DATA/'images'/a['image_filename']; shutil.copy2(source,folder/'original.jpg')
    label = DATA/'labels'/f'{source.stem}.txt'
    labels = label.read_text(encoding='utf-8').splitlines()
    im = Image.open(source).convert('RGB'); width,height=im.size
    scale = max(1.0, min(3.0, 1400 / width)); dims=(round(width*scale),round(height*scale))
    font=ImageFont.truetype(FONT,max(13,round(13*scale)))
    def scaled(b): return tuple(round(float(v)*scale) for v in b)
    def rect(draw, box, color, text):
        xy=scaled(box); draw.rectangle(xy,outline=color,width=max(2,round(scale*2)))
        bounds=draw.textbbox((xy[0],xy[1]),text,font=font); top=max(0,xy[1]-(bounds[3]-bounds[1])-5)
        draw.rectangle((xy[0],top,min(dims[0],xy[0]+bounds[2]-bounds[0]+5),xy[1]),fill=color)
        draw.text((xy[0]+2,top),text,fill='black',font=font)
    ground=im.resize(dims); d=ImageDraw.Draw(ground)
    gt_rows=[]
    for i,line in enumerate(labels,1):
        cls,xc,yc,w,h=map(float,line.split()); box=[(xc-w/2)*width,(yc-h/2)*height,(xc+w/2)*width,(yc+h/2)*height]
        name=CLASSES[int(cls)]; rect(d,box,'cyan',f'G{i} {name}');gt_rows.append(f'G{i}: {name}')
    gt_box=[int(a[k]) for k in ('x1','y1','x2','y2')]
    rect(d,gt_box,'yellow',f'{case} provisional person')
    ground.save(folder/'ground_truth_and_target.png')
    associations=entry['associations']; owners={}
    for p in associations:
        for e in p['associated_ppe']:
            owners.setdefault(int(e['detection_id']),[]).append(int(p['person_id']))
    pred=im.resize(dims); d=ImageDraw.Draw(pred)
    for person in associations:
        pid=int(person['person_id']); color='yellow' if pid==target else 'orange'
        rect(d,person['person_box'],color,f'P{pid} {person["person_confidence"]:.2f}'+(' TARGET' if pid==target else ''))
    detection_rows=[]
    for i,item in enumerate(entry['ppe_detections'],1):
        assigned=owners.get(i,[]); who=','.join('P'+str(x) for x in assigned) if assigned else 'UNASSIGNED'
        color='lime' if target in assigned else ('magenta' if assigned else 'white')
        text=f'D{i} {item["class_name"]} {item["confidence"]:.2f} -> {who}'
        rect(d,item['box'],color,text);detection_rows.append(text)
    pred.save(folder/'model_boxes_and_assignments.png')
    target_record=next(p for p in associations if p['person_id']==target)
    focused=im.resize(dims); d=ImageDraw.Draw(focused)
    rect(d,target_record['person_box'],'yellow',f'P{target} TARGET')
    for item in target_record['associated_ppe']:
        rect(d,item['box'],'lime',f'D{item["detection_id"]} {item["class_name"]} {item["confidence"]:.2f} -> P{target}')
    focused.save(folder/'target_assignments.png')
    metadata={'case_id':case,'image':a['image_filename'],'provisional_target_box':gt_box,
              'target_person_id':target,'matched_person_iou':float(match['person_iou']),
              'ground_truth_labels':gt_rows,'person_association':target_record,
              'all_person_associations':associations,'all_ppe_detections':entry['ppe_detections'],
              'detection_owners':owners,'source_settings':raw['settings']}
    (folder/'case_data.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    review.append({'case_id':case,'image_filename':a['image_filename'],'model_person_id':f'P{target}',
                   'verified_person_box':'','visible_helmet_status':'','visible_vest_status':'',
                   'ppe_detection_ownership':'','reviewer_decision':'','notes':''})
    e=html.escape; directory=folder.name
    people='; '.join(f'P{p["person_id"]}: helmet {p["helmet_status"]}, vest {p["vest_status"]}' for p in associations)
    sections.append(f'<section><h2>{case} — {e(a["image_filename"])} — target P{target}</h2>'
        f'<p>Target person box is provisional. Saved model states: {e(people)}</p>'
        f'<p>Ground truth: {e(", ".join(gt_rows)) or "none"}. Labels are dataset annotations, not verified completeness.</p>'
        f'<p>All detections: {e("; ".join(detection_rows) or "none")}</p>'
        f'<p><a href="{directory}/original.jpg">Open original full resolution</a> · '
        f'<a href="{directory}/case_data.json">Boxes, confidence and association data</a></p>'
        f'<div class="panels"><figure><img src="{directory}/ground_truth_and_target.png"><figcaption>Dataset boxes (cyan); provisional target box (yellow)</figcaption></figure>'
        f'<figure><img src="{directory}/target_assignments.png"><figcaption>Target person and PPE boxes assigned to them by the model.</figcaption></figure>'
        f'<figure><img src="{directory}/model_boxes_and_assignments.png"><figcaption>All persons: target yellow, others orange. PPE: target assigned green, other assigned magenta, unassigned white.</figcaption></figure></div></section>')

with (OUT/'review_sheet.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=review[0].keys());writer.writeheader();writer.writerows(review)
page='''<!doctype html><html><head><meta charset="utf-8"><title>PPE review: five cases</title><style>body{font:16px Arial;margin:24px;background:#f7f8fa;color:#18212e}section{background:white;padding:16px;margin:24px 0;border:1px solid #ccc} .panels{display:flex;flex-wrap:wrap;gap:14px}figure{margin:0;flex:1 1 500px}img{width:100%;height:auto}figcaption{padding:8px}p{line-height:1.5}</style></head><body><h1>Five priority cases: human review</h1><p>Inspect the original first, then compare the labelled and prediction views. Model arrows show proposed assignments, not verified ownership. Box coordinates and labels can overlap; open the image at full size and consult the JSON if needed.</p><p>Enter one row per case in <a href="review_sheet.csv">review_sheet.csv</a>. Record box as acceptable/corrected/uncertain; visible status as PRESENT/MISSING/UNKNOWN. For ownership, list D IDs as correct/wrong/uncertain for the specified person (or none); use notes for omissions and ambiguity. Leave status UNKNOWN when not visible.</p>'''+''.join(sections)+'</body></html>'
(OUT/'index.html').write_text(page,encoding='utf-8')
print('Built',len(review),'cases in',OUT)
