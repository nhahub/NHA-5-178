import csv, hashlib, json, math, platform, shutil, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import cv2, numpy as np, torch, ultralytics, yaml
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'dataset/processed/PPE'
WEIGHTS = ROOT.parent / 'runs/detect/runs/PPE/balanced_30_epochs/weights/best.pt'
OUT = Path(sys.argv[1]).resolve()
OUT.mkdir(parents=True, exist_ok=False)
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
names = {0:'Helmet',1:'NoHelmet',2:'NoVest',3:'Vest'}
cfg = yaml.safe_load((DATA/'data_balanced.yaml').read_text())
integrity, hashes, phashes = {}, defaultdict(list), []
for split in ('train','valid','test'):
    folder=DATA/split
    imgs={p.stem:p for p in (folder/'images').iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp')}
    labels={p.stem:p for p in (folder/'labels').glob('*.txt')}
    info={'images':len(imgs),'labels':len(labels),'missing_labels':sorted(imgs.keys()-labels.keys()),'orphan_labels':sorted(labels.keys()-imgs.keys()),'empty_labels':[], 'invalid_lines':[], 'out_of_bounds_boxes':[], 'unreadable_images':[], 'class_counts':Counter()}
    for stem,p in imgs.items():
        hashes[digest(p)].append(f'{split}/{p.name}')
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                pixels=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)
                low=cv2.dct(pixels)[:8,:8]; phashes.append((split,p.name,(low>np.median(low)).flatten()))
        except Exception as e: info['unreadable_images'].append([p.name,str(e)])
    for stem,p in labels.items():
        lines=p.read_text(encoding='utf-8').splitlines()
        if not any(x.strip() for x in lines): info['empty_labels'].append(p.name)
        for n,line in enumerate(lines,1):
            try:
                a=line.split()
                if len(a)!=5: raise ValueError('expected 5 fields')
                cls=int(a[0]); x,y,w,h=map(float,a[1:])
                if cls not in names or not all(math.isfinite(v) and 0<=v<=1 for v in (x,y,w,h)) or w<=0 or h<=0: raise ValueError('class or coordinate out of range')
                info['class_counts'][names[cls]]+=1
                if x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6: info['out_of_bounds_boxes'].append([p.name,n,line])
            except (ValueError,OverflowError) as e: info['invalid_lines'].append([p.name,n,str(e),line])
    info['class_counts']=dict(info['class_counts']); integrity[split]=info
cross_exact=[v for v in hashes.values() if len({x.split('/')[0] for x in v})>1]
cross_similar=[]
for i,(s,n,h) in enumerate(phashes):
    for s2,n2,h2 in phashes[i+1:]:
        distance=int(np.count_nonzero(h != h2)) if s!=s2 else 99
        if distance<=5: cross_similar.append([s,n,s2,n2,distance])
dump('dataset_integrity.json',{'splits':integrity,'exact_cross_split_sha256':cross_exact,'cross_split_phash_distance_le_5':cross_similar,'limitations':'pHash is a heuristic and cannot rule out scene/video leakage or annotation errors.'})
print('Integrity complete',flush=True)
# Stage the validation split to prevent Ultralytics cache creation or replacement in the dataset.
stage=OUT/'staged_validation'
for part in ('images','labels'):
    (stage/part).mkdir(parents=True)
    for p in (DATA/'valid'/part).iterdir():
        if p.is_file(): shutil.copy2(p,stage/part/p.name)
staged_yaml=OUT/'validation.yaml'
staged_yaml.write_text(yaml.safe_dump({'path':stage.as_posix(),'train':'images','val':'images','names':names}),encoding='utf-8')
model=YOLO(str(WEIGHTS))
report={'checkpoint':str(WEIGHTS),'checkpoint_sha256':digest(WEIGHTS),'source_yaml':str(DATA/'data_balanced.yaml'),'source_yaml_sha256':digest(DATA/'data_balanced.yaml'),'source_yaml_content':cfg,'source_validation':str(DATA/'valid'),'staged_validation':str(stage),'versions':{'python':platform.python_version(),'ultralytics':ultralytics.__version__,'torch':torch.__version__,'opencv':cv2.__version__},'settings':{'split':'val','imgsz':320,'batch':4,'device':'cpu','workers':0,'plots':False,'conf':0.001,'iou':0.7},'limitations':['Validation labels have not received complete human review.','Ultralytics precision and recall are operating-point metrics; fixed-threshold counts use a different procedure.']}
res=model.val(data=str(staged_yaml),split='val',imgsz=320,batch=4,device='cpu',workers=0,plots=False,conf=0.001,iou=0.7,project=str(OUT/'ultralytics'),name='val320',exist_ok=False,verbose=False)
report['aggregate']={'precision':float(res.box.mp),'recall':float(res.box.mr),'map50':float(res.box.map50),'map50_95':float(res.box.map)}
report['per_class']=[{'class_id':int(c),'class':names[int(c)],'precision':float(res.box.p[i]),'recall':float(res.box.r[i]),'map50':float(res.box.ap50[i]),'map50_95':float(res.box.ap[i])} for i,c in enumerate(res.box.ap_class_index)]
dump('evaluation.json',report)
with (OUT/'per_class_metrics.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=report['per_class'][0].keys()); w.writeheader(); w.writerows(report['per_class'])
print('Validation complete',flush=True)
# The all-object matching implementation counts only one prediction per GT and vice versa.
sys.path.insert(0,str(ROOT/'src/detection'))
import evaluate_small_object_errors as small
small.MODEL_PATH=WEIGHTS; small.IMAGES_DIR=stage/'images'; small.LABELS_DIR=stage/'labels'; small.IMAGE_SIZE=320
import contextlib
with (OUT/'small_object_fixed_threshold.txt').open('w',encoding='utf-8') as f:
    with contextlib.redirect_stdout(f): small.main()
print('Small object evaluation complete',flush=True)
# Review pack: missed cases across classes, plus deterministic ordinary examples.
miss=list(csv.DictReader((ROOT/'runs/PPE/small_object_analysis/missed_small_objects.csv').open(encoding='utf-8-sig')))
chosen=[]
for cls in names.values():
    chosen.extend([r['image'] for r in miss if r['class']==cls][:3])
rank=list(csv.DictReader((ROOT/'runs/PPE/annotation_audit/annotation_audit.csv').open(encoding='utf-8-sig')))
chosen += [r['image'] for r in rank[:3]]
missed_names={r['image'] for r in miss}
chosen += [p.name for p in sorted((stage/'images').iterdir()) if p.name not in missed_names][:6]
chosen=list(dict.fromkeys(chosen)); pack=OUT/'visual_review'; pack.mkdir()
rows=[]
for name in chosen:
    path=stage/'images'/name; image=cv2.imread(str(path)); h,w=image.shape[:2]
    gt=image.copy(); pred=image.copy(); label=stage/'labels'/f'{path.stem}.txt'
    for i,line in enumerate(label.read_text().splitlines(),1):
        a=line.split()
        if len(a)!=5: continue
        c=int(a[0]); x,y,bw,bh=map(float,a[1:]); box=[int((x-bw/2)*w),int((y-bh/2)*h),int((x+bw/2)*w),int((y+bh/2)*h)]
        cv2.rectangle(gt,box[:2],box[2:],(255,0,0),2); cv2.putText(gt,f'G{i} {names[c]}',(box[0],max(15,box[1]-3)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,0),1)
        rows.append({'image_filename':name,'object_id':f'G{i}','class':names[c],'ground_truth_status':'label present','prediction_status':'not assessed','suspected_issue':'','reviewer_decision':'','notes':''})
    result=model.predict(source=image,imgsz=320,conf=.25,device='cpu',verbose=False)[0]
    for i,b in enumerate(result.boxes,1):
        c=int(b.cls[0]); conf=float(b.conf[0]); box=list(map(int,b.xyxy[0].tolist()))
        cv2.rectangle(pred,box[:2],box[2:],(0,180,0),2); cv2.putText(pred,f'P{i} {names[c]} {conf:.2f}',(box[0],max(15,box[1]-3)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,180,0),1)
        rows.append({'image_filename':name,'object_id':f'P{i}','class':names[c],'ground_truth_status':'not assessed','prediction_status':f'full image conf={conf:.3f}','suspected_issue':'','reviewer_decision':'','notes':''})
    folder=pack/path.stem; folder.mkdir(); shutil.copy2(path,folder/'original.jpg'); cv2.imwrite(str(folder/'ground_truth.jpg'),gt); cv2.imwrite(str(folder/'full_prediction.jpg'),pred)
    if name.startswith('03295_jpg.rf.28082ec6b792bc923e0dcbe473c4551e'):
        shutil.copy2(ROOT/'runs/PPE/tiling_comparison/tiled_prediction.jpg',folder/'historical_tiled_prediction.jpg')
with (pack/'review.csv').open('w',newline='',encoding='utf-8-sig') as f:
    fields=['image_filename','object_id','class','ground_truth_status','prediction_status','suspected_issue','reviewer_decision','notes']; writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
dump('review_manifest.json',{'selection':chosen,'method':'First three missed rows per class, highest missed-count images, first six images absent from missed queue; deduplicated. Convenience sample, not statistically representative.','historical_tiled_prediction':'copied existing visualization; provenance not independently established','prediction_settings':{'imgsz':320,'conf':0.25},'human_review':'pending'})
print('Review pack complete',flush=True)
