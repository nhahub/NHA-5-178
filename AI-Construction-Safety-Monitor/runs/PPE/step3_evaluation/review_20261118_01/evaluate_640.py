from pathlib import Path
import json
from ultralytics import YOLO

out=Path(__file__).resolve().parent
baseline=json.loads((out/'evaluation.json').read_text(encoding='utf-8'))
model=YOLO(baseline['checkpoint'])
r=model.val(data=str(out/'validation.yaml'),split='val',imgsz=640,batch=4,device='cpu',workers=0,plots=False,conf=0.001,iou=0.7,project=str(out/'ultralytics'),name='val640',exist_ok=False,verbose=False)
result={'settings':dict(baseline['settings'],imgsz=640),'checkpoint_sha256':baseline['checkpoint_sha256'],'aggregate':{'precision':float(r.box.mp),'recall':float(r.box.mr),'map50':float(r.box.map50),'map50_95':float(r.box.map)},'per_class':[{'class_id':int(c),'class':model.names[int(c)],'precision':float(r.box.p[i]),'recall':float(r.box.r[i]),'map50':float(r.box.ap50[i]),'map50_95':float(r.box.ap[i])} for i,c in enumerate(r.box.ap_class_index)]}
(out/'evaluation_640.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
