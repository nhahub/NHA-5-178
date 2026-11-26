from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
from collections import Counter
import csv,json

out=Path(__file__).resolve().parent
prev=out.parents[1]/'review_20261118_01'
root=out.parents[4]
info=json.loads((prev/'dataset_integrity.json').read_text())
rows=[]
for split in ('train','valid','test'):
    for label,line,text in info['splits'][split]['out_of_bounds_boxes']:
        p=next((root/'dataset/processed/PPE'/split/'images').glob(Path(label).stem+'.*'))
        with Image.open(p) as im: w,h=im.size
        cls,x,y,bw,bh=map(float,text.split()); edges=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]
        over=max(0,-edges[0],-edges[1],edges[2]-w,edges[3]-h)
        # Categories describe boundary geometry, not certainty of annotation intent.
        category='minor rounding' if over<=0.5 else ('suspicious annotation' if over<=2 else 'clear invalid coordinate')
        rows.append({'split':split,'label_file':label,'line':line,'class_id':int(cls),'image_width':w,'image_height':h,'max_overflow_pixels':round(over,6),'max_overflow_normalized':round(max(0,-(x-bw/2),-(y-bh/2),x+bw/2-1,y+bh/2-1),9),'classification':category,'rule':'<=0.5px numerical boundary tolerance; >0.5 to 2px suspicious; >2px geometrically invalid','label_text':text})
with (out/'boundary_classification.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print('BOUNDARY',Counter(r['classification'] for r in rows))
print('RANGE',min(r['max_overflow_pixels'] for r in rows),max(r['max_overflow_pixels'] for r in rows))
for r in sorted(rows,key=lambda x:x['max_overflow_pixels'],reverse=True)[:8]: print(r['split'],r['label_file'],r['line'],r['max_overflow_pixels'],r['classification'])
folders=sorted(p for p in (prev/'visual_review').iterdir() if p.is_dir())
for start in range(0,len(folders),5):
    group=folders[start:start+5]; canvas=Image.new('RGB',(1500,len(group)*310),'white'); draw=ImageDraw.Draw(canvas)
    for j,folder in enumerate(group):
        for k,name in enumerate(('original.jpg','ground_truth.jpg','full_prediction.jpg')):
            with Image.open(folder/name) as image:
                image.thumbnail((490,280)); canvas.paste(image,(k*500,j*310+25))
            draw.text((k*500+4,j*310+4),f'{folder.name[:12]} {name}',fill='black')
    canvas.save(out/f'contact_sheet_{start//5+1}.jpg',quality=90)
print('SHEETS',len(folders))
