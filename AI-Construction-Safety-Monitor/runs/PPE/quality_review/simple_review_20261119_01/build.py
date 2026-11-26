from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import csv, html, json, shutil

out=Path(__file__).resolve().parent
project=out.parents[3]
old=out.parent/'priority_cases_20261118_01'
cases=[('00016_A1','A1'),('00025_B2','B2'),('00025_B3','B3'),('00783_C3','C3'),('00031_D2','D2')]
rows=[]; data=[]
for folder,case in cases:
    source=old/folder; dest=out/folder; dest.mkdir(exist_ok=False)
    for name in ('original.jpg','target_assignments.png','model_boxes_and_assignments.png'):
        shutil.copy2(source/name,dest/name)
    info=json.loads((source/'case_data.json').read_text(encoding='utf-8'))
    owners={int(k):v for k,v in info['detection_owners'].items()}
    detections=[]
    for i,d in enumerate(info['all_ppe_detections'],1):
        owners_text=', '.join('P'+str(p) for p in owners.get(i,[])) or 'No person assigned'
        detections.append({'id':i,'class_name':d['class_name'],'confidence':d['confidence'],'proposed_owner':owners_text})
        rows.append({'case':case,'item':f'D{i}','proposed_assignment':owners_text,'person_box':'','helmet':'','vest':'','ownership':'','annotation_class':'','notes':''})
    rows.append({'case':case,'item':'person','proposed_assignment':f'P{info["target_person_id"]}','person_box':'','helmet':'','vest':'','ownership':'','annotation_class':'','notes':''})
    data.append({'case':case,'folder':folder,'target':info['target_person_id'],'detections':detections})

imgpath=next((project/'dataset/processed/PPE/valid/images').glob('00269*.jpg'))
labelpath=project/'dataset/processed/PPE/valid/labels'/f'{imgpath.stem}.txt'
image=Image.open(imgpath).convert('RGB'); w,h=image.size
dest=out/'00269'; dest.mkdir(exist_ok=False); shutil.copy2(imgpath,dest/'original.jpg')
scale=3; font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
for name,classname,color in [('original_annotations.png','NoHelmet','#ffcc00'),('proposed_interpretation.png','Helmet','#00ff86')]:
    canvas=image.resize((w*scale,h*scale)); draw=ImageDraw.Draw(canvas)
    for n,line in enumerate(labelpath.read_text().splitlines(),1):
        cls,x,y,bw,bh=map(float,line.split())
        box=[int((x-bw/2)*w*scale),int((y-bh/2)*h*scale),int((x+bw/2)*w*scale),int((y+bh/2)*h*scale)]
        draw.rectangle(box,outline=color,width=3); text=f'{n}: {classname}'; tb=draw.textbbox((0,0),text,font=font); ty=max(0,box[1]-28)
        draw.rectangle([box[0],ty,box[0]+tb[2]+4,ty+27],fill=color);draw.text((box[0]+2,ty),text,font=font,fill='black')
    canvas.save(dest/name)
existing=project/'runs/PPE/step3_evaluation/review_20261118_01/visual_review'/imgpath.stem/'full_prediction.jpg'
shutil.copy2(existing,dest/'model_prediction.jpg')
for n in (1,2):
    rows.append({'case':'00269','item':f'annotation {n}','proposed_assignment':'NoHelmet -> Helmet (not applied)','person_box':'','helmet':'','vest':'','ownership':'','annotation_class':'','notes':''})
with (out/'blank_decision_sheet.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
encoded=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Simple PPE review</title><style>body{font:17px Arial,sans-serif;color:#18212a;max-width:1450px;margin:auto;padding:18px}section{border:1px solid #b6c0c7;border-radius:8px;padding:18px;margin:20px 0}h1,h2{color:#173b5e} .images{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;flex:1 1 340px}img{width:100%;max-height:640px;object-fit:contain;background:#eef1f3}figcaption{font-weight:bold;padding:6px}select,input{font:inherit;padding:6px;margin:5px}label{display:inline-block;margin-right:10px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #bbb;text-align:left}button{font:inherit;font-weight:bold;padding:12px;cursor:pointer}small{color:#394d5b} .sticky{position:sticky;top:0;background:white;padding:8px;border-bottom:1px solid #bbb;z-index:3}</style></head><body><h1>Review five people and image 00269</h1><p>First look at the original image. Then check the highlighted person and the model's proposed PPE assignments. Choose <strong>UNKNOWN</strong> when you cannot see enough. No answers are selected for you.</p><div class="sticky"><button type="button" onclick="download()">Download my answers as CSV</button> <small>Answers are not saved automatically. Download before closing or refreshing this page.</small></div><p>For ownership, <b>correct</b> means the shown assignment belongs to the indicated person; <b>wrong</b> means it does not; <b>uncertain</b> means you cannot tell; <b>unassigned</b> means no person should be assigned. The label “No person assigned” is the model output, not a reviewer answer. Click an image link to view it larger.</p><div id="cases"></div><section><h2>Image 00269 — class-label concern</h2><p>The same two existing boxes are shown with their original labels and with the proposed interpretation. The proposed labels have not been applied. For each numbered box, choose what you can actually see.</p><div class="images"><figure><a href="00269/original.jpg"><img src="00269/original.jpg"></a><figcaption>Original image</figcaption></figure><figure><a href="00269/original_annotations.png"><img src="00269/original_annotations.png"></a><figcaption>Existing labels: NoHelmet</figcaption></figure><figure><a href="00269/proposed_interpretation.png"><img src="00269/proposed_interpretation.png"></a><figcaption>Proposed interpretation: Helmet</figcaption></figure></div><p><a href="00269/model_prediction.jpg">View the existing model prediction</a></p><div id="labels00269"></div></section><p><button type="button" onclick="download()">Download my answers as CSV</button></p><script>
const cases=DATA_HERE;
function options(values){return '<option value="">Choose…</option>'+values.map(v=>'<option value="'+v+'">'+v+'</option>').join('')}
const status=['PRESENT','MISSING','UNKNOWN'], ownership=['correct','wrong','uncertain','unassigned'];
function select(id,values){return '<select id="'+id+'">'+options(values)+'</select>'}
function esc(s){return String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;')}
document.getElementById('cases').innerHTML=cases.map(c=>`<section><h2>Case ${esc(c.case)} — highlighted person P${c.target}</h2><div class="images"><figure><a href="${c.folder}/original.jpg"><img src="${c.folder}/original.jpg"></a><figcaption>Original image</figcaption></figure><figure><a href="${c.folder}/target_assignments.png"><img src="${c.folder}/target_assignments.png"></a><figcaption>Target person and their proposed PPE boxes</figcaption></figure><figure><a href="${c.folder}/model_boxes_and_assignments.png"><img src="${c.folder}/model_boxes_and_assignments.png"></a><figcaption>All people and PPE boxes</figcaption></figure></div><p>Yellow = target person. Orange = other people. Green = PPE assigned to target; magenta = assigned to someone else; white = not assigned. D numbers identify PPE detections.</p><label>Person box: ${select(c.case+'_box',['correct','incorrect','uncertain'])}</label><label>Helmet: ${select(c.case+'_helmet',status)}</label><label>Vest: ${select(c.case+'_vest',status)}</label><h3>Check each proposed PPE assignment</h3><table><thead><tr><th>Detection</th><th>Model proposal</th><th>Your ownership decision</th></tr></thead><tbody>${c.detections.map(d=>`<tr><td>D${d.id} ${esc(d.class_name)} (${d.confidence.toFixed(2)})</td><td>${esc(d.proposed_owner)}</td><td>${select(c.case+'_D'+d.id,ownership)}</td></tr>`).join('')}</tbody></table><label>Notes: <input id="${c.case}_notes" size="48"></label></section>`).join('');
document.getElementById('labels00269').innerHTML=[1,2].map(i=>`<label>Box ${i}: ${select('label_'+i,['Helmet','NoHelmet','uncertain'])}</label>`).join('')+'<p><label>Notes: <input id="label_notes" size="48"></label></p>';
function val(id){return document.getElementById(id).value} function csv(v){return '"'+String(v).replaceAll('"','""')+'"'}
function download(){let rows=[['case','item','proposed_assignment','person_box','helmet','vest','ownership','annotation_class','notes']];for(const c of cases){rows.push([c.case,'person','P'+c.target,val(c.case+'_box'),val(c.case+'_helmet'),val(c.case+'_vest'),'','',val(c.case+'_notes')]);for(const d of c.detections)rows.push([c.case,'D'+d.id,d.proposed_owner,'','','',val(c.case+'_D'+d.id),'',''])}for(const i of [1,2])rows.push(['00269','annotation '+i,'NoHelmet -> Helmet (not applied)','','','','',val('label_'+i),val('label_notes')]);let blob=new Blob(['\\ufeff'+rows.map(r=>r.map(csv).join(',')).join('\\r\\n')],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='ppe_review_decisions.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
</script></body></html>'''.replace('DATA_HERE',encoded)
(out/'index.html').write_text(page,encoding='utf-8')
print(f'Prepared {len(data)} cases, {len(rows)} blank rows and 00269 comparison: {out}')
