"""One-off: pull the six Figure 2 examples (Gemma-4-31B, seed 0) into research/paper/assets/prompts/."""
import json, glob, math, sys
from pathlib import Path
from PIL import Image, ImageOps
RUN, IMGS, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
P = {json.loads(l)['item_id']: json.loads(l) for l in open(f'{RUN}/predictions.jsonl')}
I = {json.loads(l)['item_id']: json.loads(l) for l in open(f'{RUN}/items.jsonl')}
CELLS = ['rice_leaf_bacterial_fungal-22fe9e4077:mcqa_within', 'rice_leaf_bacterial_fungal-22fe9e4077:mcqa_across',
         'rice_leaf_bacterial_fungal-22fe9e4077:oeq', 'watermelon_disease_classification-a3ac92e593:mcqa_within',
         'tomato_adaptive_yolov8_detection-d4e04ff27a:grounding', 'AgroMind-995c13c04c:native']
# fixed square crops (x0, y0, side) in source pixels, where a centred crop would cut the subject
CROP = {'AgroMind-995c13c04c:native': (0, 464, 560)}   # keeps the red box inside the circle
S = 600
out = []
for k in CELLS:
    p, it = P[k], I[k]
    src = IMGS / it['images'][0].lstrip('/')
    im = ImageOps.exif_transpose(Image.open(src)).convert('RGB')
    w, h = im.size
    boxes = [(b['bbox'], 'gt') for b in (p.get('gt_boxes') or [])] + [(b['bbox'], 'pred') for b in (p.get('pred_boxes') or [])]
    if boxes and it['task'] == 'detection':
        xs = [v for b, _ in boxes for v in (b[0], b[2])]; ys = [v for b, _ in boxes for v in (b[1], b[3])]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        r = max(math.hypot(x - cx, y - cy) for x in xs for y in ys) * 1.04   # every box corner inside the circle
        side = min(2 * r, w, h)
    elif k in CROP:
        x0, y0, side = CROP[k]; cx, cy = x0 + side / 2, y0 + side / 2
    else:
        cx, cy, side = w / 2, h / 2, min(w, h)
    x0 = min(max(cx - side / 2, 0), w - side); y0 = min(max(cy - side / 2, 0), h - side)
    im.crop((x0, y0, x0 + side, y0 + side)).resize((S, S), Image.LANCZOS).save(OUT / f"{k.split(':')[0]}.jpg", quality=90)
    f = S / side
    out.append(dict(item_id=k, file=f"{k.split(':')[0]}.jpg", task=it['task'], format=it['format'], qa_format=it.get('qa_format'),
                    dataset=it['dataset'], crop=it['crop'], prompt=it['prompt'], choices=it.get('choices'), answer=it.get('answer'),
                    gt_label=p.get('gt_label'), is_nota=it['is_nota'], response=p['answer'], correct=p['correct'],
                    judge_correct=p.get('judge_correct'), tp=p.get('tp'), fp=p.get('fp'), fn=p.get('fn'),
                    boxes=[dict(kind=kd, label=None, xyxy=[(b[0] - x0) * f, (b[1] - y0) * f, (b[2] - x0) * f, (b[3] - y0) * f]) for b, kd in boxes]))
json.dump(out, open(OUT / 'cells.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps([{k: v for k, v in c.items() if k not in ('prompt', 'boxes')} for c in out], indent=0)[:3000])
