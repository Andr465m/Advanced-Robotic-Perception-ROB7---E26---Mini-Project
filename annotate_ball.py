import os
import sys

import cv2
import pandas as pd

from common import read_video

PRE = 'data/annotations/prelabels.csv'
OUT = 'data/annotations/denmark_wales_points.csv'
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 5
Z = 2  # display scale
HELP = 'click=ball  ENTER/SPACE=accept  N=no ball  B=back  Q=save+quit'

frames, _ = read_video()
idx = list(range(0, len(frames), STEP))
pre = {r.frame: (r.x, r.y) for r in pd.read_csv(PRE).itertuples()} if os.path.exists(PRE) else {}
ann = {}
if os.path.exists(OUT):  # resume
    for r in pd.read_csv(OUT).itertuples():
        ann[r.frame] = (r.x, r.y) if r.visible else None

click, mouse = None, (0, 0)


def on_mouse(ev, x, y, flags, _):
    global click, mouse
    mouse = (x / Z, y / Z)
    if ev == cv2.EVENT_LBUTTONDOWN:
        click = mouse


def save():
    rows = [(f, int(p is not None), p[0] if p else '', p[1] if p else '') for f, p in sorted(ann.items())]
    pd.DataFrame(rows, columns=['frame', 'visible', 'x', 'y']).to_csv(OUT, index=False)


def draw(f, i):
    im = cv2.resize(frames[f], None, fx=Z, fy=Z, interpolation=cv2.INTER_CUBIC)
    sug = click or ann.get(f) or pre.get(f)
    if f in ann and ann[f] is None and click is None:
        cv2.putText(im, 'labelled: NO BALL', (10, 60), 0, 0.8, (0, 0, 255), 2)
    if sug is not None:
        col = (0, 255, 0) if click or f in ann else (0, 255, 255)
        cv2.circle(im, (int(sug[0] * Z), int(sug[1] * Z)), 12, col, 2)
    mx, my = mouse  # zoom inset
    crop = cv2.copyMakeBorder(frames[f], 40, 40, 40, 40, cv2.BORDER_CONSTANT)[int(my):int(my) + 80, int(mx):int(mx) + 80]
    if crop.shape[:2] == (80, 80):
        crop = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_NEAREST)
        cv2.drawMarker(crop, (160, 160), (255, 0, 255), cv2.MARKER_CROSS, 20, 1)
        im[10:330, -330:-10] = crop
    cv2.putText(im, f'frame {f}  ({i + 1}/{len(idx)}, done {len(ann)})', (10, 30), 0, 0.8, (255, 255, 255), 2)
    cv2.putText(im, HELP, (10, im.shape[0] - 15), 0, 0.7, (255, 255, 255), 2)
    cv2.imshow('annotate', im)


cv2.namedWindow('annotate', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('annotate', on_mouse)
i = next((k for k, f in enumerate(idx) if f not in ann), 0)
while 0 <= i < len(idx):
    f = idx[i]
    draw(f, i)
    k = cv2.waitKey(30) & 0xFF
    if k in (13, 32):  # accept
        p = click or (ann[f] if f in ann else pre.get(f))
        if p is None and f not in ann:
            continue
        ann[f], click = p, None
        save()
        i += 1
    elif k in (ord('n'), ord('N')):
        ann[f], click = None, None
        save()
        i += 1
    elif k in (ord('b'), ord('B'), 8):
        click, i = None, max(i - 1, 0)
    elif k in (ord('q'), ord('Q'), 27):
        break
save()
cv2.destroyAllWindows()
print(f'saved {len(ann)}/{len(idx)} frames to {OUT}')
