import os
import pickle
import configparser

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

ROOTS = ['data/SoccerNet/tracking/train', 'data/SoccerNet/tracking/test']
TEST = 'data/denmark_wales_clip_1min.mp4'
GT_TEST = 'data/annotations/denmark_wales_points.csv'
W, H, S = 640, 360, 3.0  # test size, SoccerNet scale
TRAIN_GAMES, VAL_GAMES = {4, 6, 7, 8, 11}, {9}
TAU = 5  # hit radius
SK = 5  # static lag


def cached(path, fn):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    out = fn()
    with open(path, 'wb') as f:
        pickle.dump(out, f)
    return out


# SoccerNet

def load_seq(d):
    cp = configparser.ConfigParser()
    cp.read(f'{d}/gameinfo.ini')
    sec = cp['Sequence']
    ids = [int(k.split('_')[1]) for k, v in sec.items() if k.startswith('trackletid') and v.strip().startswith('ball')]
    gt = np.loadtxt(f'{d}/gt/gt.txt', delimiter=',')
    ball = {}
    for r in gt[np.isin(gt[:, 1], ids)]:
        ball.setdefault(int(r[0]), np.array([r[2], r[3], r[2] + r[4], r[3] + r[5]]) / S)
    return dict(name=os.path.basename(d), game=int(sec['gameID']), dir=d, ball=ball, n=len(os.listdir(f'{d}/img1')))


def load_soccernet():
    seqs = [load_seq(f'{r}/{s}') for r in ROOTS for s in sorted(os.listdir(r))]
    return [s for s in seqs if s['game'] in TRAIN_GAMES], [s for s in seqs if s['game'] in VAL_GAMES]


def read_frame(seq, f):
    im = cv2.imread(f"{seq['dir']}/img1/{f:06d}.jpg", cv2.IMREAD_REDUCED_COLOR_2)
    return cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)


def center(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


# Tracking

class Kalman:
    def __init__(self, z, q, r):
        self.A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)  # const. velocity
        self.H = np.eye(2, 4)
        self.Q = q * np.diag([0.25, 0.25, 1, 1])
        self.R = r * np.eye(2)
        self.x = np.r_[z, 0, 0].astype(float)
        self.P = np.diag([r, r, 100, 100]).astype(float)

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[:2]

    def S(self):
        return self.H @ self.P @ self.H.T + self.R

    def update(self, z):
        K = self.P @ self.H.T @ np.linalg.inv(self.S())
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P


class BallTracker:
    def __init__(self, t_init, t_keep, t_switch, margin, gate=9.0, lam=0.05, max_miss=5, q=4.0, r=2.0):
        self.t_init, self.t_keep, self.t_switch, self.margin = t_init, t_keep, t_switch, margin
        self.gate, self.lam, self.max_miss, self.q, self.r = gate, lam, max_miss, q, r
        self.reset()

    def reset(self):
        self.kf, self.miss = None, 0

    def start(self, z):
        self.kf, self.miss = Kalman(z, self.q, self.r), 0
        return self.kf.x[:2].copy()

    def step(self, xy, sc):
        if self.kf is not None:
            p = self.kf.predict()
            if len(xy):
                e = xy - p
                d2 = np.einsum('ni,ij,nj->n', e, np.linalg.inv(self.kf.S()), e)  # Mahalanobis
                ok = (d2 < self.gate) & (sc > self.t_keep)
                j = np.argmax(sc)
                if ok.any():
                    i = np.argmax(np.where(ok, sc - self.lam * d2, -np.inf))
                    if not ok[j] and sc[j] > self.t_switch and sc[j] - sc[i] > self.margin:
                        return self.start(xy[j])
                    self.kf.update(xy[i])
                    self.miss = 0
                    return self.kf.x[:2].copy()
                if sc[j] > self.t_switch:
                    return self.start(xy[j])
            self.miss += 1
            if self.miss <= self.max_miss:
                return self.kf.x[:2].copy()  # coast
            self.kf = None
        if len(xy) and sc.max() > self.t_init:
            return self.start(xy[np.argmax(sc)])
        return None


def run_tracker(scores, cuts=None, **kw):
    tr, pred = BallTracker(**kw), {}
    for f, (xy, sc) in enumerate(scores):
        if cuts is not None and cuts[f]:
            tr.reset()
        p = tr.step(xy, sc)
        if p is not None:
            pred[f] = p
    return pred


def detect_only(scores, t):
    return {f: xy[np.argmax(sc)] for f, (xy, sc) in enumerate(scores) if len(sc) and sc.max() > t}


def shot_cut(prev, cur, t=0.3):
    h = [cv2.normalize(cv2.calcHist([cv2.cvtColor(x, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 32], [0, 180, 0, 256]), None, 1, 0, cv2.NORM_L1)
         for x in (prev, cur)]
    return cv2.compareHist(h[0], h[1], cv2.HISTCMP_BHATTACHARYYA) > t


# Static balls (spare balls beside the pitch)

def homography(a, b):
    p = cv2.goodFeaturesToTrack(a, 400, 0.01, 8)  # Shi-Tomasi
    if p is None:
        return None
    q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None)  # LK
    ok = st.ravel() == 1
    if ok.sum() < 20:
        return None
    Hm, inl = cv2.findHomography(p[ok], q[ok], cv2.RANSAC, 2.0)
    return Hm if inl is not None and inl.sum() >= 0.5 * ok.sum() else None


def homographies(grays, cuts=None):
    Hs = [None] * len(grays)
    for f in range(SK, len(grays)):
        if cuts is None or not any(cuts[f - SK + 1:f + 1]):
            Hs[f] = homography(grays[f - SK], grays[f])
    return Hs


def seq_homographies(s):
    return homographies([cv2.cvtColor(read_frame(s, f), cv2.COLOR_BGR2GRAY) for f in range(1, s['n'] + 1)])


def static_mask(scores, Hs, smin, r=2.0):
    masks = []
    for f, (xy, sc) in enumerate(scores):
        m = np.zeros(len(sc), bool)
        if Hs[f] is not None and len(sc):
            pxy, ps = scores[f - SK]
            pxy = pxy[ps >= smin]
            if len(pxy):
                w = cv2.perspectiveTransform(pxy[None].astype(np.float32), Hs[f])[0]
                m = np.hypot(w[:, None, 0] - xy[None, :, 0], w[:, None, 1] - xy[None, :, 1]).min(0) <= r
        masks.append(m)
    return masks


# Evaluation

def metrics(pred, gt, frames, tau=TAU):
    tp = fp = fn = 0
    err = []
    for f in frames:
        g, p = gt.get(f), pred.get(f)
        d = np.hypot(p[0] - g[0], p[1] - g[1]) if p is not None and g is not None else None
        if d is not None and d <= tau:
            tp += 1
            err.append(d)
        else:
            fp += p is not None
            fn += g is not None
    pr, re = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(precision=pr, recall=re, F1=2 * pr * re / max(pr + re, 1e-9), err_px=np.mean(err) if err else np.nan)


def evaluate(pred_fn, scores, gts):
    preds, gt, frames = {}, {}, []
    for n, sc in scores.items():
        p = pred_fn(sc)
        for f in range(len(sc)):
            frames.append((n, f))
            if f in p:
                preds[n, f] = p[f]
            if f in gts[n]:
                gt[n, f] = gts[n][f]
    return metrics(preds, gt, frames)


# Test clip

def read_video(path=TEST):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return frames, cap.get(cv2.CAP_PROP_FPS)


def load_gt(path=GT_TEST):
    d = pd.read_csv(path)
    return {r.frame: (r.x, r.y) for r in d.itertuples() if r.visible}, sorted(d.frame)


def save_predictions(pred, n, path):
    rows = [(f, *pred[f]) if f in pred else (f, np.nan, np.nan) for f in range(n)]
    pd.DataFrame(rows, columns=['frame', 'x', 'y']).to_csv(path, index=False)


def load_predictions(path):
    return {int(r.frame): (r.x, r.y) for r in pd.read_csv(path).dropna().itertuples()}


def write_video(frames, pred, cuts, fps, path):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    trail = []
    for f, im in enumerate(frames):
        vis = im.copy()
        if cuts[f]:
            trail = []
        if f in pred:
            x, y = map(int, pred[f])
            trail.append((x, y))
            cv2.circle(vis, (x, y), 8, (0, 0, 255), 2)
        for a, b in zip(trail[-15:-1], trail[-14:]):
            cv2.line(vis, a, b, (0, 255, 255), 1)
        cv2.putText(vis, str(f), (5, H - 8), 0, 0.5, (255, 255, 255), 1)
        vw.write(vis)
    vw.release()


def show_frames(frames, pred, show=(100, 300, 540, 800, 1100, 1420)):
    fig, ax = plt.subplots(2, 3, figsize=(18, 7))
    for a, f in zip(ax.ravel(), show):
        vis = cv2.cvtColor(frames[f], cv2.COLOR_BGR2RGB)
        if f in pred:
            cv2.circle(vis, tuple(map(int, pred[f])), 9, (255, 0, 0), 2)
        a.imshow(vis)
        a.set_title(f'frame {f}')
        a.axis('off')
    plt.tight_layout()
