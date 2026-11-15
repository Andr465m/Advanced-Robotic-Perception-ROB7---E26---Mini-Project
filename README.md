# ROB7 Advanced Robotic Perception – Mini-project: soccer ball tracking

Ball tracking with one non-NN method and one NN method. Both are trained on SoccerNet tracking data and tested on a 1-minute clip from a different environment (Denmark vs Wales, provided on Moodle).

## Files

| File | Content |
|---|---|
| `non_nn_tracking.ipynb` | Non-NN method: green field mask → DoG blob candidates (min-channel) + Hessian edge test → HOG + colour + motion features → linear-SVM → RBF-SVM cascade → static-ball penalty → Kalman filter tracker |
| `nn_tracking.ipynb` | NN method: Faster R-CNN (ResNet-50 FPN, COCO-pretrained) fine-tuned on SoccerNet ball boxes → static-ball penalty → the same Kalman filter tracker |
| `common.py` | Code shared by both methods: SoccerNet loading, Kalman filter tracker, shot-cut detection, static-ball check, evaluation, video output |
| `prelabel_test_clip.ipynb` | Proposes ball positions in the test clip with a COCO-pretrained detector, used only as suggestions while annotating |
| `annotate_ball.py` | Click tool used to annotate the test clip by hand (every 5th frame) |
| `data/annotations/denmark_wales_points.csv` | **Test annotations**: `frame, visible, x, y` for 300 frames (every 5th frame, 640×360 pixel coordinates; `visible=0` means no ball visible) |
| `data/annotations/prelabels.csv` | Pre-label suggestions shown in the annotation tool (`frame, x, y`) |
| `results/non_nn/`, `results/nn/` | Test predictions (`test_predictions.csv`), metrics (`test_metrics.csv`), tracking videos (`test_tracking.mp4`), tracker parameters, NN training loss |

## Data

- **Training/validation:** SoccerNet tracking (https://www.soccer-net.org/data), `train` and `test` splits, unzipped to `data/SoccerNet/tracking/train` and `data/SoccerNet/tracking/test`. Only the ball tracklets are used. Frames are downscaled 1920×1080 → 640×360 to match the test clip.
  - Train: games 4, 6, 7, 8, 11 (86 sequences). Validation: game 9 (20 sequences). Split by game, so the validation stadium is never seen in training.
- **Test:** `data/denmark_wales_clip_1min.mp4` (the clip from Moodle, 640×360, 25 fps, 1500 frames).

## Annotation

1. `prelabel_test_clip.ipynb` runs a COCO-pretrained Faster R-CNN ("sports ball"; not trained on SoccerNet) on the clip to suggest a position per frame.
2. `python annotate_ball.py` shows every 5th frame at 2× zoom with the suggestion. The ball centre is clicked by hand (with a zoom inset), or N is pressed if the ball is not visible. 208 of the 223 visible-ball labels were hand-clicked; the rest are suggestions checked and accepted.

## Evaluation

For each labelled frame, a prediction is a true positive if it lies within 5 px of the annotated centre. A prediction that is too far away, or a prediction in a frame with no visible ball, is a false positive. A visible ball without a correct prediction is a false negative. The notebooks report precision, recall, F1 and mean pixel error.

## Running

```
pip install -r requirements.txt
```

1. `non_nn_tracking.ipynb` (CPU, about 40 min first run; uses all cores)
2. `nn_tracking.ipynb` (GPU, about 65 min training on an RTX 2060 + about 10 min evaluation)

Run both from the project folder. Intermediate results are cached in `cache/`. If `results/nn/frcnn_ball_5games.pt` exists, NN training is skipped. The NN notebook's last cell compares both methods on the test annotations.

Model weights (`*.pt`, `*.pkl`), SoccerNet data and the test clip are not included in the repository (`.gitignore`). They are recreated by running the notebooks.
