# 🤟 Real-Time Hand Gesture Classifier

## Overview
A real-time hand gesture recognition system built with computer 
vision — from live webcam data collection through to a trained 
classifier. Uses MediaPipe's HandLandmarker to extract 21 hand 
landmark coordinates (63 features) per frame, then classifies 
them into 8 distinct gestures using a self-collected, self-labeled 
dataset of 1,863 samples.

**Final accuracy: 99.47%** across 8 gestures, achieved through an 
iterative process of finding and fixing real bugs — not by 
tuning hyperparameters, but by diagnosing root causes.

## Tech Stack
Python, OpenCV, MediaPipe (HandLandmarker Tasks API), scikit-learn, 
pandas

## Gestures
Thumbs up, thumbs down, peace, OK, open palm, fist, love (ILY 
sign), point up

## The Real Story: Three Bugs Found and Fixed

### 1. A data leakage bug hiding behind a "perfect" model
Initial training on a random 80/20 split produced 100% accuracy 
across all three models tested (Logistic Regression, Random 
Forest, Gradient Boosting) — a result treated as suspicious rather 
than a success. Investigation revealed the cause: gestures were 
recorded in continuous bursts, so consecutive frames were near-
duplicates. A random split placed near-identical frames from the 
same burst on both sides of the train/test boundary, letting the 
model "cheat" by matching memorized near-duplicates.

**Fix:** switched to a recording-order split — the first 80% of 
each gesture's frames for training, the last 20% for testing — 
ensuring the test set contained genuinely unseen hand positions. 
This dropped accuracy to an honest 94.27%, with all 15 errors 
concentrated in a single, explainable confusion: open_palm being 
mistaken for love or thumbs_down.

### 2. Live testing surfaced problems the test set missed
Running the model live on webcam revealed two additional 
confusion pairs the 262-sample formal test set hadn't caught: 
ok↔thumbs_down and love↔thumbs_down, both angle-dependent. This 
highlighted a real limitation of small, single-session datasets — 
a test set can look clean while still missing edge cases a user 
naturally produces.

### 3. Diagnosing root cause before applying a fix
Rather than guessing, two specific hypotheses were tested:
- **thumbs_down's confusion** was hypothesized to be a *data 
  coverage* problem (insufficient angle variation in training). 
  Collecting 289 additional angle-varied samples confirmed this — 
  thumbs_down reached perfect precision and recall, and its 
  confusion with open_palm disappeared entirely.
- **open_palm's remaining confusion with love** was hypothesized 
  to be either the same data-coverage issue, or a genuine 
  geometric ambiguity between the two gestures that more data 
  couldn't fix. Collecting 280 additional angle-varied open_palm 
  samples resolved it completely (55% → 100% recall), confirming 
  it was data coverage, not a fundamental limitation.

## Results

| Stage | Accuracy | Key Issue |
|---|---|---|
| Random split (before leakage fix) | 100.00% | Data leakage — invalid result |
| Recording-order split | 94.27% | open_palm confused with love/thumbs_down |
| + thumbs_down angle data | 95.31% | open_palm confusion consolidated onto love |
| + open_palm angle data | **99.47%** | Only 2 residual errors, consistent with noise |

## What This Project Demonstrates
- Recognizing that a suspiciously perfect result is a signal to 
  investigate, not celebrate
- Diagnosing a genuine data leakage bug through systematic checks 
  (duplicate detection, split logic review) rather than guessing
- Using live testing as a validation step beyond the formal test 
  set, since a controlled test set doesn't always reflect real-
  world use
- Forming specific, testable hypotheses about *why* a model fails, 
  then collecting targeted data to test them — rather than 
  applying generic fixes

## Next Steps
- Expand to a larger, more visually similar gesture set (e.g. ASL 
  alphabet) to properly stress-test model choice between 
  classifiers
- Explore a second camera angle or depth sensor to further reduce 
  reliance on data volume for resolving angle-dependent ambiguity
