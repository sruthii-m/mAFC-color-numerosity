# mAFC Perceptual Decision-Making: CNN Classifier + Human Confidence Modeling

This project studies a **multi-alternative forced-choice (mAFC) color-dominance task**: given an image
containing red, green, and blue dots in varying proportions, decide which color is most prevalent. The
repo has three components:

1. **`src/train_cnn.py`** — a CNN trained to solve the same 3-way color-dominance task from raw pixels,
   evaluated on a held-out validation set of stimulus conditions never seen during training.
2. **`confidence_modeling/`** and **`bads_fitting/`** — MATLAB code fitting three competing computational
   models of decision *confidence* (Bayesian posterior probability, peak-evidence, and evidence-difference)
   to human behavioral data from this task, using the [BADS](https://github.com/lacerbi/bads) Bayesian
   Adaptive Direct Search optimizer, with a model-recovery analysis to validate the fitting procedure.
3. **`confidence_modeling/evaluate_cnn_on_expt3.py`** — compares the trained CNN's confidence directly
   against real human confidence ratings on the same stimuli.

## Task

Each stimulus is a `100x100` image of colored dots on a gray background (see `sample_stimuli/`).
The folder name `R-B-G` (e.g. `77-62-87` means R=77, B=62, G=87 — verified against actual per-channel
pixel counts, not the R-G-B order the name visually suggests) encodes the count of dots of each color;
the label is whichever channel has the most dots. Some conditions are easy (colors far apart, e.g.
`50-77-93`), others are much harder (colors close together), which is what makes this a genuine
perceptual-difficulty gradient rather than a trivially separable classification problem.

Train and validation stimuli are pre-split into disjoint condition folders (`random_train/`,
`random_validation/`) with **no overlapping images**, so reported accuracy reflects genuine
generalization to unseen stimulus conditions.

## CNN model

A small 3-block convolutional network (Conv-Pool x3 -> GlobalAveragePooling -> Dense -> Softmax),
trained with Adam and categorical cross-entropy, with early stopping on validation loss.
Global average pooling (rather than Flatten) is used deliberately: dot positions are randomized per
image, so the task is a translation-invariant counting/density problem, not a spatial one.
See `src/train_cnn.py` for the full architecture and training loop.

```
python src/train_cnn.py \
  --train-dir /path/to/random_train \
  --val-dir /path/to/random_validation \
  --epochs 20
```

### Results

Trained on the full dataset (117,999 train / 59,900 held-out validation images across ~180 balanced
R/G/B stimulus conditions, zero overlap between splits). Early stopping (patience 3 on val loss,
best weights restored) selected epoch 7 of 10 run:

- **Final held-out validation accuracy: 98.21%** (val loss 0.081)
- Per-class confusion matrix (rows = true, columns = predicted [R, G, B]):
  R: 19525 / 410 / 65 · G: 1 / 19998 / 1 · B: 1 / 596 / 19303 — errors concentrate in R/B confusion
  when green is close behind, not random noise.

See `results/metrics.json` (full per-epoch history), `results/training_curve.png`, and
`results/confusion_matrix.png`.

## CNN vs. human confidence (Expt3)

Does the CNN's confidence track human confidence on the *same* stimuli? `training + generating codes/old
test/` turned out to contain the one stimulus set that shares exact conditions with a fitted human
experiment (Expt3, 15 subjects): 7 of its 8 condition folders reproduce Expt3's condition/config grid
exactly (verified by reversing the indexing scheme in `modelRecovery/generateData.m` and matching it
against measured per-image pixel counts; the 8th folder, `98-84-48`, is silently corrupted — its images
don't contain the color counts its own name claims, so it's excluded rather than papered over).

`confidence_modeling/compare_cnn_human_confidence.py` pulls real human accuracy/confidence per condition
from `dataForModeling_Expt3.mat`. `confidence_modeling/evaluate_cnn_on_expt3.py` runs the trained CNN on
the same images and compares:

- The CNN is **at ceiling on this stimulus set** — 100% accuracy and softmax confidence saturated to
  exactly 1.0 on all 7 conditions (`old test/` is a narrower, generally easier grid than what the CNN
  trained on, so this is an out-of-distribution generalization check the model passed outright). That
  saturation means max-softmax-probability carries no usable variance here, so it isn't a meaningful
  confidence metric for this comparison — reported as undefined rather than forced into a number.
- Pre-softmax **logit margin** (top logit minus runner-up) doesn't saturate the same way and still
  separates "obviously easy" from "less obviously easy" trials. Correlated against real human confidence
  ratings across the 7 matched conditions: **r = 0.93, p = 0.002**.

See `confidence_modeling/cnn_vs_human_results.json` and `confidence_modeling/cnn_vs_human.png`.

## Confidence modeling (human behavior)

Three models of how confidence is computed from perceptual evidence are fit per-subject via BADS:

- **Bayes** (`comp_Bayes.m`) — confidence as the Bayesian posterior probability that the choice is correct,
  computed via numerical marginalization over possible true dot counts.
- **PE / peak-evidence** (`comp_PE.m`) — confidence as a thresholded function of the strongest single-color
  signal.
- **Diff** — confidence as a thresholded function of the evidence gap between the top two colors.

Each model is fit to two behavioral datasets (`Expt2`: 25 subjects, `Expt3`: 15 subjects) by maximum
likelihood via BADS, and a full **model-recovery analysis** (`confidence_modeling/modelRecovery/`)
confirms the fitting procedure can correctly distinguish the three models when the generating model is
known — a standard sanity check before trusting the fits to real subject data. Separately,
`bads_fitting/run_fitting.m` fits a 4-parameter decision (not confidence) model per subject across four
experiments, for 40+ individual optimization runs.

## Repo layout

```
src/train_cnn.py                                CNN training script
confidence_modeling/
  helperFunctions/                                MATLAB: Bayes/PE/Diff confidence model code
  modelRecovery/                                   MATLAB: model-recovery validation
  compare_cnn_human_confidence.py                 Extracts matched human accuracy/confidence from Expt3
  evaluate_cnn_on_expt3.py                        Runs the CNN on the same stimuli, correlates confidence
bads_fitting/run_fitting.m                        MATLAB: BADS-based decision-model fitting driver
sample_stimuli/                                   One example image per dominant-color class
results/                                          Training curves, confusion matrix, metrics.json, model
```

## Note on data

Stimulus images (~200k PNGs, several hundred MB) are not checked into this repo. `sample_stimuli/`
has one example per class; the full generation code and dataset are available on request.
