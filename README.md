# Multi-Alternative Forced Choice (mAFC) Color-Numerosity Task: CNN and Confidence Modeling

## Overview

This repository contains code and analysis for a three-way color-dominance perceptual decision task in which subjects identify the most prevalent color (red, green, or blue) in dot-pattern stimuli. It includes three main components:

1. **CNN classifier** (`src/train_cnn.py`) — trained end-to-end on raw pixels to perform 3-way classification
2. **Confidence modeling** (`confidence_modeling/`) — fits three competing computational models to human confidence ratings via Bayesian Adaptive Direct Search (BADS) optimization, with model-recovery validation
3. **CNN-human confidence comparison** (`confidence_modeling/evaluate_cnn_on_expt3.py`) — compares learned model confidence against empirical human ratings

## Task Description

Stimuli are 100×100 pixel images containing colored dots on a gray background. The dominant color (highest dot count) defines the ground truth label. Conditions vary in difficulty based on color separability, creating a continuous perceptual difficulty gradient. Training and validation sets are split by condition (no image overlap), ensuring genuine generalization assessment.

## CNN Architecture and Results

The model uses a 3-block convolutional network (Conv-Pool × 3 → GlobalAveragePooling → Dense → Softmax) with global average pooling to exploit translation invariance. Training on 117,999 images with early stopping (patience=3) yielded **98.21% held-out validation accuracy** across ~180 balanced conditions.

See `results/` for detailed per-epoch history, training curves, and confusion matrices.

## Human Confidence Modeling

Three models compete to explain how confidence is derived from perceptual evidence:

- **Bayesian model**: confidence = posterior probability of correct choice via marginalization over possible true counts
- **Peak-evidence (PE) model**: confidence = thresholded function of the strongest color signal
- **Difference (Diff) model**: confidence = thresholded function of evidence gap between top two colors

Models are fit per-subject to two datasets (Expt2: *n*=25, Expt3: *n*=15) via maximum-likelihood BADS optimization. Model-recovery analysis confirms the fitting procedure can correctly distinguish generating models.

See `confidence_modeling/cnn_vs_human_results.json`.

## Repository Structure

```
src/train_cnn.py                               CNN training script
confidence_modeling/
  helperFunctions/                             Confidence model implementations
  modelRecovery/                               Model-recovery validation
  evaluate_cnn_on_expt3.py                     CNN evaluation on human stimulus set
bads_fitting/run_fitting.m                     Decision model fitting driver
sample_stimuli/                                Representative example images
results/                                       Metrics, curves, trained weights
```

## Data

Stimulus images (~200k PNGs) are not included in this repository. A single example per class is provided in `sample_stimuli/`. Full datasets and generation code available upon request.
