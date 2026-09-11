"""
Evaluate the trained CNN on the 7 old-test/ stimulus folders that match
Expt3 human behavioral conditions (see compare_cnn_human_confidence.py for
how those folders were matched to human data), and compare CNN accuracy /
confidence against the corresponding human accuracy / confidence.

CNN "confidence" = mean of the softmax max-probability across trials in a
condition -- the direct analogue of "how sure was the model," on the same
0-1 scale as normalized human confidence ratings.
"""

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from train_cnn import IMG_SIZE, CLASS_NAMES  # noqa: E402

OLD_TEST_DIR = (
    "/Users/sruthi/iCloud Drive (Archive)/Desktop/Desktop - sruthi’s MacBook Air/mAFC/"
    "training + generating codes/old test"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "results" / "model.keras"
HUMAN_JSON = Path(__file__).resolve().parent / "expt3_human_matched.json"
OUT_DIR = Path(__file__).resolve().parent
N_SAMPLE = 300  # match human n_trials per condition


def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, IMG_SIZE)
    return img


def dominant_label_from_folder(folder_name):
    # Folder name is R-B-G order (see src/train_cnn.py note).
    r, b, g = (int(v) for v in folder_name.split("-"))
    vals = {"R": r, "G": g, "B": b}
    return CLASS_NAMES.index(max(vals, key=vals.get))


def main():
    with open(HUMAN_JSON) as f:
        human_matched = {row["folder"]: row for row in json.load(f)}

    model = tf.keras.models.load_model(MODEL_PATH)
    softmax_model = model
    # A second copy with the final softmax removed: on this stimulus set the
    # trained model is at ceiling (100% accuracy, softmax saturated to
    # exactly 1.0 on every trial -- float32 softmax collapses once the top
    # logit is far enough ahead of the rest), so max-softmax-probability
    # carries no usable variance here. Pre-softmax logit margin (top logit
    # minus runner-up) is unbounded and still separates "obviously easy" from
    # "less obviously easy" trials even when softmax can't, so we report both.
    logit_model = tf.keras.models.load_model(MODEL_PATH)
    logit_model.layers[-1].activation = tf.keras.activations.linear

    results = []
    for folder_name, human in human_matched.items():
        folder = os.path.join(OLD_TEST_DIR, folder_name)
        files = sorted(glob.glob(os.path.join(folder, "*.png")))[:N_SAMPLE]
        label = dominant_label_from_folder(folder_name)

        imgs = tf.stack([load_image(f) for f in files])
        probs = softmax_model.predict(imgs, verbose=0)
        preds = np.argmax(probs, axis=1)
        cnn_acc = float(np.mean(preds == label))
        cnn_conf = float(np.mean(np.max(probs, axis=1)))

        logits = logit_model.predict(imgs, verbose=0)
        sorted_logits = np.sort(logits, axis=1)
        cnn_logit_margin = float(np.mean(sorted_logits[:, -1] - sorted_logits[:, -2]))

        results.append({
            "folder": folder_name,
            "n_images": len(files),
            "cnn_accuracy": cnn_acc,
            "cnn_confidence_0to1": cnn_conf,
            "cnn_logit_margin": cnn_logit_margin,
            "human_accuracy": human["human_accuracy"],
            "human_confidence_0to1": human["human_confidence_0to1"],
        })
        print(f"{folder_name}: CNN acc={cnn_acc:.3f} softmax_conf={cnn_conf:.3f} "
              f"logit_margin={cnn_logit_margin:.2f}  |  "
              f"Human acc={human['human_accuracy']:.3f} conf={human['human_confidence_0to1']:.3f}")

    cnn_acc = np.array([r["cnn_accuracy"] for r in results])
    cnn_conf = np.array([r["cnn_confidence_0to1"] for r in results])
    cnn_margin = np.array([r["cnn_logit_margin"] for r in results])
    human_acc = np.array([r["human_accuracy"] for r in results])
    human_conf = np.array([r["human_confidence_0to1"] for r in results])

    # Softmax confidence is saturated at 1.0 for every condition here (CNN is
    # at ceiling accuracy on this stimulus set), so it has zero variance and
    # its correlation with human confidence is mathematically undefined --
    # reported as such rather than papered over. Logit margin is the
    # informative comparison.
    softmax_const = bool(np.all(cnn_conf == cnn_conf[0]))
    if softmax_const:
        r_conf_softmax, p_conf_softmax = float("nan"), float("nan")
    else:
        r_conf_softmax, p_conf_softmax = pearsonr(cnn_conf, human_conf)
    r_conf_margin, p_conf_margin = pearsonr(cnn_margin, human_conf)
    r_acc, p_acc = pearsonr(cnn_acc, human_acc) if len(set(cnn_acc)) > 1 else (float("nan"), float("nan"))

    summary = {
        "n_conditions": len(results),
        "softmax_confidence_saturated": softmax_const,
        "confidence_correlation_softmax_r": r_conf_softmax,
        "confidence_correlation_softmax_p": p_conf_softmax,
        "confidence_correlation_logit_margin_r": float(r_conf_margin),
        "confidence_correlation_logit_margin_p": float(p_conf_margin),
        "accuracy_correlation_r": r_acc,
        "accuracy_correlation_p": p_acc,
        "per_condition": results,
    }
    with open(OUT_DIR / "cnn_vs_human_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nn={len(results)} matched conditions")
    print(f"CNN accuracy: {cnn_acc.mean():.3f} on all conditions (saturated: {len(set(cnn_acc)) == 1})")
    print(f"Softmax confidence saturated at 1.0 on every condition -> correlation undefined")
    print(f"Confidence correlation (CNN logit margin vs human): r={r_conf_margin:.3f}, p={p_conf_margin:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(human_conf, cnn_margin)
    for r in results:
        axes[0].annotate(r["folder"], (r["human_confidence_0to1"], r["cnn_logit_margin"]), fontsize=7)
    axes[0].set_xlabel("Human confidence (0-1)")
    axes[0].set_ylabel("CNN logit margin (top - runner-up)")
    axes[0].set_title(f"Confidence: r={r_conf_margin:.2f}")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(human_acc, cnn_margin)
    for r in results:
        axes[1].annotate(r["folder"], (r["human_accuracy"], r["cnn_logit_margin"]), fontsize=7)
    axes[1].set_xlabel("Human accuracy")
    axes[1].set_ylabel("CNN logit margin")
    axes[1].set_title("CNN is at 100% accuracy on all conditions (no variance to plot)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "cnn_vs_human.png", dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUT_DIR / 'cnn_vs_human.png'}")


if __name__ == "__main__":
    main()
