"""
Train a CNN to classify the dominant color (R/G/B) in mAFC dot stimuli.

Each stimulus image lives in a folder named "R-B-G" (e.g. "77-62-87" means
R=77, B=62, G=87 -- verified against actual pixel counts in the PNGs, NOT
the R-G-B order the folder name visually suggests). The label is whichever
channel has the most dots -- this mirrors the perceptual decision human
subjects make in the behavioral task ("which color is most prevalent?").

Train and validation images come from separate, pre-split directories
(random_train/ and random_validation/) so there is no train/test leakage.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

SEED = 42
IMG_SIZE = (100, 100)
CLASS_NAMES = ["R", "G", "B"]


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def collect_examples(root_dir, cap_per_folder=None):
    """Walk root_dir/<R-G-B>/stimulus*.png and return (paths, labels)."""
    root = Path(root_dir)
    paths, labels = [], []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        try:
            # Folder name order is R-B-G, not R-G-B (verified against actual
            # per-channel dot counts in the PNGs).
            r, b, g = (int(v) for v in folder.name.split("-"))
        except ValueError:
            continue
        dominant = max({"R": r, "G": g, "B": b}, key=lambda k: {"R": r, "G": g, "B": b}[k])
        label = CLASS_NAMES.index(dominant)

        files = sorted(folder.glob("stimulus*.png"))
        if cap_per_folder is not None:
            rng = random.Random(SEED)
            rng.shuffle(files)
            files = files[:cap_per_folder]

        for f in files:
            paths.append(str(f))
            labels.append(label)

    return paths, labels


def make_dataset(paths, labels, batch_size, shuffle):
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes=3)

    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        # Normalize to [0, 1] BEFORE resizing: tf.image.resize already returns
        # float32 in the original [0, 255] range, so calling
        # convert_image_dtype(..., tf.float32) afterwards is a same-dtype
        # no-op and silently skips the rescale, feeding the network raw
        # 0-255 pixel values (this was the actual reason training never
        # moved off chance-level accuracy).
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, IMG_SIZE)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels_onehot))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model():
    # This is fundamentally a counting/density task: dots are placed at random
    # positions per image, so the label depends on per-color pixel counts, not
    # spatial layout. GlobalAveragePooling2D (translation-invariant, and closer
    # to "average evidence per color channel") is a much better fit for that
    # than Flatten()+Dense, which bakes in sensitivity to absolute dot position
    # and needs far more data/epochs to learn to ignore it.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMG_SIZE, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-cap-per-folder", type=int, default=None)
    parser.add_argument("--val-cap-per-folder", type=int, default=None)
    args = parser.parse_args()

    set_seeds()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting file lists...")
    train_paths, train_labels = collect_examples(args.train_dir, args.train_cap_per_folder)
    val_paths, val_labels = collect_examples(args.val_dir, args.val_cap_per_folder)
    print(f"train: {len(train_paths)} images, val: {len(val_paths)} images")
    print("train class counts:", np.bincount(train_labels))
    print("val class counts:", np.bincount(val_labels))

    train_ds = make_dataset(train_paths, train_labels, args.batch_size, shuffle=True)
    val_ds = make_dataset(val_paths, val_labels, args.batch_size, shuffle=False)

    model = build_model()
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    history = model.fit(
        train_ds, epochs=args.epochs, validation_data=val_ds, callbacks=[early_stop]
    )

    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Final held-out validation accuracy: {val_acc * 100:.2f}%")

    model.save(out_dir / "model.keras")
    print(f"Saved trained model to {out_dir / 'model.keras'}")

    # Predictions for confusion matrix
    y_true = np.array(val_labels)
    y_pred_probs = model.predict(val_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    cm = confusion_matrix(y_true, y_pred)

    # Save metrics
    metrics = {
        "final_val_accuracy": float(val_acc),
        "final_val_loss": float(val_loss),
        "train_images": len(train_paths),
        "val_images": len(val_paths),
        "epochs": len(history.history["accuracy"]),
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Training curve plot
    epochs_range = range(1, len(history.history["accuracy"]) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history.history["accuracy"], "o-", label="Train Accuracy")
    plt.plot(epochs_range, history.history["val_accuracy"], "o-", label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs. Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(out_dir / "training_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Confusion matrix plot
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (held-out validation set)")
    plt.colorbar()
    plt.xticks(range(3), CLASS_NAMES)
    plt.yticks(range(3), CLASS_NAMES)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
