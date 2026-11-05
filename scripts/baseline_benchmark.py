#!/usr/bin/env python3
"""
Baseline benchmark (technical validation) for the de-duplicated knee X-ray release.

Fine-tunes six ImageNet-pretrained CNNs on the split made by restructure_dataset.py and writes
everything the manuscript needs to files. Nothing is printed; progress goes to benchmark.log.

    python baseline_benchmark.py <release_dir> [--out manuscript_outputs/benchmark]
           [--split splits/split_seed42.csv] [--models ResNet50,VGG16,...] [--seeds 42]

Use several seeds (e.g. --seeds 42,43,44) to report mean ± SD; each seed retrains every model.

Writes:
    table2_baseline.csv          manuscript table (mean ± SD when several seeds)
    metrics_per_run.csv          every model x seed: accuracy, macro P/R/F1, AUC, sensitivity,
                                 specificity, TN/FP/FN/TP, epochs, training time, parameters
    benchmark_facts.md           protocol and results written out for the methods section
    fig_roc.png                  ROC curves of all models (first seed)
    fig_confusion_matrices.png   confusion matrices of all models (first seed)
    fig_training_curves.png      training/validation accuracy per model (first seed)
    predictions/<model>_seed<s>.csv   test-set probabilities and predictions
    history/<model>_seed<s>.csv       per-epoch loss/accuracy
    run_info.json                versions, hardware, full configuration
    benchmark.log                progress and errors

Requires: pip install tensorflow scikit-learn pandas matplotlib
(Apple Silicon GPU, optional: pip install tensorflow-metal)
"""
import argparse
import gc
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

MODELS = ["ResNet50", "VGG16", "DenseNet121", "MobileNetV2", "Xception", "InceptionResNetV2"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # validated categorical slots 1-6
DASHES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))]
CM_CMAP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CLASS_NAMES = ["normal", "osteoarthritis"]

CFG = dict(img_size=224, batch_size=32, warmup_epochs=5, warmup_lr=1e-3, finetune_epochs=30,
           finetune_lr=1e-5, unfreeze_fraction=0.3, dropout=0.3, early_stop_patience=5,
           reduce_lr_patience=2, reduce_lr_factor=0.2, min_lr=1e-7,
           augmentation=dict(horizontal_flip=True, rotation_deg=40, translation=0.1, zoom=0.1, contrast=0.2))


class Silent:
    """Send everything written to stdout/stderr (Python and TensorFlow's C++ logs) to the log file."""

    def __init__(self, log_path):
        self.f = open(log_path, "a", buffering=1, encoding="utf-8")
        self.saved = (os.dup(1), os.dup(2))
        os.dup2(self.f.fileno(), 1)
        os.dup2(self.f.fileno(), 2)
        sys.stdout = sys.stderr = self.f

    def restore(self):
        os.dup2(self.saved[0], 1)
        os.dup2(self.saved[1], 2)
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__


def log(msg):
    sys.stdout.write(time.strftime("%H:%M:%S ") + msg + "\n")
    sys.stdout.flush()


def apps(tf):
    A = tf.keras.applications
    return {
        "ResNet50": (A.ResNet50, A.resnet50.preprocess_input),
        "VGG16": (A.VGG16, A.vgg16.preprocess_input),
        "DenseNet121": (A.DenseNet121, A.densenet.preprocess_input),
        "MobileNetV2": (A.MobileNetV2, A.mobilenet_v2.preprocess_input),
        "Xception": (A.Xception, A.xception.preprocess_input),
        "InceptionResNetV2": (A.InceptionResNetV2, A.inception_resnet_v2.preprocess_input),
    }


def make_ds(tf, df, root, preprocess, training, seed):
    S = CFG["img_size"]
    paths = [str(root / p) for p in df["file_path"]]
    labels = df["label_id"].astype("int32").tolist()
    aug = CFG["augmentation"]
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(aug["rotation_deg"] / 360, fill_mode="constant", seed=seed),
        tf.keras.layers.RandomTranslation(aug["translation"], aug["translation"], fill_mode="constant", seed=seed),
        tf.keras.layers.RandomZoom(aug["zoom"], fill_mode="constant", seed=seed),
        tf.keras.layers.RandomContrast(aug["contrast"], seed=seed),
    ])

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, dtype=tf.float32, expand_animations=False)
        img = tf.image.resize(img, (S, S), antialias=True) * 255.0
        img.set_shape((S, S, 3))
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(CFG["batch_size"])
    if training:
        ds = ds.map(lambda x, y: (tf.clip_by_value(augment(x, training=True), 0, 255), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def build(tf, ctor, weights):
    S = CFG["img_size"]
    base = ctor(include_top=False, weights=weights, input_shape=(S, S, 3))
    base.trainable = False
    inp = tf.keras.Input((S, S, 3))
    x = base(inp, training=False)  # BatchNorm stays in inference mode, also during fine-tuning
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(CFG["dropout"])(x)
    out = tf.keras.layers.Dense(2, activation="softmax")(x)
    return tf.keras.Model(inp, out), base


def run_one(tf, name, seed, frames, root, weights, out):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    ctor, pre = apps(tf)[name]
    tr = make_ds(tf, frames["train"], root, pre, True, seed)
    va = make_ds(tf, frames["val"], root, pre, False, seed)
    te = make_ds(tf, frames["test"], root, pre, False, seed)
    y_tr = frames["train"]["label_id"].astype(int).to_numpy()
    cw = {c: len(y_tr) / (2 * max(1, (y_tr == c).sum())) for c in (0, 1)}

    model, base = build(tf, ctor, weights)
    t0 = time.time()
    model.compile(tf.keras.optimizers.Adam(CFG["warmup_lr"]), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(tr, validation_data=va, epochs=CFG["warmup_epochs"], class_weight=cw, verbose=0)

    n_layers = len(base.layers)
    cut = int(n_layers * (1 - CFG["unfreeze_fraction"]))
    base.trainable = True
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= cut and not isinstance(layer, tf.keras.layers.BatchNormalization)
    model.compile(tf.keras.optimizers.Adam(CFG["finetune_lr"]), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = [tf.keras.callbacks.EarlyStopping("val_loss", patience=CFG["early_stop_patience"], restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=CFG["reduce_lr_factor"],
                                                patience=CFG["reduce_lr_patience"], min_lr=CFG["min_lr"])]
    h2 = model.fit(tr, validation_data=va, epochs=CFG["finetune_epochs"], class_weight=cw, callbacks=cbs, verbose=0)
    train_s = time.time() - t0

    prob = model.predict(te, verbose=0)[:, 1]
    y = frames["test"]["label_id"].astype(int).to_numpy()
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    m = dict(model=name, seed=seed, accuracy=accuracy_score(y, pred), precision_macro=p, recall_macro=r,
             f1_macro=f1, auc=roc_auc_score(y, prob) if len(set(y)) == 2 else float("nan"),
             sensitivity=tp / (tp + fn) if tp + fn else float("nan"),
             specificity=tn / (tn + fp) if tn + fp else float("nan"),
             TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
             epochs_warmup=len(h1.history["loss"]), epochs_finetune=len(h2.history["loss"]),
             unfrozen_layers=sum(l.trainable for l in base.layers), backbone_layers=n_layers,
             params_total=int(model.count_params()), train_seconds=round(train_s, 1))

    pd.DataFrame(dict(image_id=frames["test"]["image_id"], label_id=y, prob_osteoarthritis=prob, pred=pred)) \
        .to_csv(out / "predictions" / f"{name}_seed{seed}.csv", index=False)
    hist = {k: h1.history[k] + h2.history[k] for k in ("loss", "accuracy", "val_loss", "val_accuracy")}
    hd = pd.DataFrame(hist)
    hd.insert(0, "epoch", range(1, len(hd) + 1))
    hd.insert(1, "stage", ["warmup"] * len(h1.history["loss"]) + ["finetune"] * len(h2.history["loss"]))
    hd.to_csv(out / "history" / f"{name}_seed{seed}.csv", index=False)
    del model, base, tr, va, te
    tf.keras.backend.clear_session()
    return m, (y, prob), hd


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def figures(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MaxNLocator
    from sklearn.metrics import roc_curve
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK})
    names = list(results)

    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=300)
    for i, n in enumerate(names):
        y, prob = results[n]["scores"]
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, color=COLORS[i % 6], ls=DASHES[i % 6], lw=2,
                label=f"{n} (AUC = {results[n]['metrics']['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "fig_roc.png", facecolor="white"); plt.close(fig)

    cmap = LinearSegmentedColormap.from_list("blue", CM_CMAP)
    cols = 3; rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 2.5 * rows), dpi=300, squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= len(names):
            ax.axis("off"); continue
        m = results[names[k]]["metrics"]
        cm = [[m["TN"], m["FP"]], [m["FN"], m["TP"]]]
        tot = [sum(r) for r in cm]
        ax.imshow([[cm[i][j] / tot[i] for j in range(2)] for i in range(2)], cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                frac = cm[i][j] / tot[i]
                ax.text(j, i, f"{cm[i][j]}\n{frac:.1%}", ha="center", va="center", fontsize=8,
                        color="white" if frac > 0.55 else INK)
        ax.set_xticks([0, 1], ["normal", "OA"], fontsize=8); ax.set_yticks([0, 1], ["normal", "OA"], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8)
        ax.set_title(names[k], fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0, colors=MUTED)
    fig.tight_layout(); fig.savefig(out / "fig_confusion_matrices.png", facecolor="white"); plt.close(fig)

    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 2.4 * rows), dpi=300, squeeze=False, sharey=True)
    for k, ax in enumerate(axes.flat):
        if k >= len(names):
            ax.axis("off"); continue
        hd = results[names[k]]["history"]
        ax.plot(hd["epoch"], hd["accuracy"], color=COLORS[0], lw=2, label="training")
        ax.plot(hd["epoch"], hd["val_accuracy"], color=COLORS[1], lw=2, ls="--", label="validation")
        n_w = int((hd["stage"] == "warmup").sum())
        ax.axvline(n_w + 0.5, color=MUTED, lw=1, ls=":")
        ax.set_title(names[k], fontsize=9); ax.set_xlabel("Epoch", fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        style(ax)
    axes[0][0].set_ylabel("Accuracy", fontsize=8)
    axes[0][0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.text(0.5, 0.005, "Dotted line: end of warm-up (backbone frozen) and start of fine-tuning",
             ha="center", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out / "fig_training_curves.png", facecolor="white")
    plt.close(fig)


def main(a, out):
    import pandas as pd
    import tensorflow as tf
    import sklearn
    tf.get_logger().setLevel("ERROR")
    try:
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
    except Exception:
        pass

    (out / "predictions").mkdir(exist_ok=True)
    (out / "history").mkdir(exist_ok=True)
    root = a.release
    split_path = a.split if a.split.is_absolute() else root / a.split
    df = pd.read_csv(split_path)
    meta = pd.read_csv(root / "metadata" / "images.csv")[["image_id", "label_id"]]
    df = df.merge(meta, on="image_id", how="left")
    frames = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    models = [m.strip() for m in a.models.split(",")]
    seeds = [int(s) for s in str(a.seeds).split(",")]
    weights = None if a.no_pretrained else "imagenet"
    log(f"split {split_path.name}: " + ", ".join(f"{k} {len(v)}" for k, v in frames.items()))
    log(f"devices: {[d.name for d in tf.config.list_physical_devices()]}")

    # resumable: runs already in metrics_per_run.csv are skipped
    mp = out / "metrics_per_run.csv"
    rows = pd.read_csv(mp).to_dict("records") if mp.exists() else []
    done = {(r["model"], int(r["seed"])) for r in rows}
    for seed in seeds:
        for name in models:
            if (name, seed) in done:
                log(f"skipping {name} seed {seed} (already done)")
                continue
            log(f"training {name} seed {seed} …")
            m, _, _ = run_one(tf, name, seed, frames, root, weights, out)
            rows.append(m)
            pd.DataFrame(rows).to_csv(mp, index=False)
            log(f"  done: acc {m['accuracy']:.4f}  auc {m['auc']:.4f}  ({m['train_seconds']:.0f} s)")
            gc.collect()

    rows = [r for r in rows if r["model"] in models and int(r["seed"]) in seeds]
    first = {}
    for name in models:
        r0 = next(r for r in rows if r["model"] == name and int(r["seed"]) == seeds[0])
        pr = pd.read_csv(out / "predictions" / f"{name}_seed{seeds[0]}.csv")
        hd = pd.read_csv(out / "history" / f"{name}_seed{seeds[0]}.csv")
        first[name] = dict(metrics=r0, scores=(pr["label_id"].to_numpy(), pr["prob_osteoarthritis"].to_numpy()),
                           history=hd)

    res = pd.DataFrame(rows)
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "auc", "sensitivity", "specificity"]
    t2 = []
    for name in models:
        g = res[res["model"] == name]
        r = {"Model": name}
        for k in keys:
            r[k] = f"{g[k].mean():.4f}" if len(g) == 1 else f"{g[k].mean():.4f} ± {g[k].std(ddof=1):.4f}"
        f = first[name]["metrics"]
        r.update(TN=int(f["TN"]), FP=int(f["FP"]), FN=int(f["FN"]), TP=int(f["TP"]),
                 params_M=f"{f['params_total'] / 1e6:.1f}")
        t2.append(r)
    t2 = pd.DataFrame(t2)
    t2.to_csv(out / "table2_baseline.csv", index=False)
    figures(first, out)

    info = dict(config=CFG, models=models, seeds=seeds, weights=weights or "random init",
                split_file=str(split_path.relative_to(root) if split_path.is_relative_to(root) else split_path),
                subset_sizes={k: {c: int((v["label_id"] == i).sum()) for i, c in enumerate(CLASS_NAMES)}
                              for k, v in frames.items()},
                python=platform.python_version(), tensorflow=tf.__version__, sklearn=sklearn.__version__,
                pandas=pd.__version__, platform=platform.platform(), processor=platform.processor(),
                devices=[d.name for d in tf.config.list_physical_devices()],
                date=time.strftime("%Y-%m-%d"))
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    sz = info["subset_sizes"]
    best = t2.iloc[t2["accuracy"].str.slice(0, 6).astype(float).idxmax()]
    acc = t2["accuracy"].str.slice(0, 6).astype(float)
    L = ["# Benchmark facts for the manuscript\n",
         "## Protocol (methods: technical validation)\n",
         f"- Split: {info['split_file']} — train {sum(sz['train'].values())} "
         f"(normal {sz['train']['normal']}, OA {sz['train']['osteoarthritis']}), val {sum(sz['val'].values())} "
         f"(normal {sz['val']['normal']}, OA {sz['val']['osteoarthritis']}), test {sum(sz['test'].values())} "
         f"(normal {sz['test']['normal']}, OA {sz['test']['osteoarthritis']}); look-alike groups kept in one subset",
         f"- Models: {', '.join(models)}; weights: {info['weights']}",
         f"- Input {CFG['img_size']}×{CFG['img_size']} RGB, each model's own Keras preprocess_input",
         f"- Augmentation (training only): horizontal flip, rotation ±{CFG['augmentation']['rotation_deg']}°, "
         f"translation ±{int(CFG['augmentation']['translation'] * 100)}%, zoom ±{int(CFG['augmentation']['zoom'] * 100)}%, "
         f"contrast ±{int(CFG['augmentation']['contrast'] * 100)}%",
         f"- Head: global average pooling, dropout {CFG['dropout']}, 2-unit softmax; balanced class weights",
         f"- Warm-up: backbone frozen, Adam lr {CFG['warmup_lr']:g}, {CFG['warmup_epochs']} epochs",
         f"- Fine-tuning: top {int(CFG['unfreeze_fraction'] * 100)}% of backbone layers unfrozen (BatchNorm frozen), "
         f"Adam lr {CFG['finetune_lr']:g}, up to {CFG['finetune_epochs']} epochs, early stopping on val loss "
         f"(patience {CFG['early_stop_patience']}, best weights restored), ReduceLROnPlateau "
         f"(factor {CFG['reduce_lr_factor']}, patience {CFG['reduce_lr_patience']})",
         f"- Loss: sparse categorical cross-entropy; batch size {CFG['batch_size']}; seeds: {', '.join(map(str, seeds))}",
         f"- Software: Python {info['python']}, TensorFlow {info['tensorflow']}, scikit-learn {info['sklearn']}; "
         f"hardware: {info['platform']}; devices {', '.join(info['devices'])}\n",
         "## Results (test set)\n",
         f"- Accuracy range: {acc.min():.4f}–{acc.max():.4f}; highest: {best['Model']} ({best['accuracy']})"]
    for _, r in t2.iterrows():
        L.append(f"- {r['Model']}: acc {r['accuracy']}, macro F1 {r['f1_macro']}, AUC {r['auc']}, "
                 f"sensitivity {r['sensitivity']}, specificity {r['specificity']} "
                 f"(TN {r['TN']}, FP {r['FP']}, FN {r['FN']}, TP {r['TP']})")
    L += ["", "Positive class = osteoarthritis. Precision/recall/F1 are macro-averaged.",
          "Full numbers: table2_baseline.csv, metrics_per_run.csv."]
    (out / "benchmark_facts.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("finished")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("release", type=Path)
    ap.add_argument("--out", type=Path, default=Path("manuscript_outputs/benchmark"))
    ap.add_argument("--split", type=Path, default=Path("splits/split_seed42.csv"))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--no-pretrained", action="store_true", help=argparse.SUPPRESS)  # quick pipeline test only
    ap.add_argument("--quick-test", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.quick_test:
        CFG.update(warmup_epochs=1, finetune_epochs=1, img_size=96)
    args.out.mkdir(parents=True, exist_ok=True)
    logp = args.out / "benchmark.log"
    s = Silent(logp)
    try:
        main(args, args.out)
        ok = True
    except Exception:
        log(traceback.format_exc())
        ok = False
    s.restore()
    if not ok:
        sys.exit(f"baseline_benchmark failed - see {logp}")
