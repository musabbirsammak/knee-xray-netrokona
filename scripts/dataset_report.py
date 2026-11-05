#!/usr/bin/env python3
"""
Dataset facts, tables and figures for the Data in Brief manuscript.

Run on the RELEASE folder made by restructure_dataset.py:
    python dataset_report.py <release_dir> [--out manuscript_outputs/dataset] [--dedupe-threshold 8]

Writes (nothing is printed):
    dataset_facts.md              every number the manuscript needs, grouped by section
    table1_files.csv              files per class and format
    table_curation.csv            original collection -> removed (by reason) -> released
    table_image_properties.csv    image size, colour mode, bit depth, file size per class
    table_splits.csv              train / val / test counts per class
    removed_files.csv             every excluded original file and why
    fig_samples.png               random example images per class (with image IDs)
    fig_resolution.png            width x height of every image, by class
    report.log                    errors, if any

Requires: pip install pillow matplotlib
"""
import argparse
import csv
import json
import random
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ["normal", "osteoarthritis"]
SOURCE_TO_LABEL = {"Normal": "normal", "patient": "osteoarthritis"}
COLORS = {"normal": "#2a78d6", "osteoarthritis": "#eb6834"}   # validated categorical slots 1-2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(p, rows, fields=None):
    fields = fields or list(rows[0])
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "–"


def reason_of(status):
    s = status.lower()
    if s == "included":
        return "included"
    if "label conflict" in s:
        return "label conflict (same image in both classes)"
    if "near-identical" in s:
        return "near-identical copy"
    if "exact duplicate" in s:
        return "exact (byte-identical) copy"
    if "unreadable" in s:
        return "unreadable file"
    return "other"


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rel = a.release
    out = a.out
    images = read_csv(rel / "metadata" / "images.csv")
    mapping = read_csv(rel / "metadata" / "source_mapping.csv")
    dups = read_csv(rel / "metadata" / "duplicates.csv") if (rel / "metadata" / "duplicates.csv").exists() else []
    split_files = sorted((rel / "splits").glob("split_seed*.csv")) if (rel / "splits").exists() else []
    splits = read_csv(split_files[0]) if split_files else []

    by_label = defaultdict(list)
    for r in images:
        by_label[r["label"]].append(r)
    n_total = len(images)

    # ---------------- Table 1: files per class and format
    t1 = []
    for lab in LABELS:
        rows = by_label[lab]
        fmt = Counter(r["format"] for r in rows)
        for f in sorted(fmt):
            t1.append(dict(folder=f"images/{lab}/", label=lab, label_id=rows[0]["label_id"], format=f,
                           files=fmt[f], share_of_class=pct(fmt[f], len(rows))))
        t1.append(dict(folder=f"images/{lab}/", label=lab, label_id=rows[0]["label_id"], format="all",
                       files=len(rows), share_of_class=pct(len(rows), n_total) + " of dataset"))
    t1.append(dict(folder="", label="total", label_id="", format="all", files=n_total, share_of_class="100%"))
    write_csv(out / "table1_files.csv", t1)

    # ---------------- curation table
    orig = Counter(SOURCE_TO_LABEL.get(m["original_folder"], m["original_folder"]) for m in mapping)
    reasons = defaultdict(Counter)
    removed = []
    for m in mapping:
        lab = SOURCE_TO_LABEL.get(m["original_folder"], m["original_folder"])
        rs = reason_of(m["status"])
        reasons[rs][lab] += 1
        if rs != "included":
            removed.append(dict(original_folder=m["original_folder"], original_file_name=m["original_file_name"],
                                reason=rs, detail=m["status"].replace("excluded: ", "")))
    order = ["exact (byte-identical) copy", "near-identical copy", "label conflict (same image in both classes)",
             "unreadable file", "other"]
    cur = [dict(stage="Original collection", normal=orig["normal"], osteoarthritis=orig["osteoarthritis"],
                total=orig["normal"] + orig["osteoarthritis"])]
    for rs in order:
        if sum(reasons[rs].values()):
            cur.append(dict(stage=f"Removed: {rs}", normal=-reasons[rs]["normal"],
                            osteoarthritis=-reasons[rs]["osteoarthritis"], total=-sum(reasons[rs].values())))
    cur.append(dict(stage="Released", normal=len(by_label["normal"]), osteoarthritis=len(by_label["osteoarthritis"]),
                    total=n_total))
    write_csv(out / "table_curation.csv", cur)
    if removed:
        write_csv(out / "removed_files.csv", removed)

    # ---------------- image properties
    props = []
    for lab in LABELS + ["all"]:
        rows = images if lab == "all" else by_label[lab]
        w = [int(r["width"]) for r in rows]
        h = [int(r["height"]) for r in rows]
        mp = [int(r["width"]) * int(r["height"]) / 1e6 for r in rows]
        sz = [int(r["file_size_bytes"]) for r in rows]
        props.append(dict(
            label=lab, images=len(rows),
            width_min=min(w), width_median=int(statistics.median(w)), width_max=max(w),
            height_min=min(h), height_median=int(statistics.median(h)), height_max=max(h),
            megapixels_median=f"{statistics.median(mp):.2f}",
            colour_modes="; ".join(f"{k}: {v}" for k, v in Counter(r["mode"] for r in rows).most_common()),
            bit_depths="; ".join(f"{k}-bit: {v}" for k, v in Counter(r["bit_depth"] for r in rows).most_common()),
            file_size_median_kb=f"{statistics.median(sz) / 1024:.0f}",
            total_size_mb=f"{sum(sz) / 1024 ** 2:.1f}"))
    write_csv(out / "table_image_properties.csv", props)

    # ---------------- splits
    sp_rows = []
    if splits:
        sc = Counter((r["label"], r["split"]) for r in splits)
        for lab in LABELS:
            tot = sum(sc[(lab, s)] for s in ("train", "val", "test"))
            sp_rows.append(dict(label=lab, train=sc[(lab, "train")], val=sc[(lab, "val")], test=sc[(lab, "test")],
                                total=tot))
        sp_rows.append(dict(label="total", **{s: sum(r[s] for r in sp_rows) for s in ("train", "val", "test", "total")}))
        write_csv(out / "table_splits.csv", sp_rows)

    groups = Counter(r["duplicate_group"] for r in images if r["duplicate_group"])
    grouped_imgs = sum(groups.values())
    n_near_kept = sum(1 for d in dups if d["type"] == "near" and not d["file_a"].startswith("(excluded)")
                      and not d["file_b"].startswith("(excluded)"))

    # ---------------- figures
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                         "axes.labelcolor": INK})
    # resolution scatter
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=300)
    for lab in LABELS:
        rows = by_label[lab]
        ax.scatter([int(r["width"]) for r in rows], [int(r["height"]) for r in rows], s=10, alpha=0.55,
                   color=COLORS[lab], edgecolors="white", linewidths=0.3,
                   label=f"{lab} (n = {len(rows):,})", marker="o" if lab == "normal" else "^")
    ax.set_xlabel("Width (pixels)")
    ax.set_ylabel("Height (pixels)")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_resolution.png", facecolor="white")
    plt.close(fig)

    # sample grid
    rng = random.Random(a.seed)
    ncol = 6
    fig, axes = plt.subplots(2, ncol, figsize=(7.2, 3.3), dpi=300)
    for ri, lab in enumerate(LABELS):
        pick = rng.sample(by_label[lab], min(ncol, len(by_label[lab])))
        for ci in range(ncol):
            ax = axes[ri][ci]
            ax.axis("off")
            if ci < len(pick):
                with Image.open(rel / pick[ci]["file_path"]) as im:
                    im = im.convert("L")
                    im.thumbnail((400, 400))
                    ax.imshow(im, cmap="gray")
                ax.set_title(pick[ci]["image_id"], fontsize=7, color=MUTED, pad=2)
        axes[ri][0].text(-0.08, 0.5, f"({'ab'[ri]}) {lab}", transform=axes[ri][0].transAxes, rotation=90,
                         ha="right", va="center", fontsize=9, color=INK, fontweight="bold")
    fig.tight_layout(w_pad=0.4, h_pad=0.6)
    fig.savefig(out / "fig_samples.png", facecolor="white")
    plt.close(fig)

    # ---------------- facts file
    p_all = props[-1]
    p = {r["label"]: r for r in props}
    fmt_all = Counter(r["format"] for r in images)
    L = []
    L.append("# Dataset facts for the manuscript\n")
    L.append("Generated by dataset_report.py from the release folder. Copy the numbers into the manuscript.\n")
    L.append("## Headline (abstract, specifications table)\n")
    L.append(f"- Released images: **{n_total:,}** — normal **{len(by_label['normal']):,}** "
             f"({pct(len(by_label['normal']), n_total)}), osteoarthritis **{len(by_label['osteoarthritis']):,}** "
             f"({pct(len(by_label['osteoarthritis']), n_total)})")
    L.append("- Formats: " + ", ".join(f"{v:,} {k}" for k, v in fmt_all.most_common()))
    L.append(f"- Total size: {p_all['total_size_mb']} MB\n")
    L.append("## Curation (methods: de-duplication)\n")
    L.append(f"- Original collection: {cur[0]['total']:,} files (normal {cur[0]['normal']:,}, "
             f"osteoarthritis {cur[0]['osteoarthritis']:,})")
    for c in cur[1:-1]:
        L.append(f"- {c['stage']}: {-c['total']:,} (normal {-c['normal']:,}, osteoarthritis {-c['osteoarthritis']:,})")
    L.append(f"- Released after de-duplication: {n_total:,} "
             f"({pct(cur[0]['total'] - n_total, cur[0]['total'])} of the original files removed)")
    L.append(f"- Near-identical = 256-bit difference-hash distance ≤ {a.dedupe_threshold}; one copy kept per group "
             f"(PNG preferred, then the larger image)")
    L.append(f"- Remaining look-alike pairs kept but grouped for splitting: {n_near_kept} pairs in "
             f"{len(groups)} groups ({grouped_imgs} images)\n")
    L.append("## Image properties (data description)\n")
    for lab in LABELS + ["all"]:
        r = p[lab]
        L.append(f"- {lab}: width {r['width_min']}–{r['width_max']} px (median {r['width_median']}), "
                 f"height {r['height_min']}–{r['height_max']} px (median {r['height_median']}); "
                 f"modes {r['colour_modes']}; {r['bit_depths']}; median file {r['file_size_median_kb']} KB")
    L.append("")
    if sp_rows:
        L.append(f"## Split ({split_files[0].name}; methods: technical validation)\n")
        for r in sp_rows:
            L.append(f"- {r['label']}: train {r['train']}, val {r['val']}, test {r['test']} (total {r['total']})")
        L.append("- Look-alike groups are kept within one subset.\n")
    conflicts = [r for r in removed if r["reason"].startswith("label conflict")]
    if conflicts:
        L.append("## Label conflicts removed\n")
        for r in conflicts:
            L.append(f"- {r['original_folder']}/{r['original_file_name']}")
        L.append("")
    L.append("## Files written\n")
    for f in sorted(out.iterdir()):
        if f.name != "dataset_facts.md":
            L.append(f"- {f.name}")
    (out / "dataset_facts.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("release", type=Path)
    ap.add_argument("--out", type=Path, default=Path("manuscript_outputs/dataset"))
    ap.add_argument("--dedupe-threshold", type=int, default=8, help="value used with --dedupe-near")
    ap.add_argument("--seed", type=int, default=7, help="seed for choosing sample images")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "report.log"
    try:
        main(args)
        log.write_text("OK\n", encoding="utf-8")
    except Exception:
        log.write_text(traceback.format_exc(), encoding="utf-8")
        sys.exit(f"dataset_report failed - see {log}")
