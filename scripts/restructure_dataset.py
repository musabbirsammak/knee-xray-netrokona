#!/usr/bin/env python3
"""
Restructure the knee X-ray dataset into a clean, documented release layout.

The script COPIES files (original bytes, original format) into a new folder;
the source folder is never modified.

Source layout expected (as collected):
    <source>/Normal/normal (1).jpg ... normal (1242).png
    <source>/patient/PATIENT (1).jpg ... PATIENT (910).png

Output layout produced:
    <output>/
    ├── README.md                  dataset description, generated with the real counts
    ├── LICENSE.txt                CC BY 4.0 notice
    ├── SHA256SUMS.txt             checksums of every image (verify: sha256sum -c SHA256SUMS.txt)
    ├── folder-structure.txt       listing of the release
    ├── images/
    │   ├── normal/                normal_0001.jpg, normal_0002.png, ...
    │   └── osteoarthritis/        oa_0001.jpg, oa_0002.png, ...
    ├── metadata/
    │   ├── images.csv             one row per image: id, path, label, format, size, mode, checksum, group
    │   ├── source_mapping.csv     new file name <-> original file name (provenance)
    │   └── duplicates.csv         exact and near-duplicate pairs found (for review)
    └── splits/
        └── split_seed<seed>.csv   stratified train/val/test split; near-duplicates kept in the same subset

Usage:
    python restructure_dataset.py <source> <output> [--dry-run] [--dedupe]
                                  [--split 0.70,0.15,0.15] [--seed 42] [--no-split]
                                  [--near-threshold 12]

Requires Python 3.8+ and Pillow (pip install pillow).
"""
import argparse
import csv
import hashlib
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from PIL import Image

# ----------------------------------------------------------------------------- configuration
# source folder name -> (release folder / label, file-name prefix, numeric label)
CLASSES = {
    "Normal": ("normal", "normal", 0),
    "patient": ("osteoarthritis", "oa", 1),
}
IMAGE_EXTS = {".jpg": "jpg", ".jpeg": "jpg", ".png": "png"}
BIT_DEPTH = {"1": 1, "L": 8, "P": 8, "LA": 16, "I;16": 16, "I;16B": 16, "RGB": 24, "RGBA": 32, "I": 32, "F": 32}
HASH_SIZE = 16  # 16x16 difference hash = 256 bits


# ----------------------------------------------------------------------------- helpers
def natural_index(name):
    m = re.search(r"\((\d+)\)", name)
    return int(m.group(1)) if m else 10 ** 9


def popcount(x):
    return x.bit_count() if hasattr(x, "bit_count") else bin(x).count("1")


def dhash(img):
    g = img.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.LANCZOS)
    px = g.tobytes()
    bits = 0
    for r in range(HASH_SIZE):
        row = r * (HASH_SIZE + 1)
        for c in range(HASH_SIZE):
            bits = (bits << 1) | (px[row + c] > px[row + c + 1])
    return bits


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def tree_listing(root, max_files=5):
    """Compact tree: shows the first and last files of large folders."""
    lines = [root.name + "/"]

    def walk(d, prefix):
        entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name))
        files = [e for e in entries if e.is_file()]
        shown = entries
        if len(files) > max_files:
            keep = set(files[:3] + files[-2:])
            shown = [e for e in entries if e.is_dir() or e in keep]
        for i, e in enumerate(shown):
            last = i == len(shown) - 1
            lines.append(f"{prefix}{'└── ' if last else '├── '}{e.name}{'/' if e.is_dir() else ''}")
            if e.is_file() and len(files) > max_files and e == files[2]:
                lines.append(f"{prefix}│   … {len(files) - 5} more files …")
            if e.is_dir():
                walk(e, prefix + ("    " if last else "│   "))

    walk(root, "")
    n_dirs = sum(1 for p in root.rglob("*") if p.is_dir())
    n_files = sum(1 for p in root.rglob("*") if p.is_file())
    lines += ["", f"{n_dirs} directories, {n_files} files"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- steps
def scan(source):
    records, bad, skipped = [], [], []
    for src_folder, (label, prefix, label_id) in CLASSES.items():
        d = source / src_folder
        if not d.is_dir():
            sys.exit(f"ERROR: expected folder not found: {d}")
        files = sorted((p for p in d.iterdir() if p.is_file()),
                       key=lambda p: (IMAGE_EXTS.get(p.suffix.lower(), "zzz"), natural_index(p.name), p.name))
        for p in files:
            ext = IMAGE_EXTS.get(p.suffix.lower())
            if ext is None:
                skipped.append(p)
                continue
            data = p.read_bytes()
            try:
                with Image.open(p) as im:
                    im.load()
                    info = dict(format=im.format, width=im.width, height=im.height, mode=im.mode,
                                dhash=dhash(im))
            except Exception as e:  # corrupted / unreadable
                bad.append((p, str(e)))
                continue
            records.append(dict(src=p, src_folder=src_folder, label=label, prefix=prefix, label_id=label_id,
                                ext=ext, size=len(data), sha256=hashlib.sha256(data).hexdigest(), **info))
    return records, bad, skipped


def find_duplicates(records, threshold):
    pairs = []
    by_sha = defaultdict(list)
    for i, r in enumerate(records):
        by_sha[r["sha256"]].append(i)
    for idx in by_sha.values():
        for a in idx:
            for b in idx:
                if a < b:
                    pairs.append((a, b, "exact", 0))
    exact = {(a, b) for a, b, *_ in pairs}
    hashes = [r["dhash"] for r in records]
    n = len(records)
    for a in range(n):
        ha = hashes[a]
        for b in range(a + 1, n):
            d = popcount(ha ^ hashes[b])
            if d <= threshold and (a, b) not in exact:
                pairs.append((a, b, "near", d))
    return pairs


def assign_split(records, groups, fractions, seed):
    """Stratified, group-aware split: every duplicate group lands in one subset."""
    names = ("train", "val", "test")
    rng = random.Random(seed)
    split = {}
    by_label = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(records):
        by_label[r["label"]][groups[i]].append(i)
    for label in sorted(by_label):
        glist = list(by_label[label].values())
        rng.shuffle(glist)
        total = sum(len(g) for g in glist)
        target = [f * total for f in fractions]
        count = [0, 0, 0]
        for g in sorted(glist, key=len, reverse=True):  # big groups first; shuffle order kept among equals
            k = min(range(3), key=lambda s: (count[s] + len(g)) / target[s] if target[s] else float("inf"))
            for i in g:
                split[i] = names[k]
            count[k] += len(g)
    return split


README = """# Knee radiograph dataset for osteoarthritis detection (Netrokona, Bangladesh)

Version: 1.0 ({today})
Licence: CC BY 4.0 (see LICENSE.txt)
DOI: [10.17632/xxxxxxxxxx.1 — add after publishing on Mendeley Data]

## Summary

{n_total} de-identified knee X-ray images (anteroposterior and lateral views) collected at
Al-Nur Hospital and Diagnostic Centre, Netrokona, Bangladesh, and labelled by the hospital's
physicians into two classes:

| Class (folder)          | Label id | Images | JPEG | PNG |
|-------------------------|---------:|-------:|-----:|----:|
{class_rows}
| **Total**               |          | **{n_total}** | {n_jpg} | {n_png} |

Images are released in their original exported format and resolution; no resizing,
enhancement or format conversion was applied. Duplicate copies found in the original
collection were removed; `metadata/source_mapping.csv` lists every original file and
whether it was included.

## Folder structure

```
{tree}```

## Files

- `images/<class>/` — one folder per class; the folder name is the label. Compatible with
  `torchvision.datasets.ImageFolder` and `keras.utils.image_dataset_from_directory`.
- `metadata/images.csv` — one row per image:
  `image_id, file_path, label, label_id, format, width, height, mode, bit_depth,
  file_size_bytes, sha256, duplicate_group`.
  Images sharing a `duplicate_group` are identical or visually near-identical and should be
  kept in the same data subset.
- `metadata/source_mapping.csv` — maps each released file to its file name in the original
  collection (provenance only).
- `metadata/duplicates.csv` — exact (identical checksum) and near-duplicate
  (difference-hash distance ≤ {threshold}/256) pairs, for review.
- `splits/split_seed{seed}.csv` — a reproducible stratified train/val/test split
  ({split_desc}). Duplicate groups are never divided between subsets. The split is at image
  level: no patient identifiers are available, so images of the same person may still fall
  in different subsets.
- `SHA256SUMS.txt` — checksums; verify with `sha256sum -c SHA256SUMS.txt`.

## Class definitions

- **normal** — no radiographic signs of knee osteoarthritis, as assessed by the hospital's physicians.
- **osteoarthritis** — radiographs of patients diagnosed with knee osteoarthritis.

## Limitations

Binary labels only (no Kellgren–Lawrence grades); single centre; no patient identifiers;
mixed projections without per-image view labels; mixed file formats and image sizes.

## Citation

If you use this dataset, please cite the data article and the dataset:
[Authors], [Title], Data in Brief (2026). [DOI]
[Authors], [Dataset title], Mendeley Data, V1, 2026. [DOI]
"""

LICENSE = """This dataset is licensed under the Creative Commons Attribution 4.0 International
Licence (CC BY 4.0). You are free to share and adapt the material for any purpose, provided
you give appropriate credit, provide a link to the licence and indicate if changes were made.

Full licence text: https://creativecommons.org/licenses/by/4.0/legalcode
"""


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="folder containing Normal/ and patient/")
    ap.add_argument("output", type=Path, help="new release folder (must not exist or be empty)")
    ap.add_argument("--dry-run", action="store_true", help="scan and report only; write nothing")
    ap.add_argument("--dedupe", action="store_true",
                    help="drop exact duplicates (keep the first copy); drop all copies of an image "
                         "that appears in both classes")
    ap.add_argument("--dedupe-near", type=int, default=None, metavar="T",
                    help="also drop near-identical copies (distance <= T); keeps the PNG / largest copy")
    ap.add_argument("--split", default="0.70,0.15,0.15", help="train,val,test fractions")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-split", action="store_true", help="do not create splits/")
    ap.add_argument("--near-threshold", type=int, default=12,
                    help="max Hamming distance (of 256 bits) for near-duplicates (default 12)")
    a = ap.parse_args()

    src, out = a.source.resolve(), a.output.resolve()
    if out == src or src in out.parents:
        sys.exit("ERROR: output must be outside the source folder.")
    if out.exists() and any(out.iterdir()) and not a.dry_run:
        sys.exit(f"ERROR: output folder {out} exists and is not empty.")
    fractions = [float(x) for x in a.split.split(",")]
    if len(fractions) != 3 or abs(sum(fractions) - 1) > 1e-6:
        sys.exit("ERROR: --split needs three fractions that sum to 1.")

    print(f"Scanning {src} …")
    records, bad, skipped = scan(src)
    for p, e in bad:
        print(f"  UNREADABLE (excluded): {p.relative_to(src)}  [{e}]")
    for p in skipped:
        print(f"  not an image (ignored): {p.relative_to(src)}")
    print(f"  {len(records)} readable images")

    print("Checking for duplicates …")
    pairs = find_duplicates(records, a.near_threshold)
    n_exact = sum(1 for p in pairs if p[2] == "exact")
    cross = [(x, y, k, d) for x, y, k, d in pairs if records[x]["label"] != records[y]["label"]]
    print(f"  exact duplicate pairs: {n_exact}; near-duplicate pairs: {len(pairs) - n_exact}; "
          f"pairs across classes: {len(cross)}")

    dropped = {}
    if a.dedupe or a.dedupe_near is not None:
        t = a.dedupe_near if a.dedupe_near is not None else -1
        uf_d = UnionFind(len(records))
        for x, y, k, d in pairs:
            if k == "exact" or d <= t:
                uf_d.union(x, y)
        grp = defaultdict(list)
        for i in range(len(records)):
            grp[uf_d.find(i)].append(i)
        for g in grp.values():
            if len(g) < 2:
                continue
            if len({records[i]["label"] for i in g}) > 1:
                for i in g:
                    dropped[i] = "same image appears in both classes (label conflict)"
                continue
            best = max(g, key=lambda i: (records[i]["ext"] == "png",
                                         records[i]["width"] * records[i]["height"], -i))
            for i in g:
                if i != best:
                    kind = "exact" if records[i]["sha256"] == records[best]["sha256"] else "near-identical"
                    dropped[i] = f"{kind} duplicate of {records[best]['src_folder']}/{records[best]['src'].name}"
        print(f"  dedupe: {len(dropped)} files will be excluded")

    keep = [i for i in range(len(records)) if i not in dropped]

    # duplicate groups over kept images
    uf = UnionFind(len(records))
    for x, y, *_ in pairs:
        if x in dropped or y in dropped:
            continue
        uf.union(x, y)
    roots = Counter(uf.find(i) for i in keep)
    sizes = sorted((s for s in roots.values() if s > 1), reverse=True)
    if sizes:
        print(f"  duplicate groups: {len(sizes)} (largest has {sizes[0]} images)")
        if sizes[0] > 10:
            print("  WARNING: a very large group suggests the near-duplicate threshold is too loose; "
                  "try a lower --near-threshold.")

    # new names, per class in source order
    counters = Counter()
    per_class = Counter(records[i]["label"] for i in keep)
    width = {lab: max(4, len(str(n))) for lab, n in per_class.items()}
    for i in keep:
        r = records[i]
        counters[r["label"]] += 1
        r["image_id"] = f"{r['prefix']}_{counters[r['label']]:0{width[r['label']]}d}"
        r["new_rel"] = f"images/{r['label']}/{r['image_id']}.{r['ext']}"

    group_name, gid = {}, 0
    groups = {}
    for i in keep:
        root = uf.find(i)
        if roots[root] > 1:
            if root not in group_name:
                gid += 1
                group_name[root] = f"g{gid:04d}"
            groups[i] = group_name[root]
        else:
            groups[i] = ""

    split = {}
    if not a.no_split:
        split_groups = {i: (groups[i] or records[i]["image_id"]) for i in keep}
        split = assign_split([records[i] for i in keep], [split_groups[i] for i in keep], fractions, a.seed)
        split = {keep[j]: s for j, s in split.items()}
        sc = Counter((records[i]["label"], s) for i, s in split.items())
        for lab in sorted(per_class):
            print(f"  split {lab:15s} " + "  ".join(f"{s} {sc[(lab, s)]}" for s in ("train", "val", "test")))

    if a.dry_run:
        print("\nDry run — nothing written. Example renames:")
        for i in keep[:3] + keep[-3:]:
            r = records[i]
            print(f"  {r['src_folder']}/{r['src'].name}  ->  {r['new_rel']}")
        return

    # ------------------------------------------------------------------ write release
    for lab, *_ in CLASSES.values():
        (out / "images" / lab).mkdir(parents=True, exist_ok=True)
    (out / "metadata").mkdir(exist_ok=True)
    print(f"Copying {len(keep)} images to {out} …")
    for i in keep:
        r = records[i]
        dst = out / r["new_rel"]
        shutil.copy2(r["src"], dst)
        if hashlib.sha256(dst.read_bytes()).hexdigest() != r["sha256"]:
            sys.exit(f"ERROR: copy verification failed for {dst}")

    img_rows = [dict(image_id=records[i]["image_id"], file_path=records[i]["new_rel"],
                     label=records[i]["label"], label_id=records[i]["label_id"],
                     format=records[i]["format"], width=records[i]["width"], height=records[i]["height"],
                     mode=records[i]["mode"], bit_depth=BIT_DEPTH.get(records[i]["mode"], ""),
                     file_size_bytes=records[i]["size"], sha256=records[i]["sha256"],
                     duplicate_group=groups[i]) for i in keep]
    write_csv(out / "metadata" / "images.csv", img_rows, list(img_rows[0]))

    map_rows = []
    for i, r in enumerate(records):
        map_rows.append(dict(image_id=r.get("image_id", "") if i not in dropped else "",
                             file_path=r.get("new_rel", "") if i not in dropped else "",
                             original_folder=r["src_folder"], original_file_name=r["src"].name,
                             status="excluded: " + dropped[i] if i in dropped else "included"))
    for p, e in bad:
        map_rows.append(dict(image_id="", file_path="", original_folder=p.parent.name,
                             original_file_name=p.name, status="excluded: unreadable image"))
    write_csv(out / "metadata" / "source_mapping.csv", map_rows, list(map_rows[0]))

    ref = lambda i: records[i].get("new_rel") or f"(excluded) {records[i]['src_folder']}/{records[i]['src'].name}"
    dup_rows = [dict(file_a=ref(x), file_b=ref(y), type=k, hamming_distance=d,
                     same_class=records[x]["label"] == records[y]["label"]) for x, y, k, d in pairs]
    write_csv(out / "metadata" / "duplicates.csv", dup_rows,
              ["file_a", "file_b", "type", "hamming_distance", "same_class"])

    if split:
        (out / "splits").mkdir(exist_ok=True)
        rows = [dict(image_id=records[i]["image_id"], file_path=records[i]["new_rel"],
                     label=records[i]["label"], split=split[i]) for i in keep]
        write_csv(out / "splits" / f"split_seed{a.seed}.csv", rows, list(rows[0]))

    with (out / "SHA256SUMS.txt").open("w", encoding="utf-8") as f:
        for i in keep:
            f.write(f"{records[i]['sha256']}  {records[i]['new_rel']}\n")
    (out / "LICENSE.txt").write_text(LICENSE, encoding="utf-8")

    # README (written before the tree so it appears in it), then the tree, then README again with the tree
    fmt = Counter((records[i]["label"], records[i]["ext"]) for i in keep)
    class_rows = "\n".join(
        f"| {lab + ' (images/' + lab + '/)':23s} | {lid:8d} | {per_class[lab]:6d} | {fmt[(lab, 'jpg')]:4d} | "
        f"{fmt[(lab, 'png')]:3d} |" for lab, _, lid in CLASSES.values())
    split_desc = (f"{', '.join(f'{int(f * 100)}%' for f in fractions)}, seed {a.seed}") if split else "not created"
    readme_kw = dict(today=date.today().isoformat(), n_total=len(keep), class_rows=class_rows,
                     n_jpg=sum(v for (l, e), v in fmt.items() if e == "jpg"),
                     n_png=sum(v for (l, e), v in fmt.items() if e == "png"),
                     threshold=a.near_threshold, seed=a.seed, split_desc=split_desc)
    (out / "README.md").write_text(README.format(tree="", **readme_kw), encoding="utf-8")
    (out / "folder-structure.txt").write_text("", encoding="utf-8")
    tree = tree_listing(out)
    (out / "folder-structure.txt").write_text(tree, encoding="utf-8")
    (out / "README.md").write_text(README.format(tree=tree, **readme_kw), encoding="utf-8")

    print("Done.")
    print(tree)


if __name__ == "__main__":
    main()
