#!/usr/bin/env python3
"""
Detailed duplicate report for the knee X-ray dataset (read-only; nothing in the source is changed).

Put this file in the same folder as restructure_dataset.py and run:
    python duplicate_report.py <source> [--out dup_review]

Prints:
  1. unique images per class after removing exact (byte-identical) copies
  2. how many copies the duplicated images have
  3. which formats / file numbers the exact duplicates involve
  4. every pair that appears in BOTH classes
  5. how the near-duplicate count changes with the threshold
Writes picture sheets to <out>/ so the matches can be checked by eye:
  cross_class.png       all pairs found in both classes
  largest_groups.png    the 8 largest duplicate groups
  near_borderline.png   random near-duplicate pairs with the weakest match (distance 9-12)
  near_strong.png       random near-duplicate pairs with a strong match (distance 1-4)
"""
import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from restructure_dataset import UnionFind, find_duplicates, natural_index, popcount, scan

THRESHOLDS = [0, 2, 4, 6, 8, 10, 12]
THUMB = 150


def name(r):
    return f"{r['src_folder']}/{r['src'].name}"


def font():
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
              "C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, 11)
        except Exception:
            pass
    return ImageFont.load_default()


def sheet(rows, path, title):
    """rows: list of (caption, [records]); one row per group/pair."""
    if not rows:
        return False
    f = font()
    ncol = max(len(r[1]) for r in rows)
    cell_h = THUMB + 44
    W, H = 10 + ncol * (THUMB + 10), 30 + len(rows) * (cell_h + 18)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), title, fill="black", font=f)
    y = 30
    for caption, recs in rows:
        d.text((10, y), caption, fill=(160, 0, 0), font=f)
        for j, r in enumerate(recs):
            with Image.open(r["src"]) as im:
                im = im.convert("L")
                im.thumbnail((THUMB, THUMB))
                x = 10 + j * (THUMB + 10)
                canvas.paste(im.convert("RGB"), (x, y + 14))
            d.text((x, y + 16 + THUMB), r["src"].name[:24], fill="black", font=f)
            d.text((x, y + 30 + THUMB), f"[{r['src_folder']}]", fill=(90, 90, 90), font=f)
        y += cell_h + 18
    canvas.save(path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--out", type=Path, default=Path("dup_review"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    records, bad, _ = scan(a.source.resolve())
    n = len(records)
    pairs = find_duplicates(records, max(THRESHOLDS))
    exact = [(x, y) for x, y, k, _ in pairs if k == "exact"]
    near = [(x, y, d) for x, y, k, d in pairs if k == "near"]

    # 1-2. exact duplicates
    by_sha = defaultdict(list)
    for i, r in enumerate(records):
        by_sha[r["sha256"]].append(i)
    print("=" * 70)
    print("1. EXACT (byte-identical) DUPLICATES")
    print(f"   files scanned: {n}   unreadable: {len(bad)}")
    for lab in sorted({r["label"] for r in records}):
        files = sum(1 for r in records if r["label"] == lab)
        uniq = len({r["sha256"] for r in records if r["label"] == lab})
        print(f"   {lab:15s} files {files:5d}   unique {uniq:5d}   redundant copies {files - uniq:5d}")
    print(f"   {'ALL':15s} files {n:5d}   unique {len(by_sha):5d}   redundant copies {n - len(by_sha):5d}")
    hist = Counter(len(v) for v in by_sha.values() if len(v) > 1)
    print("\n2. COPIES PER DUPLICATED IMAGE")
    for k in sorted(hist):
        print(f"   {hist[k]:4d} images exist {k} times")

    # 3. what the exact duplicates look like
    combo = Counter()
    same_index = 0
    for x, y in exact:
        rx, ry = records[x], records[y]
        combo[f"{rx['src_folder']}/{rx['ext']} = {ry['src_folder']}/{ry['ext']}"] += 1
        same_index += natural_index(rx["src"].name) == natural_index(ry["src"].name)
    print("\n3. EXACT DUPLICATE PAIRS BY FOLDER/FORMAT")
    for k, v in combo.most_common():
        print(f"   {v:5d}  {k}")
    print(f"   pairs whose file numbers are equal: {same_index} of {len(exact)}")
    print("   examples:")
    for v in [g for g in by_sha.values() if len(g) > 1][:8]:
        print("     " + "  =  ".join(name(records[i]) for i in v))

    # 4. cross-class pairs
    cross = [(x, y, k, d) for x, y, k, d in pairs if records[x]["label"] != records[y]["label"]]
    print("\n4. PAIRS FOUND IN BOTH CLASSES (label conflicts)")
    for x, y, k, d in cross:
        print(f"   {k:5s} d={d:2d}  {name(records[x])}  ~  {name(records[y])}")
    if not cross:
        print("   none")

    # 5. threshold sweep (exact + near, transitive groups)
    print("\n5. NEAR-DUPLICATES BY THRESHOLD (distance out of 256 bits; exact copies included)")
    print("   thr  near-pairs  groups  largest  unique-after-grouping (normal / oa / all)")
    for t in THRESHOLDS:
        uf = UnionFind(n)
        for x, y in exact:
            uf.union(x, y)
        npairs = 0
        for x, y, d in near:
            if d <= t:
                uf.union(x, y)
                npairs += 1
        roots = Counter(uf.find(i) for i in range(n))
        multi = [s for s in roots.values() if s > 1]
        per = defaultdict(set)
        for i, r in enumerate(records):
            per[r["label"]].add(uf.find(i))
        labs = sorted(per)
        print(f"   {t:3d}  {npairs:10d}  {len(multi):6d}  {max(multi, default=1):7d}  "
              + " / ".join(str(len(per[l])) for l in labs) + f" / {len(roots)}")

    # picture sheets
    written = []
    if sheet([(f"{k} d={d}", [records[x], records[y]]) for x, y, k, d in cross],
             a.out / "cross_class.png", "Pairs found in both classes"):
        written.append("cross_class.png")
    uf = UnionFind(n)
    for x, y, *_ in pairs:
        uf.union(x, y)
    groups = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    big = sorted((g for g in groups.values() if len(g) > 1), key=len, reverse=True)[:8]
    if sheet([(f"group of {len(g)} (threshold 12)", [records[i] for i in g[:10]]) for g in big],
             a.out / "largest_groups.png", "Largest duplicate groups (first 10 images of each)"):
        written.append("largest_groups.png")
    for fn, lo, hi, title in (("near_borderline.png", 9, 12, "Weakest near-duplicate matches (distance 9-12)"),
                              ("near_strong.png", 1, 4, "Strong near-duplicate matches (distance 1-4)")):
        sel = [p for p in near if lo <= p[2] <= hi]
        rng.shuffle(sel)
        if sheet([(f"d={d}", [records[x], records[y]]) for x, y, d in sel[:12]], a.out / fn, title):
            written.append(fn)
    print(f"\nPicture sheets written to {a.out.resolve()}: {', '.join(written) or 'none'}")


if __name__ == "__main__":
    main()
