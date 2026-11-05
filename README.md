# Knee radiograph dataset for osteoarthritis detection (Netrokona, Bangladesh)

Code and results for the Data in Brief article describing a de-duplicated dataset of
**1,500 labelled knee X-ray images** (605 normal, 895 osteoarthritis) collected at
Al-Nur Hospital and Diagnostic Centre, Netrokona, Bangladesh.

- **Dataset:** Mendeley Data, DOI [10.17632/xxxxxxxxxx.1](https://doi.org/10.17632/xxxxxxxxxx.1) (CC BY 4.0)
- **Article:** [authors], Data in Brief (2026), DOI [to be added]

The images are **not** stored in this repository. Download them from Mendeley Data.

## Repository structure

```
.
├── scripts/
│   ├── duplicate_report.py      # finds exact and near-identical copies in the raw collection
│   ├── restructure_dataset.py   # builds the released dataset: de-duplication, renaming,
│   │                            #   metadata, checksums, stratified split
│   ├── dataset_report.py        # dataset tables and figures for the article
│   └── baseline_benchmark.py    # fine-tunes six ImageNet CNNs (technical validation)
├── results/
│   ├── dataset/                 # output of dataset_report.py
│   └── benchmark/               # output of baseline_benchmark.py (metrics, predictions,
│                                #   training histories, figures, run_info.json)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the benchmark on an NVIDIA GPU (Linux), install TensorFlow with CUDA instead:
`pip install "tensorflow[and-cuda]==2.21.0"`.

## Reproducing the results

The released dataset on Mendeley Data is the output of step 2. To reproduce only the benchmark,
download it and go to step 4.

**1. Duplicate check of the original collection** (read-only)

```bash
python scripts/duplicate_report.py koa-dataset-raw --out dup_review
```

**2. Build the released dataset** from the original collection (`koa-dataset-raw/Normal`,
`koa-dataset-raw/patient`, 2,730 files). Removes exact copies (SHA-256) and near-identical
copies (256-bit difference hash, Hamming distance ≤ 8), drops images found in both classes,
and writes metadata, checksums and a stratified 70/15/15 split (seed 42) in which look-alike
images (distance ≤ 12) are kept in the same subset.

```bash
python scripts/restructure_dataset.py koa-dataset-raw KneeXray-Netrokona-v1 --dedupe-near 8
```

**3. Dataset tables and figures**

```bash
python scripts/dataset_report.py KneeXray-Netrokona-v1 --out results/dataset
```

**4. Baseline benchmark** — six models × three seeds. About 1.5 h on an NVIDIA RTX 2000 Ada.

```bash
python scripts/baseline_benchmark.py KneeXray-Netrokona-v1 --out results/benchmark --seeds 42,43,44
```

The scripts print nothing; progress and errors go to `benchmark.log` / `report.log` in the
output folder. The benchmark can be resumed: re-running the same command skips runs already
listed in `metrics_per_run.csv`.

## Baseline results

Test subset of `splits/split_seed42.csv` (224 images: 90 normal, 134 osteoarthritis), mean ±
SD over seeds 42, 43 and 44. Osteoarthritis is the positive class.

| Model | Params (M) | Accuracy (%) | Macro F1 (%) | AUC | Sensitivity (%) | Specificity (%) |
|---|---:|---:|---:|---:|---:|---:|
| ResNet50 | 23.6 | 91.1 ± 1.3 | 90.8 ± 1.3 | 0.982 ± 0.003 | 89.3 ± 2.4 | 93.7 ± 1.3 |
| VGG16 | 14.7 | 94.3 ± 1.4 | 94.1 ± 1.5 | 0.984 ± 0.003 | 96.8 ± 0.9 | 90.7 ± 2.6 |
| DenseNet121 | 7.0 | 91.4 ± 0.7 | 91.0 ± 0.8 | 0.963 ± 0.003 | 92.5 ± 1.3 | 89.6 ± 3.6 |
| MobileNetV2 | 2.3 | 88.5 ± 0.7 | 88.3 ± 0.7 | 0.969 ± 0.001 | 85.1 ± 1.5 | 93.7 ± 0.6 |
| Xception | 20.9 | 92.1 ± 1.4 | 91.9 ± 1.4 | 0.970 ± 0.005 | 91.5 ± 2.8 | 93.0 ± 0.6 |
| InceptionResNetV2 | 54.3 | 93.3 ± 1.2 | 93.0 ± 1.2 | 0.976 ± 0.002 | 95.5 ± 2.7 | 90.0 ± 1.1 |

Full per-run metrics: `results/benchmark/metrics_per_run.csv`. Software versions and
training configuration: `results/benchmark/run_info.json`.

## Citation

If you use the dataset or this code, please cite the article and the dataset (see
`CITATION.cff`).

## Licence

Code: MIT (see `LICENSE`). Dataset: CC BY 4.0 (on Mendeley Data).
