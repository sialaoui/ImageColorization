# Scripts overview

Each script is a standalone entry point for common colorization workflows. Pass `-h` to any file for full CLI options.

- **`train.py`** — Standard training loop with early stopping, checkpointing, and optional loss curve export. Select models with `--arch` and losses with `--loss`.
- **`train_with_hints.py`** — Training variant that injects reference hints into the pipeline; useful when auxiliary chrominance cues are available.
- **`train_with_reference.py`** — Supervises colorization using paired reference images; complements the hints workflow when explicit target colors are known.
- **`test.py`** — Loads a saved checkpoint, runs the test split, and reports metrics (L1/L2, PSNR, SSIM, optional LPIPS and FID). Can save predicted images.
- **`visualize.py`** — Quick visualization utility that colorizes grayscale inputs and writes side-by-side comparisons to disk.
- **`attentionMapRefColorisation.ipynb`** — Notebook exploration of reference-based attention maps; handy for interactive experiments.