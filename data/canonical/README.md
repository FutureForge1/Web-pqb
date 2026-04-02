# Canonical Data Notes

This repository tracks the small-to-medium canonical artifacts needed to run the
annotation app and benchmark tooling out of the box:

- `human_playwright_steps.parquet`
- `human_playwright_trajectories.parquet`
- `mind2web_trajectories.parquet`
- `vwa_gpt4v_som_steps.parquet`
- `vwa_gpt4v_som_trajectories.parquet`

The full `mind2web_steps.parquet` file is intentionally not stored in normal
Git history because it exceeds the practical single-file size threshold for a
plain GitHub repository.

To restore the missing file, use one of these paths:

1. Re-run the canonical conversion pipeline from the raw data with
   `scripts/convert_canonical_training_data.py`.
2. Store `mind2web_steps.parquet` as a release asset / external data artifact
   and place it back under `data/canonical/`.

Expected path:

- `data/canonical/mind2web_steps.parquet`
