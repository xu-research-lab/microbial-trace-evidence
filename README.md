# Skin microbes as trace evidence for criminal justice

This repository contains the code and data for the study **"Skin microbes as trace evidence for criminal justice"**. It analyzes whether individuals can be identified from skin microbiome signals deposited on touched surfaces.

## Install

```bash
pip install -r requirements.txt
```

## Repository Layout

- `notebooks/` - main analysis notebooks
- `src/` - shared Python helpers
- `data/` - QIIME2 artifacts, BIOM tables, and sequence files
- `results/` - generated figures and tables
- `sourcetracker_mixed_signal/` - SourceTracker2 mixed-signal analysis

## Notebooks

Run `notebooks/general.ipynb` first. It loads shared data and imports the common helper functions from `src/`.

- `part1_material_type.ipynb` - material type classification
- `part2_touch_number.ipynb` - touch number analysis
- `part3_persistence.ipynb` - persistence analysis
- `part4_mixed_signal.ipynb` - mixed-signal analysis
- `part5_cross_year.ipynb` - cross-year generalization analysis

## External Pipelines

### Beta diversity

The QIIME2 inputs live in `data/`. Run the beta-diversity workflow from your local environment as needed.

### SourceTracker2

```bash
python sourcetracker_mixed_signal/run_sourcetracker_mixed_signal.py
bash sourcetracker_mixed_signal/run_sourcetracker.sh
python sourcetracker_mixed_signal/plot_sourcetracker_results.py
```

## Notes

- Most figures are written to `results/` as both `.pdf` and `.png`.
- Shared plotting and modeling helpers are in `src/classify.py`, `src/plot.py`, and `src/utils.py`.