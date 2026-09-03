# Sanitized notebook release

The four manuscript-analysis notebooks are distributed here as a split `tar.xz` archive because the source notebooks are larger than the text-file upload interface used for this repository update.

## Reconstruct the notebooks

From the repository root, run:

```bash
python release/reconstruct_notebooks.py
```

This reconstructs `release/analysis_notebooks.tar.xz`, verifies its SHA-256 checksum, and extracts the following sanitized notebooks into the repository root:

- `01_main_analysis.ipynb`
- `02_sensitivity_analysis.ipynb`
- `03_tables_figures.ipynb`
- `04_shap_analysis.ipynb`

The notebooks have execution outputs removed and local machine paths replaced with repository-relative paths. Patient-level data are not included.

## Archive integrity

Expected archive SHA-256:

```text
b359e06455d0af4a7f4ece631b1d1734d7dec6f4b4a9c42e72e63f3bc563bab8
```

The archive is split into four parts:

```text
analysis_notebooks.tar.xz.part00
analysis_notebooks.tar.xz.part01
analysis_notebooks.tar.xz.part02
analysis_notebooks.tar.xz.part03
```

Do not commit reconstructed outputs, source data, fitted models, or patient-level prediction files.
