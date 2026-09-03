# =============================================================================
# GitHub-ready analysis script
#
# Generated from the manuscript analysis notebook.
# Patient-level data are NOT included in this repository.
# Default paths are relative to the repository root:
#   data/Merge_20260901.xlsx
#   outputs/
#
# Before running, place an authorized local copy of the input dataset under
# data/ or edit DATA_PATH. Do not commit patient-level data or generated outputs.
# =============================================================================


# %% [markdown]
# # SHAP ONLY — Stroke landmark prediction model
#
# This notebook performs **only LightGBM SHAP calculation and figure export**.
#
# It does **not** rerun:
# - model fitting
# - Optuna
# - bootstrap
# - ROC / calibration / DCA
# - operational matrices
# - manuscript tables
#
# Required input:
# `01_main/bundles/primary_results_bundle.joblib`
#
# Final outputs:
# - `Figure_S4_SHAP_Day3.pdf`
# - `Figure_S4_SHAP_Day3.png`
# - `Figure_S5_SHAP_Day7.pdf`
# - `Figure_S5_SHAP_Day7.png`
#
# Individual standard SHAP panels are also retained in
# `03_figures_tables/figures/individual_SHAP_panels/`.

# %% 
# =============================================================================
# 1. Imports and user settings
# =============================================================================

import os

# Windows/Jupyter stability
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits


# -------------------------------------------------------------------------
# EDIT ONLY THIS PATH IF NEEDED
# -------------------------------------------------------------------------

OUTPUT_DIR = Path(
    r"outputs"
)

SEED = 20260902
SHAP_MAX_PATIENTS = 2000
MAX_DISPLAY = 15
FIGURE_DPI = 300

# False:
#   if final S4/S5 already exist, skip SHAP calculation.
# True:
#   force recalculation.
FORCE_RECALCULATE = False


BUNDLE_PATH = (
    OUTPUT_DIR
    / "01_main"
    / "bundles"
    / "primary_results_bundle.joblib"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "03_figures_tables"
    / "figures"
)

TABLE_DIR = (
    OUTPUT_DIR
    / "03_figures_tables"
    / "tables"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("Bundle :", BUNDLE_PATH)
print("Figures:", FIGURE_DIR)
print("Tables :", TABLE_DIR)

# %% 
# =============================================================================
# 2. Compatibility definitions for loading the locked Notebook 01 bundle
# =============================================================================

@dataclass
class AnalysisConfig:
    data_path: Path = Path(".")
    output_dir: Path = Path(".")
    mode: str = "FULL"

    id_col: str = "INDEX"
    patient_id_col: Optional[str] = "データ識別番号"
    admission_col: str = "入院日"
    discharge_col: str = "退院日"
    los_col: str = "在院日数"
    disposition_col: str = "転帰"
    development_start: str = "2018-04-01"
    temporal_cutoff: str = "2023-04-01"
    study_end_exclusive: str = "2025-04-01"
    enforce_first_hospitalization: bool = True
    known_duplicate_indices: tuple[Any, ...] = (4768,)

    seed: int = 20260902
    cv_folds: int = 5
    n_optuna_trials: int = 100
    n_bootstrap: int = 1000
    cv_n_jobs: int = 1
    lgbm_n_jobs: int = 1
    target_sensitivity: float = 0.80
    dca_threshold_min: float = 0.01
    dca_threshold_max: float = 0.70
    dca_threshold_step: float = 0.01

    lr_c_grid: tuple[float, ...] = (
        0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0
    )

    lgbm_learning_rate_min: float = 0.01
    lgbm_learning_rate_max: float = 0.20
    lgbm_estimators_min: int = 100
    lgbm_estimators_max: int = 800
    lgbm_max_depth_min: int = 2
    lgbm_max_depth_max: int = 6
    lgbm_min_child_samples_min: int = 20
    lgbm_min_child_samples_max: int = 100
    lgbm_subsample_min: float = 0.70
    lgbm_subsample_max: float = 1.00
    lgbm_colsample_min: float = 0.70
    lgbm_colsample_max: float = 1.00
    lgbm_regularization_min: float = 1e-4
    lgbm_regularization_max: float = 10.0

    run_los28_day3: bool = False
    run_los28_day7: bool = True
    run_death_exclusion: bool = False
    run_shap: bool = True
    shap_max_patients: int = 2000

    figure_dpi: int = 300
    figure_font: str = "Times New Roman"


@dataclass(frozen=True)
class TaskSpec:
    key: str
    label: str
    landmark_day: int
    landmark_flag: str
    outcome_col: str
    outcome_label: str
    continuous_features: tuple[str, ...]
    binary_features: tuple[str, ...]
    exclusion_col: Optional[str] = None
    exclusion_value: Optional[int] = None
    analysis_type: str = "primary"

    @property
    def features(self) -> list[str]:
        return list(
            dict.fromkeys(
                self.continuous_features
                + self.binary_features
            )
        )


# The original bundles were serialized with this canonical module name.
_CANONICAL_MODULE_NAME = "stroke_landmark_bmc_core_20260902"

sys.modules.setdefault(
    _CANONICAL_MODULE_NAME,
    sys.modules[__name__],
)

AnalysisConfig.__module__ = _CANONICAL_MODULE_NAME
TaskSpec.__module__ = _CANONICAL_MODULE_NAME


PUBLICATION_LABELS = {
    "Age": "Age, years",
    "Male": "Male sex",
    "BMI": "Body mass index, kg/m²",
    "NIHSS_total_24h後": "NIHSS score at 24 h",

    "day3_Alb": "Albumin",
    "day3_BUN": "Blood urea nitrogen",
    "day3_eGFR": "eGFR",
    "day3_Hb": "Hemoglobin",
    "day3_K": "Potassium",
    "day3_Na": "Sodium",
    "day3_WBC": "White blood cell count",
    "day3_CRP": "C-reactive protein",
    "day3_SBP": "Systolic blood pressure",
    "day3_HR": "Heart rate",
    "day3_SU日数": "Stroke-unit days",
    "day3_B項目": "Nursing dependency score",
    "day3_t-pa": "Intravenous thrombolysis",
    "day3_エダラボン": "Edaravone",
    "day3_カテコラミン": "Catecholamine administration",
    "day3_抗生剤": "Systemic antibiotic therapy",
    "day3_抗てんかん剤": "Antiepileptic medication",
    "day3_ハロペリドール_iv": "Intravenous haloperidol",
    "day3_脳血栓回収術": "Endovascular thrombectomy",
    "day3_脳手術": "Neurosurgical procedure",
    "day3_人工呼吸": "Mechanical ventilation",
    "day3_CHDF": "Continuous hemodiafiltration",
    "day3_頸動脈ステント留置術": "Carotid artery stenting",

    "day7_Alb": "Albumin",
    "day7_BUN": "Blood urea nitrogen",
    "day7_eGFR": "eGFR",
    "day7_Hb": "Hemoglobin",
    "day7_K": "Potassium",
    "day7_Na": "Sodium",
    "day7_WBC": "White blood cell count",
    "day7_CRP": "C-reactive protein",
    "day7_SBP": "Systolic blood pressure",
    "day7_HR": "Heart rate",
    "day7_SU日数": "Stroke-unit days",
    "day7_B項目": "Nursing dependency score",
    "day7_t-pa": "Intravenous thrombolysis",
    "day7_エダラボン": "Edaravone",
    "day7_カテコラミン": "Catecholamine administration",
    "day7_抗生剤": "Systemic antibiotic therapy",
    "day7_抗てんかん剤": "Antiepileptic medication",
    "day7_ハロペリドール_iv": "Intravenous haloperidol",
    "day7_脳血栓回収術": "Endovascular thrombectomy",
    "day7_脳手術": "Neurosurgical procedure",
    "day7_人工呼吸": "Mechanical ventilation",
    "day7_CHDF": "Continuous hemodiafiltration",
    "day7_頸動脈ステント留置術": "Carotid artery stenting",
}


def stable_seed(
    base_seed: int,
    text: str,
) -> int:
    return int(
        (
            base_seed
            + zlib.crc32(
                text.encode("utf-8")
            )
        )
        % (2**31 - 1)
    )


def clean_feature_name(
    name: str,
) -> str:
    return (
        name.split("__", 1)[1]
        if "__" in name
        else name
    )


def mm_to_in(
    mm: float,
) -> float:
    return mm / 25.4


plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# %% 
# =============================================================================
# 3. Load locked primary models from Notebook 01
# =============================================================================

if not BUNDLE_PATH.exists():
    raise FileNotFoundError(
        "Locked primary-results bundle was not found:\n"
        f"{BUNDLE_PATH}\n\n"
        "Run Notebook 01 first, or correct OUTPUT_DIR."
    )

print(
    "[SHAP] Loading locked primary-results bundle...",
    flush=True,
)

main_bundle = joblib.load(
    BUNDLE_PATH
)

primary_results = main_bundle[
    "primary_results"
]

required_tasks = {
    "day3_nonhome",
    "day3_los21",
    "day7_nonhome",
    "day7_los21",
}

if set(primary_results) != required_tasks:
    raise RuntimeError(
        "Unexpected primary task set:\n"
        f"{sorted(primary_results)}"
    )

for task_key in sorted(primary_results):
    if (
        "lgbm"
        not in primary_results[
            task_key
        ]["models"]
    ):
        raise RuntimeError(
            f"Locked LightGBM model missing: {task_key}"
        )

print(
    "[SHAP] Locked models loaded successfully.",
    flush=True,
)
print(
    "[SHAP] Tasks:",
    sorted(primary_results),
    flush=True,
)

# %% 
# =============================================================================
# 4. Standard SHAP functions
# =============================================================================

def prepare_standard_shap_task(
    task_key: str,
) -> dict[str, Any]:
    """
    Calculate SHAP values for one locked LightGBM model.
    SHAP is imported only here to reduce Windows/Jupyter startup risk.
    """
    import shap

    task_result = primary_results[
        task_key
    ]

    spec = task_result[
        "spec"
    ]

    pipeline: Pipeline = task_result[
        "models"
    ][
        "lgbm"
    ][
        "model"
    ]

    preprocess: ColumnTransformer = (
        pipeline.named_steps[
            "preprocess"
        ]
    )

    model: LGBMClassifier = (
        pipeline.named_steps[
            "model"
        ]
    )

    x = task_result[
        "x_temp"
    ].copy()

    if len(x) > SHAP_MAX_PATIENTS:
        x = x.sample(
            n=SHAP_MAX_PATIENTS,
            random_state=stable_seed(
                SEED,
                spec.key
                + "_shap_sample",
            ),
        )

    x_transformed = (
        preprocess.transform(x)
    )

    if hasattr(
        x_transformed,
        "toarray",
    ):
        x_transformed = (
            x_transformed.toarray()
        )

    x_transformed = np.asarray(
        x_transformed,
        dtype=float,
    )

    raw_names = [
        clean_feature_name(v)
        for v
        in preprocess.get_feature_names_out()
    ]

    display_names = [
        PUBLICATION_LABELS.get(
            name,
            name,
        )
        for name
        in raw_names
    ]

    print(
        f"[SHAP] TreeExplainer: {task_key} "
        f"(n={len(x)}, p={len(raw_names)})",
        flush=True,
    )

    explainer = shap.TreeExplainer(
        model
    )

    raw_explanation = explainer(
        x_transformed
    )

    shap_values = np.asarray(
        raw_explanation.values,
        dtype=float,
    )

    # Compatibility across SHAP / LightGBM versions.
    if shap_values.ndim == 3:
        shap_values = (
            shap_values[
                :,
                :,
                -1,
            ]
        )

    if shap_values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP shape "
            f"for {task_key}: "
            f"{shap_values.shape}"
        )

    if (
        shap_values.shape[1]
        != len(display_names)
    ):
        raise RuntimeError(
            f"SHAP feature-count mismatch "
            f"for {task_key}: "
            f"{shap_values.shape[1]} "
            f"vs {len(display_names)}"
        )

    explanation = shap.Explanation(
        values=shap_values,
        data=x_transformed,
        feature_names=display_names,
    )

    summary = pd.DataFrame(
        {
            "Task": spec.key,
            "Landmark": (
                f"Day {spec.landmark_day}"
            ),
            "Outcome": (
                spec.outcome_label
            ),
            "Raw_variable": raw_names,
            "Variable": display_names,
            "Mean_absolute_SHAP": (
                np.abs(
                    shap_values
                ).mean(
                    axis=0
                )
            ),
            "Mean_SHAP": (
                shap_values.mean(
                    axis=0
                )
            ),
            "N_SHAP_patients": len(x),
        }
    ).sort_values(
        "Mean_absolute_SHAP",
        ascending=False,
    )

    return {
        "task_key": task_key,
        "spec": spec,
        "explanation": explanation,
        "summary": summary,
    }


def save_standard_shap_panel(
    task_data: dict[str, Any],
    *,
    stem: str,
    panel_label: str,
    panel_title: str,
) -> dict[str, Path]:
    """
    One independent STANDARD shap.plots.beeswarm() panel.
    No custom dot positioning.
    No 'Sum of N other features'.
    """
    import shap

    panel_dir = (
        FIGURE_DIR
        / "individual_SHAP_panels"
    )

    panel_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = (
        panel_dir
        / f"{stem}.pdf"
    )

    png_path = (
        panel_dir
        / f"{stem}.png"
    )

    fig, ax = plt.subplots(
        figsize=(
            mm_to_in(112),
            mm_to_in(125),
        )
    )

    shap.plots.beeswarm(
        task_data[
            "explanation"
        ],
        max_display=MAX_DISPLAY,
        show=False,
        color_bar=True,
        ax=ax,
        plot_size=None,
        group_remaining_features=False,
    )

    ax.set_title(
        f"{panel_label} {panel_title}",
        loc="left",
        fontweight="bold",
        fontsize=9.0,
        pad=6,
    )

    ax.set_xlabel(
        "SHAP value (impact on model output)",
        fontsize=8.0,
    )

    ax.tick_params(
        axis="x",
        labelsize=7.0,
        direction="out",
        length=2.5,
        width=0.6,
    )

    ax.tick_params(
        axis="y",
        labelsize=7.0,
    )

    for extra_ax in fig.axes:
        if extra_ax is ax:
            continue

        extra_ax.tick_params(
            labelsize=7.0,
            length=0,
        )

        if extra_ax.get_ylabel():
            extra_ax.yaxis.label.set_size(
                7.5
            )

    fig.subplots_adjust(
        left=0.38,
        right=0.93,
        top=0.93,
        bottom=0.13,
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    fig.savefig(
        png_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )

    plt.close(fig)

    return {
        "pdf": pdf_path,
        "png": png_path,
    }


def combine_two_png_panels(
    left_png: Path,
    right_png: Path,
    output_stem: Path,
) -> dict[str, Path]:
    """
    Combine two high-resolution PNG panels.
    Output: PNG + PDF only.
    No TIFF.
    """
    from PIL import Image

    target_width_mm = 170.0
    gap_mm = 4.0
    dpi = FIGURE_DPI

    target_width_px = int(
        round(
            target_width_mm
            / 25.4
            * dpi
        )
    )

    gap_px = int(
        round(
            gap_mm
            / 25.4
            * dpi
        )
    )

    target_panel_width = (
        target_width_px
        - gap_px
    ) // 2

    left = Image.open(
        left_png
    ).convert(
        "RGB"
    )

    right = Image.open(
        right_png
    ).convert(
        "RGB"
    )

    def resize_to_width(
        image: Image.Image,
        width: int,
    ) -> Image.Image:
        scale = (
            width
            / image.width
        )

        height = max(
            1,
            int(
                round(
                    image.height
                    * scale
                )
            ),
        )

        return image.resize(
            (
                width,
                height,
            ),
            resample=Image.Resampling.LANCZOS,
        )

    left_r = resize_to_width(
        left,
        target_panel_width,
    )

    right_r = resize_to_width(
        right,
        target_panel_width,
    )

    target_height = max(
        left_r.height,
        right_r.height,
    )

    canvas = Image.new(
        "RGB",
        (
            target_width_px,
            target_height,
        ),
        color="white",
    )

    left_y = (
        target_height
        - left_r.height
    ) // 2

    right_y = (
        target_height
        - right_r.height
    ) // 2

    canvas.paste(
        left_r,
        (
            0,
            left_y,
        ),
    )

    canvas.paste(
        right_r,
        (
            target_panel_width
            + gap_px,
            right_y,
        ),
    )

    png_out = (
        output_stem.with_suffix(
            ".png"
        )
    )

    pdf_out = (
        output_stem.with_suffix(
            ".pdf"
        )
    )

    canvas.save(
        png_out,
        dpi=(
            dpi,
            dpi,
        ),
        optimize=True,
    )

    canvas.save(
        pdf_out,
        "PDF",
        resolution=float(dpi),
    )

    left.close()
    right.close()
    left_r.close()
    right_r.close()
    canvas.close()

    return {
        "png": png_out,
        "pdf": pdf_out,
    }


def export_all_shap() -> pd.DataFrame:
    """
    Calculate four LightGBM SHAP explanations and create:
      S4 = Day 3, two panels
      S5 = Day 7, two panels
    """

    task_keys = [
        "day3_nonhome",
        "day3_los21",
        "day7_nonhome",
        "day7_los21",
    ]

    prepared = {}
    summaries = []

    for task_key in task_keys:
        print(
            f"\n[SHAP] CALCULATE: "
            f"{task_key}",
            flush=True,
        )

        prepared[
            task_key
        ] = (
            prepare_standard_shap_task(
                task_key
            )
        )

        summaries.append(
            prepared[
                task_key
            ][
                "summary"
            ]
        )

    jobs = [
        (
            "day3_nonhome",
            "Figure_S4A_SHAP_Day3_nonhome",
            "(A)",
            "No direct home discharge",
        ),
        (
            "day3_los21",
            "Figure_S4B_SHAP_Day3_LOS21",
            "(B)",
            "Hospital stay >21 days",
        ),
        (
            "day7_nonhome",
            "Figure_S5A_SHAP_Day7_nonhome",
            "(A)",
            "No direct home discharge",
        ),
        (
            "day7_los21",
            "Figure_S5B_SHAP_Day7_LOS21",
            "(B)",
            "Hospital stay >21 days",
        ),
    ]

    panel_paths = {}

    for (
        task_key,
        stem,
        panel_label,
        panel_title,
    ) in jobs:
        print(
            f"[SHAP] PLOT: {stem}",
            flush=True,
        )

        panel_paths[
            task_key
        ] = (
            save_standard_shap_panel(
                prepared[
                    task_key
                ],
                stem=stem,
                panel_label=panel_label,
                panel_title=panel_title,
            )
        )

    print(
        "\n[SHAP] COMBINE: Day 3",
        flush=True,
    )

    day3_files = (
        combine_two_png_panels(
            panel_paths[
                "day3_nonhome"
            ][
                "png"
            ],
            panel_paths[
                "day3_los21"
            ][
                "png"
            ],
            FIGURE_DIR
            / "Figure_S4_SHAP_Day3",
        )
    )

    print(
        "[SHAP] COMBINE: Day 7",
        flush=True,
    )

    day7_files = (
        combine_two_png_panels(
            panel_paths[
                "day7_nonhome"
            ][
                "png"
            ],
            panel_paths[
                "day7_los21"
            ][
                "png"
            ],
            FIGURE_DIR
            / "Figure_S5_SHAP_Day7",
        )
    )

    combined_summary = pd.concat(
        summaries,
        ignore_index=True,
    )

    summary_path = (
        TABLE_DIR
        / "SHAP_summary_all_tasks.csv"
    )

    combined_summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[SHAP] COMPLETE",
        flush=True,
    )

    for path in [
        day3_files["pdf"],
        day3_files["png"],
        day7_files["pdf"],
        day7_files["png"],
        summary_path,
    ]:
        size_mb = (
            path.stat().st_size
            / (1024 ** 2)
        )

        print(
            f"  {path.name} "
            f"({size_mb:.2f} MB)",
            flush=True,
        )

    return combined_summary

# %% 
# =============================================================================
# 5. RUN SHAP ONLY
# =============================================================================

expected_final_files = [
    FIGURE_DIR
    / "Figure_S4_SHAP_Day3.pdf",

    FIGURE_DIR
    / "Figure_S4_SHAP_Day3.png",

    FIGURE_DIR
    / "Figure_S5_SHAP_Day7.pdf",

    FIGURE_DIR
    / "Figure_S5_SHAP_Day7.png",
]


all_exist = all(
    path.exists()
    for path
    in expected_final_files
)


if (
    all_exist
    and not FORCE_RECALCULATE
):
    print(
        "[SHAP] Final S4/S5 files already exist.",
        flush=True,
    )
    print(
        "[SHAP] Recalculation skipped.",
        flush=True,
    )

    for path in expected_final_files:
        print(
            f"  OK: {path}",
            flush=True,
        )

else:
    if FORCE_RECALCULATE:
        print(
            "[SHAP] FORCE_RECALCULATE=True",
            flush=True,
        )
    else:
        print(
            "[SHAP] S4/S5 are missing. "
            "Starting SHAP calculation.",
            flush=True,
        )

    with threadpool_limits(
        limits=1
    ):
        SHAP_SUMMARY = (
            export_all_shap()
        )

    display(
        SHAP_SUMMARY
        .sort_values(
            [
                "Task",
                "Mean_absolute_SHAP",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "Task",
            as_index=False,
        )
        .head(
            MAX_DISPLAY
        )
    )
