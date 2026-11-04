import os
import random

import numpy as np
import pandas as pd
import torch

from src.functions import (
    build_sensitivity_values,
    cross_validate_dnn,
    load_dataset,
    make_model_comparison,
    plot_cv_history,
    plot_model_comparison,
    plot_predictions,
    plot_search_results,
    plot_sensitivity,
    run_hyperparameter_search,
    run_polynomial_comparison,
    run_sensitivity_analysis,
)


RANDOM_STATE = 0

# Keep at 5 until the full workflow is confirmed.
# Change to 400 for the final analysis.
EPOCHS = 400

DATASET_PATH = os.path.join(
    "datasets",
    "dataset_cell_growth_7factor.csv",
)

FIGURE_DIR = "figures"
TABLE_DIR = "tables"


random.seed(
    RANDOM_STATE
)

np.random.seed(
    RANDOM_STATE
)

torch.manual_seed(
    RANDOM_STATE
)


os.makedirs(
    FIGURE_DIR,
    exist_ok=True,
)

os.makedirs(
    TABLE_DIR,
    exist_ok=True,
)


(
    df,
    x,
    y,
    features,
    targets,
) = load_dataset(
    DATASET_PATH
)


print(
    "=" * 72
)

print(
    "CELL CULTURE FULLY CONNECTED "
    "NEURAL NETWORK"
)

print(
    "=" * 72
)

print(
    f"Samples: "
    f"{df.shape[0]:,}"
)

print(
    f"Features: "
    f"{len(features)}"
)

print(
    f"Targets: "
    f"{len(targets)}"
)

print(
    f"Epochs: "
    f"{EPOCHS}"
)


# ============================================================================
# 1. Hyperparameter search
# ============================================================================

print()
print(
    "=" * 72
)

print(
    "1. HYPERPARAMETER SEARCH"
)

print(
    "=" * 72
)


(
    df_search,
    best_hyperparameters,
) = run_hyperparameter_search(
    x=x,
    y=y,
    targets=targets,
    epochs=EPOCHS,
    random_state=RANDOM_STATE,
)


df_search.to_csv(
    os.path.join(
        TABLE_DIR,
        "table_hyperparameter_search.csv",
    ),
    index=False,
)


df_best_hyperparameters = (
    pd.DataFrame(
        {
            "Hyperparameter":
                list(
                    best_hyperparameters.keys()
                ),

            "Optimal Value":
                [
                    str(value)
                    for value
                    in best_hyperparameters.values()
                ],
        }
    )
)


df_best_hyperparameters.to_csv(
    os.path.join(
        TABLE_DIR,
        "table_best_hyperparameters.csv",
    ),
    index=False,
)


plot_search_results(
    df_search=df_search,

    output_path=os.path.join(
        FIGURE_DIR,
        "figure_hyperparameter_search.png",
    ),
)


print()
print(
    "Optimal hyperparameters:"
)

for key, value in (
    best_hyperparameters.items()
):
    print(
        f"  {key}: {value}"
    )


# ============================================================================
# 2. Four-fold cross-validation
# ============================================================================

print()
print(
    "=" * 72
)

print(
    "2. FOUR-FOLD CROSS-VALIDATION"
)

print(
    "=" * 72
)


(
    df_cv_folds,
    df_cv_summary,
    df_cv_history,
    df_cv_predictions,
) = cross_validate_dnn(
    x=x,
    y=y,
    targets=targets,
    hyperparameters=best_hyperparameters,
    epochs=EPOCHS,
    n_splits=4,
    random_state=RANDOM_STATE,
)


df_cv_folds.to_csv(
    os.path.join(
        TABLE_DIR,
        "table_dnn_cv_folds.csv",
    ),
    index=False,
)


df_cv_summary.to_csv(
    os.path.join(
        TABLE_DIR,
        "table_dnn_cv_summary.csv",
    ),
    index=False,
)


df_cv_history.to_csv(
    os.path.join(
        TABLE_DIR,
        "table_dnn_training_history.csv",
    ),
    index=False,
)


df_cv_predictions.to_csv(
    os.path.join(
        TABLE_DIR,
        "table_dnn_cv_predictions.csv",
    ),
    index=False,
)


plot_cv_history(
    df_history=df_cv_history,

    output_path=os.path.join(
        FIGURE_DIR,
        "figure_dnn_training_history.png",
    ),
)


plot_predictions(
    df_predictions=df_cv_predictions,
    targets=targets,

    output_path=os.path.join(
        FIGURE_DIR,
        "figure_dnn_predicted_vs_actual.png",
    ),
)


print()
print(
    "4-fold CV summary:"
)

print(
    df_cv_summary.to_string(
        index=False
    )
)


# ============================================================================
# 3. Hyperparameter sensitivity analysis
# ============================================================================

print()
print(
    "=" * 72
)

print(
    "3. HYPERPARAMETER "
    "SENSITIVITY ANALYSIS"
)

print(
    "=" * 72
)


sensitivity_values = (
    build_sensitivity_values(
        best_hyperparameters
    )
)


print()
print(
    "Calculating 3-fold baseline..."
)


(
    df_baseline_folds,
    df_baseline_summary,
    df_baseline_history,
    df_baseline_predictions,
) = cross_validate_dnn(
    x=x,
    y=y,
    targets=targets,
    hyperparameters=best_hyperparameters,
    epochs=EPOCHS,
    n_splits=3,
    random_state=RANDOM_STATE,
)


baseline_result = {
    "Mean NRMSE":
        df_baseline_summary[
            "NRMSE Mean"
        ].mean(),

    "Total Training Time [s]":
        (
            df_baseline_folds
            .groupby(
                "Fold"
            )[
                "Training Time [s]"
            ]
            .first()
            .sum()
        ),
}


for _, row in (
    df_baseline_summary.iterrows()
):
    target = row[
        "Target"
    ]

    if target.startswith(
        "Specific Growth"
    ):
        short = "Growth"

    elif target.startswith(
        "Specific Death"
    ):
        short = "Death"

    elif target.startswith(
        "Specific Production"
    ):
        short = "Production"

    else:
        short = target

    baseline_result[
        f"{short} RMSE Mean"
    ] = row[
        "RMSE Mean"
    ]

    baseline_result[
        f"{short} NRMSE Mean"
    ] = row[
        "NRMSE Mean"
    ]


all_sensitivity_tables = []


for (
    parameter_name,
    values,
) in sensitivity_values.items():

    df_sensitivity = (
        run_sensitivity_analysis(
            x=x,
            y=y,
            targets=targets,
            best_hyperparameters=best_hyperparameters,
            parameter_name=parameter_name,
            values=values,
            baseline_result=baseline_result,
            epochs=EPOCHS,
            n_splits=3,
            random_state=RANDOM_STATE,
        )
    )

    safe_name = (
        parameter_name.lower()
    )

    df_sensitivity.to_csv(
        os.path.join(
            TABLE_DIR,
            (
                "table_sensitivity_"
                f"{safe_name}.csv"
            ),
        ),
        index=False,
    )

    plot_sensitivity(
        df_sensitivity=df_sensitivity,
        parameter_name=parameter_name,

        output_path=os.path.join(
            FIGURE_DIR,
            (
                "figure_sensitivity_"
                f"{safe_name}.png"
            ),
        ),
    )

    all_sensitivity_tables.append(
        df_sensitivity
    )


df_sensitivity_all = pd.concat(
    all_sensitivity_tables,
    ignore_index=True,
)


df_sensitivity_all.to_csv(
    os.path.join(
        TABLE_DIR,
        "table_sensitivity_all.csv",
    ),
    index=False,
)


# ============================================================================
# 4. Polynomial regression comparison
# ============================================================================

print()
print(
    "=" * 72
)

print(
    "4. POLYNOMIAL REGRESSION "
    "COMPARISON"
)

print(
    "=" * 72
)


(
    df_polynomial,
    df_polynomial_summary,
    best_degree,
) = run_polynomial_comparison(
    x=x,
    y=y,
    targets=targets,
    degrees=(
        1,
        2,
        3,
    ),
    n_splits=4,
    random_state=RANDOM_STATE,
)


df_polynomial.to_csv(
    os.path.join(
        TABLE_DIR,
        "table_polynomial_cv.csv",
    ),
    index=False,
)


df_polynomial_summary.to_csv(
    os.path.join(
        TABLE_DIR,
        "table_polynomial_summary.csv",
    ),
    index=False,
)


df_model_comparison = (
    make_model_comparison(
        df_dnn_summary=df_cv_summary,
        df_dnn_folds=df_cv_folds,
        df_poly_degree_summary=df_polynomial_summary,
        best_degree=best_degree,
    )
)


df_model_comparison.to_csv(
    os.path.join(
        TABLE_DIR,
        "table_model_comparison.csv",
    ),
    index=False,
)


plot_model_comparison(
    df_comparison=df_model_comparison,

    output_path=os.path.join(
        FIGURE_DIR,
        "figure_model_comparison.png",
    ),
)


print()
print(
    f"Best polynomial degree: "
    f"{best_degree}"
)

print()

print(
    "Model comparison:"
)

print(
    df_model_comparison.to_string(
        index=False
    )
)


print()
print(
    "=" * 72
)

print(
    "PROCESS FINISHED SUCCESSFULLY"
)

print(
    "=" * 72
)