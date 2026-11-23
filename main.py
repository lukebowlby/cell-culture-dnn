"""Cell-culture DNN model search, cross-validation, and analysis.

Full training:
    python -u main.py

Regenerate reports without retraining:
    python main.py --results-only

Short test:
    python main.py --epochs 5
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

from src.functions import (
    add_architecture_parameter_counts,
    build_sensitivity_values,
    compare_target_performance,
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
    summarize_cv_folds,
)

RANDOM_STATE = 0
EPOCHS = 400
DATASET_PATH = os.path.join(
    "datasets", "dataset_cell_growth_7factor.csv"
)
FIGURE_DIR = "figures"
TABLE_DIR = "tables"


def table_path(table_dir, name):
    return os.path.join(table_dir, name)


def figure_path(figure_dir, name):
    return os.path.join(figure_dir, name)


def store_table(df, table_dir, name):
    df.to_csv(table_path(table_dir, name), index=False)


def load_saved_table(table_dir, name):
    path = table_path(table_dir, name)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Cannot find {path}. "
            "Results-only mode requires the saved "
            "400-epoch tables and does not retrain."
        )

    return pd.read_csv(path)


def print_section(number, name):
    print("\n" + "=" * 72)
    print(f"{number}. {name}")
    print("=" * 72)


def run_full_analysis(
    x, y, targets, n_features, epochs, figure_dir, table_dir
):
    print_section(1, "HYPERPARAMETER SEARCH")

    df_search, best_hyperparameters = run_hyperparameter_search(
        x=x,
        y=y,
        targets=targets,
        epochs=epochs,
        random_state=RANDOM_STATE,
    )

    store_table(
        df_search, table_dir, "table_hyperparameter_search.csv"
    )

    df_best_hyperparameters = pd.DataFrame({
        "Hyperparameter": list(best_hyperparameters.keys()),
        "Optimal Value": [
            str(value) for value in best_hyperparameters.values()
        ],
    })

    store_table(
        df_best_hyperparameters,
        table_dir,
        "table_best_hyperparameters.csv",
    )

    print("\nOptimal hyperparameters:")
    for name, value in best_hyperparameters.items():
        print(f"  {name}: {value}")

    print_section(2, "FOUR-FOLD CROSS-VALIDATION")

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
        epochs=epochs,
        n_splits=4,
        random_state=RANDOM_STATE,
    )

    df_cv_summary = summarize_cv_folds(
        df_cv_folds, targets
    )

    store_table(
        df_cv_folds, table_dir, "table_dnn_cv_folds.csv"
    )
    store_table(
        df_cv_summary, table_dir, "table_dnn_cv_summary.csv"
    )
    store_table(
        df_cv_history, table_dir, "table_dnn_training_history.csv"
    )
    store_table(
        df_cv_predictions, table_dir, "table_dnn_cv_predictions.csv"
    )

    print("\n4-fold CV summary (sample standard deviation):")
    print(df_cv_summary.to_string(index=False))

    print_section(3, "HYPERPARAMETER SENSITIVITY ANALYSIS")

    sensitivity_values = build_sensitivity_values(
        best_hyperparameters
    )

    print("\nCalculating 3-fold baseline...")

    baseline_folds, baseline_summary, _, _ = cross_validate_dnn(
        x=x,
        y=y,
        targets=targets,
        hyperparameters=best_hyperparameters,
        epochs=epochs,
        n_splits=3,
        random_state=RANDOM_STATE,
    )

    baseline_result = {
        "Mean NRMSE": baseline_summary["NRMSE Mean"].mean(),
        "Total Training Time [s]": (
            baseline_folds
            .groupby("Fold")["Training Time [s]"]
            .first()
            .sum()
        ),
    }

    from src.functions import short_target_name

    for _, row in baseline_summary.iterrows():
        short = short_target_name(row["Target"])

        baseline_result[
            f"{short} RMSE Mean"
        ] = row["RMSE Mean"]

        baseline_result[
            f"{short} NRMSE Mean"
        ] = row["NRMSE Mean"]

    sensitivity_tables = []

    for parameter_name, values in sensitivity_values.items():
        df_sensitivity = run_sensitivity_analysis(
            x=x,
            y=y,
            targets=targets,
            best_hyperparameters=best_hyperparameters,
            parameter_name=parameter_name,
            values=values,
            baseline_result=baseline_result,
            epochs=epochs,
            n_splits=3,
            random_state=RANDOM_STATE,
        )

        if parameter_name == "hidden_layers":
            df_sensitivity = add_architecture_parameter_counts(
                df_sensitivity,
                n_features=n_features,
                n_targets=len(targets),
            )

            df_architecture = df_sensitivity[
                [
                    "Value",
                    "Trainable Parameters",
                    "Mean NRMSE",
                    "Total Training Time [s]",
                ]
            ].rename(columns={"Value": "Hidden Layers"})

            store_table(
                df_architecture,
                table_dir,
                "table_architecture_complexity.csv",
            )

        store_table(
            df_sensitivity,
            table_dir,
            f"table_sensitivity_{parameter_name}.csv",
        )

        sensitivity_tables.append(df_sensitivity)

    store_table(
        pd.concat(sensitivity_tables, ignore_index=True),
        table_dir,
        "table_sensitivity_all.csv",
    )

    print_section(4, "POLYNOMIAL REGRESSION COMPARISON")

    (
        df_polynomial,
        df_poly_summary,
        best_degree,
    ) = run_polynomial_comparison(
        x=x,
        y=y,
        targets=targets,
        degrees=(1, 2, 3),
        n_splits=4,
        random_state=RANDOM_STATE,
    )

    store_table(
        df_polynomial, table_dir, "table_polynomial_cv.csv"
    )
    store_table(
        df_poly_summary, table_dir, "table_polynomial_summary.csv"
    )

    build_reporting_outputs(
        df_search=df_search,
        df_cv_folds=df_cv_folds,
        df_cv_history=df_cv_history,
        df_cv_predictions=df_cv_predictions,
        df_polynomial=df_polynomial,
        df_poly_summary=df_poly_summary,
        sensitivity_tables=sensitivity_tables,
        targets=targets,
        n_features=n_features,
        figure_dir=figure_dir,
        table_dir=table_dir,
    )

    print(f"\nBest polynomial degree: {best_degree}")
    print("\nPROCESS FINISHED SUCCESSFULLY")


def build_reporting_outputs(
    df_search,
    df_cv_folds,
    df_cv_history,
    df_cv_predictions,
    df_polynomial,
    df_poly_summary,
    sensitivity_tables,
    targets,
    n_features,
    figure_dir,
    table_dir,
):
    """Generate reports from data without retraining models."""

    df_cv_summary = summarize_cv_folds(
        df_cv_folds, targets
    )

    store_table(
        df_cv_summary,
        table_dir,
        "table_dnn_cv_summary.csv",
    )

    best_degree = int(
        df_poly_summary.loc[
            df_poly_summary["Mean_NRMSE"].idxmin(),
            "Degree",
        ]
    )

    df_model_comparison = make_model_comparison(
        df_dnn_summary=df_cv_summary,
        df_dnn_folds=df_cv_folds,
        df_poly_degree_summary=df_poly_summary,
        best_degree=best_degree,
    )

    store_table(
        df_model_comparison,
        table_dir,
        "table_model_comparison.csv",
    )

    df_target_comparison = compare_target_performance(
        df_dnn_summary=df_cv_summary,
        df_polynomial_folds=df_polynomial,
        best_degree=best_degree,
        targets=targets,
    )

    store_table(
        df_target_comparison,
        table_dir,
        "table_target_comparison.csv",
    )

    plot_search_results(
        df_search=df_search,
        output_path=figure_path(
            figure_dir, "figure_hyperparameter_search.png"
        ),
    )

    plot_cv_history(
        df_history=df_cv_history,
        output_path=figure_path(
            figure_dir, "figure_dnn_training_history.png"
        ),
    )

    plot_predictions(
        df_predictions=df_cv_predictions,
        targets=targets,
        output_path=figure_path(
            figure_dir, "figure_dnn_predicted_vs_actual.png"
        ),
    )

    plot_model_comparison(
        df_comparison=df_model_comparison,
        output_path=figure_path(
            figure_dir, "figure_model_comparison.png"
        ),
    )

    for df_sensitivity in sensitivity_tables:
        parameter = str(
            df_sensitivity["Parameter"].iloc[0]
        )

        plot_sensitivity(
            df_sensitivity=df_sensitivity,
            parameter_name=parameter,
            output_path=figure_path(
                figure_dir,
                f"figure_sensitivity_{parameter}.png",
            ),
        )

        if parameter == "hidden_layers":
            df_architecture = add_architecture_parameter_counts(
                df_sensitivity,
                n_features=n_features,
                n_targets=len(targets),
            )

            df_architecture = df_architecture[
                [
                    "Value",
                    "Trainable Parameters",
                    "Mean NRMSE",
                    "Total Training Time [s]",
                ]
            ].rename(columns={"Value": "Hidden Layers"})

            store_table(
                df_architecture,
                table_dir,
                "table_architecture_complexity.csv",
            )

    print("\n4-fold CV summary (sample SD, IQR-normalized NRMSE):")
    print(df_cv_summary.to_string(index=False))

    print("\nTarget-by-target comparison:")
    print(df_target_comparison.to_string(index=False))

    print("\nOverall model comparison:")
    print(df_model_comparison.to_string(index=False))

    print("\nReporting tables and figures updated without training.")


def reuse_saved_results(
    targets, n_features, figure_dir, table_dir
):
    print("\nRESULTS-ONLY MODE: reading saved CSV files.")

    df_search = load_saved_table(
        table_dir, "table_hyperparameter_search.csv"
    )
    df_cv_folds = load_saved_table(
        table_dir, "table_dnn_cv_folds.csv"
    )
    df_cv_history = load_saved_table(
        table_dir, "table_dnn_training_history.csv"
    )
    df_cv_predictions = load_saved_table(
        table_dir, "table_dnn_cv_predictions.csv"
    )
    df_polynomial = load_saved_table(
        table_dir, "table_polynomial_cv.csv"
    )
    df_poly_summary = load_saved_table(
        table_dir, "table_polynomial_summary.csv"
    )

    sensitivity_tables = [
        load_saved_table(
            table_dir, f"table_sensitivity_{parameter}.csv"
        )
        for parameter in (
            "hidden_layers",
            "activation",
            "learning_rate",
            "batch_size",
            "weight_decay",
        )
    ]

    build_reporting_outputs(
        df_search=df_search,
        df_cv_folds=df_cv_folds,
        df_cv_history=df_cv_history,
        df_cv_predictions=df_cv_predictions,
        df_polynomial=df_polynomial,
        df_poly_summary=df_poly_summary,
        sensitivity_tables=sensitivity_tables,
        targets=targets,
        n_features=n_features,
        figure_dir=figure_dir,
        table_dir=table_dir,
    )

    print("\nPROCESS FINISHED SUCCESSFULLY (RESULTS ONLY)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--results-only",
        action="store_true",
        help="Rebuild outputs without retraining",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of epochs for full training",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be positive")

    if args.results_only and args.epochs != EPOCHS:
        parser.error(
            "--results-only reads the final tables/ directory"
        )

    table_dir = TABLE_DIR
    figure_dir = FIGURE_DIR

    if args.epochs != EPOCHS:
        table_dir = f"tables_test_{args.epochs}"
        figure_dir = f"figures_test_{args.epochs}"

    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    df, x, y, features, targets = load_dataset(
        DATASET_PATH
    )

    print("=" * 72)
    print("CELL CULTURE FULLY CONNECTED NEURAL NETWORK")
    print("=" * 72)
    print(f"Samples: {df.shape[0]:,}")
    print(f"Features: {len(features)}")
    print(f"Targets: {len(targets)}")
    print(f"Epochs: {args.epochs}")

    if args.results_only:
        reuse_saved_results(
            targets=targets,
            n_features=len(features),
            figure_dir=figure_dir,
            table_dir=table_dir,
        )
    else:
        run_full_analysis(
            x=x,
            y=y,
            targets=targets,
            n_features=len(features),
            epochs=args.epochs,
            figure_dir=figure_dir,
            table_dir=table_dir,
        )


if __name__ == "__main__":
    main()