import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import (
    KFold,
    ParameterGrid,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
)

from src.classes import FullyConnectedNeuralNetwork


def load_dataset(dataset_path):
    df = pd.read_csv(
        dataset_path
    )

    targets = (
        df.columns[:3]
        .to_list()
    )

    features = (
        df.columns[3:]
        .to_list()
    )

    x = (
        df.loc[:, features]
        .to_numpy()
    )

    y = (
        df.loc[:, targets]
        .to_numpy()
    )

    return (
        df,
        x,
        y,
        features,
        targets,
    )


def calculate_metrics(
    y_true,
    y_pred,
    targets,
):
    records = []

    iqr = (
        np.quantile(
            y_true,
            q=0.75,
            axis=0,
        )
        -
        np.quantile(
            y_true,
            q=0.25,
            axis=0,
        )
    )

    for j, target in enumerate(
        targets
    ):
        rmse = root_mean_squared_error(
            y_true[:, j],
            y_pred[:, j],
        )

        nrmse = (
            rmse
            / iqr[j]
        )

        records.append(
            {
                "Target":
                    target,

                "RMSE":
                    rmse,

                "NRMSE":
                    nrmse,
            }
        )

    return (
        pd.DataFrame.from_records(
            records
        )
    )


def short_target_name(
    target
):
    if target.startswith(
        "Specific Growth"
    ):
        return "Growth"

    if target.startswith(
        "Specific Death"
    ):
        return "Death"

    if target.startswith(
        "Specific Production"
    ):
        return "Production"

    return target


def flatten_metrics(
    df_metrics,
    targets,
):
    record = {}

    for target in targets:
        row = (
            df_metrics.loc[
                df_metrics[
                    "Target"
                ]
                == target
            ]
            .iloc[0]
        )

        short = (
            short_target_name(
                target
            )
        )

        record[
            f"{short} RMSE"
        ] = row[
            "RMSE"
        ]

        record[
            f"{short} NRMSE"
        ] = row[
            "NRMSE"
        ]

    record[
        "Mean NRMSE"
    ] = (
        df_metrics[
            "NRMSE"
        ].mean()
    )

    return record


def train_one_model(
    x_train,
    y_train,
    x_test,
    y_test,
    targets,
    hyperparameters,
    epochs,
    random_state,
):
    model = FullyConnectedNeuralNetwork(
        epochs=epochs,
        verbose=False,
        random_state=random_state,
        **hyperparameters,
    )

    start = (
        time.perf_counter()
    )

    model.fit(
        x_train,
        y_train,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    y_pred = model.predict(
        x_test
    )

    df_metrics = (
        calculate_metrics(
            y_true=y_test,
            y_pred=y_pred,
            targets=targets,
        )
    )

    return (
        model,
        y_pred,
        df_metrics,
        elapsed,
    )


def run_hyperparameter_search(
    x,
    y,
    targets,
    epochs=400,
    random_state=0,
):
    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    param_grid_stage_1 = dict(
        hidden_layers=[
            (5,),
            (6, 5, 4),
            (
                8,
                16,
                32,
                16,
                8,
                4,
            ),
        ],

        activation=[
            "tanh",
            "relu",
        ],

        learning_rate=[
            0.001,
            0.01,
        ],
    )

    combinations_stage_1 = list(
        ParameterGrid(
            param_grid_stage_1
        )
    )

    records_stage_1 = []

    print()
    print(
        "Stage 1: architecture, activation, "
        "and learning rate"
    )

    for i, hyperparameters in enumerate(
        combinations_stage_1,
        start=1,
    ):
        hyperparameters = dict(
            hyperparameters
        )

        hyperparameters[
            "batch_size"
        ] = 1024

        hyperparameters[
            "weight_decay"
        ] = 0.0

        (
            model,
            y_pred,
            df_metrics,
            elapsed,
        ) = train_one_model(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            targets=targets,
            hyperparameters=hyperparameters,
            epochs=epochs,
            random_state=random_state,
        )

        record = {
            "Stage":
                1,

            "hidden_layers":
                hyperparameters[
                    "hidden_layers"
                ],

            "activation":
                hyperparameters[
                    "activation"
                ],

            "learning_rate":
                hyperparameters[
                    "learning_rate"
                ],

            "batch_size":
                hyperparameters[
                    "batch_size"
                ],

            "weight_decay":
                hyperparameters[
                    "weight_decay"
                ],

            "Training Time [s]":
                elapsed,
        }

        record.update(
            flatten_metrics(
                df_metrics,
                targets,
            )
        )

        records_stage_1.append(
            record
        )

        print(
            f"{i:02d}/"
            f"{len(combinations_stage_1):02d}"
            f" | Mean NRMSE = "
            f"{record['Mean NRMSE']:.4f}"
            f" | {hyperparameters}"
        )

    df_stage_1 = (
        pd.DataFrame.from_records(
            records_stage_1
        )
    )

    idx_best_stage_1 = (
        df_stage_1[
            "Mean NRMSE"
        ].idxmin()
    )

    best_stage_1 = (
        df_stage_1.loc[
            idx_best_stage_1
        ]
    )

    param_grid_stage_2 = dict(
        batch_size=[
            256,
            1024,
            4096,
        ],

        weight_decay=[
            0.0,
            0.001,
            0.01,
        ],
    )

    combinations_stage_2 = list(
        ParameterGrid(
            param_grid_stage_2
        )
    )

    records_stage_2 = []

    base_hyperparameters = {
        "hidden_layers":
            tuple(
                best_stage_1[
                    "hidden_layers"
                ]
            ),

        "activation":
            best_stage_1[
                "activation"
            ],

        "learning_rate":
            float(
                best_stage_1[
                    "learning_rate"
                ]
            ),
    }

    print()
    print(
        "Stage 2: batch size "
        "and weight decay"
    )

    for i, values_stage_2 in enumerate(
        combinations_stage_2,
        start=1,
    ):
        hyperparameters = deepcopy(
            base_hyperparameters
        )

        hyperparameters.update(
            values_stage_2
        )

        (
            model,
            y_pred,
            df_metrics,
            elapsed,
        ) = train_one_model(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            targets=targets,
            hyperparameters=hyperparameters,
            epochs=epochs,
            random_state=random_state,
        )

        record = {
            "Stage":
                2,

            "hidden_layers":
                hyperparameters[
                    "hidden_layers"
                ],

            "activation":
                hyperparameters[
                    "activation"
                ],

            "learning_rate":
                hyperparameters[
                    "learning_rate"
                ],

            "batch_size":
                hyperparameters[
                    "batch_size"
                ],

            "weight_decay":
                hyperparameters[
                    "weight_decay"
                ],

            "Training Time [s]":
                elapsed,
        }

        record.update(
            flatten_metrics(
                df_metrics,
                targets,
            )
        )

        records_stage_2.append(
            record
        )

        print(
            f"{i:02d}/"
            f"{len(combinations_stage_2):02d}"
            f" | Mean NRMSE = "
            f"{record['Mean NRMSE']:.4f}"
            f" | {hyperparameters}"
        )

    df_stage_2 = (
        pd.DataFrame.from_records(
            records_stage_2
        )
    )

    idx_best_stage_2 = (
        df_stage_2[
            "Mean NRMSE"
        ].idxmin()
    )

    best_stage_2 = (
        df_stage_2.loc[
            idx_best_stage_2
        ]
    )

    best_hyperparameters = {
        "hidden_layers":
            tuple(
                best_stage_2[
                    "hidden_layers"
                ]
            ),

        "activation":
            best_stage_2[
                "activation"
            ],

        "learning_rate":
            float(
                best_stage_2[
                    "learning_rate"
                ]
            ),

        "batch_size":
            int(
                best_stage_2[
                    "batch_size"
                ]
            ),

        "weight_decay":
            float(
                best_stage_2[
                    "weight_decay"
                ]
            ),
    }

    df_search = pd.concat(
        [
            df_stage_1,
            df_stage_2,
        ],
        ignore_index=True,
    )

    return (
        df_search,
        best_hyperparameters,
    )


def cross_validate_dnn(
    x,
    y,
    targets,
    hyperparameters,
    epochs=400,
    n_splits=4,
    random_state=0,
):
    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    metric_records = []
    history_records = []
    prediction_records = []

    for fold, (
        idx_train,
        idx_test,
    ) in enumerate(
        kfold.split(x),
        start=1,
    ):
        model = FullyConnectedNeuralNetwork(
            epochs=epochs,
            verbose=False,
            random_state=(
                random_state
                + fold
            ),
            **hyperparameters,
        )

        start = (
            time.perf_counter()
        )

        model.fit(
            x[idx_train],
            y[idx_train],
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        y_pred = model.predict(
            x[idx_test]
        )

        df_metrics = (
            calculate_metrics(
                y_true=y[
                    idx_test
                ],
                y_pred=y_pred,
                targets=targets,
            )
        )

        for _, row in (
            df_metrics.iterrows()
        ):
            metric_records.append(
                {
                    "Fold":
                        fold,

                    "Target":
                        row[
                            "Target"
                        ],

                    "RMSE":
                        row[
                            "RMSE"
                        ],

                    "NRMSE":
                        row[
                            "NRMSE"
                        ],

                    "Best Epoch":
                        model.best_epoch_,

                    "Best Validation Loss":
                        model.best_loss_valid_,

                    "Training Time [s]":
                        elapsed,
                }
            )

        for (
            epoch,
            loss_train,
            loss_valid,
        ) in zip(
            model.history_[
                "epoch"
            ],
            model.history_[
                "loss_train"
            ],
            model.history_[
                "loss_valid"
            ],
        ):
            history_records.append(
                {
                    "Fold":
                        fold,

                    "Epoch":
                        epoch,

                    "Loss Train":
                        loss_train,

                    "Loss Valid":
                        loss_valid,
                }
            )

        for (
            row_position,
            sample_index,
        ) in enumerate(
            idx_test
        ):
            record = {
                "Fold":
                    fold,

                "Sample Index":
                    int(
                        sample_index
                    ),
            }

            for j, target in enumerate(
                targets
            ):
                short = short_target_name(
                    target
                )

                record[
                    f"Actual {short}"
                ] = y[
                    sample_index,
                    j,
                ]

                record[
                    f"Predicted {short}"
                ] = y_pred[
                    row_position,
                    j,
                ]

            prediction_records.append(
                record
            )

        print(
            f"Completed fold "
            f"{fold}/{n_splits}"
        )

    df_folds = (
        pd.DataFrame.from_records(
            metric_records
        )
    )

    df_history = (
        pd.DataFrame.from_records(
            history_records
        )
    )

    df_predictions = (
        pd.DataFrame.from_records(
            prediction_records
        )
    )

    summary_records = []

    for target in targets:
        df_target = (
            df_folds.loc[
                df_folds[
                    "Target"
                ]
                == target
            ]
        )

        summary_records.append(
            {
                "Target":
                    target,

                "RMSE Mean":
                    df_target[
                        "RMSE"
                    ].mean(),

                "RMSE SD":
                    df_target[
                        "RMSE"
                    ].std(
                        ddof=0
                    ),

                "NRMSE Mean":
                    df_target[
                        "NRMSE"
                    ].mean(),

                "NRMSE SD":
                    df_target[
                        "NRMSE"
                    ].std(
                        ddof=0
                    ),
            }
        )

    df_summary = (
        pd.DataFrame.from_records(
            summary_records
        )
    )

    return (
        df_folds,
        df_summary,
        df_history,
        df_predictions,
    )


def build_sensitivity_values(
    best_hyperparameters,
):
    baseline_layers = tuple(
        best_hyperparameters[
            "hidden_layers"
        ]
    )

    architecture_map = {
        (5,): [
            (3,),
            (4,),
            (5,),
            (6, 5, 4),
            (
                8,
                16,
                32,
                16,
                8,
                4,
            ),
        ],

        (6, 5, 4): [
            (4,),
            (5,),
            (6, 5, 4),
            (
                8,
                16,
                32,
                16,
                8,
                4,
            ),
            (
                16,
                32,
                64,
                32,
                16,
                8,
            ),
        ],

        (
            8,
            16,
            32,
            16,
            8,
            4,
        ): [
            (5,),
            (6, 5, 4),
            (
                8,
                16,
                32,
                16,
                8,
                4,
            ),
            (
                16,
                32,
                64,
                32,
                16,
                8,
            ),
            (
                32,
                64,
                128,
                64,
                32,
                16,
            ),
        ],
    }

    hidden_layers_values = (
        architecture_map[
            baseline_layers
        ]
    )

    learning_rate = (
        best_hyperparameters[
            "learning_rate"
        ]
    )

    if np.isclose(
        learning_rate,
        0.001,
    ):
        learning_rate_values = [
            0.0001,
            0.0005,
            0.001,
            0.005,
            0.01,
        ]

    else:
        learning_rate_values = [
            0.001,
            0.005,
            0.01,
            0.02,
            0.05,
        ]

    batch_size = int(
        best_hyperparameters[
            "batch_size"
        ]
    )

    batch_size_values = [
        max(
            16,
            batch_size // 4,
        ),

        max(
            32,
            batch_size // 2,
        ),

        batch_size,

        batch_size * 2,

        batch_size * 4,
    ]

    batch_size_values = list(
        dict.fromkeys(
            batch_size_values
        )
    )

    weight_decay = float(
        best_hyperparameters[
            "weight_decay"
        ]
    )

    if np.isclose(
        weight_decay,
        0.0,
    ):
        weight_decay_values = [
            0.0,
            1e-5,
            1e-4,
            1e-3,
            1e-2,
        ]

    elif np.isclose(
        weight_decay,
        0.001,
    ):
        weight_decay_values = [
            1e-5,
            1e-4,
            1e-3,
            1e-2,
            1e-1,
        ]

    else:
        weight_decay_values = [
            1e-4,
            1e-3,
            1e-2,
            5e-2,
            1e-1,
        ]

    return {
        "hidden_layers":
            hidden_layers_values,

        "activation": [
            "sigmoid",
            "tanh",
            "relu",
            "leakyrelu",
        ],

        "learning_rate":
            learning_rate_values,

        "batch_size":
            batch_size_values,

        "weight_decay":
            weight_decay_values,
    }


def run_sensitivity_analysis(
    x,
    y,
    targets,
    best_hyperparameters,
    parameter_name,
    values,
    baseline_result,
    epochs=400,
    n_splits=3,
    random_state=0,
):
    records = []

    print()
    print(
        f"Sensitivity analysis: "
        f"{parameter_name}"
    )

    baseline_value = (
        best_hyperparameters[
            parameter_name
        ]
    )

    for value in values:

        if value == baseline_value:
            record = {
                "Parameter":
                    parameter_name,

                "Value":
                    value,

                "Mean NRMSE":
                    baseline_result[
                        "Mean NRMSE"
                    ],

                "Total Training Time [s]":
                    baseline_result[
                        "Total Training Time [s]"
                    ],
            }

            for target in targets:
                short = (
                    short_target_name(
                        target
                    )
                )

                record[
                    f"{short} RMSE Mean"
                ] = baseline_result[
                    f"{short} RMSE Mean"
                ]

                record[
                    f"{short} NRMSE Mean"
                ] = baseline_result[
                    f"{short} NRMSE Mean"
                ]

            records.append(
                record
            )

            print(
                f"{parameter_name} = "
                f"{value}"
                f" | Mean NRMSE = "
                f"{record['Mean NRMSE']:.4f}"
                f" | baseline reused"
            )

            continue

        hyperparameters = deepcopy(
            best_hyperparameters
        )

        hyperparameters[
            parameter_name
        ] = value

        (
            df_folds,
            df_summary,
            df_history,
            df_predictions,
        ) = cross_validate_dnn(
            x=x,
            y=y,
            targets=targets,
            hyperparameters=hyperparameters,
            epochs=epochs,
            n_splits=n_splits,
            random_state=random_state,
        )

        record = {
            "Parameter":
                parameter_name,

            "Value":
                value,

            "Mean NRMSE":
                df_summary[
                    "NRMSE Mean"
                ].mean(),
        }

        for _, row in (
            df_summary.iterrows()
        ):
            short = (
                short_target_name(
                    row[
                        "Target"
                    ]
                )
            )

            record[
                f"{short} RMSE Mean"
            ] = row[
                "RMSE Mean"
            ]

            record[
                f"{short} NRMSE Mean"
            ] = row[
                "NRMSE Mean"
            ]

        record[
            "Total Training Time [s]"
        ] = (
            df_folds
            .groupby(
                "Fold"
            )[
                "Training Time [s]"
            ]
            .first()
            .sum()
        )

        records.append(
            record
        )

        print(
            f"{parameter_name} = "
            f"{value}"
            f" | Mean NRMSE = "
            f"{record['Mean NRMSE']:.4f}"
        )

    return (
        pd.DataFrame.from_records(
            records
        )
    )


def run_polynomial_comparison(
    x,
    y,
    targets,
    degrees=(1, 2, 3),
    n_splits=4,
    random_state=0,
):
    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    records = []

    for degree in degrees:
        start = (
            time.perf_counter()
        )

        fold_nrmse = []
        degree_records = []

        for fold, (
            idx_train,
            idx_test,
        ) in enumerate(
            kfold.split(x),
            start=1,
        ):
            model = make_pipeline(
                StandardScaler(),

                PolynomialFeatures(
                    degree=degree,
                    include_bias=False,
                ),

                LinearRegression(),
            )

            model.fit(
                x[idx_train],
                y[idx_train],
            )

            y_pred = model.predict(
                x[idx_test]
            )

            df_metrics = (
                calculate_metrics(
                    y_true=y[
                        idx_test
                    ],
                    y_pred=y_pred,
                    targets=targets,
                )
            )

            fold_nrmse.append(
                df_metrics[
                    "NRMSE"
                ].mean()
            )

            for _, row in (
                df_metrics.iterrows()
            ):
                degree_records.append(
                    {
                        "Degree":
                            degree,

                        "Fold":
                            fold,

                        "Target":
                            row[
                                "Target"
                            ],

                        "RMSE":
                            row[
                                "RMSE"
                            ],

                        "NRMSE":
                            row[
                                "NRMSE"
                            ],
                    }
                )

        elapsed = (
            time.perf_counter()
            - start
        )

        mean_nrmse = np.mean(
            fold_nrmse
        )

        for record in (
            degree_records
        ):
            record[
                "Degree Total Time [s]"
            ] = elapsed

            records.append(
                record
            )

        print(
            f"Polynomial degree "
            f"{degree}"
            f" | Mean NRMSE = "
            f"{mean_nrmse:.4f}"
            f" | Time = "
            f"{elapsed:.2f} s"
        )

    df_polynomial = (
        pd.DataFrame.from_records(
            records
        )
    )

    df_degree_summary = (
        df_polynomial
        .groupby(
            "Degree",
            as_index=False,
        )
        .agg(
            Mean_NRMSE=(
                "NRMSE",
                "mean",
            ),

            Total_Training_Time_s=(
                "Degree Total Time [s]",
                "first",
            ),
        )
    )

    best_degree = int(
        df_degree_summary.loc[
            df_degree_summary[
                "Mean_NRMSE"
            ].idxmin(),
            "Degree",
        ]
    )

    return (
        df_polynomial,
        df_degree_summary,
        best_degree,
    )


def make_model_comparison(
    df_dnn_summary,
    df_dnn_folds,
    df_poly_degree_summary,
    best_degree,
):
    dnn_mean_nrmse = (
        df_dnn_summary[
            "NRMSE Mean"
        ].mean()
    )

    dnn_training_time = (
        df_dnn_folds
        .groupby(
            "Fold"
        )[
            "Training Time [s]"
        ]
        .first()
        .sum()
    )

    poly_row = (
        df_poly_degree_summary.loc[
            df_poly_degree_summary[
                "Degree"
            ]
            == best_degree
        ]
        .iloc[0]
    )

    return pd.DataFrame(
        {
            "Model": [
                (
                    "Optimized Fully Connected "
                    "Neural Network"
                ),

                (
                    "Polynomial Regression "
                    f"(degree {best_degree})"
                ),
            ],

            "Mean 4-Fold CV NRMSE": [
                dnn_mean_nrmse,

                poly_row[
                    "Mean_NRMSE"
                ],
            ],

            (
                "Approximate CV "
                "Training Time [s]"
            ): [
                dnn_training_time,

                poly_row[
                    "Total_Training_Time_s"
                ],
            ],
        }
    )


def plot_cv_history(
    df_history,
    output_path,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 7),
        layout="constrained",
    )

    axes = axes.ravel()

    for fold in range(
        1,
        5,
    ):
        ax = axes[
            fold - 1
        ]

        df_fold = (
            df_history.loc[
                df_history[
                    "Fold"
                ]
                == fold
            ]
        )

        ax.plot(
            df_fold[
                "Epoch"
            ],
            df_fold[
                "Loss Train"
            ],
            label="Training",
        )

        ax.plot(
            df_fold[
                "Epoch"
            ],
            df_fold[
                "Loss Valid"
            ],
            label="Validation",
        )

        ax.set_title(
            f"Fold {fold}"
        )

        ax.set_xlabel(
            "Epoch"
        )

        ax.set_ylabel(
            "Loss"
        )

        ax.legend()

    fig.savefig(
        output_path,
        dpi=300,
    )

    plt.close(
        fig
    )


def plot_predictions(
    df_predictions,
    targets,
    output_path,
):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
        layout="constrained",
    )

    for i, target in enumerate(
        targets
    ):
        short = (
            short_target_name(
                target
            )
        )

        actual = (
            df_predictions[
                f"Actual {short}"
            ]
        )

        predicted = (
            df_predictions[
                f"Predicted {short}"
            ]
        )

        lower = min(
            actual.min(),
            predicted.min(),
        )

        upper = max(
            actual.max(),
            predicted.max(),
        )

        axes[i].scatter(
            actual,
            predicted,
            s=6,
            alpha=0.25,
        )

        axes[i].plot(
            [
                lower,
                upper,
            ],
            [
                lower,
                upper,
            ],
            linestyle="--",
        )

        axes[i].set_xlabel(
            "Actual"
        )

        axes[i].set_ylabel(
            "Predicted"
        )

        axes[i].set_title(
            short
        )

    fig.savefig(
        output_path,
        dpi=300,
    )

    plt.close(
        fig
    )


def plot_search_results(
    df_search,
    output_path,
):
    df_plot = (
        df_search
        .reset_index(
            drop=True
        )
        .copy()
    )

    df_plot[
        "Search"
    ] = np.arange(
        1,
        df_plot.shape[0]
        + 1,
    )

    fig, ax = plt.subplots(
        figsize=(8, 4),
        layout="constrained",
    )

    ax.scatter(
        df_plot[
            "Search"
        ],
        df_plot[
            "Mean NRMSE"
        ],
    )

    ax.set_xlabel(
        "Hyperparameter Set"
    )

    ax.set_ylabel(
        "Mean NRMSE"
    )

    ax.set_title(
        "Held-Out Hyperparameter Search"
    )

    fig.savefig(
        output_path,
        dpi=300,
    )

    plt.close(
        fig
    )


def plot_sensitivity(
    df_sensitivity,
    parameter_name,
    output_path,
):
    values = (
        df_sensitivity[
            "Value"
        ]
        .astype(str)
        .to_list()
    )

    x_pos = np.arange(
        len(values)
    )

    fig, ax = plt.subplots(
        figsize=(7, 4),
        layout="constrained",
    )

    ax.plot(
        x_pos,
        df_sensitivity[
            "Mean NRMSE"
        ],
        marker="o",
    )

    ax.set_xticks(
        x_pos
    )

    ax.set_xticklabels(
        values,
        rotation=30,
        ha="right",
    )

    ax.set_xlabel(
        parameter_name
    )

    ax.set_ylabel(
        "Mean 3-Fold CV NRMSE"
    )

    ax.set_title(
        f"Sensitivity to "
        f"{parameter_name}"
    )

    fig.savefig(
        output_path,
        dpi=300,
    )

    plt.close(
        fig
    )


def plot_model_comparison(
    df_comparison,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(7, 4),
        layout="constrained",
    )

    ax.bar(
        df_comparison[
            "Model"
        ],
        df_comparison[
            "Mean 4-Fold CV NRMSE"
        ],
    )

    ax.set_ylabel(
        "Mean 4-Fold CV NRMSE"
    )

    ax.set_title(
        "Model Comparison"
    )

    ax.tick_params(
        axis="x",
        labelrotation=15,
    )

    fig.savefig(
        output_path,
        dpi=300,
    )

    plt.close(
        fig
    )