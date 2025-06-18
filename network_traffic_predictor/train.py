"""
Trains an LSTM model to forecast network traffic based on the preprocessed data.
This script handles data preparation, model training, evaluation, and result generation.
"""

import os
import random
import sys
from argparse import ArgumentParser
from pathlib import Path
from signal import SIGINT
from typing import Any, TypedDict

# pyright: reportMissingModuleSource=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential, save_model
from tensorflow.random import set_seed

from network_traffic_predictor.utils import cli_colors
from network_traffic_predictor.utils.plotting import set_theme

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
set_theme()

random.seed(0x80251BC46E68743C)
np.random.seed(0x8A8EAB5A)
set_seed(0x985E79749991772482D363C286ADBD2A)


def _extract_traffic_series(df: pl.DataFrame) -> pl.Series:
    """
    Aggregates packet data into a bytes_per_second time series.
    """
    print('Aggregating data to create bytes_per_second time series...')
    return (
        df.lazy()
        .group_by_dynamic('Timestamp', every='1s')
        .agg(bytes_per_second=pl.col('Size (bytes)').sum())
        .collect()['bytes_per_second']
    )


def _create_sequences(data: np.ndarray, look_back: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates sliding window sequences for the LSTM model.
    """
    xx: list[np.ndarray] = []
    y: list[np.ndarray] = []
    for i in range(len(data) - look_back):
        xx.append(data[i : (i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(xx), np.array(y)


class _Results(TypedDict):
    """
    Traning and evaluation results for the model.
    """

    look_back: int
    mse: float
    model: 'Model[Any, Any]'
    y_test_orig: np.ndarray
    test_predict_orig: np.ndarray


def _train_and_evaluate_model(traffic_series: pl.Series, look_back: int, epochs: int, batch_size: int) -> _Results:
    """
    Handles the full pipeline for a single look_back configuration:
    scaling, splitting, model building, training, and evaluation.
    """
    print(f'--- Processing with look_back = {look_back} ---')
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(traffic_series.to_numpy().reshape(-1, 1))

    # Create sequences
    xx, y = _create_sequences(scaled_data, look_back)
    xx = np.reshape(xx, (xx.shape[0], xx.shape[1], 1))

    # Split into 80% training / 20% test sets
    train_size = int(len(xx) * 0.8)
    X_train, X_test = xx[:train_size], xx[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    print(f'Train/Test split: {len(X_train)} / {len(X_test)} samples.')

    # Build LSTM model
    model = Sequential([
        Input((look_back, 1)),
        LSTM(50),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Training
    print(f'Training model for look_back={look_back}...')
    _ = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Evaluation
    test_predict_scaled = model.predict(X_test, verbose=0)
    test_predict_orig = scaler.inverse_transform(test_predict_scaled)
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(y_test_orig, test_predict_orig)
    print(f'Evaluation complete. Test MSE: {mse:.2f}')

    return {
        'look_back': look_back,
        'mse': mse,
        'model': model,
        'y_test_orig': y_test_orig,
        'test_predict_orig': test_predict_orig,
    }


def _find_best_model(*, input_file: Path, output_dir: Path, epochs: int, batch_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = output_dir / input_file.name

    df = pl.read_parquet(input_file)
    traffic_series = _extract_traffic_series(df)

    # --- Experiment Loop ---
    look_back_configs = (10, 20, 30)
    results = [
        _train_and_evaluate_model(traffic_series, look_back, epochs, batch_size) for look_back in look_back_configs
    ]
    best_model_info = min(results, key=lambda result: result['mse'])

    # --- Generate Final Report Assets ---
    best_model_path = basename.parent / f'{basename.stem}.best_model.e{epochs}.b{batch_size}.keras'
    save_model(best_model_info['model'], best_model_path)
    print(f'Best model (look_back={best_model_info["look_back"]}) saved to {best_model_path}')

    results_df = pl.from_dict({
        'Look Back': [result['look_back'] for result in results],
        'MSE': [result['mse'] for result in results],
    })

    print('--- MSE for Different Window Configurations ---')
    print(results_df)
    results_df.to_pandas().to_latex(
        buf=basename.parent / f'{basename.stem}.mse_results.e{epochs}.b{batch_size}.tex',
        float_format='%.2f',
        index=False,
        escape=True,
    )

    # Prediction plot
    _ = plt.figure(figsize=(18, 8))
    _ = plt.plot(best_model_info['y_test_orig'], label='Tráfego Real (Bytes/s)', color='royalblue')
    _ = plt.plot(
        best_model_info['test_predict_orig'], label='Tráfego Esperado (Bytes/s)', color='darkorange', alpha=0.9
    )
    params = f'look_back={best_model_info["look_back"]}, {epochs=}, {batch_size=}'
    _ = plt.title(rf'Tráfego Real vs. Modelado (Modelo: \texttt{{{params}}})', fontsize=16)
    _ = plt.xlabel('Passo (Segundos)', fontsize=12)
    _ = plt.ylabel('Bytes por Segundo', fontsize=12)
    _ = plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plot_path = basename.parent / f'{basename.stem}.prediction.e{epochs}.b{batch_size}.png'
    plt.savefig(plot_path, dpi=300)
    print(f'Prediction plot saved to {plot_path}')
    plt.close()


def main() -> int:
    """
    Train and evaluate an LSTM model for network traffic forecasting.
    """
    parser = ArgumentParser('train', description='Train and evaluate an LSTM model for network traffic forecasting.')
    _ = parser.add_argument('parquet_file', type=Path, help='Path to the preprocessed parquet file.')
    _ = parser.add_argument(
        '-o',
        '--output-dir',
        type=Path,
        default=Path('results'),
        help='Directory to save the model, plots, and results.',
    )
    _ = parser.add_argument('-e', '--epochs', type=int, default=20, help='Number of epochs to run.')
    _ = parser.add_argument('-b', '--batch-size', type=int, default=32, help='Number of samples per epoch.')
    _ = cli_colors.add_color_option(parser)

    args = parser.parse_args()
    try:
        _find_best_model(
            input_file=args.parquet_file,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        return 0
    except KeyboardInterrupt:
        return SIGINT


if __name__ == '__main__':
    sys.exit(main())
