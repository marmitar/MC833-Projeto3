# Network Traffic Prediction with LSTM

See [report.pdf](docs/report.pdf).

## Project Overview

This project aims to predict network traffic volume using a Long Short-Term Memory (LSTM) Recurrent Neural Network. The
core task is to build a model that forecasts the `bytes_per_second` in a time series generated from a public MAWI/CAIDA
dataset. The workflow includes data preparation, exploratory data analysis (EDA), model implementation, and a critical
analysis of the results.

## Methodology and Data

- **Dataset**: The model is trained on a time series derived from a PCAP file from the MAWI/CAIDA database.

- **Model Architecture**: A lean architecture consisting of a single LSTM layer followed by a Dense output layer is
  implemented using TensorFlow/Keras.

- **Technique**: The model uses a sliding window of past data (e.g., 10, 20, or 30 seconds) as input to predict the
  traffic volume for the subsequent second.

- **Evaluation**: The model's predictive performance is evaluated using the Mean Squared Error (MSE) metric on a test
  set comprising 20% of the data.

## Usage

### Install dependencies:

```sh
uv sync --all-extras
```

### Download and preprocess the dataset:

```sh
uv run data/download.py data/
```

```sh
uv run data/preprocess.py data/200701251400.dump
```

### Run the analysis:

```sh
uv run data/analysis.py data/200701251400.parquet
```

### Train the LSTM

```sh
uv run data/train.py data/200701251400.parquet
```

## Development

### Install dev tools

```sh
uv sync --all-extras --all-groups
```

```sh
pre-commit install
```

### Linting

```sh
uv run ruff check --fix
```

```sh
uv run pyright
```

### Formatting

```sh
uv run ruff format
```

```sh
pre-commit run -a
```
