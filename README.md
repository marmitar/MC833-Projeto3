# Network Traffic Prediction with LSTM

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
uv sync --all-extras --all-groups
```

### Download the dataset:

```sh
uv run data/download.py
```

### Run the analysis and training scripts:

...

## Disclaimer

This repository contains the implementation for Practical Assignment 3 of the MC833 - Computer Network Programming
course at UNICAMP.

More details on [project description](docs/description.md) and [working with LSTM](docs/lstm.md).
