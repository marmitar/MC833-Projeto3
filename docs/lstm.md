# Network Traffic Prediction with LSTM

Rómulo W.C. Bustincio

Institute of Computing - UNICAMP

April 2025

## Objective

To present the theoretical and practical foundations for modeling network traffic with LSTM:

- Traffic time series.
- Recurrent neural networks (RNNs) and LSTM.
- Implementation and results with TensorFlow.

## Time Series in Networks

- Continuous data: bytes_per_second.
- Characteristics: seasonality, peaks, TCP burst.
- Objective: predict the next value in the sequence.

## RNN vs LSTM

- RNNs: effective, but limited by long dependencies.
- LSTM: introduces long-term memory.
- Better performance on sequential data.

## How LSTM Works

- Three main gates: input, forget, output.
- Maintains cell state between timesteps.
- Learns complex temporal patterns.

## LSTM Cell Diagram

```raw
                   C_{t-1}
                     ▲
               Célula│LSTM
       ┌─────────────────────────────┐
       │ ┌─────┐  ┌──────┐  ┌──────┐ │
       │ │Input│⟶│Forget│⟶│Output│ │
       │ │ Gate│  │ Gate │  │ Gate │ │
  X_t  │ └─────┘  └──────┘  └──────┘ │  h_t
──────►│                             │──────►
       │                             │
       │                             │
       │       Cell State C_t        │
       │                             │
       └─────────────────────────────┘
                     ▲
                     │C_t
                     │
```

## Model Architecture

- Layers: LSTM followed by dense.
- Input: windows of 10 to 30 seconds.
- Output: predicted value for the next second.

## Code Example (TensorFlow)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=32)
```

**Listing:** Example code in Keras

## Metrics and Evaluation

- Main metric: Mean Squared Error (MSE).
- Evaluation by window size.
- Comparison: actual value vs. prediction.

## Critical Discussion

- Limitations: TCP burst and sudden peaks.
- Possible improvements: attention and Transformers.
- Conclusion: LSTM is effective, but not definitive.
