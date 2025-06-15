# Project 3 - Network traffic prediction with Recurrent Neural Networks

**Course:** Computer Networks Programming (MC833) - UNICAMP

## Introduction

Predicting traffic volume in computer networks is fundamental for capacity planning, anomaly detection, and load
balancing. This project proposes that you work with a public file (PCAP) from the MAWI/CAIDA database, build a
second-by-second time series based on the traffic data, and use a neural network-based forecasting approach.

## LSTM (Recurrent Neural Networks)

LSTM (Long Short-Term Memory) models belong to the class of recurrent neural networks (RNNs) and are especially
effective for handling problems with sequential data, such as network traffic time series. LSTMs are capable of learning
complex temporal patterns, maintaining an internal "memory" that considers both short and long-term trends.

In this assignment, the LSTM will be used to predict the traffic volume per second, using a sliding window of previous
values (for example, the last 10, 20, or 30 seconds) as input. The network should predict the next value of the time
series constructed from the PCAP file.

### How to use LSTM:

The implementation should be done with TensorFlow/Keras (or alternatively PyTorch). The recommended architecture is
lean: one LSTM layer followed by a dense (fully connected) layer. The input will be a sequence of `bytes_per_second`,
generated from the time series. The output will be the predicted value for the following second. Hyperparameter
adjustments such as `epochs`, `batch_size`, and `look_back` should be considered and recorded.

In addition to training and evaluating the model, you should perform an exploratory data analysis (EDA), focusing on
identifying patterns in the traffic, such as peaks, periodic behaviors, and the distribution of protocols (e.g., TCP and
UDP).

## Task Details

### 1. Data acquisition and preparation

- What to do:
  - **Download the PCAP file:** A pre-processed file is already available for download at the following link: 👉
    [link](https://drive.google.com/drive/folders/1DU2usbXuR4u4rllxp2wLQkXtIhf5p7JQ?usp=sharing) It contains the traffic
    time series in a compact format, facilitating analysis and model training.

  - **Alternative option:** Students who wish to can directly download a raw PCAP file from the public MAWI database
    (<https://mawi.wide.ad.jp/mawi/>), specifically the following
    [link](https://mawi.wide.ad.jp/mawi/samplepoint-C/2007/200701011800.html), and process it on their own. However, be
    aware: processing large PCAP files can be time-consuming and require more computational resources.

  - **Supporting code:** In the project's shared folder, two Python code examples are available:

    - A script to read files in Parquet format containing the pre-processed data.
    - An alternative script that reads and processes PCAP files directly, generating the `bytes_per_second` time series.
    - ▲ This second script may take longer to run, especially with large files.

### 2. Exploratory data analysis (EDA)

- **What to do:**

  - Generate descriptive statistics: mean, standard deviation, quartiles, maximum and minimum values.

  - Produce visualizations: histograms of byte distribution for different protocols (TCP, UDP).

- **Expected result:** A section "Exploratory data analysis" in the report, containing all figures and interpretive
  comments.

### 3. Predictive modeling: LSTM

- **What to do:**

  - **Series preparation** - Generate the training (80%) and test (20%) sets from the `bytes_per_second` series,
    extracted from the PCAP file. Create sliding windows (`look-back`) of 10 seconds as input for the LSTM model. Then,
    test other windows (e.g., 20 s and 30 s) to evaluate the network's sensitivity to the amount of history.

  - **Neural network (LSTM)** - Implement a lean architecture with one LSTM layer followed by a dense (fully connected)
    layer responsible for predicting the next point in the series. Adjust essential hyperparameters such as the number
    of epochs, batch size, and window size (`look-back`). Save and record the best model.

  - **Training and evaluation** - Train the model on the training data and evaluate it on the test set, using the
    [MSE](https://www.freecodecamp.org/news/machine-learning-mean-squared-error-regression-line-c7dde9a26b93/) (Mean
    Squared Error) metric to measure predictive performance.

  - **Visualization and analysis:**

    - A table with the MSE values for different window configurations;
    - A "real vs. predicted value" graph showing the curve predicted by the LSTM compared to the actual data in the test
      set;
    - Brief comments highlighting the sections where the model shows good adherence or significant deviations (e.g.,
      peaks, periods of stability, or rapid variations).

- **Expected result:**

  - A section of the report titled "Predictive modeling" containing:
    - Description of the adopted LSTM approach;
    - Table of MSE per window configuration;
    - Comparative graph of real vs. predicted values;
    - Preliminary analysis of the LSTM model's performance.

### 4. Critical discussion

- **What to do:**

  - Explain why the model had the observed performance: influence of TCP burst, hourly patterns, stationarity of the
    series, etc.

  - Relate the results to typical network phenomena (traffic peaks, periodic behaviors, possible attacks).

- **Expected result:**

  - Clear conclusions in the report, pointing out advantages, limitations, and suggestions for future work.

## Deliverables

Single submission: report in PDF, via Google Classroom.

## Essential best practices

- Fix seeds (numpy, random, tensorflow or pytorch) for reproducibility.
- Comment the code.
- Do not version the PCAP file; use `.gitignore`.
- Discuss ethical implications (privacy of flows).
