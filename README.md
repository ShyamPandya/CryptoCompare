# CryptoCompare RNN Model

This time series model is designed to predict the trend of CryptoCurrencies over a period of time based on the history of prices as well as twitter sentiment analysis and predict the duration of the trends.

## Getting Started

There are `3` major files:
  * `app.py`: Fetches data from SQL database for the latest price history, normalizes it, adds twitter sentiment features, calculates features for trend based on price history, splits it into test and train data and passes the values for trend and trend duration prediction.
  * `mainModel.py`:   Calling function for getData as well as for the RNN model. Defines all the tf.FLAGS as well as defines parameters passed to the RNN model
  * `trendPredictionModel.py`: Has functions to define the RNN modules and LSTM cells as well as functions to train, test, save and display plots based on the requirement

### Prerequisites

* Python
* Tensorflow
