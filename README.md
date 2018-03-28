# CryptoCompare RNN Model

This time series model is designed to predict the trend of CryptoCurrencies over a period of time based on the history of prices as well as twitter sentiment analysis.

## Getting Started

There are `3` major files:
  * `getData.py`: Fetches data from csv file, normalizes it, splits it into test and train data and stores it in the variables of                         CryptoData class and can be accessed by using `self`.______
  * `model.py`:   Calling function for getData as well as for the RNN model. Defines all the tf.FLAGS as well as defines parameters                       passed to the RNN model
  * `model_rnn.py`: Has functions to define the RNN modules and LSTM cells as well as functions to train, test, save and display plots                       based on the requirement

### Prerequisites

* Python
* Tensorflow


## Train the model

Define the following flags:
* `input_size` to be 7 as the number of features/columns in our dataset is 7.
* `output_size` to be 4 as we expect to get outputs of `Low`, `High`, `Open`, `Close` values for the coin
* `num_steps` to be 1 as we feed data for a single day to the LSTM and is a single day time based operation
* `max_epoch` depending on how many epochs do you want to train the model for

