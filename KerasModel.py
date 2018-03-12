import mysql.connector
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
import tensorflow as tf

def main(_):
    connection = mysql.connector.connect(user='root', database='CryptoCompare', password='Greenacres29')
    cursor = connection.cursor()
    df = pd.read_sql('SELECT * FROM storage where fsym=%(from)s and tsym=%(to)s order by time asc', con=connection,params={"from":"BTC","to":"USD"})
    training_set = df.iloc[2500:, 2:3]
    testing_set = df.iloc[288:, 2:3]
    training_set = training_set.values
    testing_set = testing_set.values
    sc = MinMaxScaler()
    training_set = sc.fit_transform(training_set)
    # Getting the inputs and the ouputs

    X_train = training_set[0:len(training_set)-1]
    y_train = training_set[1:len(training_set)]

    # Reshaping into required shape for Keras
    X_train = np.reshape(X_train, (len(training_set)-1, 1, 1))


    # Initializing the Recurrent Neural Network
    regressor = Sequential()

    # Adding the input layer and the LSTM layer
    regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))

    # Adding the output layer
    regressor.add(Dense(units=1))

    # Compiling the Recurrent Neural Network
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the Recurrent Neural Network to the Training set
    regressor.fit(X_train, y_train, batch_size=32, epochs=200)

    inputs = testing_set
    inputs = np.reshape(inputs, (len(testing_set), 1, 1))
    predicted_stock_price = regressor.predict(inputs)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    # Visualizing the results
    plt.plot(testing_set, color='red', label='Real Tesla Stock Price')
    plt.plot(predicted_stock_price, color='blue', label='Predicted Tesla Stock Price')
    plt.title('Tesla Stock Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Tesla Stock Price')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    tf.app.run()