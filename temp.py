import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scipy
import mysql.connector
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
import urllib.parse
import requests
import json

now_datetime = datetime.datetime.now().strftime("%b+%d,+%Y")
BASE_URL = "https://finance.google.com/finance/historical?output=csv&q=GOOG&startdate=Jan+1%2C+1980&enddate={0}"
symbol_url = BASE_URL.format(
        urllib.parse.quote(now_datetime, '+')
    )
f = requests.get(symbol_url)
dataa =f.text
print(type(dataa))
dataa = dataa.split()
print(type(dataa))
print(dataa)
connection = mysql.connector.connect(user='root', database='CryptoCompare', password='Greenacres29')
cursor = connection.cursor()
# Get Data
query = "SELECT * FROM storage order by time desc"
cursor.execute(query)
rows = cursor.fetchall()
X_data =[]
y_data = []
print(len(rows))
i=0
for row in rows:
    raw_seq = [x for tuple in row[['open','high','low','close']].values for x in tuple]
    row = list(row)
    i=i+1
    if row[7] == 'BTC' and row[8] == 'USD':
        row.pop(0)
        row.pop(6)
        row.pop(6)
        row[0]= datetime.datetime.combine(row[0].date(), row[0].time()).timestamp()
        y_data.append((row[1]+row[4])/2)
        row.pop(5)
        X_data.append(row)
X_data = np.asarray(X_data)
y_data = np.asarray(y_data)






# Import data
data = pd.read_csv('/home/maverick/Downloads/^GSPC.csv')
print(type(data))
raw_seq = [price for tup in data[['Open', 'Close']].values for price in tup]
print(raw_seq)
input_size=1
num_steps=30
raw_seq = np.array(raw_seq)
print(raw_seq.shape)
seq = [np.array(raw_seq[i * input_size: (i + 1) * input_size])
               for i in range(len(raw_seq) // input_size)]

seq = [seq[0] / seq[0][0] - 1.0] + [
                curr / seq[i][-1] - 1.0 for i, curr in enumerate(seq[1:])]


# split into groups of num_steps
X = np.array([seq[i: i + num_steps] for i in range(len(seq) - num_steps)])
y = np.array([seq[i + num_steps] for i in range(len(seq) - num_steps)])
print(X.shape)
print(y.shape)



# Drop date variable
data = data.drop(['DATE'], 1)
# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]
# Make data a numpy array
data = data.values
print(n)
print(p)
print(type(data))
print(type(data[0]))
print(data[0].shape)

# Training and test data
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end + 1
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]
print(X_train.shape)
print(y_train.shape)
print(type(X_train))
print(y_train)


scaler = MinMaxScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)
# Build X and y
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]




# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
n_dim = features.shape[1]
learning_rate = 0.05
training_epochs = 5000
cost_history = np.empty(shape=[1],dtype=float)
X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,1])
W = tf.Variable(tf.ones([n_dim,1]))


init = tf.initialize_all_variables()


y_ = tf.matmul(X, W)
cost = tf.reduce_mean(tf.square(y_ - Y))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={X:X_train,Y:y_train})
    cost_history = np.append(cost_history,sess.run(cost,feed_dict={X: X_train,Y: y_train}))

plt.plot(range(len(cost_history)),cost_history)
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()

correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("accuracy")
print(sess.run(accuracy, feed_dict={X: X_test, Y: y_test}))
prediction=tf.argmax(y_,1)
print("predictions")
print(prediction.eval(feed_dict={X: X_test}, session=sess))


fdfd =[]
pred_y = sess.run(y_, feed_dict={X: X_test})
fdfd.append(pred_y)
mse = tf.reduce_mean(tf.square(pred_y - y_test))
print("MSE: %.4f" % sess.run(mse))
print(y_test)
print(fdfd)

fig, ax = plt.subplots()
ax.scatter(y_test, pred_y)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

sess.close()


        row = list(row)
        i=i+1
        if row[7] == 'BTC' and row[8] == 'USD':
            row.pop(0)
            row.pop(6)
            row.pop(6)
            row[0]= datetime.datetime.combine(row[0].date(), row[0].time()).timestamp()
            y_data.append((row[1]+row[4])/2)
            row.pop(5)
            X_data.append(row)
    X_data = np.asarray(X_data)
    y_data = np.asarray(y_data)
    print(y_data)
    normalized_X = feature_normalize(X_data)
    f, l = append_bias_reshape(normalized_X, y_data)
    print(l)
    return f, l


my_dict = {"BTC": 1,
           "ETH": 2,
           "USD": 3,
           "CND": 4,
           "IOTA": 5,
           "XRP": 6,
           "LTC": 7,
           "DASH": 8,
           "EOS": 9,
           "GAS": 10,
           "COE": 11,
           "POE": 12,
           "ADA": 13,
           "DEEP": 14,
           "AIR": 15
        }


