import tensorflow as tf
from getData import CryptoData
from model_rnn import LstmRNN
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


flags = tf.app.flags
flags.DEFINE_integer("crypto_count", 1, "Crypto count [1]")
flags.DEFINE_integer("input_size", 7, "Input size [4]")
flags.DEFINE_integer("output_size", 4, "Output size [1]")
flags.DEFINE_integer("num_steps", 1, "Num of steps [30]")
flags.DEFINE_integer("num_layers", 1, "Num of layer [1]")
flags.DEFINE_integer("lstm_size", 128, "Size of one LSTM cell [128]")
flags.DEFINE_integer("batch_size", 62, "The size of batch images [64]")
flags.DEFINE_float("keep_prob", 0.8, "Keep probability of dropout layer. [0.8]")
flags.DEFINE_float("init_learning_rate", 0.001, "Initial learning rate at early stage. [0.001]")
flags.DEFINE_float("learning_rate_decay", 0.99, "Decay rate of learning rate. [0.99]")
flags.DEFINE_integer("init_epoch", 5, "Num. of epoches considered as early stage. [5]")
flags.DEFINE_integer("max_epoch", 1000, "Total training epoches. [50]")
flags.DEFINE_integer("embed_size", None, "If provided, use embedding vector of this size. [None]")
flags.DEFINE_string("stock_symbol", None, "Target stock symbol [None]")
flags.DEFINE_integer("sample_size", 4, "Number of stocks to plot during training. [4]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")

FLAGS = flags.FLAGS

def load_data(input_size, num_steps, fsym, tsym, target_symbol=None, test_ratio=0.05):
        return [
            CryptoData(
                target_symbol,
                fsym=fsym,
                tsym=tsym,
                input_size=input_size,
                num_steps=num_steps,
                test_ratio=test_ratio)
        ]

def main(_):
    with tf.Session() as sess:
        rnn_model = LstmRNN(
            sess,
            FLAGS.crypto_count,
            lstm_size=FLAGS.lstm_size,
            num_layers=FLAGS.num_layers,
            num_steps=FLAGS.num_steps,
            input_size=FLAGS.input_size,
            embed_size=FLAGS.embed_size,
            output_size=FLAGS.output_size
        )

        '''coin_data_list1 = load_data(
        FLAGS.input_size,
        FLAGS.num_steps,
        'ETH',
        'USD',
        target_symbol = FLAGS.stock_symbol
        )
        coin_data_list2 = load_data(
            FLAGS.input_size,
            FLAGS.num_steps,
            'XRP',
            'USD',
            target_symbol=FLAGS.stock_symbol
        )
        coin_data_list3 = load_data(
            FLAGS.input_size,
            FLAGS.num_steps,
            'CND',
            'USD',
            target_symbol=FLAGS.stock_symbol
        )'''
        coin_data_list4 = load_data(
            FLAGS.input_size,
            FLAGS.num_steps,
            'BTC',
            'USD',
            target_symbol=FLAGS.stock_symbol
        )

    if FLAGS.train:
        #rnn_model.train(coin_data_list1, FLAGS,'ETH')
        #rnn_model.train(coin_data_list2, FLAGS,'XRP')
        #rnn_model.train(coin_data_list3, FLAGS,'CND')
        rnn_model.train(coin_data_list4, FLAGS,'BTC',1)
    else:
        if not rnn_model.load()[0]:
            raise Exception("[!] Train a model first, then run test mode")


if __name__ == '__main__':
    tf.app.run()
