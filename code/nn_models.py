import os
from tflearn import lstm, input_data, regression, fully_connected, conv_2d, max_pool_2d, dropout, DNN
import tensorflow as tf

tf.set_random_seed(0)


def get_checkpoint_path(model_type):
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models', model_type)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    return os.path.join(checkpoint_path, '{}_checkpoint'.format(model_type))


def create_model(net, model_type, save_model):
    net = regression(net, optimizer='adam', loss='categorical_crossentropy', name='output')

    if save_model:
        return DNN(net, checkpoint_path=get_checkpoint_path(model_type))
    else:
        return DNN(net)


def get_rnn_layers(save_model):
    input_layer = input_data(shape=[None, 28, 28])
    net = lstm(input_layer, n_units=128, activation='tanh', inner_activation='sigmoid', return_seq=True)
    net = lstm(net, n_units=128, activation='tanh', inner_activation='sigmoid')
    net = fully_connected(net, n_units=10, activation='softmax')

    return create_model(net, 'rnn', save_model)


def get_cnn_layers(save_model):
    input_layer = input_data(shape=[None, 28, 28, 1])
    dropout_layer = dropout(input_layer, keep_prob=0.5)
    net = conv_2d(dropout_layer, nb_filter=32, filter_size=(3,3))
    net = max_pool_2d(net, kernel_size=(2,2))
    net = conv_2d(net, nb_filter=32, filter_size=(3,3))
    net = max_pool_2d(net, kernel_size=(2,2))

    net = fully_connected(net, n_units=256)

    net = dropout(net, keep_prob=0.5)

    net = fully_connected(net, n_units=10, activation='softmax')

    return create_model(net, 'cnn', save_model)


def get_fc_layers(save_model):
    input_layer = input_data(shape=[None, 784])

    net = fully_connected(input_layer, n_units=1000, regularizer='L2', activation='relu')
    net = fully_connected(net, n_units=1000, regularizer='L2', activation='relu')
    net = fully_connected(net, n_units=10, regularizer='L2', activation='softmax')

    return create_model(net, 'fc', save_model)
