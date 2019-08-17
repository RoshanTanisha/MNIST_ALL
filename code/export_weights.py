from ast import literal_eval
import tensorflow as tf
import os
from nn_models import get_fc_layers
import pandas as pd


def export_model_weights(meta_file_path):
    model = get_fc_layers(save_model=False)
    model.load(model_file=meta_file_path)

    variable_dictionary = {}

    for each_tf_variable in tf.trainable_variables():
        variable_dictionary[each_tf_variable.name] = (str(model.get_weights(each_tf_variable).tolist()), each_tf_variable.shape)

    pd.DataFrame(variable_dictionary).to_hdf('../saved_weights/fc.h5', key='df')


if __name__ == "__main__":
    model_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models', 'fc')
    model_meta_file_path = os.path.join(model_dir_path, 'checkpoint')

    with open(model_meta_file_path, 'r') as fin:
        value = literal_eval(fin.readline().split(': ')[1].strip())

    print(value)
    export_model_weights(value)
