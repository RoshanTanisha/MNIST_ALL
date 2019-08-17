import os
from ast import literal_eval
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

import yaml
config = None


def get_classifier(model_class_name):
    global config
    if config is None:
        config = load_config('config.yaml')
    arguments = get_args_for_classifier(config, model_class_name)
    if arguments is None:
        train_object = eval(model_class_name)()
    else:
        train_object = eval(model_class_name)(**arguments)
    return train_object


def load_config(config_filename):
    config_file_path = os.path.join(os.path.dirname(__file__), config_filename)
    with open(config_file_path, 'r') as fin:
        configuration = yaml.load(fin)

    return configuration


def get_args_for_classifier(configuration, classifier_class_name):
    try:
        return configuration[classifier_class_name]
    except KeyError:
        return None


if __name__ == '__main__':
    classifier_class_name = 'SVC'
    train_object = get_classifier(classifier_class_name)
