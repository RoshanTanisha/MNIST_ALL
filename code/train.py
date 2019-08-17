from load_data import get_train_test_data
from models import get_classifier
from nn_models import get_cnn_layers, get_fc_layers, get_rnn_layers
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_score
import numpy as np


def train(model, train_x, train_y, reshape):
    if reshape:
        return model.fit(train_x.reshape((-1, 28, 28)), train_y)
    return model.fit(train_x, train_y)


def predict(model, test_x, reshape):
    if reshape:
        return model.predict(test_x.reshape((-1, 28, 28)))
    return model.predict(test_x)


def evaluate(test_y, pred_y):
    if test_y.shape[-1] != 1:
        print(np.unique(np.argmax(pred_y, axis=1), return_counts=True))
        print(np.unique(np.argmax(test_y, axis=1), return_counts=True))
        print('Accuracy: ', accuracy_score(np.argmax(test_y, axis=1), np.argmax(pred_y, axis=1)))
        print('Precision: ', precision_score(np.argmax(test_y, axis=1), np.argmax(pred_y, axis=1), average='weighted'))
    else:
        print(np.unique(pred_y, return_counts=True))
        print(np.unique(test_y, return_counts=True))
        print('Accuracy : ', accuracy_score(test_y, pred_y))
        print('Precision: ', precision_score(test_y, pred_y, average='weighted'))


if __name__ == "__main__":
    print('loading data...')
    train_x, train_y, test_x, test_y = get_train_test_data(normalize=True, onehot=False, reshape=False)

    print(train_x.shape)
    print(train_y.shape)

    model_object = get_classifier('DecisionTreeClassifier')
    print(model_object)
    train(model_object, train_x, train_y, reshape=False)
    evaluate(test_y, predict(model_object, test_x, reshape=False))
