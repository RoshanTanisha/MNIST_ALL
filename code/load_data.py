import pandas as pd
import os


def load_data(filepath):
    data = pd.read_hdf(filepath, key='df', start=0, stop=2000)
    X = data.filter(like='pixel').reset_index(drop=True)
    Y = data.filter(like='Label').reset_index(drop=True)

    return X, Y


def normalize_data(X):
    X = (X - X.min().min()) / X.max().max()

    return X


def complete_path(filename):
    root_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = 'dataset/hdf5'
    return os.path.join(root_dir, dataset_dir, filename)


def get_train_test_data(normalize=True, onehot=True, reshape=False):
    train_filename = 'train_data_mnist.h5'
    test_filename = 'test_data_mnist.h5'
    
    train_X, train_Y = load_data(complete_path(train_filename))
    if normalize:
        train_X = normalize_data(train_X)
    if onehot:
        train_Y = pd.get_dummies(train_Y['Label'].apply(lambda x: str(x)))

    test_X, test_Y = load_data(complete_path(test_filename))
    if normalize:
        test_X = normalize_data(test_X)
    if onehot:
        test_Y = pd.get_dummies(test_Y['Label'].apply(lambda x: str(x)))

    train_X = train_X.values
    train_Y = train_Y.values
    test_X = test_X.values
    test_Y = test_Y.values

    if reshape:
        train_X = train_X.reshape((-1, 28, 28, 1))
        test_X = test_X.reshape((-1, 28, 28, 1))

    return train_X, train_Y, test_X, test_Y


if __name__ == "__main__":
    get_train_test_data(onehot=True)
