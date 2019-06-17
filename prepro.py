import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import hilbert, chirp


def unpickle(f):
    import _pickle as cPickle
    fo = open(f, 'rb')
    d = cPickle.load(fo, encoding='latin1')
    fo.close()
    return d


def load_dataset(data_dir):
    train_img = []
    train_label = []

    # load train data
    for i in range(1, 6):
        train_data = unpickle("%s/data_batch_%d" % (data_dir, i))
        train_img.extend(train_data["data"])
        train_label.extend(train_data["labels"])

    # load test data
    test_data = unpickle("%s/test_batch" % (data_dir))
    test_img = test_data["data"]
    test_label = test_data["labels"]

    # transform images to ndarray of float32 and labels to ndarray of int32
    train_img = np.array(train_img, dtype=np.float32)
    train_label = np.array(train_label, dtype=np.int32)
    test_img = np.array(test_img, dtype=np.float32)
    test_label = np.array(test_label, dtype=np.int32)

    return train_img, test_img, train_label, test_label


def load_dataset_ex(data_dir):
    train_img = []
    train_label = []

    # load train data (data_batch_1, 10000)
    train_data = unpickle("%s/data_batch_1" % (data_dir))
    train_img.extend(train_data["data"])
    train_label.extend(train_data["labels"])

    # load test data
    test_data = unpickle("%s/test_batch" % (data_dir))
    test_img = test_data["data"]
    test_label = test_data["labels"]

    # transform images to ndarray of float32 and labels to ndarray of int32
    train_img = np.array(train_img, dtype=np.float32)
    train_label = np.array(train_label, dtype=np.int32)
    test_img = np.array(test_img, dtype=np.float32)
    test_label = np.array(test_label, dtype=np.int32)

    return train_img, test_img, train_label, test_label


def monotone(train_img, test_img):
    train_data = []
    for i in range(len(train_img)):
        X_train_ = train_img[i].reshape((3, 32, 32)).transpose(1, 2, 0)
        # transform train color images to monochrome image
        X_train_ = np.array(Image.fromarray(np.uint8(X_train_)).convert("L")) / 255
        train_data.append(X_train_)

    test_data = []
    for i in range(len(test_img)):
        X_test_ = test_img[i].reshape((3, 32, 32)).transpose(1, 2, 0)
        # transform test color image to monochrome image
        X_test_ = np.array(Image.fromarray(np.uint8(X_test_)).convert("L")) / 255
        test_data.append(X_test_)

    return train_data, test_data


def hilbert_transform(train_img, test_img):
    train_data = []
    for i in range(len(train_img)):
        # transform train images to time series like
        train_signal = train_img[i].reshape(32 * 32, 1)
        train_signal = np.array(train_signal, dtype="float64")

        # normalize time series to the range [0,1]
        mms_train = MinMaxScaler()
        train_signal = mms_train.fit_transform(train_signal)
        # do Hilbert transform to the time series and transform time series to image
        train_signal = hilbert(train_signal, axis=0).reshape(32, 32)
        train_data.append(train_signal)

    test_data = []
    for i in range(len(test_img)):
        # transform test images to time series like
        test_signal = test_img[i].reshape(32 * 32, 1)
        test_signal = np.array(test_signal, dtype="float64")

        # normalize time series to the range [0,1]
        mms_test = MinMaxScaler()
        test_signal = mms_test.fit_transform(test_signal)
        # do Hilbert transform to the time series and transform time series to image
        test_signal = hilbert(test_signal, axis=0).reshape(32, 32)
        test_data.append(test_signal)

    return train_data, test_data


def monotone_i(train_img, test_img):
    train_data = []
    for i in range(len(train_img)):
        X_train_ = train_img[i].reshape((3, 32, 32)).transpose(1, 2, 0)
        # transform train color images to monochrome image
        X_train_ = np.array(Image.fromarray(np.uint8(X_train_)).convert("L")) / 255 + 0.01j
        train_data.append(X_train_)

    test_data = []
    for i in range(len(test_img)):
        X_test_ = test_img[i].reshape((3, 32, 32)).transpose(1, 2, 0)
        # transform test color image to monochrome image
        X_test_ = np.array(Image.fromarray(np.uint8(X_test_)).convert("L")) / 255 + 0.01j
        test_data.append(X_test_)

    return train_data, test_data