import tensorflow as tf


def normalise(X_train, X_test):
    X_train = X_train / 255.0
    X_train = tf.cast(X_train, dtype=tf.float32)
    
    X_test = X_test / 255.0
    X_test = tf.cast(X_test, dtype=tf.float32)

    return X_train, X_test


def limit_size(train, test, size):
    train = (train[0][:size[0]], train[1][:size[0]])
    test = (test[0][:size[1]], test[1][:size[1]])
    return train, test


def load_mnist(size=()):
    (X_train, y_train), (X_test , y_test) = tf.keras.datasets.mnist.load_data()
    X_train, X_test = normalise(X_train, X_test)

    # shape : (None, 28,28) to (None, 28,28,1)
    X_train = tf.expand_dims(X_train, axis=-1)
    X_test = tf.expand_dims(X_test, axis=-1)

    if size:
        return limit_size((X_train, y_train), (X_test , y_test), size) 
    return (X_train, y_train), (X_test , y_test)


def load_fashion_mnist(size=()):
    (X_train, y_train), (X_test , y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train, X_test = normalise(X_train, X_test)

    # shape : (None, 28,28) to (None, 28,28,1)
    X_train = tf.expand_dims(X_train, axis=-1)
    X_test = tf.expand_dims(X_test, axis=-1)

    if size:
        return limit_size((X_train, y_train), (X_test , y_test), size) 
    return (X_train, y_train), (X_test , y_test)


def load_cifar_10(size=()):
    (X_train, y_train), (X_test , y_test) = tf.keras.datasets.cifar10.load_data()
    X_train, X_test = normalise(X_train, X_test)
    y_train = y_train.reshape((len(y_train),))
    y_test = y_test.reshape((len(y_test),))
    if size:
        return limit_size((X_train, y_train), (X_test , y_test), size)
    return (X_train, y_train), (X_test , y_test)


def load_cifar_100(size=()):
    (X_train, y_train), (X_test , y_test) = tf.keras.datasets.cifar100.load_data()
    X_train, X_test = normalise(X_train, X_test)
    y_train = y_train.reshape((len(y_train),))
    y_test = y_test.reshape((len(y_test),))
    if size:
        return limit_size((X_train, y_train), (X_test , y_test), size)
    return (X_train, y_train), (X_test , y_test)
