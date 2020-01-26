from __future__ import print_function, division

import numpy as np

from keras.datasets import mnist, cifar10, cifar100, imdb
from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.core import Dropout, SpatialDropout1D
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.preprocessing import sequence
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler

from loss import (crossentropy, robust, unhinged, sigmoid, ramp, savage,
                  boot_soft)


# losses that need sigmoid on top of last layer
yes_softmax = ['crossentropy', 'forward', 'est_forward', 'backward',
               'est_backward', 'boot_soft', 'savage']
# unhinged needs bounded models or it diverges
yes_bound = ['unhinged', 'ramp', 'sigmoid']


class KerasModel():

    def get_data(self):

        (X_train, y_train), (X_test, y_test) = self.load_data()

        idx_perm = np.random.RandomState(101).permutation(X_train.shape[0])
        X_train, y_train = X_train[idx_perm], y_train[idx_perm]

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        return X_train, X_test, y_train, y_test

    # custom losses for the CNN
    def make_loss(self, loss, P=None):

        if loss == 'crossentropy':
            return crossentropy
        elif loss in ['forward', 'backward']:
            return robust(loss, P)
        elif loss == 'unhinged':
            return unhinged
        elif loss == 'sigmoid':
            return sigmoid
        elif loss == 'ramp':
            return ramp
        elif loss == 'savage':
            return savage
        elif loss == 'boot_soft':
            return boot_soft
        else:
            ValueError("Loss unknown.")

    def compile(self, model, loss, P=None):

        if self.optimizer is None:
            ValueError()

        metrics = ['accuracy']

        model.compile(loss=self.make_loss(loss, P),
                      optimizer=self.optimizer, metrics=metrics)

        model.summary()
        self.model = model

    def load_model(self, file):
        self.model.load_weights(file)
        print('Loaded model from %s' % file)

    def fit_model(self, model_file, X_train, Y_train, validation_split=None,
                  validation_data=None):

        # cannot do both
        if validation_data is not None and validation_split is not None:
            return ValueError()

        callbacks = []
        monitor = 'val_loss'
        # monitor = 'val_acc'

        mc_callback = ModelCheckpoint(model_file, monitor=monitor,
                                      verbose=1, save_best_only=True)
        callbacks.append(mc_callback)

        if hasattr(self, 'scheduler'):
            callbacks.append(self.scheduler)

        # use data augmentation
        if hasattr(self, 'data_generator'):
            print('DATA GENERATOR DISABLED!')
            return 0
            
            # hack for using validation with data augmentation
            idx_val = np.round(validation_split * X_train.shape[0]).astype(int)
            X_val, Y_val = X_train[:idx_val], Y_train[:idx_val]
            X_train_local, Y_train_local = X_train[idx_val:], Y_train[idx_val:]

            self.data_generator.fit(X_train_local)

            history = \
                self.model.fit_generator(
                    self.data_generator.flow(X_train_local, Y_train_local,
                                             batch_size=self.num_batch),
                    steps_per_epoch=X_train.shape[0] // self.num_batch,
                    epochs=self.epochs, max_q_size=100,
                    validation_data=(X_val, Y_val),
                    verbose=1, callbacks=callbacks)

        else:

            history = self.model.fit(
                        X_train, Y_train, batch_size=self.num_batch,
                        epochs=self.epochs,
                        validation_split=validation_split,
                        validation_data=validation_data,
                        verbose=1, callbacks=callbacks)

        # use the model that reached the lowest loss at training time
        self.load_model(model_file)

        return history.history

    def evaluate_model(self, X, Y):
        score = self.model.evaluate(X, Y, batch_size=self.num_batch, verbose=1)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        return score[1]

    def predict_proba(self, X):
        pred = self.model.predict(X, batch_size=self.num_batch, verbose=1)
        return pred


class PriceModel(KerasModel):

    def __init__(self, num_batch=32):
        self.num_batch = num_batch
        self.classes = 6
        self.epochs = 100
        self.normalize = True
        self.optimizer = None
        self.scaler = StandardScaler()

    def load_data(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        if self.normalize:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        return (X_train, y_train), (X_test, y_test)

    def build_model(self, loss, P=None):

        input = Input(shape=(463,))

        x = Dense(128, kernel_initializer='he_normal')(input)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(self.classes, kernel_initializer='he_normal')(x)

        if loss in yes_bound:
            output = BatchNormalization(axis=1)(output)

        if loss in yes_softmax:
            output = Activation('softmax')(output)

        model = Model(inputs=input, outputs=output)
        self.compile(model, loss, P)

class NoiseEstimator():

    def __init__(self, classifier, row_normalize=True, alpha=0.0,
                 filter_outlier=False, cliptozero=False, verbose=0):
        """classifier: an ALREADY TRAINED model. In the ideal case, classifier
        should be powerful enough to only make mistakes due to label noise."""

        self.classifier = classifier
        self.row_normalize = row_normalize
        self.alpha = alpha
        self.filter_outlier = filter_outlier
        self.cliptozero = cliptozero
        self.verbose = verbose

    def fit(self, X):

        # number of classes
        c = self.classifier.classes
        T = np.empty((c, c))

        # predict probability on the fresh sample
        eta_corr = self.classifier.predict_proba(X)

        # find a 'perfect example' for each class
        for i in np.arange(c):

            if not self.filter_outlier:
                idx_best = np.argmax(eta_corr[:, i])
            else:
                eta_thresh = np.percentile(eta_corr[:, i], 97,
                                           interpolation='higher')
                robust_eta = eta_corr[:, i]
                robust_eta[robust_eta >= eta_thresh] = 0.0
                idx_best = np.argmax(robust_eta)

            for j in np.arange(c):
                T[i, j] = eta_corr[idx_best, j]

        self.T = T
        return self

    def predict(self):

        T = self.T
        c = self.classifier.classes

        if self.cliptozero:
            idx = np.array(T < 10 ** -6)
            T[idx] = 0.0

        if self.row_normalize:
            row_sums = T.sum(axis=1)
            T /= row_sums[:, np.newaxis]

        if self.verbose > 0:
            print(T)

        if self.alpha > 0.0:
            T = self.alpha * np.eye(c) + (1.0 - self.alpha) * T

        if self.verbose > 0:
            print(T)
            print(np.linalg.inv(T))

        return T
