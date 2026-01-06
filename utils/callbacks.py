#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 17:55:04 2018

@author: teelbo0
"""
import os
import warnings
import numpy as np
from matplotlib import pyplot as plt
import itertools
import io
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from utils.losses import categorical_accuracy

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """

    buf = io.BytesIO()

    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)

    return image

def ConfusionMatrixLogger(model, ds, log_dir, batch_size, class_names):
    '''
    Returns a keras callback function that runs predictions for ds and stores the confusion matrix.

    Note: ds should be finite.
    '''
    file_writer_cm = tf.summary.create_file_writer(os.path.join(log_dir, 'cm'))
    def log_confusion_matrix(epoch, logs):

        # Use the model to predict the values from the test_images.
        test_pred_raw = model.predict(ds)
        test_pred = tf.argmax(test_pred_raw, axis=1)

        true_categories = tf.concat([y for x, y in ds], axis=0)
        true_categories = tf.argmax(true_categories, axis=1)

        # Calculate the confusion matrix using sklearn.metrics
        cm = confusion_matrix(true_categories, test_pred, normalize='true')

        figure = plot_confusion_matrix(cm, class_names=class_names)
        cm_image = plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)
    return tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)


def TripletConfusionMatrixLogger(model, ds, log_dir, batch_size, class_names):
    '''
    Returns a keras callback function that runs predictions for ds and stores the confusion matrix.

    Note: ds should be finite.
    '''
    #TODO: You don't know the anchor's class at runtime, only if the classification is correct or not...
    # so can't create a confusion matrix
    file_writer_cm = tf.summary.create_file_writer(os.path.join(log_dir, 'cm'))
    def log_confusion_matrix(epoch, logs):

        # Use the model to predict the values from the test_images.
        _, _, p_dist, n_dist = model.predict(ds)

        true_categories = [1] * len(p_dist)
        pred_categories = p_dist < n_dist

        # Calculate the confusion matrix using sklearn.metrics
        cm = confusion_matrix(true_categories, pred_categories, normalize='true')

        figure = plot_confusion_matrix(cm, class_names=class_names)
        cm_image = plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)
    return tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

class MultiGPUCheckpointCallback(Callback):

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MultiGPUCheckpointCallback, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)


class FinetuneScheduler(Callback):
    """Finetuning scheduler.

    Takes care of setting the right layers to trainable.

    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns the number of layers
            that need to be frozen. Returns -1 if no action is required.
    """

    def __init__(self, schedule, verbose=0):
        super(FinetuneScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        nb_layers = self.schedule(epoch)

        if nb_layers >= 0:
            for layer in self.model.get_layer('vgg16').layers[:nb_layers]:
                layer.trainable = False
            for layer in self.model.get_layer('vgg16').layers[nb_layers:]:
                layer.trainable = True

            self.model.compile(self.model.optimizer, self.model.loss, self.model.metrics)

            if self.verbose > 0:
                print('\nEpoch %05d: FinetuneScheduler freezing %04d layers.' % (epoch + 1, nb_layers))
            if self.verbose > 1:
                trainable_count = int(np.sum([K.count_params(p) for p in set(self.model.trainable_weights)]))
                non_trainable_count = int(np.sum([K.count_params(p) for p in set(self.model.non_trainable_weights)]))

                print('Total params: {:,}'.format(trainable_count + non_trainable_count))
                print('Trainable params: {:,}'.format(trainable_count))
                print('Non-trainable params: {:,}'.format(non_trainable_count))
                print('Last frozen layer is: ', self.model.layers[nb_layers - 1].name)

