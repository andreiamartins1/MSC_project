#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:17:59 2018

@author: teelbo0
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
import numpy as np


def weighted_binary_crossentropy(weights=[.5, 1]):
    def weighted_binary_crossentropy(y_true, y_pred):
        weight_mask = (1-y_true) * weights[0] + y_true * weights[1]
        return K.mean(weight_mask*K.binary_crossentropy(y_pred, y_true), axis=-1)
    return weighted_binary_crossentropy


# def weighted_categorical_crossentropy(nb_classes=3, weights=None):
#     if weights is None:
#         weights = np.ones((1,nb_classes))
    
#     def wce(y_true, y_pred):


#     return NotImplementedError()



def balanced_categorical_crossentropy(batch_size, nb_classes):
    def cross_entropy(y_true, y_pred):
        y_true = tf.one_hot(y_true, nb_classes, axis=-1, on_value=1.0, off_value=0.0)
        s = tf.reduce_sum(y_true, axis=1)
        s = 1.0 - s/batch_size
        s = K.tile(s, [batch_size,1])
        l = K.log(y_pred)
        l = tf.multiply(y_true, l)
        return K.mean(tf.multiply(-s, l), axis=-1)
    return cross_entropy



class BalancedCategoricalAccuracy(Metric):

    def __init__(self, nb_classes, name='balanced_categorical_accuracy' ,**kwargs):
        super(BalancedCategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.recalls = [tf.keras.metrics.Recall(class_id=i) for i in range(nb_classes)]
        self.balanced_categorical_accuracy = self.add_weight(name='balanced_cat_acc', initializer='zeros')

    def reset_state(self):
        for r in self.recalls:
            r.reset_state()
        self.balanced_categorical_accuracy = self.add_weight(name='balanced_cat_acc', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        result = 0
        for r in self.recalls:
            r.update_state(y_true, y_pred, sample_weight)
            result += r.result()

        self.balanced_categorical_accuracy.assign(result /float(len(self.recalls)))

    def result(self):
        return self.balanced_categorical_accuracy

    def get_config(self):
        config = super(BalancedCategoricalAccuracy, self).get_config()
        config['nb_classes'] = len(self.recalls)
        return config


class FocalLoss(Loss):

    def __init__(self,gamma,**kwargs):
        self.gamma = gamma
        super(FocalLoss, self).__init__(**kwargs)


    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        # losses = K.categorical_crossentropy(y_true, y_pred)
        # ones = tf.ones_like()
        pow = tf.ones_like(y_pred) * self.gamma
        out = -tf.math.pow(1.-y_pred, pow) * tf.math.log(y_pred) * y_true
        out = tf.reduce_sum(out, axis=-1)
        
        return K.mean(out)


class DropoutCategoricalCrossEntropy(Loss):

    def __init__(self, alpha=1.0, **kwargs):
        self.alpha = alpha
        super(DropoutCategoricalCrossEntropy, self).__init__(**kwargs)


    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        losses = -K.categorical_crossentropy(y_true, y_pred)

        k = tf.cast(tf.math.round(tf.math.multiply(self.alpha, tf.cast(tf.shape(y_pred)[-2], dtype=tf.float32))),
                    dtype=tf.int32)
        result = tf.math.top_k(losses, k)

        out = tf.ones(k, dtype=tf.float32)
        out = tf.scatter_nd(indices=tf.expand_dims(result.indices, axis=[-1]), updates=out, shape=tf.shape(losses))

        return K.mean(-losses * out)

def trimmed_loss(y_true, y_pred, alpha=0.75):
    y_true = tf.cast(y_true, dtype=tf.float32)
    losses = -K.categorical_crossentropy(y_true, y_pred)
    k = tf.cast(tf.math.round(tf.math.multiply(alpha, tf.cast(tf.shape(y_pred)[-2], dtype=tf.float32))), dtype=tf.int32)
    result = tf.math.top_k(losses, k)
    out = tf.ones(k, dtype=tf.float32)
    out = tf.scatter_nd(indices=tf.expand_dims(result.indices, axis=[-1]), updates=out, shape=tf.shape(losses))

    return K.mean(-losses * out)


def categorical_accuracy(y_true, y_pred):
    return tf.keras.metrics.categorical_accuracy(y_true, y_pred)


def categorical_accuracy_label_smoothing(label_smoothing=0.1, nb_classes=2):
    def categorical_accuracy(y_true, y_pred):
        y_true = (1 - label_smoothing) * y_true + label_smoothing / nb_classes
        return tf.keras.metrics.categorical_accuracy(y_true, y_pred)


def binary_accuracy(y_true, y_pred):
    y_true = tf.math.argmax(y_true, axis=1)
    y_true = tf.cast(y_true, tf.float32)
    return tf.keras.metrics.binary_accuracy(y_true, y_pred)


def dummy_loss(y_true, y_pred):
    return 0.0


class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'alpha': self.alpha
            })
        return config

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

class SemiHardHardTripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(SemiHardHardTripletLossLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'alpha': self.alpha
            })
        return config

    def semi_hard_hard_triplet_loss(self, inputs):
        anchor, positive, negative = inputs

        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)

        # Semi hard hard examples satisfy:
        # abs(n_dist - p_dist) < margin
        # in other words: they are already classified correctly, but are still in the margin
        semi_hard_hard_loss = tf.where(tf.math.abs(n_dist - p_dist) < self.alpha, K.maximum(p_dist - n_dist + self.alpha, 0), 0)

        return K.sum(semi_hard_hard_loss)

    def call(self, inputs):
        loss = self.semi_hard_hard_triplet_loss(inputs)
        self.add_loss(loss)
        return loss

class SemiHardTripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(SemiHardTripletLossLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'alpha': self.alpha
            })
        return config

    def semi_hard_triplet_loss(self, inputs):
        anchor, positive, negative = inputs

        p_dist = K.sum(K.square(anchor - positive), axis=-1)
        n_dist = K.sum(K.square(anchor - negative), axis=-1)

        # Semi hard examples satisfy:
        # n_dist - p_dist < margin
        # AND
        # p_dist < n_dist
        # in other words: they are already classified correctly, but are still in the margin
        semi_hard_loss = tf.where(n_dist - p_dist < self.alpha, K.maximum(p_dist - n_dist + self.alpha, 0), 0)
        semi_hard_loss = tf.where(p_dist < n_dist, semi_hard_loss, 0)

        return K.sum(semi_hard_loss)

    def call(self, inputs):
        loss = self.semi_hard_triplet_loss(inputs)
        self.add_loss(loss)
        return loss


class TripletPosDistanceLayer(Layer):
    def __init__(self, **kwargs):
        super(TripletPosDistanceLayer, self).__init__(**kwargs)

    def triplet_distance(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        return p_dist

    def call(self, inputs):
        metric = self.triplet_distance(inputs)
        self.add_metric(metric, name=self.name)
        return metric


class TripletNegDistanceLayer(Layer):
    def __init__(self, **kwargs):
        super(TripletNegDistanceLayer, self).__init__(**kwargs)

    def triplet_distance(self, inputs):
        anchor, positive, negative = inputs
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        return n_dist

    def call(self, inputs):
        metric = self.triplet_distance(inputs)
        self.add_metric(metric, name=self.name)
        return metric


class TripletAccuracyLayer(Layer):
    def __init__(self, **kwargs):
        super(TripletAccuracyLayer, self).__init__(**kwargs)

    def triplet_accuracy(y_true, y_pred):
        anchor, positive, negative = y_pred
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        return K.mean(p_dist < n_dist)

    def call(self, inputs):
        metric = self.triplet_accuracy(inputs)
        self.add_metric(metric, name=self.name)
        return metric


class EasyTripletAccuracyLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(EasyTripletAccuracyLayer, self).__init__(**kwargs)

    def easy_triplet_accuracy(y_true, y_pred):
        anchor, positive, negative = y_pred
        p_dist = K.sum(K.square(anchor - positive), axis=-1)
        n_dist = K.sum(K.square(anchor - negative), axis=-1)

        accuracy = p_dist < n_dist
        return K.mean(tf.where(p_dist - n_dist > self.alpha, accuracy, 0))

    def call(self, inputs):
        metric = self.easy_triplet_accuracy(inputs)
        self.add_metric(metric, name=self.name)
        return metric
