# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
import sys

import numpy as np

sys.path.append('.')
sys.path.append('..')

import tensorflow as tf
from utils.losses import TripletLossLayer, TripletAccuracyLayer, TripletPosDistanceLayer, TripletNegDistanceLayer, DropoutCategoricalCrossEntropy, BalancedCategoricalAccuracy
from tensorflow.keras.regularizers import l2


# +

# # from wavetf import WaveTFFactory

# def input_block(image_dimensions=(384, 288, 3), bn=False):
#     inputs = tf.keras.Input(shape=image_dimensions)
#     outputs = tf.keras.layers.Conv2D(32, kernel_size=3, activation=tf.nn.relu)(inputs)
#     outputs = tf.keras.layers.BatchNormalization()(outputs) if bn else outputs
#     outputs = tf.keras.layers.MaxPooling2D()(outputs)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs, name='invididual_branch_network')
#     return model


# def input_block_vgg16(nb_classes, image_dimensions):
#     inputs = tf.keras.Input(shape=image_dimensions)
#     x = tf.keras.applications.vgg16.preprocess_input(inputs)

#     base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=image_dimensions, pooling='avg')

#     base_model.summary()
#     outputs = base_model(x, training=False)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model

# def input_block_mobilenetV2(image_dimensions):
#     inputs = tf.keras.Input(shape=image_dimensions)

#     x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
#     base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', input_shape=image_dimensions, pooling='avg', alpha=0.5)

#     base_model.summary()
#     outputs = base_model(x, training=False)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model


def input_block_efficientnet(image_dimensions):
    inputs = tf.keras.Input(shape=image_dimensions)

    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights='imagenet', input_shape=image_dimensions, pooling='avg')

    # base_model.summary()
    outputs = base_model(x, training=False)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# def resnet(nb_classes, image_dimensions=(384, 288, 3), bn=False, fc_dropout=0., reg_constant=0., nf=64):
#     inputs = tf.keras.Input(shape=image_dimensions)

#     # rescale = tf.keras.layers.experimental.preprocessing.Rescaling(2., offset= -1) # xception, the input images are in range [0, 1)

#     rescale = tf.keras.layers.experimental.preprocessing.Rescaling(255.) # resnet, the input images are in range [0, 1)
#     preprocess = tf.keras.applications.resnet.preprocess_input

#     base_model = tf.keras.applications.ResNet50(
#         weights='imagenet',  # Load weights pre-trained on ImageNet.
#         include_top=False, pooling='avg')  # Do not include the ImageNet classifier at the top.
#     base_model.trainable = True

#     preprocessing = preprocess(inputs)
#     # for layer in base_model.layers[-3:]:
#     #     layer.trainable = True


#     base_model.summary()
#     base_model.trainable = False

#     top_left = preprocessing[:, :224, :224, :]
#     top_right = preprocessing[:, -224:, :224, :]
#     bottom_left = preprocessing[:, :224, -224:, :]
#     bottom_right = preprocessing[:, -224:, -224:, :]
#     offset_x = round((image_dimensions[0] - 224) / 2)
#     offset_y = round((image_dimensions[1] - 224) / 2)
#     center = preprocessing[:, offset_x:offset_x + 224, offset_y:offset_y + 224, :]

#     layers = [top_left, top_right, bottom_right, bottom_left, center]
#     final_outputs = []
#     seq = tf.keras.Sequential()
#     seq.add(base_model)
#     seq.add(tf.keras.layers.Flatten())
#     seq.add(tf.keras.layers.Dense(nf * 4, activation='relu', kernel_initializer='he_uniform'))
#     if fc_dropout > 0.:
#         seq.add(tf.keras.layers.Dropout(fc_dropout))
#     seq.add(tf.keras.layers.BatchNormalization())

#     seq.add(tf.keras.layers.Dense(nf * 2, activation='relu', kernel_initializer='he_uniform'))
#     if fc_dropout > 0.:
#         seq.add(tf.keras.layers.Dropout(fc_dropout))
#     seq.add(tf.keras.layers.BatchNormalization())

#     seq.add(tf.keras.layers.Dense(nf, activation='relu', kernel_initializer='he_uniform'))
#     if fc_dropout > 0.:
#         seq.add(tf.keras.layers.Dropout(fc_dropout))
#     seq.add(tf.keras.layers.BatchNormalization())

#     seq.add(tf.keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='he_uniform'))
#     for layer in layers:
#         outputs = seq(layer)
#         final_outputs.append(outputs)
#     outputs = tf.keras.layers.average(final_outputs)
#     model = tf.keras.Model(inputs, outputs)

#     return model


# def resnet(nb_classes, image_dimensions=(384, 288, 3), bn=False, fc_dropout=0., reg_constant=0., nf=64, resnet_layers=50):
#     inputs = tf.keras.Input(shape=image_dimensions)

#     # rescale = tf.keras.layers.experimental.preprocessing.Rescaling(2., offset= -1) # xception, the input images are in range [0, 1)

#     # rescale = tf.keras.layers.experimental.preprocessing.Rescaling(255.) # resnet, the input images are in range [0, 1)
#     preprocess = tf.keras.applications.resnet_v2.preprocess_input

#     if resnet_layers == 152:
#         base_model = tf.keras.applications.ResNet152V2(
#             weights='imagenet',  # Load weights pre-trained on ImageNet.
#             include_top=False, pooling='avg')  # Do not include the ImageNet classifier at the top.
#     elif resnet_layers == 101:
#         base_model = tf.keras.applications.ResNet101V2(
#             weights='imagenet',  # Load weights pre-trained on ImageNet.
#             include_top=False, pooling='avg')  # Do not include the ImageNet classifier at the top.
#     else:
#         base_model = tf.keras.applications.ResNet50V2(
#             weights='imagenet',  # Load weights pre-trained on ImageNet.
#             include_top=False, pooling='avg')  # Do not include the ImageNet classifier at the top.

#     preprocessing = preprocess(inputs)
#     # for layer in base_model.layers[-3:]:
#     #     layer.trainable = True


#     base_model.summary()
#     base_model.trainable = False

#     top_left = preprocessing[:, :224, :224, :]
#     top_right = preprocessing[:, -224:, :224, :]
#     bottom_left = preprocessing[:, :224, -224:, :]
#     bottom_right = preprocessing[:, -224:, -224:, :]
#     offset_x = round((image_dimensions[0] - 224) / 2)
#     offset_y = round((image_dimensions[1] - 224) / 2)
#     center = preprocessing[:, offset_x:offset_x + 224, offset_y:offset_y + 224, :]

#     layers = [top_left, top_right, bottom_right, bottom_left, center]
#     # layers = [center]
#     final_outputs = []
#     seq = tf.keras.Sequential()
#     seq.add(base_model)
#     seq.add(tf.keras.layers.Flatten())
#     for layer in layers:
#         outputs = seq(layer)
#         final_outputs.append(outputs)
#     outputs = tf.keras.layers.concatenate([tf.keras.layers.average(final_outputs), tf.keras.layers.add(final_outputs)])
#     outputs = tf.keras.layers.Dense(nf * 4, activation='relu', kernel_initializer='he_uniform')(outputs)
#     if fc_dropout > 0.:
#         outputs = tf.keras.layers.Dropout(fc_dropout)(outputs)
#     outputs = tf.keras.layers.BatchNormalization()(outputs)

#     outputs = tf.keras.layers.Dense(nf * 2, activation='relu', kernel_initializer='he_uniform')(outputs)
#     if fc_dropout > 0.:
#         outputs = tf.keras.layers.Dropout(fc_dropout)(outputs)
#     outputs = tf.keras.layers.BatchNormalization()(outputs)

#     outputs = tf.keras.layers.Dense(nf, activation='relu', kernel_initializer='he_uniform')(outputs)
#     if fc_dropout > 0.:
#         outputs = tf.keras.layers.Dropout(fc_dropout)(outputs)
#     outputs = tf.keras.layers.BatchNormalization()(outputs)

#     # outputs = tf.keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='he_uniform')(outputs)
    
    
#     outputs = tf.keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='he_uniform')(outputs)
#     model = tf.keras.Model(inputs, outputs)

#     return model

def vgg16(nb_classes, nf=64, image_dimensions=(384, 288, 3), fc_dropout=0., reg_constant=0., frozen_layers=1):
    inputs = tf.keras.Input(shape=image_dimensions)
    # x = inputs
    x = tf.keras.applications.vgg16.preprocess_input(inputs)

    base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=image_dimensions,
                                                   pooling=None)

    base_model.summary()

    outputs = base_model(x, training=True)
    # base_model.trainable = False
    for layer in base_model.layers[:-frozen_layers]:
        layer.trainable = False

    # Embedding part
    outputs = tf.keras.layers.GlobalAveragePooling2D()(outputs)
    outputs = tf.keras.layers.Flatten()(outputs)

    # if reg_constant != 0:
    #     outputs = tf.keras.layers.Dense(nf * 4, activation='relu', kernel_initializer='he_uniform',
    #                                     kernel_regularizer=l2(reg_constant))(outputs)
    # else:
    #     outputs = tf.keras.layers.Dense(nf * 4, activation='relu', kernel_initializer='he_uniform')(outputs)
    # outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs
    # outputs = tf.keras.layers.BatchNormalization()(outputs)

    # if reg_constant != 0:
    #     outputs = tf.keras.layers.Dense(nf * 2, activation='relu', kernel_initializer='he_uniform',
    #                                     kernel_regularizer=l2(reg_constant))(outputs)
    # else:
    #     outputs = tf.keras.layers.Dense(nf * 2, activation='relu', kernel_initializer='he_uniform')(outputs)
    # outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs
    # outputs = tf.keras.layers.BatchNormalization()(outputs)

    # if reg_constant != 0:
    #     outputs = tf.keras.layers.Dense(nf, activation='relu', kernel_initializer='he_uniform',
    #                                     kernel_regularizer=l2(reg_constant))(outputs)
    # else:
    #     outputs = tf.keras.layers.Dense(nf, activation='relu', kernel_initializer='he_uniform')(outputs)
    # outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs
    # outputs = tf.keras.layers.BatchNormalization()(outputs)

    if reg_constant != 0:
        outputs = tf.keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='he_uniform',
                                        kernel_regularizer=l2(reg_constant))(outputs)
    else:
        outputs = tf.keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='he_uniform')(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# def xception(nb_classes, nf=64, image_dimensions=(384, 288, 3), fc_dropout=0., reg_constant=0.):
#     inputs = tf.keras.Input(shape=image_dimensions)
#     x = tf.keras.applications.xception.preprocess_input(inputs)

#     base_model = tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=image_dimensions,
#                                                    pooling='avg')

#     base_model.summary()

#     outputs = base_model(x, training=True)
#     # base_model.trainable = False
#     for layer in base_model.layers[:-5]:
#         layer.trainable = False

#     # if reg_constant != 0:
#     #     outputs = tf.keras.layers.Dense(nf * 4, activation='relu', kernel_initializer='he_uniform',
#     #                                     kernel_regularizer=l2(reg_constant))(outputs)
#     # else:
#     #     outputs = tf.keras.layers.Dense(nf * 4, activation='relu', kernel_initializer='he_uniform')(outputs)
#     # outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs
#     # outputs = tf.keras.layers.BatchNormalization()(outputs)

#     # if reg_constant != 0:
#     #     outputs = tf.keras.layers.Dense(nf * 2, activation='relu', kernel_initializer='he_uniform',
#     #                                     kernel_regularizer=l2(reg_constant))(outputs)
#     # else:
#     #     outputs = tf.keras.layers.Dense(nf * 2, activation='relu', kernel_initializer='he_uniform')(outputs)
#     # outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs
#     # outputs = tf.keras.layers.BatchNormalization()(outputs)

#     # if reg_constant != 0:
#     #     outputs = tf.keras.layers.Dense(nf, activation='relu', kernel_initializer='he_uniform',
#     #                                     kernel_regularizer=l2(reg_constant))(outputs)
#     # else:
#     #     outputs = tf.keras.layers.Dense(nf, activation='relu', kernel_initializer='he_uniform')(outputs)
#     # outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs
#     # outputs = tf.keras.layers.BatchNormalization()(outputs)

#     if reg_constant != 0:
#         outputs = tf.keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='he_uniform',
#                                         kernel_regularizer=l2(reg_constant))(outputs)
#     else:
#         outputs = tf.keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='he_uniform')(outputs)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model


# def simple_linear(nb_classes, nf=64, image_dimensions=(384, 288, 3), fc_dropout=0.):
#     inputs = tf.keras.Input(shape=image_dimensions)

#     outputs = tf.keras.layers.Flatten()(inputs)
#     outputs = tf.keras.layers.Dense(nf, activation='relu', kernel_initializer='he_uniform',
#                                     kernel_regularizer=l2(5e-2))(outputs)
#     outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs

#     outputs = tf.keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='he_uniform',
#                                     kernel_regularizer=l2(5e-2))(outputs)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model

# def vgg19(nb_classes, nf=64, image_dimensions=(384, 288, 3), fc_dropout=0.):
#     inputs = tf.keras.Input(shape=image_dimensions)
#     x = tf.keras.applications.vgg19.preprocess_input(inputs)
#     base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=image_dimensions,
#                                                    pooling='avg')

#     base_model.summary()

#     outputs = base_model(x, training=False)
#     base_model.trainable = False

#     # Embedding part
#     outputs = tf.keras.layers.Flatten()(outputs)
#     outputs = tf.keras.layers.Dense(nf, activation='selu', kernel_initializer='he_uniform',
#                                     kernel_regularizer=l2(5e-2))(outputs)
#     outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs

#     outputs = tf.keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='he_uniform',
#                                     kernel_regularizer=l2(5e-2))(outputs)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model

# def mobilenet(nb_classes, nf=64, image_dimensions=(384, 288, 3), fc_dropout=0., alpha=1.0):
#     inputs = tf.keras.Input(shape=image_dimensions)

#     preprocessing = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
#     base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg', alpha=alpha)

#     base_model.summary()
#     base_model.trainable = False

#     top_left = preprocessing[:,:224, :224, :]
#     top_right = preprocessing[:,-224:, :224, :]
#     bottom_left = preprocessing[:,:224, -224:, :]
#     bottom_right = preprocessing[:,-224:, -224:, :]
#     offset_x = round((image_dimensions[0]-224)/2)
#     offset_y = round((image_dimensions[1]-224)/2)
#     center = preprocessing[:,offset_x:offset_x+224, offset_y:offset_y+224, :]

#     layers = [top_left, top_right, bottom_right, bottom_left, center]
#     final_outputs = []
#     seq = tf.keras.Sequential()
#     seq.add(base_model)
#     seq.add(tf.keras.layers.Flatten())
#     seq.add(tf.keras.layers.Dense(nf * 4, activation='relu', kernel_initializer='he_uniform'))
#     if fc_dropout > 0.:
#         seq.add(tf.keras.layers.Dropout(fc_dropout))
#     seq.add(tf.keras.layers.BatchNormalization())

#     seq.add(tf.keras.layers.Dense(nf * 2, activation='relu', kernel_initializer='he_uniform'))
#     if fc_dropout > 0.:
#         seq.add(tf.keras.layers.Dropout(fc_dropout))
#     seq.add(tf.keras.layers.BatchNormalization())

#     seq.add(tf.keras.layers.Dense(nf, activation='relu', kernel_initializer='he_uniform'))
#     if fc_dropout > 0.:
#         seq.add(tf.keras.layers.Dropout(fc_dropout))
#     seq.add(tf.keras.layers.BatchNormalization())

#     seq.add(tf.keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='he_uniform'))
#     for layer in layers:

#         outputs = seq(layer)
#         final_outputs.append(outputs)
#     outputs = tf.keras.layers.average(final_outputs)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model

# def mobilenet_full(nb_classes, nf=64, image_dimensions=(224, 224, 3), fc_dropout=0., alpha=1.0):
#     inputs = tf.keras.Input(shape=image_dimensions)

#     preprocessing = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
#     base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, input_shape=image_dimensions, alpha=alpha)

#     base_model.summary()
#     base_model.trainable = True
#     outputs = base_model(preprocessing)

#     # Embedding part
#     outputs = tf.keras.layers.Flatten()(outputs)
#     # outputs = tf.keras.layers.Dense(nf*2, activation='selu', kernel_initializer='he_uniform',
#     #                                 kernel_regularizer=l2(5e-4))(outputs)
#     # outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs
#     outputs = tf.keras.layers.Dense(nf, activation='selu', kernel_initializer='he_uniform', kernel_regularizer=l2(5e-4))(outputs)
#     outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs

#     outputs = tf.keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='he_uniform', kernel_regularizer=l2(5e-4))(outputs)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model



# def MIclass(nb_inputs, nb_classes, image_dimensions=(384, 288, 3), bn=False, conv_dropout=0., fc_dropout=0., nf=64, multi_output=False, reg_constant=0.):
#     '''
#     Multi-input network for classification with separate branches for each input (shared weights)
#     after which all brances are merged and continue via a single branch.

#     Arguments:
#     nb_inputs - number of inputs, i.e. separate branches
#     nb_classes - number of output classes for softmax layer
#     image_dimensions - dimensions for each input
#     bn - Do batchnormalization if bn is True
#     fc_dropout - Must be float 0.-1. for dropout rate for fully connected layers
#     '''
#     individual_input_network = input_block_efficientnet(image_dimensions)
#     # individual_input_network = input_block_efficientnet(image_dimensions)
#     individual_input_network.trainable = False
#     if nb_inputs > 1:
#         inputs = tf.keras.Input(shape=(nb_inputs, ) + image_dimensions)

#         # Process branches separately with shared weights
#         intermediate = [individual_input_network(inputs[:, i, :, :, :]) for i in range(nb_inputs)]

#         # Merge branches
#         outputs_max = tf.keras.layers.Maximum()(intermediate)
#         outputs_avg = tf.keras.layers.Average()(intermediate)
#         # outputs = tf.keras.layers.concatenate(intermediate, axis=3)
#         outputs = tf.keras.layers.concatenate([outputs_avg, outputs_max], axis=1)

#     else:
#         inputs = tf.keras.Input(shape=image_dimensions)
#         outputs = individual_input_network(inputs)

#     # Continue with single branch
#     outputs = tf.keras.layers.Flatten()(outputs)
#     # outputs = tf.keras.layers.Dense(nf, activation='relu', kernel_initializer='he_uniform',
#     #                                 kernel_regularizer=l2(5e-2))(outputs)
#     # outputs = tf.keras.layers.concatenate([outputs, tf.keras.layers.GlobalAveragePooling2D()(preprocessing)])
#     if reg_constant != 0:
#         outputs = tf.keras.layers.Dense(nf*4, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(reg_constant))(outputs)
#     else:
#         outputs = tf.keras.layers.Dense(nf*4, activation='relu', kernel_initializer='he_uniform')(outputs)
#     outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs
#     outputs = tf.keras.layers.BatchNormalization()(outputs) if bn else outputs

#     if reg_constant != 0:
#         outputs = tf.keras.layers.Dense(nf*2, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(reg_constant))(outputs)
#     else:
#         outputs = tf.keras.layers.Dense(nf*2, activation='relu', kernel_initializer='he_uniform')(outputs)
#     outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs
#     outputs = tf.keras.layers.BatchNormalization()(outputs) if bn else outputs

#     if reg_constant != 0:
#         outputs = tf.keras.layers.Dense(nf, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(reg_constant))(outputs)
#     else:
#         outputs = tf.keras.layers.Dense(nf, activation='relu', kernel_initializer='he_uniform')(outputs)
#     outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs
#     outputs = tf.keras.layers.BatchNormalization()(outputs) if bn else outputs

#     if multi_output:
#         reshape = tf.keras.layers.Reshape((2,1))
#         outputs = [reshape(tf.keras.layers.Dense(2, activation='softmax', kernel_initializer='he_uniform', kernel_regularizer=l2(1e-1))(outputs)) for _ in range(nb_classes)]
#         outputs = tf.keras.layers.concatenate(outputs, -1)
#     else:
#         outputs = tf.keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='he_uniform')(outputs)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model


# def embedding_network(nf=64, image_dimensions=(384, 288, 3), embedding_size=64, bn=False, conv_dropout=0., fc_dropout=0.):
#     inputs = tf.keras.Input(shape=image_dimensions)

#     outputs = tf.keras.layers.Conv2D(nf, kernel_size=3, activation=tf.nn.relu, kernel_initializer='he_uniform', kernel_regularizer=l2(2e-4))(inputs)
#     outputs = tf.keras.layers.Dropout(conv_dropout)(outputs) if conv_dropout > 0. else outputs
#     outputs = tf.keras.layers.BatchNormalization()(outputs) if bn else outputs
#     outputs = tf.keras.layers.MaxPooling2D()(outputs)

#     outputs = tf.keras.layers.Conv2D(nf, kernel_size=3, activation=tf.nn.relu, kernel_initializer='he_uniform', kernel_regularizer=l2(2e-4))(outputs)
#     outputs = tf.keras.layers.Dropout(conv_dropout)(outputs) if conv_dropout > 0. else outputs
#     outputs = tf.keras.layers.BatchNormalization()(outputs) if bn else outputs
#     outputs = tf.keras.layers.MaxPooling2D()(outputs)

#     outputs = tf.keras.layers.Conv2D(2 * nf, kernel_size=3, activation=tf.nn.relu, kernel_initializer='he_uniform', kernel_regularizer=l2(2e-4))(outputs)
#     outputs = tf.keras.layers.Dropout(conv_dropout)(outputs) if conv_dropout > 0. else outputs
#     outputs = tf.keras.layers.BatchNormalization()(outputs) if bn else outputs
#     outputs = tf.keras.layers.MaxPooling2D()(outputs)

#     outputs = tf.keras.layers.Flatten()(outputs)
#     outputs = tf.keras.layers.Dense(2 * nf, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(2e-4))(outputs)
#     outputs = tf.keras.layers.Dense(embedding_size, activation=None, kernel_initializer='he_uniform', kernel_regularizer=l2(2e-4))(outputs)

#     #Force the encoding to live on the d-dimentional hypershpere
#     outputs = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(outputs)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model


# def embedding_vgg(nf=64, image_dimensions=(384, 288, 3), embedding_size=64, fc_dropout=0.):
#     inputs = tf.keras.Input(shape=image_dimensions)

#     base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=image_dimensions, pooling='avg')

#     # base_model.trainable = False
#     # for layer in base_model.layers[:15]:
#     #     layer.trainable = False
#     base_model.summary()

#     outputs = base_model(inputs, training=True)

#     # Embedding part
#     outputs = tf.keras.layers.Flatten()(outputs)
#     outputs = tf.keras.layers.Dense(nf, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(2e-4))(outputs)
#     outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs

#     outputs = tf.keras.layers.Dense(embedding_size, activation=None, kernel_initializer='he_uniform', kernel_regularizer=l2(2e-4), use_bias=False)(outputs)
#     # Force the encoding to live on the d-dimentional hypershpere
#     outputs = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(outputs)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model


# def embedding_vgg_nodropout(nf=64, image_dimensions=(384, 288, 3), embedding_size=64, fc_dropout=0.):
#     inputs = tf.keras.Input(shape=image_dimensions)

#     base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=image_dimensions, pooling='avg')

#     # base_model.trainable = False
#     # for layer in base_model.layers[:15]:
#     #     layer.trainable = False
#     base_model.summary()

#     outputs = base_model(inputs, training=True)

#     # Embedding part
#     outputs = tf.keras.layers.Flatten()(outputs)
#     outputs = tf.keras.layers.Dense(nf, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(2e-4))(outputs)

#     outputs = tf.keras.layers.Dense(embedding_size, activation=None, kernel_initializer='he_uniform', kernel_regularizer=l2(2e-4), use_bias=False)(outputs)
#     # Force the encoding to live on the d-dimentional hypershpere
#     outputs = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(outputs)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model


# def triplet_model(nf=64, image_dimensions=(384, 288, 3), embedding_size=64, bn=False, conv_dropout=0., fc_dropout=0., margin=0.2):
#     # Define the tensors for the three input images
#     anchor_input = tf.keras.layers.Input(image_dimensions, name="anchor_input")
#     positive_input = tf.keras.layers.Input(image_dimensions, name="positive_input")
#     negative_input = tf.keras.layers.Input(image_dimensions, name="negative_input")

#     # Generate the encodings (feature vectors) for the three images
#     # embedder = embedding_network(nf=nf, image_dimensions=image_dimensions, embedding_size=embedding_size, bn=bn, conv_dropout=conv_dropout, fc_dropout=fc_dropout)
#     embedder = embedding_vgg(nf=nf, image_dimensions=image_dimensions, embedding_size=embedding_size, fc_dropout=fc_dropout)
#     embedder.summary()
#     encoded_a = embedder(anchor_input)
#     encoded_p = embedder(positive_input)
#     encoded_n = embedder(negative_input)

#     # TripletLoss Layer
#     loss_layer = TripletLossLayer(alpha=margin, name='triplet_loss')([encoded_a, encoded_p, encoded_n])
#     accuracy_layer = TripletAccuracyLayer(name='triplet_accuracy')([encoded_a, encoded_p, encoded_n])
#     pdist_layer = TripletPosDistanceLayer(name='triplet_pdist')([encoded_a, encoded_p, encoded_n])
#     ndist_layer = TripletNegDistanceLayer(name='triplet_ndist')([encoded_a, encoded_p, encoded_n])

#     # Connect the inputs with the outputs
#     network = tf.keras.Model(inputs=[anchor_input,positive_input,negative_input],outputs=[loss_layer, accuracy_layer, pdist_layer, ndist_layer])

#     # return the model
#     return network


# def tuned_effnet(image_dimensions, fold='0'):

#     model = tf.keras.models.load_model(os.path.join(f'data/snapshots/hypVSadnVSssp_HDall2022_efficientnetFull_regularized0.0_256x256_1in_nf64_bnTrue_fcdo0.5_convdo0.0_lossls0.4_lossDropout1.0_fold{fold}_best'), compile=False)
#     model.trainable = False
#     inputs = tf.keras.Input(shape=image_dimensions)
#     x = inputs
#     for layer in model.layers[1:2]:
#         x = layer(x)
#     return tf.keras.Model(inputs=inputs, outputs=x)

def efficientnet(nb_classes, nf=64, image_dimensions=(384, 288, 3), fc_dropout=0., multi_output=False, reg_constant=0., nb_inputs=1, fold='0', b="0", frozen_layers=1):

    # base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights='imagenet',
    #                                                                input_shape=image_dimensions, pooling='avg')
    if b == '7':
        base_model = tf.keras.applications.efficientnet.EfficientNetB7(include_top=False, weights='imagenet',
                                                                   input_shape=image_dimensions, pooling='avg')
    elif b == '6':
        base_model = tf.keras.applications.efficientnet.EfficientNetB6(include_top=False, weights='imagenet',
                                                                   input_shape=image_dimensions, pooling='avg')
    elif b == '5':
        base_model = tf.keras.applications.efficientnet.EfficientNetB5(include_top=False, weights='imagenet',
                                                                   input_shape=image_dimensions, pooling='avg')
    elif b == '4':
        base_model = tf.keras.applications.efficientnet.EfficientNetB4(include_top=False, weights='imagenet',
                                                                   input_shape=image_dimensions, pooling='avg')
    elif b == '3':
        base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights='imagenet',
                                                                   input_shape=image_dimensions, pooling='avg')
    elif b == '2':
        base_model = tf.keras.applications.efficientnet.EfficientNetB2(include_top=False, weights='imagenet',
                                                                   input_shape=image_dimensions, pooling='avg')
    elif b == '1':
        base_model = tf.keras.applications.efficientnet.EfficientNetB1(include_top=False, weights='imagenet',
                                                                   input_shape=image_dimensions, pooling='avg')
    else:
        base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights='imagenet',
                                                                   input_shape=image_dimensions, pooling='avg')
    # base_model = tuned_effnet(image_dimensions, fold)
    # base_model.trainable = False
    for layer in base_model.layers[:-frozen_layers]:
        layer.trainable = False
    # base_model.summary()
    # outputs = base_model(preprocessing, training=False)

    if nb_inputs > 1:
        inputs = tf.keras.Input(shape=(nb_inputs, ) + image_dimensions)
        preprocessing = tf.keras.applications.efficientnet.preprocess_input(inputs)

        # Process branches separately with shared weights
        intermediate = [base_model(preprocessing[:, f, :, :], training=False) for f in range(nb_inputs)]

        # Merge branches
        outputs_max = tf.keras.layers.Maximum()(intermediate)
        outputs_avg = tf.keras.layers.Average()(intermediate)
        # outputs = tf.keras.layers.concatenate(intermediate, axis=3)
        outputs = tf.keras.layers.concatenate([outputs_avg, outputs_max], axis=1)

    else:
        inputs = tf.keras.Input(shape=image_dimensions)
        preprocessing = tf.keras.applications.efficientnet.preprocess_input(inputs)

        outputs = base_model(preprocessing, training=False)

    # Continue with single branch
    outputs = tf.keras.layers.Flatten()(outputs)
    # outputs = tf.keras.layers.Dense(nf, activation='relu', kernel_initializer='he_uniform',
    #                                 kernel_regularizer=l2(5e-2))(outputs)
    # outputs = tf.keras.layers.concatenate([outputs, tf.keras.layers.GlobalAveragePooling2D()(preprocessing)])
    # if reg_constant != 0:
    #     outputs = tf.keras.layers.Dense(nf*4, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(reg_constant))(outputs)
    # else:
    #     outputs = tf.keras.layers.Dense(nf*4, activation='relu', kernel_initializer='he_uniform')(outputs)
    # outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs
    # outputs = tf.keras.layers.BatchNormalization()(outputs)

    # if reg_constant != 0:
    #     outputs = tf.keras.layers.Dense(nf*2, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(reg_constant))(outputs)
    # else:
    #     outputs = tf.keras.layers.Dense(nf*2, activation='relu', kernel_initializer='he_uniform')(outputs)
    # outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs
    # outputs = tf.keras.layers.BatchNormalization()(outputs)

    # if reg_constant != 0:
    #     outputs = tf.keras.layers.Dense(nf, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(reg_constant))(outputs)
    # else:
    #     outputs = tf.keras.layers.Dense(nf, activation='relu', kernel_initializer='he_uniform')(outputs)
    # outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs
    # outputs = tf.keras.layers.BatchNormalization()(outputs)

    # outputs = tf.keras.layers.experimental.RandomFourierFeatures(output_dim=4096, scale=10., kernel_initializer='gaussian')(outputs)

    if multi_output:
        reshape = tf.keras.layers.Reshape((2, 1))
        outputs = [reshape(tf.keras.layers.Dense(2, activation='softmax', kernel_initializer='he_uniform',
                                                 kernel_regularizer=l2(5e-2))(outputs)) for _ in range(nb_classes)]
        outputs = tf.keras.layers.concatenate(outputs, -1)
    else:
        # outputs = tf.keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='he_uniform',
        #                                 kernel_regularizer=l2(5e-2))(outputs)
        if reg_constant != 0:
            outputs = tf.keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='he_uniform', kernel_regularizer=l2(reg_constant))(outputs)
        else:
            outputs = tf.keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='he_uniform')(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# def efficientnet_full(nb_classes, b='0', nf=64,image_dimensions=(384, 288, 3), fc_dropout=0., reg_constant=0.):
#     inputs = tf.keras.Input(shape=image_dimensions)

#     preprocessing = tf.keras.applications.efficientnet.preprocess_input(inputs)

#     if b == '7':
#         base_model = tf.keras.applications.efficientnet.EfficientNetB7(include_top=False, weights='imagenet',
#                                                                    input_shape=image_dimensions, pooling='avg')
#     elif b == '6':
#         base_model = tf.keras.applications.efficientnet.EfficientNetB6(include_top=False, weights='imagenet',
#                                                                    input_shape=image_dimensions, pooling='avg')
#     elif b == '5':
#         base_model = tf.keras.applications.efficientnet.EfficientNetB5(include_top=False, weights='imagenet',
#                                                                    input_shape=image_dimensions, pooling='avg')
#     elif b == '4':
#         base_model = tf.keras.applications.efficientnet.EfficientNetB4(include_top=False, weights='imagenet',
#                                                                    input_shape=image_dimensions, pooling='avg')
#     elif b == '3':
#         base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights='imagenet',
#                                                                    input_shape=image_dimensions, pooling='avg')
#     elif b == '2':
#         base_model = tf.keras.applications.efficientnet.EfficientNetB2(include_top=False, weights='imagenet',
#                                                                    input_shape=image_dimensions, pooling='avg')
#     elif b == '1':
#         base_model = tf.keras.applications.efficientnet.EfficientNetB1(include_top=False, weights='imagenet',
#                                                                    input_shape=image_dimensions, pooling='avg')
#     else:
#         base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights='imagenet',
#                                                                    input_shape=image_dimensions, pooling='avg')
#     # base_model.summary()
#     outputs = base_model(preprocessing, training=True)

#     # Continue with single branch
#     outputs = tf.keras.layers.Flatten(name='extra_part')(outputs)
#     # outputs = tf.keras.layers.Dense(nf, activation='relu', kernel_initializer='he_uniform',
#     #                                 kernel_regularizer=l2(5e-2))(outputs)
#     # outputs = tf.keras.layers.concatenate([outputs, tf.keras.layers.GlobalAveragePooling2D()(preprocessing)])
#     if reg_constant != 0:
#         outputs = tf.keras.layers.Dense(nf * 4, activation='relu', kernel_initializer='he_uniform',
#                                         kernel_regularizer=l2(reg_constant))(outputs)
#     else:
#         outputs = tf.keras.layers.Dense(nf * 4, activation='relu', kernel_initializer='he_uniform')(outputs)
#     outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs
#     outputs = tf.keras.layers.BatchNormalization()(outputs)

#     if reg_constant != 0:
#         outputs = tf.keras.layers.Dense(nf * 2, activation='relu', kernel_initializer='he_uniform',
#                                         kernel_regularizer=l2(reg_constant))(outputs)
#     else:
#         outputs = tf.keras.layers.Dense(nf * 2, activation='relu', kernel_initializer='he_uniform')(outputs)
#     outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs
#     outputs = tf.keras.layers.BatchNormalization()(outputs)

#     if reg_constant != 0:
#         outputs = tf.keras.layers.Dense(nf, activation='relu', kernel_initializer='he_uniform',
#                                         kernel_regularizer=l2(reg_constant))(outputs)
#     else:
#         outputs = tf.keras.layers.Dense(nf, activation='relu', kernel_initializer='he_uniform')(outputs)
#     outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs
#     outputs = tf.keras.layers.BatchNormalization()(outputs)

#     if reg_constant != 0:
#         outputs = tf.keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='he_uniform',
#                                         kernel_regularizer=l2(reg_constant))(outputs)
#     else:
#         outputs = tf.keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='he_uniform')(outputs)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model


# def detnet_bottleneck(input, name ,with_conv=False):

#     reg_cst = 1e-3
#     # First Conv
#     x = tf.keras.layers.Conv2D(64, (1,1), name=name+'_1_conv', activation='relu')(input)
#     x = tf.keras.layers.BatchNormalization(name=name+'_1_bn')(x)
#     # x = tf.keras.layers.LeakyReLU(name=name+'_1_elu')(x)

#     #Dilated conv
#     x = tf.keras.layers.Conv2D(64, (3, 3), dilation_rate=(2,2), padding='same' ,name=name + '_2_conv', activation='relu')(x)
#     x = tf.keras.layers.BatchNormalization(name=name + '_2_bn')(x)
#     # x = tf.keras.layers.LeakyReLU(name=name + '_2_elu')(x)

#     # Last Conv
#     x = tf.keras.layers.Conv2D(256, (1, 1), name=name + '_3_conv')(x)
#     x = tf.keras.layers.BatchNormalization(name=name + '_3_bn')(x)

#     if with_conv:
#         passing_conv = tf.keras.layers.Conv2D(256, (1, 1), name=name + '_0_conv', activation='relu')(input)
#         passing_conv = tf.keras.layers.BatchNormalization(name=name + '_0_bn')(passing_conv)
#         output = tf.keras.layers.Add(name=name+'_add')([x, passing_conv])
#     else:
#         output = tf.keras.layers.Add(name=name+'_add')([x, input])

#     return tf.keras.layers.LeakyReLU(name=name + '_out')(output)


# def basic_cnn(nb_classes, nf=64, image_dimensions=(224, 224, 3), fc_dropout=0., conv_dropout=0.):
#     inputs = tf.keras.Input(shape=image_dimensions)

#     x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
#     # x = inputs

#     l = 4

#     for i in range(l):
#         x = tf.keras.layers.Conv2D(16*2**(i+1), (1,1), activation='relu')(x)
#         x = tf.keras.layers.Conv2D(16 * 2 ** (i + 1), (3, 3), activation='relu')(x)
#         x = tf.keras.layers.Conv2D(32 * 2 ** (i + 1), (3, 3), activation='relu')(x)

#         x = tf.keras.layers.MaxPooling2D((2, 2), (2, 2))(x)
#         if i != l-1:
#             x = tf.keras.layers.SpatialDropout2D(conv_dropout)(x) if conv_dropout > 0. else x


#     x = tf.keras.layers.GlobalAveragePooling2D()(x)
#     # Embedding part
#     outputs = tf.keras.layers.Flatten()(x)
#     outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs

#     outputs = tf.keras.layers.Dense(nf, activation='relu', kernel_initializer='he_uniform',
#                                     kernel_regularizer=l2(2e-4))(outputs)
#     outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs

#     outputs = tf.keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='he_uniform',
#                                     kernel_regularizer=l2(2e-4))(outputs)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model

# def detnet(nb_classes, image_dimensions=(384, 288, 3), nf=64, fc_dropout=0., conv_dropout=0.):
#     inputs = tf.keras.Input(shape=image_dimensions)



#     # rescale = tf.keras.layers.experimental.preprocessing.Rescaling(2., offset= -1) # xception, the input images are in range [0, 1)

#     preprocess = tf.keras.applications.resnet50.preprocess_input



#     x = preprocess(inputs)
#     base_model = tf.keras.applications.ResNet50(
#         weights='imagenet',  # Load weights pre-trained on ImageNet.
#         input_shape=image_dimensions,
#         include_top=False,
#         input_tensor=x)  # Do not include the ImageNet classifier at the top.
#     base_model.trainable = False

#     x = base_model.get_layer('conv4_block6_out').output

#     x = detnet_bottleneck(x, 'conv5_block1', True)
#     x = tf.keras.layers.SpatialDropout2D(conv_dropout)(x) if conv_dropout > 0. else x
#     x = detnet_bottleneck(x, 'conv5_block2')
#     x = tf.keras.layers.SpatialDropout2D(conv_dropout)(x) if conv_dropout > 0. else x
#     x = detnet_bottleneck(x, 'conv5_block3')
#     x = tf.keras.layers.SpatialDropout2D(conv_dropout)(x) if conv_dropout > 0. else x

#     x = detnet_bottleneck(x, 'conv6_block1', True)
#     x = tf.keras.layers.SpatialDropout2D(conv_dropout)(x) if conv_dropout > 0. else x
#     x = detnet_bottleneck(x, 'conv6_block2')
#     x = tf.keras.layers.SpatialDropout2D(conv_dropout)(x) if conv_dropout > 0. else x
#     x = detnet_bottleneck(x, 'conv6_block3')

#     # Classifying part
#     outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
#     outputs = tf.keras.layers.Dense(nf * 4, activation='relu', kernel_initializer='he_uniform')(outputs)
#     outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs
#     outputs = tf.keras.layers.BatchNormalization()(outputs)

#     outputs = tf.keras.layers.Dense(nf * 2, activation='relu', kernel_initializer='he_uniform')(outputs)
#     outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs
#     outputs = tf.keras.layers.BatchNormalization()(outputs)

#     outputs = tf.keras.layers.Dense(nf, activation='relu', kernel_initializer='he_uniform')(outputs)
#     outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs
#     outputs = tf.keras.layers.BatchNormalization()(outputs)

#     outputs = tf.keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='he_uniform')(outputs)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs)

#     return model


# def wavelet_network(nf=64, image_dimensions=(384, 288, 3), nb_classes=2, bn=False, conv_dropout=0., wave_kern='db2'):
#     inputs = tf.keras.Input(shape=image_dimensions)
#     rescale = tf.keras.layers.experimental.preprocessing.Rescaling(255.)(inputs)


#     wave = WaveTFFactory.build(wave_kern)(rescale)
#     wave = tf.keras.layers.BatchNormalization()(wave)

#     conv = tf.keras.layers.Conv2D(nf, kernel_size=1)(inputs)
#     conv = tf.keras.layers.BatchNormalization()(conv)
#     conv = tf.keras.layers.ReLU()(conv)
#     conv = tf.keras.layers.Conv2D(nf, kernel_size=3, padding='same')(conv)
#     conv = tf.keras.layers.BatchNormalization()(conv)
#     conv = tf.keras.layers.ReLU()(conv)
#     conv = tf.keras.layers.Conv2D(nf, kernel_size=1)(conv)
#     conv = tf.keras.layers.BatchNormalization()(conv)
#     conv = tf.keras.layers.ReLU()(conv)
#     conv = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv)

#     conv = tf.keras.layers.concatenate([wave, conv])

#     conv = tf.keras.layers.Conv2D(nf*2, kernel_size=3, padding='same')(conv)
#     conv = tf.keras.layers.BatchNormalization()(conv)
#     conv = tf.keras.layers.ReLU()(conv)
#     conv = tf.keras.layers.Conv2D(nf * 2, kernel_size=3, padding='same')(conv)
#     conv = tf.keras.layers.BatchNormalization()(conv)
#     conv = tf.keras.layers.ReLU()(conv)
#     conv = tf.keras.layers.Conv2D(nf * 2, kernel_size=1, padding='same')(conv)
#     conv = tf.keras.layers.BatchNormalization()(conv)
#     conv = tf.keras.layers.ReLU()(conv)
#     pool = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv)

#     conv = tf.keras.layers.Conv2D(nf*4, kernel_size=3, padding='same')(pool)
#     conv = tf.keras.layers.BatchNormalization()(conv)
#     conv = tf.keras.layers.ReLU()(conv)
#     conv = tf.keras.layers.Conv2D(nf * 4, kernel_size=3, padding='same')(conv)
#     conv = tf.keras.layers.BatchNormalization()(conv)
#     conv = tf.keras.layers.ReLU()(conv)
#     conv = tf.keras.layers.Conv2D(nf * 4, kernel_size=1, padding='same')(conv)
#     conv = tf.keras.layers.BatchNormalization()(conv)
#     conv = tf.keras.layers.ReLU()(conv)
#     conv = tf.keras.layers.concatenate([pool, conv])
#     pool = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv)

#     conv = tf.keras.layers.Conv2D(nf*8, kernel_size=3, padding='same')(pool)
#     conv = tf.keras.layers.BatchNormalization()(conv)
#     conv = tf.keras.layers.ReLU()(conv)
#     conv = tf.keras.layers.Conv2D(nf * 8, kernel_size=3, padding='same')(conv)
#     conv = tf.keras.layers.BatchNormalization()(conv)
#     conv = tf.keras.layers.ReLU()(conv)
#     conv = tf.keras.layers.concatenate([pool, conv])
#     conv = tf.keras.layers.Conv2D(nf * 8, kernel_size=1, padding='same')(conv)
#     conv = tf.keras.layers.BatchNormalization()(conv)
#     conv = tf.keras.layers.ReLU()(conv)
#     conv = tf.keras.layers.GlobalMaxPooling2D()(conv)

#     outputs = tf.keras.layers.Dropout(conv_dropout)(conv)

#     outputs = tf.keras.layers.Dense(2 * nf, activation='relu')(outputs)
#     if nb_classes == 2:
#         activation = 'sigmoid'
#     else:
#         activation = 'softmax'
#     outputs = tf.keras.layers.Dense(nb_classes, activation=activation)(outputs)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model

# def transfer(model_name, nb_classes=2, image_dimensions=(384, 288, 3)):
#     model = tf.keras.models.load_model(os.path.join('data/snapshots/', model_name + '_best.hdf5'))
#     model.trainable = False

#     model.layers[-2].trainable = True
#     x = model.layers[-2].output

#     if nb_classes == 2:
#         activation = 'sigmoid'
#     else:
#         activation = 'softmax'
#     output = tf.keras.layers.Dense(nb_classes, activation=activation, name='dense_classification')(x)
#     model = tf.keras.Model(inputs=model.inputs, outputs=output)
#     return model


# def darknet_block(i, filters_conv1):
#     x = tf.keras.layers.Conv2D(filters_conv1, 1)(i)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU()(x)
#     x = tf.keras.layers.Conv2D(filters_conv1*2, 3, padding='same')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU()(x)

#     return tf.keras.layers.concatenate([i,x])


# def darknet(nb_classes, image_dimensions=(384, 288, 3), nf=64, fc_dropout=0., conv_dropout=0.):
#     inputs = tf.keras.Input(shape=image_dimensions)



#     x = tf.keras.layers.Conv2D(32, 3, padding='same')(inputs)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU()(x)

#     x = tf.keras.layers.Conv2D(64, 3,strides=(2,2) ,padding='same')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU()(x)

#     x = darknet_block(x, 64)

#     x = tf.keras.layers.Conv2D(128, 3, strides=(2, 2), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU()(x)

#     x = darknet_block(x, 64)
#     x = darknet_block(x, 64)
#     x = tf.keras.layers.Dropout(conv_dropout)(x)

#     x = tf.keras.layers.Conv2D(128, 3, strides=(2, 2), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU()(x)

#     x = darknet_block(x, 128)
#     x = darknet_block(x, 128)
#     x = tf.keras.layers.Dropout(conv_dropout)(x)
#     x = darknet_block(x, 128)
#     x = darknet_block(x, 128)
#     x = tf.keras.layers.Dropout(conv_dropout)(x)
#     x = darknet_block(x, 128)
#     x = darknet_block(x, 128)
#     x = tf.keras.layers.Dropout(conv_dropout)(x)
#     x = darknet_block(x, 128)
#     x = darknet_block(x, 128)
#     x = tf.keras.layers.Dropout(conv_dropout)(x)

#     x = tf.keras.layers.Conv2D(256, 3, strides=(2, 2), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU()(x)

#     x = darknet_block(x, 256)
#     x = darknet_block(x, 256)
#     x = tf.keras.layers.Dropout(conv_dropout)(x)
#     x = darknet_block(x, 256)
#     x = darknet_block(x, 256)
#     x = tf.keras.layers.Dropout(conv_dropout)(x)
#     x = darknet_block(x, 256)
#     x = darknet_block(x, 256)
#     x = tf.keras.layers.Dropout(conv_dropout)(x)
#     x = darknet_block(x, 256)
#     x = darknet_block(x, 256)
#     x = tf.keras.layers.Dropout(conv_dropout)(x)

#     x = tf.keras.layers.Conv2D(512, 3, strides=(2, 2), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU()(x)

#     x = darknet_block(x, 512)
#     x = darknet_block(x, 512)
#     x = tf.keras.layers.Dropout(conv_dropout)(x)
#     x = darknet_block(x, 512)
#     x = darknet_block(x, 512)
#     x = tf.keras.layers.Dropout(conv_dropout)(x)

#     x = tf.keras.layers.GlobalAveragePooling2D()(x)
#     x = tf.keras.layers.Flatten()(x)
#     tf.keras.layers.Dropout(fc_dropout)(x)
#     x = tf.keras.layers.Dense(nf, activation='relu')(x)
#     x = tf.keras.layers.Dense(nf, activation='relu')(x)

#     outputs = tf.keras.layers.Dense(nb_classes, activation='softmax')(x)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model


# def inception_block(i, nb_conv1=64, nb_conv13=96, nb_conv3=128, nb_conv15=16, nb_conv5=32, nb_pool1=32):

#     conv1 = tf.keras.layers.Conv2D(nb_conv1, 1, activation='relu')(i)

#     conv13 = tf.keras.layers.Conv2D(nb_conv13, 1, activation='relu')(i)
#     conv3 = tf.keras.layers.Conv2D(nb_conv3, 3, activation='relu', padding='same')(conv13)

#     conv15 = tf.keras.layers.Conv2D(nb_conv15, 1, activation='relu')(i)
#     conv5 = tf.keras.layers.Conv2D(nb_conv5, 5, activation='relu', padding='same')(conv15)

#     pool1 = tf.keras.layers.MaxPool2D((3,3), strides=(1,1), padding='same')(i)
#     pool1 = tf.keras.layers.Conv2D(nb_pool1, 1, activation='relu')(pool1)

#     return tf.keras.layers.concatenate([conv1, conv3, conv5, pool1])

# def inceptionV4_block(i, nb_conv1=64, nb_conv13=96, nb_conv3=128, nb_conv15=16, nb_conv5=32, nb_pool1=64):

#     conv1 = tf.keras.layers.Conv2D(nb_conv1, 1)(i)

#     conv13 = tf.keras.layers.Conv2D(nb_conv13, 1)(i)
#     conv3 = tf.keras.layers.Conv2D(nb_conv3, 3,  padding='same')(conv13)

#     conv15 = tf.keras.layers.Conv2D(nb_conv15, 1)(i)
#     conv5 = tf.keras.layers.Conv2D(nb_conv5, 3, padding='same')(conv15)
#     conv5 = tf.keras.layers.Conv2D(nb_conv5, 3, padding='same')(conv5)

#     out = tf.keras.layers.concatenate([conv1, conv3, conv5])
#     out = tf.keras.layers.Conv2D(nb_pool1, 1)(out)

#     out = tf.keras.layers.Add()([i, out])

#     return tf.keras.layers.ReLU()(out)


# def googlenet(nb_classes=2, image_dimensions=(384, 288, 3), output_weights=(1, 0.3, 0.3)):
#     inputs = tf.keras.layers.Input(image_dimensions)

#     x = tf.keras.layers.Conv2D(64, 7, strides=(2,2), padding='same')(inputs)
#     x = tf.keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same')(x)

#     x = tf.keras.layers.Conv2D(64, 1, padding='same')(x)
#     x = tf.keras.layers.Conv2D(196, 3, padding='same')(x)
#     x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = tf.keras.layers.Dropout(0.2)(x)

#     x = inception_block(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = inception_block(x, 128, 128, 192, 32, 96, 64)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = tf.keras.layers.Dropout(0.2)(x)

#     x = inception_block(x, 192, 96, 208, 16, 48, 64)
#     x = tf.keras.layers.Dropout(0.2)(x)

#     output1 = tf.keras.layers.AvgPool2D((5, 5), strides=(3,3), padding='same')(x)
#     output1 = tf.keras.layers.Conv2D(128, 1, activation='relu')(output1)
#     output1 = tf.keras.layers.BatchNormalization()(output1)
#     output1 = tf.keras.layers.Flatten()(output1)
#     output1 = tf.keras.layers.Dense(1024, activation='relu')(output1)
#     output1 = tf.keras.layers.Dropout(0.7)(output1)
#     output1 = tf.keras.layers.Dense(nb_classes, activation='softmax')(output1)

#     x = inception_block(x, 160, 112, 224, 24, 64, 64)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = inception_block(x, 128, 128, 256, 24, 64, 64)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = inception_block(x, 112, 144, 288, 32, 64, 64)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     output2 = tf.keras.layers.AvgPool2D((5, 5), strides=(3, 3), padding='same')(x)
#     output2 = tf.keras.layers.Conv2D(128, 1, activation='relu')(output2)
#     output2 = tf.keras.layers.BatchNormalization()(output2)
#     output2 = tf.keras.layers.Flatten()(output2)
#     output2 = tf.keras.layers.Dense(1024, activation='relu')(output2)
#     output2 = tf.keras.layers.Dropout(0.7)(output2)
#     output2 = tf.keras.layers.Dense(nb_classes, activation='softmax')(output2)

#     x = inception_block(x, 256, 160, 320, 32, 128, 128)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)

#     x = inception_block(x, 256, 160, 320, 32, 128, 128)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = inception_block(x, 384, 192, 384, 48, 128, 128)
#     x = tf.keras.layers.GlobalAveragePooling2D()(x)

#     x = tf.keras.layers.Dropout(0.4)(x)

#     outputs = tf.keras.layers.Dense(nb_classes, activation='softmax')(x)


#     outputs = tf.keras.layers.concatenate([outputs, output1, output2])
#     outputs = tf.keras.layers.Dense(nb_classes, activation='softmax')(outputs)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model


# def googlenet_v2(nb_classes=2, image_dimensions=(384, 288, 3), output_weights=(1, 0.3, 0.3)):
#     inputs = tf.keras.layers.Input(image_dimensions)

#     x = tf.keras.layers.Conv2D(64, 7, strides=(2,2), padding='same')(inputs)
#     x = tf.keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same')(x)

#     x = tf.keras.layers.Conv2D(64, 1, padding='same')(x)
#     x = tf.keras.layers.Conv2D(196, 3, padding='same')(x)
#     x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = tf.keras.layers.Dropout(0.2)(x)

#     x = inceptionV4_block(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = inceptionV4_block(x, 128, 128, 192, 32, 96, 64)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = tf.keras.layers.Dropout(0.2)(x)

#     x = inceptionV4_block(x, 192, 96, 208, 16, 48, 64)
#     x = tf.keras.layers.Dropout(0.2)(x)

#     output1 = tf.keras.layers.AvgPool2D((5, 5), strides=(3,3), padding='same')(x)
#     output1 = tf.keras.layers.Conv2D(128, 1, activation='relu')(output1)
#     output1 = tf.keras.layers.BatchNormalization()(output1)
#     output1 = tf.keras.layers.Flatten()(output1)
#     output1 = tf.keras.layers.Dense(1024, activation='relu')(output1)
#     output1 = tf.keras.layers.Dropout(0.7)(output1)
#     output1 = tf.keras.layers.Dense(nb_classes, activation='softmax')(output1)

#     x = inceptionV4_block(x, 160, 112, 224, 24, 64, 64)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = inceptionV4_block(x, 128, 128, 256, 24, 64, 64)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = inceptionV4_block(x, 112, 144, 288, 32, 64, 64)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     output2 = tf.keras.layers.AvgPool2D((5, 5), strides=(3, 3), padding='same')(x)
#     output2 = tf.keras.layers.Conv2D(128, 1, activation='relu')(output2)
#     output2 = tf.keras.layers.BatchNormalization()(output2)
#     output2 = tf.keras.layers.Flatten()(output2)
#     output2 = tf.keras.layers.Dense(1024, activation='relu')(output2)
#     output2 = tf.keras.layers.Dropout(0.7)(output2)
#     output2 = tf.keras.layers.Dense(nb_classes, activation='softmax')(output2)

#     x = inceptionV4_block(x, 256, 160, 320, 32, 128, 128)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)

#     x = inceptionV4_block(x, 256, 160, 320, 32, 128, 128)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = inceptionV4_block(x, 384, 192, 384, 48, 128, 128)
#     x = tf.keras.layers.GlobalAveragePooling2D()(x)

#     x = tf.keras.layers.Dropout(0.4)(x)

#     outputs = tf.keras.layers.Dense(nb_classes, activation='softmax')(x)


#     outputs = tf.keras.layers.concatenate([outputs, output1, output2])
#     outputs = tf.keras.layers.Dense(nb_classes, activation='softmax')(outputs)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model

# def multimodel(model_names, nb_classes=3, image_dimensions=(384, 288, 3), weighted_output=True):

#     same_base = True
#     check_vars = model_names[0].split('_')[1:]
#     types = []
#     for model_name in model_names:
#         vars = model_name.split('_')
#         types.append(vars[0])
#         vars = vars[1:]
#         if check_vars != vars:
#             same_base = False


#     model = tf.keras.models.load_model(os.path.join('../data/snapshots/', model_names[0] + '_best.hdf5'))
#     if same_base:
#         for i, layer in enumerate(model.layers):
#             if layer.trainable:
#                 layer._name = types[0] + "_" + layer._name
#     outputs = [model.layers[-1].output]
#     i = 1
#     for model_name in model_names[1:]:
#         extra_model = tf.keras.models.load_model(os.path.join('../data/snapshots/', model_name + '_best.hdf5'))
#         extra_model._layers.pop(0)
#         for layer in extra_model.layers:
#             layer._name = types[i] + "_" + layer._name


#         i += 1

#         outputs.append(extra_model(model.inputs))

#     output = tf.keras.layers.concatenate(outputs)
#     output = tf.keras.layers.Dense(nb_classes)(output)
#     new_model = tf.keras.Model(inputs=model.inputs, outputs=output)
#     new_model.summary()
#     return new_model


# def multimodel_same_base(model_names, nb_classes=3, image_dimensions=(384, 288, 3), weighted_output=True):

#     same_base = True
#     check_vars = model_names[0].split('_')[1:]
#     types = []
#     for model_name in model_names:
#         vars = model_name.split('_')
#         types.append(vars[0])
#         vars = vars[1:]
#         if check_vars != vars:
#             same_base = False

#     assert same_base, "Not the same base"

#     model = tf.keras.models.load_model(os.path.join('data/snapshots/', model_names[0] + '_best.hdf5'))
#     output_layer_idx = 0
#     models = [model]
#     config = model.get_config()

#     old_to_new = {}
#     new_to_old = {model_names[0]: {}}
#     last_non_trainable = ''
#     new_name = ''
#     model.summary()

#     for layer in config['layers']:
#         if 'trainable' in layer['config'].keys() and layer['config']['trainable']:
#             # print(layer)
#             new_name = types[0] + '_' + layer['name']
#             layer['config']['trainable'] = False
#         else:
#             new_name = layer['name']
#             last_non_trainable = new_name

#         old_to_new[layer['name']], new_to_old[model_names[0]][new_name] = new_name, layer['name']
#         layer['name'] = new_name
#         layer['config']['name'] = new_name


#         if len(layer['inbound_nodes']) > 0:
#             for in_node in layer['inbound_nodes'][0]:
#                 if type(in_node) == list:
#                     in_node[0] = old_to_new[in_node[0]]

#     final_layers = [new_name]
#     for i, model_name in enumerate(model_names[1:]):
#         extra_model = tf.keras.models.load_model(os.path.join('data/snapshots/', model_name + '_best.hdf5'))
#         extra_model.summary()
#         models.append(extra_model)
#         new_to_old[model_name] = {}
#         old_to_new[model_name] = {}
#         extra_config = extra_model.get_config()
#         new_name = ''
#         for layer in extra_config['layers']:
#             if 'trainable' in layer['config'].keys() and layer['config']['trainable']:
#                 new_name = types[i+1] + '_' + layer['name']
#                 layer['config']['trainable'] = False

#                 old_to_new[model_name][layer['name']], new_to_old[model_name][new_name] = new_name, layer['name']
#                 layer['name'] = new_name
#                 layer['config']['name'] = new_name

#                 if len(layer['inbound_nodes']) > 0:
#                     for in_node in layer['inbound_nodes'][0]:
#                         if in_node[0] not in old_to_new[model_name].keys():
#                             if 'model' in in_node[0]:
#                                 in_node[0] = old_to_new['model_6']
#                             else:
#                                 in_node[0] = old_to_new[in_node[0]]
#                         else:
#                             in_node[0] = old_to_new[model_name][in_node[0]]
#                 config['layers'].append(layer)

#         final_layers.append(new_name)


#     config['layers'].append({'class_name': 'Concatenate',
#                              'config': {'name': 'concatenate_final', 'trainable': True, 'dtype': 'float32', 'axis': -1},
#                              'name': 'concatenate_final',
#                              'inbound_nodes': [[[node, 0, 0, {}] for node in final_layers]]})
#     config['layers'].append({'class_name': 'Dense',
#                              'config': {'name': 'dense_out', 'trainable': True, 'dtype': 'float32', 'units': nb_classes,
#                                         'activation': 'softmax', 'use_bias': True,
#                                         'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}},
#                                         'bias_initializer': {'class_name': 'Zeros', 'config': {}},
#                                         'kernel_regularizer': None, 'bias_regularizer': None,
#                                         'activity_regularizer': None, 'kernel_constraint': None,
#                                         'bias_constraint': None},
#                              'name': 'dense_out',
#                              'inbound_nodes': [[['concatenate_final', 0, 0, {}]]]})

#     for input_layer in config['input_layers']:
#         input_layer[0] = old_to_new[input_layer[0]]

#     config['output_layers'][0][0] = 'dense_out'

#     new_model = tf.keras.Model().from_config(config)

#     for layer in new_model.layers[:-2]:
#         t = layer.name.split('_')[0]
#         if t in types:
#             idx = types.index(t)
#         else:
#             idx = 0
#         layer.set_weights(models[idx].get_layer(new_to_old[model_names[idx]][layer.name]).get_weights())
#     return new_model


# def multimodel_same_base_weighted_output(model_names, nb_classes=3, image_dimensions=(384, 288, 3), weighted_output=True, class_labels=None):

#     possible_weights = {
#         'hypVSadn': [[1., 0., 0.], [0., 1., 0.]],
#         'hypVSssp': [[1., 0., 0.], [0., 0., 1.]],
#         'adnVSssp': [[0., 1., 0.], [0., 0., 1.]],
#         'hypVSall': [[1., 0., 0.], [0., 0.5, 0.5]],
#         'adnVSall': [[0., 1., 0.], [0.5, 0., 0.5]],
#         'sspVSall': [[0., 0., 1.], [0.5, 0.5, 0.]],
#     }


#     same_base = True
#     check_vars = model_names[0].split('_')[1:]
#     types = []
#     for model_name in model_names:
#         vars = model_name.split('_')
#         types.append(vars[0])
#         vars = vars[1:]
#         if check_vars != vars:
#             same_base = False

#     assert same_base, "Not the same base"

#     model = tf.keras.models.load_model(os.path.join('data/snapshots/', model_names[0] + '_best.hdf5'))
#     output_layer_idx = 0
#     models = [model]
#     config = model.get_config()

#     old_to_new = {}
#     new_to_old = {model_names[0]: {}}
#     last_non_trainable = ''
#     new_name = ''
#     model.summary()

#     for layer in config['layers']:
#         if 'trainable' in layer['config'].keys() and layer['config']['trainable']:
#             # print(layer)
#             new_name = types[0] + '_' + layer['name']
#             layer['config']['trainable'] = False
#         else:
#             new_name = layer['name']
#             last_non_trainable = new_name

#         old_to_new[layer['name']], new_to_old[model_names[0]][new_name] = new_name, layer['name']
#         layer['name'] = new_name
#         layer['config']['name'] = new_name


#         if len(layer['inbound_nodes']) > 0:
#             for in_node in layer['inbound_nodes'][0]:
#                 if type(in_node) == list:
#                     in_node[0] = old_to_new[in_node[0]]

#     config['layers'].append({'class_name': 'Reshape',
#                              'config': {'name': types[0] +'_reshape',
#                                         'trainable': True,
#                                         'dtype': 'float32',
#                                         'target_shape': (1, 2)},
#                              'name': types[0] +'_reshape',
#                              'inbound_nodes': [[[new_name, 1, 0, {}]]]})
#     config['layers'].append({'class_name': 'Dot',
#                              'config': {'name': types[0] +'_dot',
#                                         'trainable': True,
#                                         'dtype': 'float32',
#                                         'axes': (1, 2),
#                                         'normalize': False},
#                              'name': types[0] +'_dot',
#                              'inbound_nodes': [[['_CONSTANT_VALUE', -1, possible_weights[types[0]], {}],
#                                                 [types[0] +'_reshape', 0, 0, {}]]]})
#     config['layers'].append({'class_name': 'Reshape',
#                              'config': {'name': types[0] +'_reshape_1',
#                                         'trainable': True,
#                                         'dtype': 'float32',
#                                         'target_shape': (3,)},
#                              'name': types[0] +'_reshape_1',
#                              'inbound_nodes': [[[types[0] +'_dot', 0, 0, {}]]]})

#     final_layers = [types[0] +'_reshape_1']
#     print(possible_weights[types[0]])

#     for i, model_name in enumerate(model_names[1:]):
#         extra_model = tf.keras.models.load_model(os.path.join('data/snapshots/', model_name + '_best.hdf5'))
#         extra_model.summary()
#         models.append(extra_model)
#         new_to_old[model_name] = {}
#         old_to_new[model_name] = {}
#         extra_config = extra_model.get_config()
#         new_name = ''
#         for layer in extra_config['layers']:
#             if 'trainable' in layer['config'].keys() and layer['config']['trainable']:
#                 new_name = types[i+1] + '_' + layer['name']
#                 layer['config']['trainable'] = False

#                 old_to_new[model_name][layer['name']], new_to_old[model_name][new_name] = new_name, layer['name']
#                 layer['name'] = new_name
#                 layer['config']['name'] = new_name

#                 if len(layer['inbound_nodes']) > 0:
#                     for in_node in layer['inbound_nodes'][0]:
#                         if in_node[0] not in old_to_new[model_name].keys():
#                             if 'model' in in_node[0]:
#                                 in_node[0] = old_to_new['model_6']
#                             else:
#                                 in_node[0] = old_to_new[in_node[0]]
#                         else:
#                             in_node[0] = old_to_new[model_name][in_node[0]]
#                 config['layers'].append(layer)

#         config['layers'].append({'class_name': 'Reshape',
#                                  'config': {'name': types[i] + '_reshape',
#                                             'trainable': True,
#                                             'dtype': 'float32',
#                                             'target_shape': (1, 2)},
#                                  'name': types[i] + '_reshape',
#                                  'inbound_nodes': [[[new_name, 1, 0, {}]]]})
#         config['layers'].append({'class_name': 'Dot',
#                                  'config': {'name': types[i] + '_dot',
#                                             'trainable': True,
#                                             'dtype': 'float32',
#                                             'axes': (1, 2),
#                                             'normalize': False},
#                                  'name': types[i] + '_dot',
#                                  'inbound_nodes': [[['_CONSTANT_VALUE', -1, possible_weights[types[i]], {}],
#                                                     [types[i] + '_reshape', 0, 0, {}]]]})
#         print(possible_weights[types[i]])
#         config['layers'].append({'class_name': 'Reshape',
#                                  'config': {'name': types[i] + '_reshape_1',
#                                             'trainable': True,
#                                             'dtype': 'float32',
#                                             'target_shape': (3,)},
#                                  'name': types[i] + '_reshape_1',
#                                  'inbound_nodes': [[[types[i] + '_dot', 0, 0, {}]]]})
#         final_layers.append(types[i] + '_reshape_1')


#     config['layers'].append({'class_name': 'Concatenate',
#                              'config': {'name': 'avg_final', 'trainable': True, 'dtype': 'float32', 'axis': -1},
#                              'name': 'avg_final',
#                              'inbound_nodes': [[[node, 0, 0, {}] for node in final_layers]]})

#     for input_layer in config['input_layers']:
#         input_layer[0] = old_to_new[input_layer[0]]

#     config['output_layers'][0][0] = 'avg_final'

#     new_model = tf.keras.Model().from_config(config)

#     for layer in new_model.layers:
#         t = layer.name.split('_')[0]
#         if t in types:
#             idx = types.index(t)
#         else:
#             idx = 0
#         ns = new_to_old[model_names[idx]]
#         print(ns)
#         if layer.name in ns.keys():
#             layer.set_weights(models[idx].get_layer(ns[layer.name]).get_weights())
#     return new_model


def random_crop_sampling_branch(x, crop_size=(112,112), nb_samples=1, nb_classes=2, conv_dropout=0., fc_dropout=0.):

    # if nb_samples > 1:
    #     x = tf.keras.layers.concatenate([tf.keras.layers.experimental.preprocessing.RandomCrop(crop_size[0], crop_size[1])(x) for _ in range(nb_samples)])
    # else:
    x = tf.keras.layers.experimental.preprocessing.RandomCrop(crop_size[0], crop_size[1])(x)

    base_model = input_block_vgg16(nb_classes=nb_classes, image_dimensions=crop_size + (3,))
    base_model.trainable = False

    x = base_model(x)


    return x


def random_crop_model(image_dimensions=(384, 288, 3), nb_classes=2, nb_branches=5, conv_dropout=0., fc_dropout=0., nb_samples=1, crop_size=(64,64), backbone='vgg'):
    input = tf.keras.layers.Input(image_dimensions)

    x_list = [tf.keras.layers.experimental.preprocessing.RandomCrop(crop_size[0], crop_size[1])(input, training=True) for i in range(nb_samples)]

    if backbone == 'mobilenet':
        assert crop_size[0] in [96, 128, 160, 192, 224], "If not in [96, 128, 160, 192, 224], then mobilenet weights don't exist"
        base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', input_shape=crop_size+(3,), pooling='avg')
    else:
        base_model = input_block_vgg16(nb_classes=nb_classes, image_dimensions=crop_size + (3,))
    base_model.trainable = False

    output = []
    for x in x_list:
        outputs = base_model(x)
        outputs = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if conv_dropout > 0. else outputs
        outputs = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform',
                                        kernel_regularizer=l2(5e-2))(outputs)
        outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs

        outputs = tf.keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='he_uniform',
                                        kernel_regularizer=l2(5e-2))(outputs)

        output.append(outputs)

    outputs = tf.keras.layers.average(output)

    model = tf.keras.Model(inputs=input, outputs=outputs)

    return model



def segment_model(image_dimensions=(384, 288, 3), nb_classes=2, segments=(2,2), conv_dropout=0., fc_dropout=0.):

    input = tf.keras.layers.Input(image_dimensions)

    x_step, y_step = image_dimensions[0] // segments[0], image_dimensions[1] // segments[1]
    assert x_step >=32 and y_step > 32, "The amount of segments is too large to train succesfully"

    rest_x, rest_y = int((image_dimensions[0] % segments[0])/2), int((image_dimensions[1] % segments[1])/2)

    base_model = input_block_vgg16(nb_classes=nb_classes, image_dimensions=(x_step, y_step, 3))
    base_model.trainable = False

    x_list = []
    for i in range(segments[0]):
        for j in range(segments[1]):
            x_list.append(base_model(input[:,rest_x+ i*x_step:rest_x+(i+1)*x_step,rest_y+ j*y_step:rest_y+(j+1)*y_step, :]))
    x = tf.keras.layers.average(x_list)

    outputs = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if conv_dropout > 0. else outputs
    outputs = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform',
                                    kernel_regularizer=l2(5e-2))(outputs)
    outputs = tf.keras.layers.Dropout(fc_dropout)(outputs) if fc_dropout > 0. else outputs

    outputs = tf.keras.layers.Dense(2, activation='softmax', kernel_initializer='he_uniform',
                                    kernel_regularizer=l2(5e-2))(outputs)



    model = tf.keras.Model(inputs=input, outputs=outputs)
    return model


def vote_net_same_base(model_names, nb_classes=3, image_dimensions=(384, 288, 3), output_weights=(1.0,1.0,1.0), labels=None, gather_type='avg', voting_th=0., num_model_types=2):
    ignored_layers = ['tf.__operators__.getitem', 'tf.__operators__.getitem_1','tf.__operators__.getitem_2','tf.__operators__.getitem_3','tf.__operators__.getitem_4', 'tf.nn.bias_add', 'tf.math.truediv', 'tf.math.subtract']

    same_base = True
    check_vars = model_names[0].split('_')[2:-2]
    types = []
    for model_name in model_names:
        vars = model_name.split('_')
        types.append(vars[1] + "_" + vars[-2])
        vars = vars[2:-2]
        if check_vars != vars:
            same_base = False

    assert same_base, "Not the same base"

    model = tf.keras.models.load_model(os.path.join('data/snapshots/', model_names[0]), compile=False)
    # model = tf.keras.models.load_model(os.path.join('data/snapshots/', model_names[0] + '_final'), compile=False)
    output_layer_idx = 0
    models = [model]
    config = model.get_config()

    old_to_new = {}
    new_to_old = {model_names[0]: {}}
    last_non_trainable = ''
    new_name = ''
    model.summary()
    first_model_name = model_names[0]

    for layer in config['layers']:
        if 'trainable' in layer['config'].keys() and layer['config']['trainable'] and layer['config']['name'] not in ignored_layers:
            # print(layer)
            new_name = types[0] + '_' + layer['name']
            layer['config']['trainable'] = False
        else:
            new_name = layer['name']
            last_non_trainable = new_name

        old_to_new[layer['name']], new_to_old[model_names[0]][new_name] = new_name, layer['name']
        layer['name'] = new_name
        layer['config']['name'] = new_name


        if len(layer['inbound_nodes']) > 0:
            for in_node in layer['inbound_nodes'][0]:
                if type(in_node) == list:
                    in_node[0] = old_to_new[in_node[0]]
    # print(last_non_trainable)
    final_layers = [new_name]
    for i, model_name in enumerate(model_names[1:]):
        print(model_name)
        extra_model = tf.keras.models.load_model(os.path.join('data/snapshots/', model_name), compile=False)
        # extra_model = tf.keras.models.load_model(os.path.join('data/snapshots/', model_name + '_final'), compile=False)
        # extra_model.summary()
        models.append(extra_model)
        new_to_old[model_name] = {}
        old_to_new[model_name] = {}
        extra_config = extra_model.get_config()
        new_name = ''
        for layer in extra_config['layers']:
            if 'trainable' in layer['config'].keys() and layer['config']['trainable'] and layer['config']['name'] not in ignored_layers:
                new_name = types[i+1] + '_' + layer['name']
                layer['config']['trainable'] = False

                old_to_new[model_name][layer['name']], new_to_old[model_name][new_name] = new_name, layer['name']
                layer['name'] = new_name
                layer['config']['name'] = new_name

                if len(layer['inbound_nodes']) > 0:
                    for in_node in layer['inbound_nodes'][0]:
                        # print(in_node[0])
                        # print(old_to_new)
                        if in_node[0] not in old_to_new[model_name].keys():
                            # print("here")
                            in_node[0] = old_to_new[in_node[0]]
                        else:
                            in_node[0] = old_to_new[model_name][in_node[0]]
                config['layers'].append(layer)

        final_layers.append(new_name)


    config['layers'].append({'class_name': 'Concatenate',
                             'config': {'name': 'concatenate_final', 'trainable': False, 'dtype': 'float32', 'axis': -1},
                             'name': 'concatenate_final',
                             'inbound_nodes': [[[node, 0, 0, {}] for node in final_layers]]})

    for input_layer in config['input_layers']:
        input_layer[0] = old_to_new[input_layer[0]]

    config['output_layers'][0][0] = 'concatenate_final'

    new_model = tf.keras.Model().from_config(config)
    # print([i.name for i in new_model.layers])
    # print(new_to_old)
    for layer in new_model.layers[:-1]:
        t = "_".join(layer.name.split('_')[:2])
        if t in types:
            idx = types.index(t)
        else:
            idx = 0
        layer.set_weights(models[idx].get_layer(new_to_old[model_names[idx]][layer.name]).get_weights())

    new_model.summary()

    input = tf.keras.Input(image_dimensions)
    x = new_model(input, training=False)
    cst = tf.tile(tf.reshape(tf.constant([0.,]), (-1, 1)), [tf.shape(x)[0], 1])
    x = tf.keras.layers.concatenate([cst, x], axis=-1)
    # x = tf.keras.layers.Reshape((-1, 1))(x)
    # x = tf.keras.layers.ZeroPadding1D(padding=(1, 0))(x)
    # x = tf.keras.layers.Flatten()(x)

    weigths = tf.tile(tf.reshape(tf.constant(output_weights), (-1, len(output_weights), 1)), [tf.shape(x)[0], 1, len(model_names)])

    indices = np.zeros((nb_classes,len(model_names)), dtype=int)
    for i, label in enumerate(labels):
        for j, cls in enumerate(label):
            if len(cls) == 1:
                indices[int(cls), i] = int(j+i*2)+1
            else:
                for c in cls:
                    indices[int(c), i] = int(j + i * 2) + 1
    print(indices)
    outputs = tf.gather(x, indices, axis=-1)
    

    if gather_type == 'sum':
        outputs = tf.math.reduce_sum(outputs, axis=-1)
    elif gather_type == 'avg':
        outputs = tf.math.reduce_mean(outputs, axis=-1)
    elif gather_type == 'logsumexp':
        outputs = tf.math.reduce_logsumexp(outputs, axis=-1)
    elif gather_type == 'eucl_norm':
        outputs = tf.math.reduce_euclidean_norm(outputs, axis=-1)
    elif gather_type == 'max':
        outputs = tf.math.reduce_max(outputs, axis=-1)
    elif gather_type == 'vote':
        if voting_th > 0 and voting_th < 0.5:
            outputs -= voting_th
        outputs = tf.math.round(outputs)
        outputs = tf.math.multiply(outputs, weigths)
        outputs = tf.math.reduce_sum(outputs, axis=-1)
    elif gather_type == 'sum_2':
        if voting_th > 0 and voting_th < 0.5:
            outputs -= voting_th
        outputs = tf.math.round(outputs)
        total_types = int(len(labels) / num_model_types)
        outputs = tf.keras.layers.concatenate([tf.expand_dims(tf.nn.softmax(tf.math.reduce_sum(outputs[:,:, i*total_types:(i+1)*total_types], axis=-1)), axis=-1) for i in range(num_model_types)])
        outputs = tf.math.reduce_sum(outputs, axis=-1)
    elif gather_type == 'vote_2':
        if voting_th > 0 and voting_th < 0.5:
            outputs -= voting_th
        outputs = tf.math.round(outputs)
        total_types = int(len(labels) / num_model_types)
        outputs = tf.keras.layers.concatenate([tf.expand_dims(tf.math.round(tf.nn.softmax(tf.math.reduce_sum(tf.math.round(outputs[:,:, i*total_types:(i+1)*total_types]), axis=-1))), axis=-1) for i in range(num_model_types)])
        outputs = tf.math.reduce_sum(outputs, axis=-1)
    else:
        outputs = tf.math.reduce_mean(outputs, axis=-1)

    outputs = tf.nn.softmax(outputs)
    final_model = tf.keras.Model(inputs = input, outputs=outputs)
    final_model.summary()
    return final_model




# -


# if __name__ == '__main__':
#     # Definindo parâmetros para o modelo ResNet
#     nb_classes = 3
#     image_dimensions = (256, 256, 3)  # Exemplo de entrada para o modelo
#     fc_dropout = 0.5
#     resnet_layers = 50  # Usando ResNet50 por padrão

#     # Criar o modelo ResNet
#     model = resnet(nb_classes, image_dimensions, fc_dropout=fc_dropout, resnet_layers=resnet_layers)
#     model.summary()  # Exibir o resumo do modelo

if __name__ == '__main__':
    # Define model parameters
    nb_classes = 2  # Number of classes
    nf = 64  # Number of filters
    image_dimensions = (384, 288, 3)  # Image dimensions
    fc_dropout = 0.5  # Dropout in the fully connected layer
    multi_output = False  # Whether the model has multiple outputs
    reg_constant = 0.01  # Regularization strength
    nb_inputs = 1  # Number of inputs
    fold = '0'  # Fold identifier (if applicable)
    b = "0"  # EfficientNet version ('0' for B0, '7' for B7, etc.)
    frozen_layers = 1  # Number of layers to freeze

    # Instantiate the model
    model = efficientnet(
        nb_classes=nb_classes, nf=nf, image_dimensions=image_dimensions,
        fc_dropout=fc_dropout, multi_output=multi_output,
        reg_constant=reg_constant, nb_inputs=nb_inputs,
        fold=fold, b=b, frozen_layers=frozen_layers
    )

    # Print model summary
    model.summary()

# +

# if __name__ == '__main__':
#     names = ['adnVSall_HDall2022_efficientnet_regularized0.0_256x256_1in_nf128_bnTrue_fcdo0.5_convdo0.0_lossls0.5_lossDropout1.0_shuffle_dsc09_reflect_5fold0',
#              'hypVSall_HDall2022_efficientnet_regularized0.0_256x256_1in_nf128_bnTrue_fcdo0.5_convdo0.0_lossls0.5_lossDropout1.0_shuffle_dsc09_reflect_5fold0']

#     labels = [('1', ('0', '2')), ('0', ('1', '2'))]
#     # b = tf.constant([[[1.,2.]]])
#     #
#     # a = tf.constant([[[1., 0., 0.], [0., 1., 0.]]])
#     # print(tf.keras.layers.Reshape((3,))(tf.keras.layers.Dot(axes=(1,2))([a,b])))
#     model = vote_net_same_base(names, 3, (256,256,3), labels=labels)
#     model.summary()
#     # model = mobilenet(2, image_dimensions=(256, 256, 3), nf=32)
#     # model.summary()
# -


