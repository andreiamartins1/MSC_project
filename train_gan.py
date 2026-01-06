#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:02:15 2018

@author: teelbo0
"""
import os
import random
import time
import shutil
from utils.Dataset import Dataset
from tqdm import tqdm, trange
import numpy as np
from joblib import dump, load

from sklearn.svm import SVC

import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from utils.networks import MIclass, resnet, vgg16, efficientnet, detnet, wavelet_network, transfer, darknet, googlenet, random_crop_model, mobilenet, segment_model, mobilenet_full, basic_cnn, vgg19, simple_linear, xception, efficientnet_full
# from utils.callbacks import ConfusionMatrixLogger
from utils.losses import categorical_accuracy_label_smoothing

from tensorboard.plugins.hparams import api as hp



scope = 'manual'
assert scope in ['all', 'manual']

def main(logdir, name, hparams, train_imageset='data/imagesets_1080p/{}_split_0.7.txt'.format(scope), val_imageset='data/imagesets_1080p/{}_split_0.1.txt'.format(scope)):
    image_size = (int(1920),int(1080))
    target_size = (int(hparams['target_width']), int(hparams['target_height']))
    loading_size = target_size if hparams['do_cropping'] == 'preloaded' else image_size

    if hparams["selected_labels"] is not None:
        assert len(hparams["selected_labels"]) == hparams["nb_classes"], "The nb_classes should be the same as the amount of selected labels"

    # make backup of training script
    source = __file__
    destination = 'logs/' + name + '.txt'
    shutil.copyfile(source, destination)
    print("Training script backed up in " + destination)

    input = tf.keras.layers.Input(target_size + (3,))

    augmentations = {
        "flip_horizontal": True,
        "flip_vertical": True,
        "cutout": 10,
        "shear": 0.2,
        "rotate": 90.,
        "translate": 20,  # max number of pixels
        "color": 0.2,
        "blur": 1.3
    }

    # ds_train = Dataset(train_imageset, batch_size=hparams['batch_size'], target_size=target_size, loading_size=target_size, augmentations=augmentations, nb_inputs=hparams['nb_inputs'], balanced=hparams['balanced'], do_cropping='preloaded', balance_polypids=True, selected_labels=hparams['selected_labels'])
    # ds_val = Dataset(val_imageset, batch_size=hparams['batch_size'], target_size=target_size, loading_size=target_size, augmentations=None, nb_inputs=hparams['nb_inputs'], shuffle=True, repeat=True, balanced=hparams['balanced'], do_cropping='preloaded', balance_polypids=True, selected_labels=hparams['selected_labels'])

    ds_train = Dataset(train_imageset, batch_size=hparams['batch_size'], target_size=target_size,
                       loading_size=loading_size, augmentations=None, nb_inputs=hparams['nb_inputs'],
                       balanced=hparams['balanced'], do_cropping=True, balance_polypids=True,
                       selected_labels=hparams['selected_labels'])
    ds_val = Dataset(val_imageset, batch_size=hparams['batch_size'], target_size=target_size, loading_size=loading_size,
                     augmentations=None, nb_inputs=hparams['nb_inputs'], shuffle=True, repeat=True,
                     balanced=hparams['balanced'], do_cropping=True, balance_polypids=True,
                     selected_labels=hparams['selected_labels'])
    if hparams['network'] == 'resnet':
        model = resnet(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,))
    elif hparams['network'] == 'cnn':
        model = basic_cnn(nb_classes=hparams['nb_classes'], nf=hparams['nf'], image_dimensions=target_size + (3,), fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'])
    elif hparams['network'] == 'miclass':
        model = MIclass(nb_inputs=hparams['nb_inputs'], nf=hparams['nf'], nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,), bn=hparams['batch_normalization'], fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'], multi_output=hparams['multi_output'])
    elif hparams['network'] == 'vgg19':
        preprocessed = tf.keras.applications.vgg19.preprocess_input(input)
        x = tf.keras.applications.vgg19.VGG19(include_top=False, pooling='avg')(preprocessed)
        model = tf.keras.Model(inputs=input, outputs=x)
    elif hparams['network'] == 'linear':
        model = simple_linear(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,), fc_dropout=hparams['fc_dropout'], nf=hparams['nf'])
    elif hparams['network'] == 'vgg16':
        preprocessed = tf.keras.applications.vgg16.preprocess_input(input)
        x = tf.keras.applications.vgg16.VGG16(include_top=False, pooling='avg')(preprocessed)
        model = tf.keras.Model(inputs=input, outputs=x)
    elif hparams['network'] == 'xception':
        model = xception(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,), fc_dropout=hparams['fc_dropout'], nf=hparams['nf'])
    elif hparams['network'] == 'mobilenet':
        model = mobilenet(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,), fc_dropout=hparams['fc_dropout'], alpha=hparams['alpha'], nf=hparams['nf'])
    elif hparams['network'] == 'mobilenetFull':
        model = mobilenet_full(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,), fc_dropout=hparams['fc_dropout'], alpha=hparams['alpha'])
    elif hparams['network'] == 'efficientnet':
        model = efficientnet(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,),
                          fc_dropout=hparams['fc_dropout'], nf=hparams['nf'], multi_output=hparams['multi_output'])
    elif hparams['network'] == 'efficientnetFull':
        model = efficientnet_full(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,),
                          fc_dropout=hparams['fc_dropout'], b=hparams['b'], nf=hparams['nf'])
    elif hparams['network'] == 'detnet':
        model = detnet(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,),
                          fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'], nf=hparams['nf'])
    elif hparams['network'] == 'wavelet':
        model = wavelet_network(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,),
                          bn=hparams['batch_normalization'], conv_dropout=hparams['fc_dropout'], nf=hparams['nf'])
    elif hparams['network'] == 'transfer':
        model = transfer('hypVSadn_HDall_detnet_256x256_miclass_1in_nf32_bnTrue_fcdo0.5_convdo0.0_lossls0.1', nb_classes=hparams['nb_classes'],image_dimensions=target_size + (3,))
    elif hparams['network'] == 'darknet':
        model = darknet(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,), fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'])
    elif hparams['network'] == 'googlenet':
        model = googlenet(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,))
    elif hparams['network'] == 'randomCrop':
        model = random_crop_model(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,),
                                  fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'],
                                  nb_branches=hparams['nb_branches'], nb_samples=hparams['nb_samples'],
                                  backbone=hparams['backbone'], crop_size=hparams['crop_size'])
    elif hparams['network'] == 'segment':
        model = segment_model(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,),
                                  fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'])
    elif hparams['network'] == 'reuse':
        pass
    else:
            raise ValueError("hparams['network'] = {} is not defined".format(hparams['network']))
    samples = ds_train.tfdataset
    results = None
    labels = None
    total_len = len(ds_train)// hparams['batch_size']

    # print(feat_cols)
    iterator = iter(samples)
    for _ in trange(total_len):
        sample, label = next(iterator)
        result = np.array(model.predict(sample))

        label = np.array(label)
        label = label * 2
        label = label - 1

        if results is None:
            results = result
            labels = label
        else:
            results = np.vstack([results, result])
            labels = np.vstack([labels, label])

    start_time = time.time()
    clf = SVC(kernel='rbf',gamma='auto', verbose=True, probability=True)
    clf.fit(results, labels[:, 1])

    print('Total time: {}'.format(time.time()-start_time))
    dump(clf, 'data/svc2.joblib')
    clf = load('data/svc2.joblib')


    samples = ds_val.tfdataset
    results = None
    labels = None
    probabilities = None

    start_time_total = time.time()
    total_cnn_time = 0
    total_svm_time = 0
    # print(feat_cols)
    total_len = len(ds_val) // hparams['batch_size']
    iterator = iter(samples)
    for _ in trange(total_len):
        sample, label = next(iterator)
        start_time = time.time()
        result = np.array(model.predict(sample))
        total_cnn_time += time.time() - start_time
        start_time = time.time()
        prediction = clf.predict(result)
        total_svm_time += time.time() - start_time
        probability = clf.predict_proba(result)



        # print('Miss-rate: {}'.format(np.count_nonzero(prediction - label[:, 1]) / hparams['batch_size']))

        if results is None:
            results = prediction
            labels = label
            probabilities = probability
        else:
            results = np.hstack([results, prediction])
            labels = np.vstack([labels, label])
            probabilities = np.vstack([probabilities, probability])

    labels = labels * 2
    labels = labels - 1

    dump(results, 'data/predictions_svc2_balanced.joblib')
    dump(labels, 'data/labels_svc2_balanced.joblib')
    dump(probabilities, 'data/probabilities_svc2_balanced.joblib')

    total_time = time.time() - start_time_total
    difference = labels[:, 1] - results
    print('Accuracy: {}'.format(1 - np.count_nonzero(difference) / (total_len * hparams['batch_size'])))
    print('Total time: {}, with {}% in model prediction and {}% in svm for batch size {}'.format(total_time, 100* total_cnn_time/total_time, 100* total_svm_time/total_time, hparams['batch_size']))


if __name__ == '__main__':
    hparams = {
            'learning_rate': 1e-3,
            'target_width': 256,
            'target_height': 256,
            'nb_inputs': 1,
            'nb_classes': 2,
            'batch_normalization': True,
            'conv_dropout': 0.,
            'fc_dropout': 0.5,
            'batch_size': 64,
            'balanced': True,
            'network': 'vgg16',
            'do_cropping': True,
            'nf': 512,
            'label_smoothing': 0.1,
            'selected_labels': ["0","1"],
            'alpha': 0.5,
            'multi_output': False,
            'b': '0'
        }
    # name = 'hypVSadn_HD{scope}_{network}_{target_width}x{target_height}_miclass_{nb_inputs}in_nf{nf}_bn{batch_normalization}_fcdo{fc_dropout}_convdo{conv_dropout}_lossls{label_smoothing}_avgfusion_l25e-2'.format(scope=scope, network=hparams['network'], target_width=hparams['target_width'], target_height=hparams['target_height'], nb_inputs=hparams['nb_inputs'], nf=hparams['nf'], batch_normalization=hparams['batch_normalization'], fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'], label_smoothing=hparams['label_smoothing'])
    # main('logs/', name, hparams)
    s = ''

    first = True
    for i in hparams['selected_labels']:
        if not first:
            s += 'VS'

        if len(i) > 1:
            s += 'all'
        else:
            if i == '0':
                s += 'hyp'
            elif i == '1':
                s += 'adn'
            elif i == '2':
                s += 'ssp'
        first = False

    tf.keras.backend.clear_session()
    name = s + '_HD{scope}2022_{network}_svm'.format(scope=scope, network=hparams['network'])
    # with k as range(0,1):
    k = 0#random.randint(0, 6)
    train_imageset = 'data/imagesets_1080p/5bootstrap_{}/{}k_train.txt'.format(scope, k)
    val_imageset = 'data/imagesets_1080p/5bootstrap_{}/{}k_val.txt'.format(scope, k)
    # train_imageset = 'data/imagesets_1080p/manual_split_0.7.txt'.format(scope, k)
    # val_imageset = 'data/imagesets_1080p/manual_split_0.1.txt'.format(scope, k)
    main('logs/', name, hparams, train_imageset, val_imageset)
