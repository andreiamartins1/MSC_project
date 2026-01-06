#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:02:15 2018

@author: teelbo0
"""
import os
import random
import shutil
from utils.Dataset import Dataset
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from utils.networks import vgg16
# from utils.callbacks import ConfusionMatrixLogger
from utils.losses import categorical_accuracy_label_smoothing, trimmed_loss, DropoutCategoricalCrossEntropy, balanced_categorical_crossentropy, BalancedCategoricalAccuracy, FocalLoss

import datetime
my_date = datetime.date.today() # if date is 01/01/2018
year, week_num, day_of_week = my_date.isocalendar()
logdir = f'logs/{year}_week{week_num}/'
os.makedirs(logdir, exist_ok=True)

scope = 'all'
assert scope in ['all', 'manual']

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
def main(logdir, name, hparams, train_imageset, val_imageset):
    # image_size = (int(1920),int(1080))
    image_size = (int(1080),int(1920))
    target_size = (int(hparams['target_width']), int(hparams['target_height']))
    loading_size = target_size if hparams['do_cropping'] == 'preloaded' else image_size
    #hparams["selected_labels"] = [tuple(i) if len(i)>1 else i for i in hparams["selected_labels"].split("vs") ]
    hparams["selected_labels"] = [tuple(i) if isinstance(i, str) and len(i) > 1 else i for i in hparams["selected_labels"]]
    # k = hparams["k"]
    #train_imageset = 'data/imagesets_1080p/5fold_neg_all/{}k_train.txt'.format(k)
    #val_imageset = 'data/imagesets_1080p/5fold_neg_all/{}k_val.txt'.format(k)

    # if hparams["manual_dataset"]:
    #     train_imageset = train_imageset.replace("_all", "_manual")
    #     val_imageset = val_imageset.replace("_all", "_manual")
    hparams["nb_classes"] = len(hparams["selected_labels"])
    if hparams["selected_labels"] is not None:
        assert len(hparams["selected_labels"]) == hparams["nb_classes"], "The nb_classes should be the same as the amount of selected labels"

    print(hparams["selected_labels"])
    
    # make backup of training script
    source = __file__
    destination = logdir + name + '.txt'
    shutil.copyfile(source, destination)

    print("Training script backed up in " + destination)
    augmentations = None
    augmentations = {
        "flip_horizontal": True,
        "flip_vertical": True,
        # "cutout": 10,
        "shear": 0.2,
        "rotate": 90.,
        "translate": 20,  # max number of pixels
        "color": 0.2,
        "blur": 1.3
    }
    # ds_train = Dataset(train_imageset, batch_size=hparams['batch_size'], target_size=target_size, loading_size=target_size, augmentations=augmentations, nb_inputs=hparams['nb_inputs'], balanced=hparams['balanced'], do_cropping='preloaded', balance_polypids=True, selected_labels=hparams['selected_labels'])
    # ds_val = Dataset(val_imageset, batch_size=hparams['batch_size'], target_size=target_size, loading_size=target_size, augmentations=None, nb_inputs=hparams['nb_inputs'], shuffle=True, repeat=True, balanced=hparams['balanced'], do_cropping='preloaded', balance_polypids=True, selected_labels=hparams['selected_labels'])

    ds_train = Dataset(train_imageset, batch_size=hparams['batch_size'], target_size=target_size,
                       loading_size=loading_size, augmentations=augmentations, nb_inputs=hparams['nb_inputs'],
                       balanced=hparams['balanced'], do_cropping=hparams['do_cropping'], balance_polypids=hparams['balanced'],
                       selected_labels=hparams['selected_labels'], nb_classes=hparams['nb_classes'], shuffle=True)

    # ds_train = Dataset(train_imageset, batch_size=hparams['batch_size'], target_size=target_size,
    #                    loading_size=loading_size, augmentations=augmentations, nb_inputs=hparams['nb_inputs'],
    #                    balanced=hparams['balanced'], do_cropping=hparams['do_cropping'], balance_polypids=True,
    #                    selected_labels=hparams['selected_labels'], nb_classes=hparams['nb_classes'], shuffle=True)
    ds_val = Dataset(val_imageset, batch_size=hparams['batch_size'], target_size=target_size, loading_size=loading_size,
                     augmentations=None, nb_inputs=hparams['nb_inputs'], shuffle=True, repeat=True,
                     balanced=hparams['balanced'], do_cropping=hparams['do_cropping'], balance_polypids=hparams['balanced'],
                     selected_labels=hparams['selected_labels'], nb_classes=hparams['nb_classes'])
    # ds_val = Dataset(val_imageset, batch_size=hparams['batch_size'], target_size=target_size, loading_size=loading_size,
    #                  augmentations=None, nb_inputs=hparams['nb_inputs'], shuffle=True, repeat=True,
    #                  balanced=True, do_cropping=hparams['do_cropping'], balance_polypids=True,
    #                  selected_labels=hparams['selected_labels'], nb_classes=hparams['nb_classes'])
    if hparams['network'] == 'resnet':
        model = resnet(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,), fc_dropout=hparams['fc_dropout'],reg_constant=hparams['reg_constant'], nf=hparams['nf'], resnet_layers=hparams['resnet_layers'])
    elif hparams['network'] == 'cnn':
        model = basic_cnn(nb_classes=hparams['nb_classes'], reg_constant=hparams['reg_constant'], nf=hparams['nf'], image_dimensions=target_size + (3,), fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'])
    elif hparams['network'] == 'miclass':
        model = MIclass(nb_inputs=hparams['nb_inputs'], nf=hparams['nf'], nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,), bn=hparams['batch_normalization'], fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'], multi_output=hparams['multi_output'])
    elif hparams['network'] == 'vgg16':
        model = vgg16(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,), fc_dropout=hparams['fc_dropout'], nf=hparams['nf'],reg_constant=hparams['reg_constant'], frozen_layers=hparams["locked_layers"])
    elif hparams['network'] == 'linear':
        model = simple_linear(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,), fc_dropout=hparams['fc_dropout'], nf=hparams['nf'])
    elif hparams['network'] == 'vgg19':
        model = vgg19(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,), fc_dropout=hparams['fc_dropout'], nf=hparams['nf'])
    elif hparams['network'] == 'xception':
        model = xception(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,), fc_dropout=hparams['fc_dropout'], nf=hparams['nf'],reg_constant=hparams['reg_constant'])
    elif hparams['network'] == 'mobilenet':
        model = mobilenet(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,), fc_dropout=hparams['fc_dropout'], alpha=hparams['alpha'], nf=hparams['nf'])
    elif hparams['network'] == 'mobilenetFull':
        model = mobilenet_full(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,), fc_dropout=hparams['fc_dropout'], alpha=hparams['alpha'])
    elif hparams['network'] == 'efficientnet':
        model = efficientnet(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,),
                          fc_dropout=hparams['fc_dropout'], nf=hparams['nf'], multi_output=False, reg_constant=hparams['reg_constant'], nb_inputs=hparams['nb_inputs'], fold=str(k), b='0', frozen_layers=1)
    elif hparams['network'] == 'efficientnetFull':
        model = efficientnet_full(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,),
                          fc_dropout=hparams['fc_dropout'], b=hparams['b'], reg_constant=hparams['reg_constant'],nf=hparams['nf'])
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
    # model.load_weights('data/snapshots/cvc_vgg16_256x256_miclass_1in_bnTrue_fcdo0.3_convdo0.2_maxmerge_best.hdf5')

    callbacks = []
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('data/snapshots/all/' + name + '_best.h5', monitor='val_balanced_categorical_accuracy', mode='max', verbose=0, save_best_only=True, save_freq='epoch'))
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=os.path.join(logdir, name), profile_batch=0))

    # model.summary()

    # 
    if hparams["loss_type"] == "fl":
        loss = FocalLoss(hparams['gamma_loss'])
    elif hparams["loss_type"] == "dce":
        loss = DropoutCategoricalCrossEntropy(hparams['alpha_loss'])
    else:
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=hparams['label_smoothing'])
    # loss = balanced_categorical_crossentropy(hparams['batch_size'], hparams['nb_classes'])
    # if hparams['alpha_loss'] is not None and hparams['alpha_loss'] != 1:
    #     loss = DropoutCategoricalCrossEntropy(hparams['alpha_loss'])
    metrics = [tf.keras.metrics.CategoricalAccuracy(),]
    #metrics = [BalancedCategoricalAccuracy(hparams['nb_classes']),]
    for s in range(hparams['nb_classes']):
        metrics += [ tf.keras.metrics.Precision(class_id=s, name=f'precision_class_{s}'), tf.keras.metrics.Recall(class_id=s, name=f'recall_class_{s}')]

    # optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate'])
    optimizer = tf.keras.optimizers.SGD(learning_rate=hparams['learning_rate'], momentum=0.9)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.summary()
    length_train = 100#len(ds_train) // (16* hparams['batch_size'])
    length_val = 40#len(ds_val) // (16* hparams['batch_size'])
    model.fit(ds_train.tfdataset,
              epochs=30,
              steps_per_epoch=length_train,
              validation_data=ds_val.tfdataset,
              validation_steps=length_val,
              verbose=1,
              callbacks=callbacks)


    model.save('data/snapshots/all/' + name + '_final.h5')
    tf.keras.backend.clear_session()
if __name__ == '__main__':
    hparams = {
            'learning_rate': 1e-3,
            'target_width': 256,
            'target_height': 256,
            'nb_inputs': 1,
            'nb_classes': 2,
            'batch_normalization': True,
            'conv_dropout': 0.,
            'fc_dropout': 0.,
            'batch_size': 64,
            'balanced': True,
            'network': 'vgg16',
            'nf': 64,
            'label_smoothing': 0.4,
            'selected_labels': ["0","1"],
            #'selected_labels': ["0",("1", "2")],
            # 'selected_labels': ["0","1", "2"],
            'alpha_loss': 0.5,
            'gamma_loss': 1.0,
            "loss_type": "ce",
            'multi_output': False,
            'b': '0',
            #'do_cropping': True,
            'do_cropping': 'preloaded',
            'resnet_layers': '',
            'reg_constant': 0.,
            'locked_layers': 1,
        }

    split_type = 'fold'
    #dsc = '09'
    total_k = 1

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
    # name = s + '_HD{scope}2022_{network}{resnet_layers}_regularized{reg_constant}_{target_width}x{target_height}_{nb_inputs}in_nf{nf}_bn{batch_normalization}_fcdo{fc_dropout}_convdo{conv_dropout}_lossls{label_smoothing}_lossDropout{alpha}_shuffle_dsc{dsc}_reflect'.format(scope=scope, network=hparams['network'],b=hparams['b'], reg_constant=hparams['reg_constant'], target_width=hparams['target_width'], target_height=hparams['target_height'], nb_inputs=hparams['nb_inputs'], nf=hparams['nf'], batch_normalization=hparams['batch_normalization'], fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'], label_smoothing=hparams['label_smoothing'], alpha=hparams['alpha_loss'], dsc=dsc, resnet_layers=hparams['resnet_layers'])
    loss_descriptor = hparams["loss_type"]
    if loss_descriptor == "fl":
        loss_descriptor +="_gamma" + str(hparams['gamma_loss'])
    elif loss_descriptor == "dce":
        loss_descriptor += "_alpha" + str(hparams["alpha_loss"])
    else:
        loss_descriptor = "ce_lossls{label_smoothing}".format(label_smoothing=hparams['label_smoothing'])
    name = s + '_enhancedContrast_HD{scope}2024_{network}_{b}_regularized{reg_constant}_{target_width}x{target_height}_{nb_inputs}in_nf{nf}_bn{batch_normalization}_fcdo{fc_dropout}_balanced{balanced}_loss_{loss}_sgd'.format(scope=scope, network=hparams['network'],b=hparams['b'], reg_constant=hparams['reg_constant'], target_width=hparams['target_width'], target_height=hparams['target_height'], nb_inputs=hparams['nb_inputs'], nf=hparams['nf'], batch_normalization=hparams['batch_normalization'], fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'], alpha=hparams['alpha_loss'],  resnet_layers=hparams['b'], balanced=hparams["balanced"], loss=loss_descriptor, locked_layers=hparams["locked_layers"])

    # with k as range(0,1):
    k = 0#random.randint(0, 6)
    #for k in range(total_k):
    # train_imageset = 'data/imagesets_characterisation_1080p/{}{}_{}{}/{}k_train.txt'.format(total_k, split_type, scope, dsc,k)
    # val_imageset = 'data/imagesets_characterisation_1080p/{}{}_{}{}/{}k_val.txt'.format(total_k, split_type, scope, dsc, k)
    #train_imageset = 'data/imagecomsets_1080p/{}{}_{}/{}k_train.txt'.format(total_k, split_type, scope,k)
    #val_imageset = 'data/imagesets_1080p/{}{}_{}/{}k_val.txt'.format(total_k, split_type, scope, k)
    # train_imageset = 'data/imagesets_filtered_size_smaller_th_0.1/{}{}_{}/{}k_train.txt'.format(total_k, split_type, scope,k)
    # val_imageset = 'data/imagesets_filtered_size_smaller_th_0.1/{}{}_{}/{}k_val.txt'.format(total_k, split_type, scope, k)
    # train_imageset = 'data/imagesets_filtered_size/{}{}_{}_th_0.1/{}k_train.txt'.format(total_k, split_type, scope,k)
    # val_imageset = 'data/imagesets_filtered_size/{}{}_{}_th_0.1/{}k_val.txt'.format(total_k, split_type, scope, k)
    # train_imageset = 'data/imagesets_1080p/{}_split_0.7.txt'.format(scope)
    # val_imageset = 'data/imagesets_1080p/{}_split_0.1.txt'.format(scope)

    train_imageset = '/DATASERVER/MIC/GENERAL/STUDENTS/amartic/mscthesis/polypclassificationmi/data/imagesets_characterisation/all_split_0.7_balanced.txt'

    val_imageset = '/DATASERVER/MIC/GENERAL/STUDENTS/amartic/mscthesis/polypclassificationmi/data/imagesets_characterisation/all_split_0.1_balanced.txt'
    
    main(logdir, name+f'_{total_k}{split_type}{k}', hparams, train_imageset, val_imageset)
