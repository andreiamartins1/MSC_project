#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:02:15 2018

@author: teelbo0
"""
import os
import shutil
from utils.Dataset import Dataset

import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from utils.networks import MIclass, resnet, vgg16, efficientnet, detnet, wavelet_network, transfer, darknet, googlenet, multimodel_same_base
from utils.callbacks import ConfusionMatrixLogger
from utils.losses import categorical_accuracy_label_smoothing

from tensorboard.plugins.hparams import api as hp


scope = 'all'
assert scope in ['all', 'manual']

def main(logdir, name, hparams, train_imageset='data/imagesets_1080p/{}_split_0.7.txt'.format(scope), val_imageset='data/imagesets_1080p/{}_split_0.1.txt'.format(scope)):
    image_size = (int(1920),int(1080))
    target_size = (int(hparams['target_width']), int(hparams['target_height']))

    if hparams["selected_labels"] is not None:
        assert len(hparams["selected_labels"]) == hparams["nb_classes"], "The nb_classes should be the same as the amount of selected labels"


    # Check whether model has been trained before
    if os.path.isfile('data/snapshots/' + name + '_final.h5'):
        print(f'Model {name} exists already')
        return

    # make backup of training script
    source = __file__
    destination = 'logs/' + name + '.txt'
    shutil.copyfile(source, destination)
    print("Training script backed up in " + destination)

    augmentations = {
            "rotate": 90.,
            "flip_horizontal": True,
            "flip_vertical": True,
            "color": 0.2,
            "cropbox": 2.0 # augment the cropbox padding by +-200%
            }
    # if hparams['network'] == 'resnet':
    #     augmentations = {}

    ds_train = Dataset(train_imageset, batch_size=hparams['batch_size'], target_size=target_size, loading_size=image_size, augmentations=augmentations, nb_inputs=hparams['nb_inputs'], balanced=hparams['balanced'], do_cropping=True, balance_polypids=True, nb_classes=hparams['nb_classes'], selected_labels=hparams['selected_labels'])
    ds_val = Dataset(val_imageset, batch_size=hparams['batch_size'], target_size=target_size, loading_size=image_size, augmentations=None, nb_inputs=hparams['nb_inputs'], shuffle=True, repeat=True, balanced=hparams['balanced'], do_cropping=True, balance_polypids=True, nb_classes=hparams['nb_classes'], selected_labels=hparams['selected_labels'])

    if hparams['network'] == 'resnet':
            model = resnet(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,))
    elif hparams['network'] == 'miclass':
            model = MIclass(nb_inputs=hparams['nb_inputs'], nf=hparams['nf'], nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,), bn=hparams['batch_normalization'], fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'])
    elif hparams['network'] == 'vgg16':
            model = vgg16(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,), fc_dropout=hparams['fc_dropout'])
    elif hparams['network'] == 'detnet':
            model = detnet(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,),
                          fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'])
    elif hparams['network'] == 'efficientnet':
        model = efficientnet(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,),
                          fc_dropout=hparams['fc_dropout'], nf=hparams['nf'], multi_output=False)
    elif hparams['network'] == 'darknet':
            model = darknet(nb_classes=hparams['nb_classes'], image_dimensions=target_size + (3,), fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'])
    elif hparams['network'] == 'mixed':
        model = multimodel_same_base(hparams['names'], nb_classes=3, image_dimensions=target_size + (3,))
    else:
            raise ValueError("hparams['network'] = {} is not defined".format(hparams['network']))
    # model.summary()
    # model.load_weights('data/snapshots/cvc_vgg16_256x256_miclass_1in_bnTrue_fcdo0.3_convdo0.2_maxmerge_best.hdf5')

    callbacks = []
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('data/snapshots/mm_' + name + '_best.hdf5', monitor='val_categorical_accuracy', mode='max', verbose=0, save_best_only=True, save_freq='epoch'))
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=os.path.join(logdir, name), profile_batch=0))
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True))

    # callbacks.append(hp.KerasCallback(os.path.join(logdir, name), hparams, trial_id=name))
    # callbacks.append(ConfusionMatrixLogger(model=model, ds=ds_val.tfdataset, log_dir=os.path.join(logdir, name), batch_size=hparams['batch_size'], class_names=list(ds_val.class_names.values())))

    # model.get_layer('model').trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate']), loss=CategoricalCrossentropy(label_smoothing=hparams['label_smoothing']), metrics=['categorical_accuracy'])
    model.summary()
    model.fit(ds_train.tfdataset,
              epochs=200,
              steps_per_epoch=100,
              validation_data=ds_val.tfdataset,
              validation_steps=40,
              verbose=1,
              callbacks=callbacks)

    # for layer in model.get_layer('model').layers[15:]:
    #     layer.trainable = True
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate']/10), loss=CategoricalCrossentropy(label_smoothing=hparams['label_smoothing']), metrics=['categorical_accuracy'])
    # model.summary()
    # model.fit(ds_train.tfdataset,
    #           epochs=100,
    #           steps_per_epoch=100,
    #           validation_data=ds_val.tfdataset,
    #           validation_steps=40,
    #           verbose=1,
    #           callbacks=callbacks)

    model.save('data/snapshots/mm_' + name + '_final.h5')
    del model
    del ds_train
    del ds_val

if __name__ == '__main__':
    hparams = {
            'learning_rate': 1e-3,
            'target_width': 256,
            'target_height': 256,
            'nb_inputs': 1,
            'nb_classes': 2,
            'batch_normalization': True,
            'conv_dropout': 0.0,
            'fc_dropout': 0.5,
            'batch_size': 64,
            'balanced': True,
            'network': 'miclass',
            'nf': 256,
            'label_smoothing': 0.1,
        }
    # name = 'hypVSadn_HD{scope}_{network}_{target_width}x{target_height}_miclass_{nb_inputs}in_nf{nf}_bn{batch_normalization}_fcdo{fc_dropout}_convdo{conv_dropout}_lossls{label_smoothing}_avgfusion_l25e-2'.format(scope=scope, network=hparams['network'], target_width=hparams['target_width'], target_height=hparams['target_height'], nb_inputs=hparams['nb_inputs'], nf=hparams['nf'], batch_normalization=hparams['batch_normalization'], fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'], label_smoothing=hparams['label_smoothing'])
    # main('logs/', name, hparams)

    tf.keras.backend.clear_session()


    selected_labels = {'hypVSall': ["0", ("1", "2") ], 'adnVSall':["1", ("0", "2") ], 'sspVSall': ["2", ("0", "1") ], 'hypVSadn': ["0", "1"], 'adnVSssp': ["1", "2"], 'hypVSssp': ["0", "2"]}#, 'hypVSall': ["0", ("1", "2") ], 'adnVSall':["1", ("0", "2") ], 'sspVSall': ["2", ("0", "1") ]}
    names = []
    k = 0
    for selected, labels in selected_labels.items():
        hparams["selected_labels"] = labels
        train_imageset = 'data/imagesets_1080p/6bootstrap_{}/{}k_train.txt'.format(scope, k)
        val_imageset = 'data/imagesets_1080p/6bootstrap_{}/{}k_val.txt'.format(scope, k)
        name = '{test}_HD{scope}_{network}_{target_width}x{target_height}_miclass_{nb_inputs}in_nf{nf}_bn{batch_normalization}_fcdo{fc_dropout}_convdo{conv_dropout}_lossls{label_smoothing}'.format(test=selected,scope=scope, network=hparams['network'], target_width=hparams['target_width'], target_height=hparams['target_height'], nb_inputs=hparams['nb_inputs'], nf=hparams['nf'], batch_normalization=hparams['batch_normalization'], fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'], label_smoothing=hparams['label_smoothing'])
        print("###################################################")
        print("#                                                 #")
        print(f"#                    {selected}                     #")
        print("#                                                 #")
        print("###################################################")
        main('logs/', name, hparams, train_imageset, val_imageset)
        names.append(name)
        k += 1
        if k == len(selected_labels.keys()):
            k = 0


    hparams['learning_rate'] = 1e-4
    hparams['names'] = names
    hparams['nb_classes'] = 3
    hparams['selected_labels'] = ("0", "1", "2")
    hparams['network'] = 'mixed'
    print(names)
    name = 'mixed_eff_net_model_HD{scope}_{network}_{target_width}x{target_height}_miclass_{nb_inputs}in_nf{nf}_bn{batch_normalization}_fcdo{fc_dropout}_convdo{conv_dropout}_lossls{label_smoothing}'.format(
        scope=scope, network=hparams['network'], target_width=hparams['target_width'],
        target_height=hparams['target_height'], nb_inputs=hparams['nb_inputs'], nf=hparams['nf'],
        batch_normalization=hparams['batch_normalization'], fc_dropout=hparams['fc_dropout'],
        conv_dropout=hparams['conv_dropout'], label_smoothing=hparams['label_smoothing'])
    main('logs/', name, hparams, train_imageset, val_imageset)
