#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:02:15 2018

@author: teelbo0
"""
import os
import shutil
from utils.Dataset_catsndogs import Dataset

import tensorflow as tf
from utils.networks import MIclass, resnet, wavelet_network
from utils.callbacks import ConfusionMatrixLogger

from tensorboard.plugins.hparams import api as hp

def main(logdir, name, hparams):
        target_size = (int(hparams['target_width']), int(hparams['target_height']))

        # make backup of training script
        source = __file__
        destination = os.path.join(logdir, name + '.txt')
        shutil.copyfile(source, destination)
        print("Training script backed up in " + destination)

        augmentations = {
                "rotate": 90.,
                "flip_horizontal": True,
                "flip_vertical": True,
                "color": 0.2,
                "zoom": 0.25
                }
        ds_train = Dataset('data/catsndogs/train/', batch_size=hparams['batch_size'], target_size=target_size, augmentations=augmentations, nb_inputs=hparams['nb_inputs'])
        ds_val = Dataset('data/catsndogs/val/', batch_size=hparams['batch_size'], target_size=target_size, augmentations=None, nb_inputs=hparams['nb_inputs'], shuffle=False)

        if hparams['network'] == 'resnet':
                model = resnet(nb_classes=2, image_dimensions=target_size + (3,))
        elif hparams['network'] == 'miclass':
                model = MIclass(nb_inputs=hparams['nb_inputs'], nb_classes=2, image_dimensions=target_size + (3,), bn=hparams['batch_normalization'], fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'])
        elif hparams['network'] == 'wavelet':
                model = wavelet_network(nb_classes=2, image_dimensions=target_size + (3,),
                                        bn=hparams['batch_normalization'], conv_dropout=hparams['fc_dropout'])
        else:
                raise ValueError("hparams['network'] = {} is not defined".format(hparams['network']))
        model.summary()

        opt = tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate'])
        model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

        callbacks = []
        callbacks.append(tf.keras.callbacks.ModelCheckpoint('data/snapshots/' + name + '_best.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_freq='epoch'))
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=os.path.join(logdir, name), profile_batch=0))
        callbacks.append(hp.KerasCallback(os.path.join(logdir, name), hparams, trial_id=name))
        callbacks.append(ConfusionMatrixLogger(model=model, ds=ds_val.tfdataset, log_dir=os.path.join(logdir, name), batch_size=hparams['batch_size'], class_names=list(ds_val.class_names.values())))

        hist = model.fit(ds_train.tfdataset,
                epochs=150,
                steps_per_epoch=len(ds_train) // hparams['batch_size'],
                validation_data=ds_val.tfdataset,
                validation_steps=len(ds_val) // hparams['batch_size'],
                callbacks=callbacks)
        print(hist.history.keys())
        model.save('data/snapshots/' + name + '_final.h5')

if __name__ == '__main__':
        hparams = {
                'learning_rate': 1e-3,
                'target_width': 128,
                'target_height': 128,
                'nb_inputs': 1,
                'batch_normalization': True,
                'conv_dropout': 0.2,
                'fc_dropout': 0.5,
                'batch_size': 64,
                'network': 'resnet'
        }
        name = 'catsndogs_{network}_{target_width}x{target_height}_miclass_{nb_inputs}in_bn{batch_normalization}_fcdo{fc_dropout}_convdo{conv_dropout}_maxmerge'.format(network=hparams['network'],target_width=hparams['target_width'], target_height=hparams['target_height'], nb_inputs=hparams['nb_inputs'], batch_normalization=hparams['batch_normalization'], fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'])
        main('logs/', name, hparams)
