#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:02:15 2018

@author: teelbo0
"""
import os
import shutil
from utils.Dataset_cvc import Dataset

import tensorflow as tf
from utils.networks import MIclass, resnet, vgg16
from utils.callbacks import ConfusionMatrixLogger

from tensorboard.plugins.hparams import api as hp

def main(logdir, name, hparams):
        image_size = (int(1920),int(1080))
        target_size = (int(hparams['target_width']), int(hparams['target_height']))

        # make backup of training script
        source = __file__
        destination = 'logs/' + name + '.txt'
        shutil.copyfile(source, destination)
        print("Training script backed up in " + destination)

        augmentations = {
                "rotate": 10.,
                "flip_horizontal": True,
                "flip_vertical": True,
                "color": 0.2,
                "cropbox": 0.5 # augment the cropbox padding by +-50%
                }
        ds_train = Dataset('data/cvc/train/', batch_size=hparams['batch_size'], target_size=target_size, loading_size=image_size, augmentations=augmentations, nb_inputs=hparams['nb_inputs'], balanced=hparams['balanced'], do_cropping=True)
        ds_val = Dataset('data/cvc/val/', batch_size=hparams['batch_size'], target_size=target_size, loading_size=image_size, augmentations=None, nb_inputs=hparams['nb_inputs'], shuffle=True, repeat=True, balanced=hparams['balanced'], do_cropping=True)
        ds_val_fin = Dataset('data/cvc/val/', batch_size=hparams['batch_size'], target_size=target_size, loading_size=image_size, augmentations=None, nb_inputs=hparams['nb_inputs'], shuffle=True, repeat=False, balanced=hparams['balanced'], do_cropping=True)

        if hparams['network'] == 'resnet':
                model = resnet(nb_classes=2, image_dimensions=target_size + (3,))
        elif hparams['network'] == 'miclass':
                model = MIclass(nb_inputs=hparams['nb_inputs'], nb_classes=2, image_dimensions=target_size + (3,), bn=hparams['batch_normalization'], fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'])
        elif hparams['network'] == 'vgg16':
                model = vgg16(nb_classes=2, image_dimensions=target_size + (3,), fc_dropout=hparams['fc_dropout'])
        else:
                raise ValueError("hparams['network'] = {} is not defined".format(hparams['network']))
        model.summary()

        callbacks = []
        callbacks.append(tf.keras.callbacks.ModelCheckpoint('data/snapshots/' + name + '_best.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_freq='epoch'))
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=os.path.join(logdir, name), profile_batch=0))
        # callbacks.append(hp.KerasCallback(os.path.join(logdir, name), hparams, trial_id=name))
        # callbacks.append(ConfusionMatrixLogger(model=model, ds=ds_val_fin.tfdataset, log_dir=os.path.join(logdir, name), batch_size=hparams['batch_size'], class_names=list(ds_val.class_names.values())))

        model.get_layer('model').trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate']), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.summary()
        model.fit(ds_train.tfdataset,
                  epochs=50,
                  steps_per_epoch=len(ds_train) // hparams['batch_size'],
                  validation_data=ds_val.tfdataset,
                  validation_steps=20,
                  verbose=1,
                  callbacks=callbacks)

        for layer in model.get_layer('model').layers[15:]:
            layer.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate'] / 10), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.summary()
        model.fit(ds_train.tfdataset,
                  epochs=100,
                  steps_per_epoch=len(ds_train) // hparams['batch_size'],
                  validation_data=ds_val.tfdataset,
                  validation_steps=20,
                  verbose=1,
                  callbacks=callbacks)

        model.save('data/snapshots/' + name + '_final.h5')


if __name__ == '__main__':
        hparams = {
                'learning_rate': 1e-3,
                'target_width': 256,
                'target_height': 256,
                'nb_inputs': 1,  # not useful because only 1 image per polyp in this dataset
                'batch_normalization': True,
                'conv_dropout': 0.2,
                'fc_dropout': 0.3,
                'batch_size': 64,
                'balanced': True,
                'network': 'miclass'
        }
        name = 'cvc_{network}_{target_width}x{target_height}_miclass_{nb_inputs}in_bn{batch_normalization}_fcdo{fc_dropout}_convdo{conv_dropout}_maxmerge'.format(network=hparams['network'], target_width=hparams['target_width'], target_height=hparams['target_height'], nb_inputs=hparams['nb_inputs'], batch_normalization=hparams['batch_normalization'], fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'])
        main('logs/', name, hparams)
