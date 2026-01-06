#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:02:15 2018

@author: teelbo0
"""
import os
import shutil
from utils.Dataset import TripletDataset

import tensorflow as tf
from utils.networks import triplet_model

scope = 'all'
assert scope in ['all', 'manual']


def main(logdir, name, hparams):
    image_size = (int(1920), int(1080))
    target_size = (int(hparams['target_width']), int(hparams['target_height']))

    # make backup of training script
    source = __file__
    destination = os.path.join(logdir, name + '.txt')
    shutil.copyfile(source, destination)
    print("Training script backed up in " + destination)

    augmentations = {"rotate": 10.,
                     "flip_horizontal": True,
                     "flip_vertical": True,
                     "color": 0.2,
                     "zoom": 0.1,
                     "cropbox": 0.5}  # augment the cropbox padding by +-50%

    ds_train = TripletDataset('data/imagesets/{}_split_0.7.txt'.format(scope), batch_size=hparams['batch_size'], target_size=target_size, loading_size=image_size, augmentations=augmentations, nb_inputs=hparams['nb_inputs'], balanced=hparams['balanced'], do_cropping=True)
    ds_val = TripletDataset('data/imagesets/{}_split_0.1.txt'.format(scope), batch_size=hparams['batch_size'], target_size=target_size, loading_size=image_size, augmentations=None, nb_inputs=hparams['nb_inputs'], shuffle=True, do_cropping=True)

    model = triplet_model(nf=hparams['n_filters'], image_dimensions=target_size + (3,), embedding_size=hparams['embedding_size'], bn=hparams['batch_normalization'], fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'], margin=hparams['triplet_margin'])

    callbacks = []
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('data/snapshots/' + name + '_best.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_freq='epoch'))
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=os.path.join(logdir, name), profile_batch=0))
    # callbacks.append(hp.KerasCallback(os.path.join(logdir, name), hparams, trial_id=name))
    # callbacks.append(TripletConfusionMatrixLogger(model=model, ds=ds_val.tfdataset.take(1), log_dir=os.path.join(logdir, name), batch_size=hparams['batch_size'], class_names=list(ds_val.class_names.values())))

    # first train embedding space alone
    model.get_layer('model').get_layer('vgg16').trainable = False
    opt = tf.keras.optimizers.Adagrad(learning_rate=hparams['learning_rate'])
    model.compile(optimizer=opt, loss=None, metrics=[])
    model.summary()

    model.fit(ds_train.tfdataset,
              epochs=20,
              steps_per_epoch=40,  # len(ds_train) // hparams['batch_size'],
              validation_data=ds_val.tfdataset,
              validation_steps=20,  # len(ds_val) // hparams['batch_size'],
              callbacks=callbacks)

    # # now also finetune encoder
    model.trainable = True
    for layer in model.get_layer('model').get_layer('vgg16').layers[:15]:
        layer.trainable = False
    opt = tf.keras.optimizers.Adagrad(learning_rate=hparams['learning_rate'])
    model.compile(optimizer=opt, loss=None, metrics=[])
    model.summary()
    model.fit(ds_train.tfdataset,
              epochs=150,
              steps_per_epoch=40,  # len(ds_train) // hparams['batch_size'],
              validation_data=ds_val.tfdataset,
              validation_steps=20,  # len(ds_val) // hparams['batch_size'],
              callbacks=callbacks)

    model.save('data/snapshots/' + name + '_final.h5')


if __name__ == '__main__':
    hparams = {
        'learning_rate': 0.001,  # noqa: E126
        'target_width': 256,
        'target_height': 256,
        'nb_inputs': 1,
        'batch_normalization': True,
        'conv_dropout': 0.,
        'fc_dropout': 0.3,
        'batch_size': 64,
        'network': 'triplet',
        'balanced': False,  # if you balance the data before making triplets, you'll eventually create triplets with anchor and positive being the same image (i.e. trivial triplets)
        'embedding_size': 4,
        'triplet_margin': 0.2,
        'n_filters': 16
    }
    name = 'hypVSadn_{scope}_{network}-smallvgg16FT2stepPP_{target_width}x{target_height}_{nb_inputs}in_emb{embedding_size}_margin{triplet_margin}_bn{batch_normalization}_fcdo{fc_dropout}_convdo{conv_dropout}_nf{n_filters}'.format(scope=scope, network=hparams['network'],target_width=hparams['target_width'], target_height=hparams['target_height'], nb_inputs=hparams['nb_inputs'], embedding_size=hparams['embedding_size'], triplet_margin=hparams['triplet_margin'], batch_normalization=hparams['batch_normalization'], fc_dropout=hparams['fc_dropout'], conv_dropout=hparams['conv_dropout'], n_filters=hparams['n_filters'])
    main('logs/', name, hparams)
