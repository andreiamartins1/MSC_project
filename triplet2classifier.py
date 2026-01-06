import os
import argparse
import re
import shutil

import tensorflow as tf
from utils.callbacks import FinetuneScheduler
from utils.losses import TripletLossLayer, SemiHardTripletLossLayer, SemiHardHardTripletLossLayer, TripletAccuracyLayer, TripletPosDistanceLayer, TripletNegDistanceLayer, categorical_accuracy, binary_accuracy
custom_objects = {'tf': tf, 'TripletLossLayer': TripletLossLayer, 'SemiHardTripletLossLayer': SemiHardTripletLossLayer, 'SemiHardHardTripletLossLayer': SemiHardHardTripletLossLayer, 'TripletAccuracyLayer': TripletAccuracyLayer, 'TripletPosDistanceLayer': TripletPosDistanceLayer, 'TripletNegDistanceLayer': TripletNegDistanceLayer}
from utils.networks import embedding_vgg_nodropout

# Some general parameters
image_size = (int(1920), int(1080))
batch_size = 64
nb_classes = 2
augmentations = {
    "rotate": 10.,
    "flip_horizontal": True,
    "flip_vertical": True,
    "color": 0.2,
    "zoom": 0.1,
    "cropbox": 0.5}  # augment the cropbox padding by +-50%


def get_model():
    print("Transforming network for classification")
    # model = tf.keras.models.load_model(os.path.join('data/snapshots/', args.model_name + '_best.hdf5'), custom_objects=custom_objects)
    # single_branch = model.get_layer('model')  # extract single embedding network

    # remove dropout layers
    nf = int(args.model_name.split('_')[-1][-2:])
    input_shape = tuple([int(x) for x in re.search(r'_\d{1,}x\d{1,}_', args.model_name).group()[1:-1].split('x')]) + (3,)
    embedding_size = int(re.search(r'emb\d{1,}', args.model_name).group()[3:])
    single_branch = embedding_vgg_nodropout(nf, input_shape, embedding_size)
    single_branch.load_weights(os.path.join('data/snapshots/', args.model_name + '_best.hdf5'), by_name=True)

    # fix original network
    # single_branch.trainable = True
    single_branch.get_layer('vgg16').summary()
    # for layer in single_branch.get_layer('vgg16').layers[:15]:
    #     layer.trainable = False
    single_branch.summary()

    # append class layers
    # outputs = tf.keras.layers.Dense(args.nf, activation='relu', name='class_fc')(single_branch.output)
    outputs = single_branch.output
    outputs = tf.keras.layers.Dense(nb_classes, activation=tf.nn.softmax, name='class_output')(outputs)
    model_class = tf.keras.Model(single_branch.inputs, outputs)
    model_class.summary()
    return model_class


def get_linear_model():
    print("Transforming network for classification")
    model = tf.keras.models.load_model(os.path.join('data/snapshots/', args.model_name + '_best.hdf5'), custom_objects=custom_objects)
    single_branch = model.get_layer('model')  # extract single embedding network

    # fix original network
    single_branch.trainable = False
    single_branch.summary()

    # append linear svm layer
    outputs = tf.keras.layers.Dense(1, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(), name='svm_layer')(single_branch.output)
    model_class = tf.keras.Model(single_branch.inputs, outputs)
    model_class.summary()
    return model_class


# Hinge Loss
def hinge_loss(y_true, y_pred):
    y_true = tf.math.argmax(y_true, axis=1)
    y_true = tf.cast(y_true, tf.float32)
    y_true = y_true * 2 - 1  # labels must be -1 and 1
    return tf.maximum(0., 1 - y_true * y_pred)


def train_linear_model():
    # infer parameters from model filename
    nb_inputs = int(re.search(r'\d{1,}in', args.model_name).group()[:-2])
    input_size = tuple([int(x) for x in re.search(r'_\d{1,}x\d{1,}_', args.model_name).group()[1:-1].split('x')])
    dataset = args.model_name.split('_')[0]

    if dataset == 'catsndogs':
        from utils.Dataset_catsndogs import Dataset
        labels = {0: 'cat', 1: 'dog'}
        imageset_dir_train = 'data/catsndogs/train/'
        imageset_dir_val = 'data/catsndogs/val/'
        loading_size = None
    elif dataset == 'polyps':
        from utils.Dataset import Dataset
        labels = {0: 'HP', 1: 'AD', 2: 'SSA'}
        imageset_dir_train = 'data/imagesets/all_split_0.7.txt'
        imageset_dir_val = 'data/imagesets/all_split_0.1.txt'
        loading_size = image_size
    elif dataset == 'cvc':
        from utils.Dataset_cvc import Dataset
        labels = {0: 'ADN', 1: 'nonADN'}
        imageset_dir_train = 'data/cvc/train/'
        imageset_dir_val = 'data/cvc/val/'
        loading_size = image_size

    # Load and create class model
    model = get_linear_model()
    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(optimizer=opt, loss=hinge_loss, metrics=[binary_accuracy])

    # Create datasets
    ds_train = Dataset(imageset_dir_train, batch_size=batch_size, target_size=input_size, loading_size=loading_size, augmentations=augmentations, nb_inputs=nb_inputs, balanced=args.balanced, do_cropping=True)
    ds_val = Dataset(imageset_dir_val, batch_size=batch_size, target_size=input_size, loading_size=loading_size, augmentations=None, nb_inputs=nb_inputs, shuffle=True, repeat=True, balanced=args.balanced, do_cropping=True)

    # Fit model
    callbacks = []
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('data/snapshots/' + args.model_name + '_class_best.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_freq='epoch'))
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=os.path.join('logs/', args.model_name + '_class'), profile_batch=0))

    model.fit(ds_train.tfdataset,
              epochs=20,
              steps_per_epoch=len(ds_train) // batch_size,
              validation_data=ds_val.tfdataset,
              validation_steps=20,
              callbacks=callbacks)
    model.save('data/snapshots/' + args.model_name + '_class_final.h5')


def main():
    # make backup of training script
    source = __file__
    destination = os.path.join('logs/', args.model_name + '.txt')
    shutil.copyfile(source, destination)
    print("Training script backed up in " + destination)

    # infer parameters from model filename
    nb_inputs = int(re.search(r'\d{1,}in', args.model_name).group()[:-2])
    input_size = tuple([int(x) for x in re.search(r'_\d{1,}x\d{1,}_', args.model_name).group()[1:-1].split('x')])
    dataset = args.model_name.split('_')[0]

    if dataset == 'catsndogs':
        from utils.Dataset_catsndogs import Dataset
        labels = {0: 'cat', 1: 'dog'}
        imageset_dir_train = 'data/catsndogs/train/'
        imageset_dir_val = 'data/catsndogs/val/'
        loading_size = None
    elif dataset == 'hypVSadn':
        from utils.Dataset import Dataset
        labels = {0: 'HP', 1: 'AD', 2: 'SSA'}
        imageset_dir_train = 'data/imagesets/all_split_0.7.txt'
        imageset_dir_val = 'data/imagesets/all_split_0.1.txt'
        loading_size = image_size
    elif dataset == 'cvc':
        from utils.Dataset_cvc import Dataset
        labels = {0: 'ADN', 1: 'nonADN'}
        imageset_dir_train = 'data/cvc/train/'
        imageset_dir_val = 'data/cvc/val/'
        loading_size = image_size

    # Load and create class model
    model = get_model()

    # Create datasets
    ds_train = Dataset(imageset_dir_train, batch_size=batch_size, target_size=input_size, loading_size=loading_size, augmentations=augmentations, nb_inputs=nb_inputs, balanced=args.balanced, do_cropping=True)
    ds_val = Dataset(imageset_dir_val, batch_size=batch_size, target_size=input_size, loading_size=loading_size, augmentations=None, nb_inputs=nb_inputs, shuffle=True, repeat=True, balanced=args.balanced, do_cropping=True)

    # Fit model
    callbacks = []
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('data/snapshots/' + args.model_name + '_' + args.appendix + '_class_best.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_freq='epoch'))
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=os.path.join('logs/', args.model_name + '_' + args.appendix + '_class'), profile_batch=0))

    # First train only class layers
    model.trainable = True
    for layer in model.layers[:6]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-1), loss='categorical_crossentropy', metrics=[categorical_accuracy])
    model.summary()
    model.fit(ds_train.tfdataset,
              epochs=20,
              steps_per_epoch=200,
              validation_data=ds_val.tfdataset,
              validation_steps=20,
              callbacks=callbacks)

    # Now finetune embedding space
    model.trainable = True
    for layer in model.layers[:5]:
        layer.trainable = False
    model.training = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=[categorical_accuracy])
    model.summary()
    model.fit(ds_train.tfdataset,
              epochs=10,
              steps_per_epoch=200,
              validation_data=ds_val.tfdataset,
              validation_steps=20,
              callbacks=callbacks)

    # Now finetune encoder
    model.trainable = True
    for layer in model.layers[:3]:
        layer.trainable = False
    model.training = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=[categorical_accuracy])
    model.summary()
    model.fit(ds_train.tfdataset,
              epochs=20,
              steps_per_epoch=200,
              validation_data=ds_val.tfdataset,
              validation_steps=20,
              callbacks=callbacks)

    model.save('data/snapshots/' + args.model_name + '_' + args.appendix + '_class_final.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", action="store", default="cvc_triplet_128x128_1in_emb32_margin0.5_bnTrue_fcdo0.5_convdo0.2_maxmerge", dest="model_name")
    parser.add_argument("-nf", "--n_filters", action="store", default=16, dest="nf")
    parser.add_argument("--balanced", action="store_true", dest="balanced")
    parser.add_argument("-a", "--appendix", action="store", default="", dest="appendix")

    args = parser.parse_args()

    main()
    # train_linear_model()
