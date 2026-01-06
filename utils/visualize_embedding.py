import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import argparse
import re

from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

import tensorflow as tf
from losses import TripletLossLayer, SemiHardTripletLossLayer, SemiHardHardTripletLossLayer, TripletAccuracyLayer, TripletPosDistanceLayer, TripletNegDistanceLayer
custom_objects = {'tf': tf, 'TripletLossLayer': TripletLossLayer, 'SemiHardTripletLossLayer': SemiHardTripletLossLayer, 'SemiHardHardTripletLossLayer': SemiHardHardTripletLossLayer, 'TripletAccuracyLayer': TripletAccuracyLayer, 'TripletPosDistanceLayer': TripletPosDistanceLayer, 'TripletNegDistanceLayer': TripletNegDistanceLayer}

image_size = (int(1920), int(1080))
batch_size = 16


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


def polyps_data(n_samples=1000):
    # predict for subset of images and keep embedding vectors
    model = tf.keras.models.load_model(os.path.join('data/snapshots/', args.model_name + '_best.hdf5'), custom_objects=custom_objects)
    model = model.get_layer('model')
    model.summary()

    nb_inputs = int(re.search(r'\d{1,}in', args.model_name).group()[:-2])
    input_size = tuple([int(x) for x in re.search(r'_\d{1,}x\d{1,}_', args.model_name).group()[1:-1].split('x')])
    embedding_size = int(re.search(r'emb\d{1,}', args.model_name).group()[3:])

    ds = Dataset(imageset_dir, batch_size=batch_size, target_size=input_size, loading_size=loading_size, augmentations={}, nb_inputs=nb_inputs, shuffle=True, repeat=False, do_cropping=True, balanced=True)

    sample_idx = 0
    embeddings = np.empty((0, embedding_size), dtype=np.float32)
    labels = np.empty((0, 1), dtype=np.uint8)
    for img, label in ds.tfdataset:
        embeddings = np.append(embeddings, model.predict(img), axis=0)
        labels = np.append(labels, np.expand_dims(np.argmax(label, axis=-1), axis=-1), axis=0)

        # stop criterium
        if sample_idx > n_samples / batch_size:
            break
        sample_idx += 1

    plt.scatter(embeddings[:, 0], embeddings[:, 1],
                c=labels, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('Accent', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.savefig('embeddings_01.png')

    # pca for the embedding space
    pca = PCA(2)  # project from 64 to 2 dimensions
    projected = pca.fit_transform(embeddings)
    print(embeddings.shape)
    print(projected.shape)

    plt.figure()
    plt.scatter(projected[:, 0], projected[:, 1],
                c=labels, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('Accent', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.savefig('embeddings_pca.png')

    # See how many components are really necessary
    plt.figure()
    pca = PCA().fit(embeddings)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig('explained_variance.png')


def digits_data():
    digits = load_digits()
    digits.data.shape

    pca = PCA(2)  # project from 64 to 2 dimensions
    projected = pca.fit_transform(digits.data)
    print(digits.data.shape)
    print(projected.shape)
    print(digits.target.shape, digits.target)

    plt.scatter(projected[:, 0], projected[:, 1],
                c=digits.target, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('Accent', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()

    plt.figure()
    pca = PCA().fit(digits.data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')

    plt.show()


def random_data():
    rng = np.random.RandomState(1)
    X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
    plt.scatter(X[:, 0], X[:, 1])
    plt.axis('equal')

    pca = PCA(n_components=2)
    pca.fit(X)

    plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 3 * np.sqrt(length)
        draw_vector(pca.mean_, pca.mean_ + v)
    plt.axis('equal')

    pca = PCA(n_components=1)
    pca.fit(X)
    X_pca = pca.transform(X)
    print("original shape:   ", X.shape)
    print("transformed shape:", X_pca.shape)

    X_new = pca.inverse_transform(X_pca)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
    plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
    plt.axis('equal')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", action="store", default="cvc_triplet_128x128_1in_emb32_margin0.5_bnTrue_fcdo0.5_convdo0.2_maxmerge", dest="model_name")
    parser.add_argument("-ds", "--dataset", action="store", default="cvc_triplet", dest="dataset")
    parser.add_argument("--balanced", action="store_true", dest="balanced")
    args = parser.parse_args()

    assert args.dataset in ['catsndogs', 'polyps', 'cvc']

    if args.dataset == 'catsndogs':
        from Dataset_catsndogs import Dataset
        labels = {0: 'cat', 1: 'dog'}
        imageset_dir = 'data/catsndogs/val/'
        loading_size = None
    elif args.dataset == 'polyps':
        from Dataset import Dataset
        labels = {0: 'HP', 1: 'AD', 2: 'SSA'}
        imageset_dir = 'data/imagesets/all_split_0.7.txt'
        loading_size = image_size
    elif args.dataset == 'cvc':
        from Dataset_cvc import Dataset
        labels = {0: 'ADN', 1: 'nonADN'}
        imageset_dir = 'data/cvc/train/'
        loading_size = image_size

    polyps_data()
