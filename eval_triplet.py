import os
import sys
sys.path.append('.')
sys.path.append('..')

import numpy as np
import cv2
import argparse
import pandas as pd
import glob
import pickle
import re
import random
random.seed(1994)
from matplotlib import pyplot as plt

import tensorflow as tf
from utils.losses import TripletLossLayer, SemiHardTripletLossLayer, SemiHardHardTripletLossLayer, TripletAccuracyLayer, TripletPosDistanceLayer, TripletNegDistanceLayer
custom_objects = {'tf': tf, 'TripletLossLayer': TripletLossLayer, 'SemiHardTripletLossLayer': SemiHardTripletLossLayer, 'SemiHardHardTripletLossLayer': SemiHardHardTripletLossLayer, 'TripletAccuracyLayer': TripletAccuracyLayer, 'TripletPosDistanceLayer': TripletPosDistanceLayer, 'TripletNegDistanceLayer': TripletNegDistanceLayer}
from utils.Dataset import crop_to_bbox

image_size = (int(1920), int(1080))
batch_size = 16

results_struct = [
                'anchor_fn',
                'pos_fn',
                'neg_fn',
                'anchor_label',
                'pos_label',
                'neg_label',

                'pdist',
                'ndist']


class Evaluator(object):
    def __init__(self, model, ds):
        self.model = model
        self.ds = ds
        self.result = pd.DataFrame(columns=results_struct)

    def run(self):
        for triplet in self.ds.tfdataset:
            pred = self._forward_nn(triplet)  # output of form [loss, accuracy, [pdist], [ndist]]
            self._calculate_metrics(triplet, pred)

    def _forward_nn(self, frame):
        return self.model.predict(frame)

    def _calculate_metrics(self, triplet, pred):
        # loop over batch dimension
        for b in range(len(triplet['anchor_fn'])):
            self.result = self.result.append({
                'anchor_fn': triplet['anchor_fn'][b].numpy(),
                'pos_fn': triplet['pos_fn'][b].numpy(),
                'neg_fn': triplet['neg_fn'][b].numpy(),
                'anchor_label': triplet['anchor_label'][b].numpy(),
                'pos_label': triplet['pos_label'][b].numpy(),
                'neg_label': triplet['neg_label'][b].numpy(),

                'pdist': pred[2][b],
                'ndist': pred[3][b]}, ignore_index=True)


def analyse_df(df_fn):
    df = pd.read_pickle(df_fn)

    alphas = np.linspace(0, 0.5, 20)
    accuracies = []
    inclusions = []
    for alpha in alphas:
        accuracies.append((df['ndist'] - df['pdist'] > alpha).sum() / ((df['ndist'] - df['pdist'] > alpha).sum() + (df['ndist'] - df['pdist'] < -alpha).sum()))
        inclusions.append(((df['ndist'] - df['pdist'] > alpha).sum() + (df['ndist'] - df['pdist'] < -alpha).sum()) / len(df))

    plt.plot(alphas, accuracies)
    plt.plot(alphas, inclusions)
    plt.legend()
    plt.savefig('accVSincl.png')


def main():
    model = tf.keras.models.load_model(os.path.join('data/snapshots/', args.model_name + '_best.hdf5'), custom_objects=custom_objects)
    model.summary()

    nb_inputs = int(re.search(r'\d{1,}in', args.model_name).group()[:-2])
    input_size = tuple([int(x) for x in re.search(r'_\d{1,}x\d{1,}_', args.model_name).group()[1:-1].split('x')])
    ds = Dataset(imageset_dir, batch_size=batch_size, target_size=input_size, loading_size=loading_size, augmentations={}, nb_inputs=nb_inputs, shuffle=True, repeat=False, do_cropping=True, balanced=False)

    evaluator = Evaluator(model, ds)
    evaluator.run()
    result = evaluator.result

    print(result)
    result.to_excel(os.path.join("logs/", args.model_name + ".xlsx"), columns=results_struct)
    result.to_pickle(os.path.join("logs/", args.model_name + ".pkl"))


def eval_tfdataset():
    model = tf.keras.models.load_model(os.path.join('data/snapshots/', args.model_name + '_best.hdf5'), custom_objects=custom_objects)
    model.summary()

    nb_inputs = int(re.search(r'\d{1,}in', args.model_name).group()[:-2])
    input_size = tuple([int(x) for x in re.search(r'_\d{1,}x\d{1,}_', args.model_name).group()[1:-1].split('x')])
    ds = Dataset(imageset_dir, batch_size=batch_size, target_size=input_size, loading_size=loading_size, augmentations={}, nb_inputs=nb_inputs, shuffle=True, repeat=True, do_cropping=True, balanced=args.balanced)

    result = model.evaluate(ds.tfdataset, steps=150)
    print(dict(zip(model.metrics_names, result)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", action="store", default="cvc_triplet_128x128_1in_emb32_margin0.5_bnTrue_fcdo0.5_convdo0.2_maxmerge", dest="model_name")
    parser.add_argument("-ds", "--dataset", action="store", default="cvc_triplet", dest="dataset")
    parser.add_argument("--balanced", action="store_true", dest="balanced")
    args = parser.parse_args()

    assert args.dataset in ['catsndogs', 'polyps', 'cvc']

    if args.dataset == 'catsndogs':
        from utils.Dataset_catsndogs import TripletDataset as Dataset
        labels = {0: 'cat', 1: 'dog'}
        imageset_dir = 'data/catsndogs/val/'
        loading_size = None
    elif args.dataset == 'polyps':
        from utils.Dataset import TripletDataset as Dataset
        labels = {0: 'HP', 1: 'AD', 2: 'SSA'}
        imageset_dir = 'data/imagesets/all_split_0.7.txt'
        loading_size = image_size
    elif args.dataset == 'cvc':
        from utils.Dataset_cvc import TripletDataset as Dataset
        labels = {0: 'ADN', 1: 'nonADN'}
        imageset_dir = 'data/cvc/val/'
        loading_size = image_size

    eval_tfdataset()
    main()
    analyse_df(os.path.join("logs/", args.model_name + ".pkl"))
