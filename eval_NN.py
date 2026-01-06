import math
import os
import sys
sys.path.append('.')
sys.path.append('..')

import numpy as np
import argparse
import pandas as pd
import glob
import pickle
import re
import random
random.seed(1994)
import cv2
from operator import itemgetter
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf

from utils.Dataset import crop_to_bbox
from utils.losses import DropoutCategoricalCrossEntropy
#from wavetf import WaveTFFactory

img_dir = '/DATASERVER/MIC/SHARED/ENDOSCOPY/EUROPOL/JPEGImages/1080p/'
ann_dir = '/DATASERVER/MIC/SHARED/ENDOSCOPY/EUROPOL/Annotations/480p/'

image_size = (int(1920), int(1080))
exp_alpha = 0.1

results_struct = [
                'seq',
                'light',
                'seqs',  # the seq names that exist for this polyp
                'img_fns',  # list of N original image filenames
                'ann_fns',  # list of N image annotations
                'key',  # list of frame keyness
                'pred',  # prediction softmax output
                'pred_class',  # argmax of pred
                'gt_class',
                'match',
                'confidence',
                'pred_th',  # prediction softmax output
                'pred_class_th',  # argmax of pred
                'match_th',
                'confidence_th',
                'exp_smooth',
                'cum_exp_smooth',
                'exp_smooth_th',
                'cum_exp_smooth_th'
]


def get_seq_length(seq, correct_seq=False):
    '''
    Returns the number of frames of the given seq
    '''
    if correct_seq:
        return NotImplementedError('automatic sequence name correction is not yet implemented.')

    if not os.path.exists(ann_dir + seq + '/frames_to_use.pickle'):
        raise ValueError("frames_to_use does not exist.")
    validity = pickle.load(open(ann_dir + seq + '/frames_to_use.pickle', 'rb'))
    return len(validity)


def get_filename(seq, idx, return_type='img', check_valid=False, check_keyness=False, correct_seq=False):
    '''
    Returns the full path to the image or annotation for a given sequence and frame idx.

    Arguments:
    seq - A string with the full name of the sequence, e.g. 'p049_02_B_WLI'
    idx - An integer indicating the frame index to retrieve filename for
    type - either img | ann | both

    check_valid - Return whether the requested filename is for a valid image (i.e. with a temporally stable annotation)
    check_keyness - Return whether the requested filename is a key frame (i.e. if the annotation was manually performed)
    correct_seq - Automatically corrects the sequence metadata in the name if unsure about correctness

    Returns a dictionary with the following keys (* if optional):
    img_fn
    ann_fn*
    valid*
    key*
    corrected_seq*
    '''
    if correct_seq:
        return NotImplementedError('automatic sequence name correction is not yet implemented.')

    img_fn = img_dir + seq + "/%05d" % (idx + 1) + ".jpg"
    ann_fn = ann_dir + seq + "/%05d" % (idx + 1) + ".png"

    if not os.path.exists(img_fn):
        img_fn = img_fn.replace('.jpg', '.png')
    if not os.path.exists(img_fn):
        # try easics format
        img_fn = img_dir + seq + "/" + "input_%05d" % (idx) + ".png"
        if not os.path.exists(ann_fn):
            ann_fn = ann_dir + seq + "/" + seq + ".mp4-%05d" % (idx + 1) + ".png"
    if not os.path.exists(img_fn):
        # try the other format
        img_fn = img_dir + seq + "/" + seq + ".mp4-%05d" % (idx + 1) + ".jpg"
        ann_fn = ann_dir + seq + "/" + seq + ".mp4-%05d" % (idx + 1) + ".png"
    if os.path.exists(img_fn) and not os.path.exists(ann_fn):
        ann_fn = ann_dir + seq + "/" + seq + ".mp4-%05d" % (idx + 1) + ".png"

    assert os.path.exists(img_fn), img_fn + " does not exist."
    assert os.path.exists(ann_fn), ann_fn + " does not exist."

    result = {'img_fn': img_fn}
    if return_type != 'img':
        result['ann_fn'] = ann_fn

    if check_valid:
        # Check valid
        if not os.path.exists(ann_dir + seq + '/frames_to_use.pickle'):
            print(ann_dir + seq + '/frames_to_use.pickle')
            raise ValueError("frames_to_use does not exist.")
        result['valid'] = pickle.load(open(ann_dir + seq + '/frames_to_use.pickle', 'rb'))[idx]
    if check_keyness:
        # Check if key frame
        candidates_pattern = glob.glob(img_dir + seq + "/00000_ann*%05d" % (idx + 1) + ".jpg")
        result['key'] = len(candidates_pattern) > 0

    return result


class Evaluator(object):
    def __init__(self, model, seqs, label, polyp_id, types, n_samples=20, n_classes=2, labels=None, th=0.5):

        if type(model) != list:
            self.model = [model,]
        self.model = model


        if labels is None:
            self.labels = [np.arange(0, n_classes) for j in range(len(self.model))]
        else:
            self.labels = []

            for z in labels:
                lb = np.ones_like(range(n_classes)) * -1
                for key, l in z.items():
                    lb[int(key)] = int(l)
                self.labels.append(lb)

        self.types = types

        self.seqs = seqs
        self.label = label
        self.polyp_id = polyp_id
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.th = th

        self.result = pd.DataFrame(columns=results_struct)

        self.nb_inputs = int(re.search(r'\d{1,}in', model_name).group()[:-2])
        self.input_size = tuple([int(x) for x in re.search(r'_\d{1,}x\d{1,}_', model_name).group()[1:-1].split('x')])
        self.valid_img_fns = []
        self.__populate_img_fns__()

    def __populate_img_fns__(self):
        '''
        Populate self.results with self.n_samples lists of filenames to run prediction on.
        '''
        # build list of valid img_fns
        valid_img_fns = []
        valid_ann_fns = []
        valid_img_keyness = []
        for seq in self.seqs:
            for i in range(get_seq_length(seq)):
                frame_metadata = get_filename(seq, i, return_type='both', check_valid=True, check_keyness=True)

                if frame_metadata['valid']:
                    valid_img_fns.append(frame_metadata['img_fn'])
                    valid_ann_fns.append(frame_metadata['ann_fn'])
                    valid_img_keyness.append(frame_metadata['key'])

        # take self.n_samples of self.n_inputs images
        # for idx in range(self.n_samples):
        #     sample_fns, sample_anns, sample_keyness = zip(*random.choices(list(zip(valid_img_fns, valid_ann_fns, valid_img_keyness)), k=self.nb_inputs))
        #     self.result = self.result.append({'seqs': self.seqs, 'img_fns': sample_fns, 'ann_fns': sample_anns, 'key': sample_keyness}, ignore_index=True)

        for i in range(len(valid_img_fns)):
            self.result = self.result.append({'seqs': self.seqs, 'img_fns': [valid_img_fns[i]], 'ann_fns': [valid_ann_fns[i]], 'key': [valid_img_keyness[i]]}, ignore_index=True)
        self.valid_img_fns = valid_img_fns

    def run(self):
        for idx, row in self.result.iterrows():
            input_batch = []
            # if not row['key']:
            #     continue

            for img_fn, ann_fn in zip(row['img_fns'], row['ann_fns']):
                img = self._preprocess(img_fn, ann_fn)
                input_batch.append(img)
            pred = self._forward_nn(np.asarray(input_batch))  # select first image from batch size 1
            self._calculate_metrics(idx, pred)

    def run_batch_mode(self, batch_size):
        def batcher(seq, size):
            return ((pos, seq[pos:pos + size]) for pos in range(0, len(seq), size))

        for pos, rows in batcher(self.result, batch_size):
            input_batch = []

            for subpos, row in rows.iterrows():
                img = self._preprocess(row['img_fns'][0], row['ann_fns'][0])
                input_batch.append(img)

            preds = self._forward_nn(np.asarray(input_batch))
            for subpos, pred in enumerate(preds):
                self._calculate_metrics(pos + subpos, pred)

    def _load_img(self, fn):
        img = tf.io.read_file(fn)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, (1920,  1080))

    def _load_ann(self, fn):
        ann = tf.io.read_file(fn)
        ann = tf.image.decode_jpeg(ann, channels=1)
        ann = tf.image.convert_image_dtype(ann, tf.float32)
        return ann

    def _get_filepath_from_same_polyp(self, file_path):
        return random.choice(self.valid_img_fns)

    def _preprocess(self, img_file_path: tf.Tensor, ann_file_path: tf.Tensor):
        imgs = []
        for i in range(self.nb_inputs):
            img = self._load_img(img_file_path)
            ann = self._load_ann(ann_file_path)
            ann = tf.image.resize(ann, img.shape[:2])

            padding = 0.2
            img = tf.convert_to_tensor(crop_to_bbox(img.numpy(), ann.numpy(), padding=padding, output_size=self.input_size))
            img.set_shape(self.input_size + (3,))
            # img = tf.keras.applications.vgg16.preprocess_input((255 * img))
            img *= 255
            imgs.append(img)

            img_file_path = self._get_filepath_from_same_polyp(img_file_path)
            ann_file_path = img_file_path.replace(img_dir, ann_dir).replace('.jpg', '.png')

        if self.nb_inputs > 1:
            imgs = tf.convert_to_tensor(imgs)
            return imgs
        else:
            return tf.convert_to_tensor(imgs[0])

    def _forward_nn(self, frame):
        predictions = [m.predict(frame) for m in self.model]
        return predictions

    def _calculate_metrics(self, idx, pred):
        pred_vote = np.zeros_like(range(self.n_classes), dtype=np.float64)
        pred_vote_count = np.zeros_like(range(self.n_classes), dtype=np.float64)
        pred_vote_th = np.zeros_like(pred_vote, dtype=np.float64)
        pred_vote_count_th = np.zeros_like(range(self.n_classes), dtype=np.float64)
        # TODO: Fix vote for ssp's
        print(self.model)
        for k, prediction in enumerate(pred):
            prediction = prediction[0]

            v = np.argmax(prediction)
            conf = prediction[v]
            pred_vote += prediction[self.labels[k]]
            pred_vote_count[self.labels[k] == v] += 1.0/float(np.count_nonzero(self.labels[k] == v))
            # print(pred_vote, pred_vote_count)
            self.result.loc[self.result.index[idx], self.types[k] + '_pred'] = prediction
            self.result.loc[self.result.index[idx], self.types[k] + '_pred_class'] = np.argmax(self.labels[k] == v)
            self.result.loc[self.result.index[idx], self.types[k] + '_confidence'] = conf
            if conf > self.th:
                pred_vote_th += prediction[self.labels[k]]
                pred_vote_count_th[self.labels[k] == v] += 1.0/float(np.count_nonzero(self.labels[k] == v))
        pred_vote /= np.sum(pred_vote)
        s = np.sum(pred_vote_th)
        if s > 0:
            pred_vote_th /= s

        self.result.loc[self.result.index[idx], 'pred'] = pred_vote
        self.result.loc[self.result.index[idx], 'pred_class'] = np.argmax(pred_vote_count)
        self.result.loc[self.result.index[idx], 'gt_class'] = self.label
        self.result.loc[self.result.index[idx], 'match'] = int(np.argmax(pred_vote_count)) == int(self.label)
        self.result.loc[self.result.index[idx], 'confidence'] = pred_vote[np.argmax(pred_vote_count)]

        self.result.loc[self.result.index[idx], 'pred_th'] = pred_vote_th
        self.result.loc[self.result.index[idx], 'pred_class_th'] = np.argmax(pred_vote_count_th)
        self.result.loc[self.result.index[idx], 'match_th'] = int(np.argmax(pred_vote_th)) == int(self.label)
        self.result.loc[self.result.index[idx], 'confidence_th'] = pred_vote_th[np.argmax(pred_vote_count_th)]


def analyse_df(df_fn, types=()):
    df = pd.read_pickle(df_fn)
    df = df.reset_index()

    #TODO: for 3-way confidence, change high conf value and starting value of
    preds = [t+'_pred' for t in types]
    pred_classes = [t+'_pred_class' for t in types]

    for i, pred_class in enumerate(pred_classes):
        print("Problem {}".format(types[i]))
        for confidence in np.arange(0.3, 1.0, 0.05):
            df_high_conf = df.loc[df[preds[i]].apply(np.max) > confidence]

            tp = ((df_high_conf['gt_class'] == 1) & (df_high_conf[pred_class] == 1)).sum()
            fp = ((df_high_conf['gt_class'] == 0) & (df_high_conf[pred_class] == 1)).sum()
            fn = ((df_high_conf['gt_class'] == 1) & (df_high_conf[pred_class] == 0)).sum()
            tn = ((df_high_conf['gt_class'] == 0) & (df_high_conf[pred_class] == 0)).sum()
            print(tp, fp, fn, tn)
            print("At {} % confidence with {} % included".format(100*confidence, 100*len(df_high_conf)/len(df)))
            print("Accuracy    - ", (tp+tn) / len(df_high_conf))
            print("Sensitivity - ", tp / (tp + fn))
            print("Specificity - ", tn / (tn + fp))
            print("Precision   - ", tp / (tp + fp))

    confidence_threshold = 0.45
    names = {0: "hyp", 1: "adn", 2: "ssp"}

    print("### Per frame analysis ###")

    print("Accuracy ", df['match'].sum()/len(df))
    for confidence in np.arange(0.3, 1.0, 0.05):
        df_high_conf = df.loc[df['confidence'] > confidence]
        if len(df_high_conf) != 0:
            print(df_high_conf['match'].sum() / len(df_high_conf), "at {} % confidence with {} % included".format(100*confidence, 100*len(df_high_conf)/len(df)))

    if 'hc_match' in df.keys():
        print("Accuracy ", df['hc_match'].sum() / len(df))
        for confidence in np.arange(0.3, 1.0, 0.05):
            df_high_conf = df.loc[df['confidence'] > confidence]
            if len(df_high_conf) != 0:
                print(df_high_conf['hc_match'].sum() / len(df_high_conf),
                      "at {} % confidence with {} % included".format(100 * confidence,
                                                                     100 * len(df_high_conf) / len(df)))





    print("### Per class analysis ###")

    print("Confusion matrix for whole dataset", "\n")
    sums = df.groupby(['gt_class']).size()
    cm1 = df.groupby(['gt_class', 'pred_class']).size()

    print(cm1.unstack(fill_value=0).rename(columns=names, index=names), "\n")
    print("Normalized based on gt", "\n")
    print((cm1/sums).unstack(fill_value=0).rename(columns=names, index=names), "\n")

    df_high_conf = df.loc[df['confidence'] > confidence_threshold]
    print(f"Confusion matrix for high confidence (confidence>{confidence_threshold}) dataset")
    sums = df_high_conf.groupby(['gt_class']).size()
    cm1 = df_high_conf.groupby(['gt_class', 'pred_class']).size()

    print(cm1.unstack(fill_value=0).rename(columns=names, index=names), "\n")
    print("Normalized based on gt")
    print((cm1 / sums).unstack(fill_value=0).rename(columns=names, index=names), "\n")

    lights = ['WLI', 'BLI', 'LCI']
    for i in range(3):
        df_light = df.loc[df['light'] == i]
        print(f"Confusion matrix for light modality {lights[i]}")
        sums = df_light.groupby(['gt_class']).size()
        cm1 = df_light.groupby(['gt_class', 'pred_class']).size()

        print(cm1.unstack(fill_value=0).rename(columns=names, index=names), "\n")
        print("Normalized based on gt")
        print((cm1 / sums).unstack(fill_value=0).rename(columns=names, index=names), "\n")

    if 'hc_vote' in df.keys():
        print('## High confidence voting ##', "\n")

        print("### Per class analysis ###")

        print("Confusion matrix for whole dataset", "\n")
        sums = df.groupby(['gt_class']).size()
        cm1 = df.groupby(['gt_class', 'hc_vote']).size()

        print(cm1.unstack(fill_value=0).rename(columns=names, index=names), "\n")
        print("Normalized based on gt", "\n")
        print((cm1 / sums).unstack(fill_value=0).rename(columns=names, index=names), "\n")

        df_high_conf = df.loc[df['confidence'] > confidence_threshold]
        print(f"Confusion matrix for high confidence (confidence>{confidence_threshold}) dataset")
        sums = df_high_conf.groupby(['gt_class']).size()
        cm1 = df_high_conf.groupby(['gt_class', 'hc_vote']).size()

        print(cm1.unstack(fill_value=0).rename(columns=names, index=names), "\n")
        print("Normalized based on gt")
        print((cm1 / sums).unstack(fill_value=0).rename(columns=names, index=names), "\n")

        for i in range(3):
            df_light = df.loc[df['light'] == i]
            print(f"Confusion matrix for light modality {lights[i]}")
            sums = df_light.groupby(['gt_class']).size()
            cm1 = df_light.groupby(['gt_class', 'hc_vote']).size()

            print(cm1.unstack(fill_value=0).rename(columns=names, index=names), "\n")
            print("Normalized based on gt")
            print((cm1 / sums).unstack(fill_value=0).rename(columns=names, index=names), "\n")


    print("### Per polyp analysis ###")
    # df = df.reset_index()
    # print(df.groupby('polyp_id')['match'].sum() / df.groupby('polyp_id')['match'].count())
    print("sensitivity ", ((df.groupby('polyp_id')['match'].sum() / df.groupby('polyp_id')['match'].count()) > 0.5).sum() / df['polyp_id'].nunique())
    # print("sensitivity for consistent cases ", ((df.groupby('polyp_id')['match'].sum() / df.groupby('polyp_id')['match'].count()) > 0.7 + (df.groupby('polyp_id')['match'].sum() / df.groupby('polyp_id')['match'].count()) < 0.3).sum())
    # print(((df.groupby('polyp_id')['match'].sum() / df.groupby('polyp_id')['match'].count()) > 0.7).sum())
    # print((((df.groupby('polyp_id')['match'].sum() / df.groupby('polyp_id')['match'].count()) > 0.7).sum() + ((df.groupby('polyp_id')['match'].sum() / df.groupby('polyp_id')['match'].count()) < 0.3).sum()))
    # print(df['polyp_id'].nunique())

    df_high_conf = df.loc[df['confidence'] > confidence_threshold]
    if len(df_high_conf) > 0:
        # print(df_high_conf.groupby("polyp_id")['match'].sum())
        # print(df.loc[df['confidence'] > 0.7].groupby('polyp_id')['match'].sum(), df.loc[df['confidence'] > 0.7].groupby('polyp_id')['match'].count())
        # print('final', df_high_conf.groupby("polyp_id").filter(lambda x: len(x) >= 10).groupby('polyp_id')['match'].sum() / df_high_conf.groupby("polyp_id").filter(lambda x: len(x) >= 10).groupby('polyp_id')['match'].count())
        # print(df_high_conf.groupby("polyp_id").filter(lambda x: len(x) < 10))
        df_high_conf_long = df_high_conf.groupby("polyp_id").filter(lambda x: len(x) >= 10)
        print("sensitivity high conf", ((df_high_conf_long.groupby('polyp_id')['match'].sum() / df_high_conf_long.groupby('polyp_id')['match'].count()) > 0.5).sum() / df_high_conf_long['polyp_id'].nunique())

    print("### Per seq analysis ###")
    # print(df.groupby('polyp_id')['match'].sum() / df.groupby('polyp_id')['match'].count())
    print("sensitivity ", ((df.groupby('seq')['match'].sum() / df.groupby('seq')['match'].count()) > 0.5).sum() / df['seq'].nunique())
    # print("sensitivity for consistent cases ", ((df.groupby('seq')['match'].sum() / df.groupby('seq')['match'].count()) > 0.7 + (df.groupby('seq')['match'].sum() / df.groupby('seq')['match'].count()) < 0.3).sum())
    # print(((df.groupby('seq')['match'].sum() / df.groupby('seq')['match'].count()) > 0.7).sum())
    # print((((df.groupby('seq')['match'].sum() / df.groupby('seq')['match'].count()) > 0.7).sum() + ((df.groupby('seq')['match'].sum() / df.groupby('seq')['match'].count()) < 0.3).sum()))
    # print(df['seq'].nunique())

    df_high_conf = df.loc[df['confidence'] > confidence_threshold]
    if len(df_high_conf) > 0:
        # print(df_high_conf.groupby("seq")['match'].sum())
        # print(df.loc[df['confidence'] > 0.7].groupby('seq')['match'].sum(), df.loc[df['confidence'] > 0.7].groupby('seq')['match'].count())
        # print('final', df_high_conf.groupby("seq").filter(lambda x: len(x) >= 10).groupby('seq')['match'].sum() / df_high_conf.groupby("seq").filter(lambda x: len(x) >= 10).groupby('seq')['match'].count())
        # print(df_high_conf.groupby("seq").filter(lambda x: len(x) < 10))
        df_high_conf_long = df_high_conf.groupby("seq").filter(lambda x: len(x) >= 10)
        print("sensitivity high conf", ((df_high_conf_long.groupby('seq')['match'].sum() / df_high_conf_long.groupby('seq')['match'].count()) > 0.5).sum() / df_high_conf_long['seq'].nunique())

    print("### Per polyp exp smoothed analysis ###")
    df_final_conf = df.groupby('polyp_id').tail(1)
    for confidence in np.arange(0.3, 1.0, 0.05):
        df_final_conf_high = df_final_conf.loc[df_final_conf['cum_exp_smooth'].apply(np.max) > confidence]
        if len(df_final_conf_high) > 0:
            print("High confidence accuracy {} % at {} % included".format((df_final_conf_high['cum_exp_smooth'].apply(np.argmax) == df_final_conf_high['gt_class'].astype(int)).sum() / (df_final_conf_high['cum_exp_smooth'].apply(np.argmax) == df_final_conf_high['gt_class'].astype(int)).count(), len(df_final_conf_high) / len(df_final_conf)))

    print("### Per seq exp smoothed analysis ###")
    df_final_conf = df.groupby('seq').tail(1)
    for confidence in np.arange(0.3, 1.0, 0.05):
        df_final_conf_high = df_final_conf.loc[df_final_conf['cum_exp_smooth'].apply(np.max) > confidence]
        if len(df_final_conf_high) > 0:
            print("High confidence accuracy {} % at {} % included".format((df_final_conf_high['cum_exp_smooth'].apply(np.argmax) == df_final_conf_high['gt_class'].astype(int)).sum() / (df_final_conf_high['cum_exp_smooth'].apply(np.argmax) == df_final_conf_high['gt_class'].astype(int)).count(), len(df_final_conf_high) / len(df_final_conf)))


def create_video(df_fn, voting_types=()):
    df = pd.read_pickle(df_fn)
    # print(list(df['hypVSadnVSssp_pred']))
    video_folder = 'data/inferenced/vote_ensembles'
    # TODO: Add blue color for ssp and add extra frame to compare th vote and normal vote

    os.makedirs(video_folder, exist_ok=True)


    cls_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]


    for polyp_id, group in tqdm(df.groupby('polyp_id'), desc="Creating videos"):
        gt_class = int(group.iloc[0]['gt_class'])
        for seq in group['seqs'].iloc[0]:
            pred_class = int(np.argmax(group.loc[group['seq'] == seq].iloc[-1]['cum_exp_smooth_th']))
            confidence = int(100 * np.max(group.loc[group['seq'] == seq].iloc[-1]['cum_exp_smooth_th']))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_shape = cv2.imread(group.iloc[0]['img_fns'][0]).shape[:2][::-1]
            writer = cv2.VideoWriter(os.path.join(video_folder, str(seq) + '_{}_{}_{}.mp4'.format(gt_class, pred_class, confidence)), fourcc, 25., video_shape)

            cum_exp_pred = [0.33, 0.33, 0.33]

            img_fn_df = group['img_fns'].apply(itemgetter(0)).astype(str)
            for frame_idx in trange(get_seq_length(seq), leave=False, desc=seq):
                frame_metadata = get_filename(seq, frame_idx, return_type='both', check_valid=True)

                frame_fn = frame_metadata['img_fn']
                ann_fn = frame_metadata['ann_fn']

                exp_pred = group.loc[img_fn_df == frame_fn]['exp_smooth'].values[0] if len(group.loc[img_fn_df == frame_fn]['exp_smooth'].values) > 0 else None
                cum_exp_pred = group.loc[img_fn_df == frame_fn]['cum_exp_smooth'].values[0] if len(group.loc[img_fn_df == frame_fn]['cum_exp_smooth'].values) > 0 else cum_exp_pred

                exp_pred_th = group.loc[img_fn_df == frame_fn]['exp_smooth_th'].values[0] if len(
                    group.loc[img_fn_df == frame_fn]['exp_smooth_th'].values) > 0 else None
                cum_exp_pred_th = group.loc[img_fn_df == frame_fn]['cum_exp_smooth_th'].values[0] if len(
                    group.loc[img_fn_df == frame_fn]['cum_exp_smooth_th'].values) > 0 else cum_exp_pred

                cls = np.argmax(cum_exp_pred)
                cls_th = np.argmax(cum_exp_pred_th)

                frame = cv2.imread(frame_fn)


                if frame_metadata['valid']:
                    # Draw contours of annotation
                    mask = cv2.imread(ann_fn, 0)
                    mask = cv2.resize(mask, frame.shape[:2][::-1])
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        bbox = cv2.boundingRect(cnt)

                        frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), cls_colors[cls], 5)


                        frame = cv2.rectangle(frame, (int(bbox[0]-15), int(bbox[1]-15)),
                                              (int(bbox[0] + bbox[2]+15), int(bbox[1] + bbox[3]+15)), cls_colors[cls_th], 5)
                        frame = cv2.putText(frame, str('{:.2f}'.format(np.max(cum_exp_pred))),
                                            (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                            color=(0, 0, 0), thickness=5)
                        frame = cv2.putText(frame, str('{:.2f}'.format(np.max(cum_exp_pred))),
                                            (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                            color=cls_colors[cls], thickness=2)
                        frame = cv2.putText(frame, str('{:.2f}'.format(np.max(cum_exp_pred_th))),
                                            (int(bbox[0] + 80), int(bbox[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX,
                                            fontScale=1,
                                            color=(0, 0, 0), thickness=5)
                        frame = cv2.putText(frame, str('{:.2f}'.format(np.max(cum_exp_pred_th))),
                                            (int(bbox[0]+80), int(bbox[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                            color=cls_colors[cls_th], thickness=2)

                frame = cv2.putText(frame, 'GT -   {}'.format(gt_class), (int(video_shape[0]*0.67), int(video_shape[1]*0.9-80)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=cls_colors[gt_class], thickness=2)
                if exp_pred is not None:
                    frame = cv2.putText(frame, 'pred - {}'.format(['{:.2f}'.format(p) for p in exp_pred]), (int(video_shape[0]*0.67), int(video_shape[1]*0.9-40)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=cls_colors[np.argmax(exp_pred)], thickness=2)
                frame = cv2.putText(frame, 'cum pred - {}'.format(['{:.2f}'.format(p) for p in cum_exp_pred]), (int(video_shape[0]*0.67), int(video_shape[1]*0.9)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=cls_colors[cls], thickness=2)

                if exp_pred_th is not None:
                    frame = cv2.putText(frame, 'pred th - {}'.format(['{:.2f}'.format(p) for p in exp_pred_th]), (int(video_shape[0]*0.67), int(video_shape[1]*0.9+40)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=cls_colors[np.argmax(exp_pred_th)], thickness=2)
                frame = cv2.putText(frame, 'cum pred th - {}'.format(['{:.2f}'.format(p) for p in cum_exp_pred_th]), (int(video_shape[0]*0.67), int(video_shape[1]*0.9+80)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=cls_colors[cls_th], thickness=2)

                if len(voting_types) > 0:
                    for j, t in enumerate(voting_types):
                        pred = group.loc[img_fn_df == frame_fn][t + '_pred'].values[0] if len(group.loc[img_fn_df == frame_fn][t + '_pred'].values) > 0 else None
                        cls = group.loc[img_fn_df == frame_fn][t + '_pred_class'].values[0] if len(group.loc[img_fn_df == frame_fn][t + '_pred_class'].values) > 0 else None
                        if pred is not None:
                            frame = cv2.putText(frame, '{} pred - {}'.format(t,['{:.2f}'.format(p) for p in pred]),
                                                (int(video_shape[0] * 0.05), int(video_shape[1]- 10 - 20*j)),
                                                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                                                color=cls_colors[cls], thickness=1)

                writer.write(frame)
        writer.release()


def main(model_names, types, selected_labels=('0', '1'), output_fn="logs/output"):
    sequences_fn = imageset_dir.replace('.txt', '_sequences.txt')
    assert os.path.exists(sequences_fn), sequences_fn

    all_cls = []
    sls = []

    for selected_label in selected_labels:
        sl = {}
        for cls, i in enumerate(selected_label):
            for j in i:
                if j not in all_cls:
                    all_cls.append(j)
                sl[j] = cls
        sls.append(sl)


    # Read all sequences from file
    with open(sequences_fn) as f:
        content = f.readlines()

    # Read all sequences from file
    with open('data/imagesets_1080p/total_BLI_all_sequences.txt') as f:
        bli = set([x.strip().split(' ')[0] for x in f.readlines()])
    # Read all sequences from file
    with open('data/imagesets_1080p/total_WLI_all_sequences.txt') as f:
        wli = set([x.strip().split(' ')[0] for x in f.readlines()])
    # Read all sequences from file
    with open('data/imagesets_1080p/total_LCI_all_sequences.txt') as f:
        lci = set([x.strip().split(' ')[0] for x in f.readlines()])
    sequences = [x.strip().split(' ')[0] for x in content]
    labels = [x.strip().split(' ')[1] for x in content]
    polyp_ids = [int(x.strip().split(' ')[2]) for x in content]

    models = [tf.keras.models.load_model(os.path.join('data/snapshots/', n + '_best.hdf5'), {'DropoutCategoricalCrossEntropy': DropoutCategoricalCrossEntropy(float(n.split('_')[-2].replace('lossDropout', '')))}) for n in model_names]#, {'DaubWaveLayer2D':WaveTFFactory.build('db2')})
    result = None

    for polyp_id in sorted(set(polyp_ids)):
        polyp_seqs = [seq for idx, seq in enumerate(sequences) if polyp_ids[idx] == polyp_id]
        label = labels[polyp_ids.index(polyp_id)]
        if label not in all_cls:
            continue
        print("Processing polyp {} with label {} and sequences {}.".format(polyp_id, label, polyp_seqs))
        try:
            evaluator = Evaluator(models, polyp_seqs, label, polyp_id, types,  n_classes=len(all_cls), labels=sls, th=0.6)
            if args.batch_size > 1:
                evaluator.run_batch_mode(args.batch_size)
            else:
                evaluator.run()
            evaluator.result['polyp_id'] = polyp_id
            evaluator.result = evaluator.result.set_index('polyp_id')
            if result is not None:
                result = pd.concat([result, evaluator.result])
            else:
                result = evaluator.result

        except ValueError as e:
            print(f"Something went wrong: {e}")  # sequence not complete

        # reset recurrent network state before starting new sequence
        for m in evaluator.model:
            m.reset_states()
        print(evaluator.result)




    # Get sequence column
    result = result.reset_index()
    result['seq'] = ""
    for i, row in result.iterrows():
        result.at[i, 'seq'] = result.at[i, 'img_fns'][0].split('/')[-2]
        if result.at[i, 'seq'] in wli:
            result.at[i, 'light'] = 0
        elif result.at[i, 'seq'] in bli:
            result.at[i, 'light'] = 1
        elif result.at[i, 'seq'] in lci:
            result.at[i, 'light'] = 2
        else:
            result.at[i, 'light'] = 0
    # Calculate exponential smoothing and cumulative exp smoothed value
    result['exp_smooth'] = np.nan
    result['exp_smooth'] = result['exp_smooth'].astype(object)
    result['cum_exp_smooth'] = np.nan
    result['cum_exp_smooth'] = result['cum_exp_smooth'].astype(object)

    result['exp_smooth_th'] = np.nan
    result['exp_smooth_th'] = result['exp_smooth_th'].astype(object)
    result['cum_exp_smooth_th'] = np.nan
    result['cum_exp_smooth_th'] = result['cum_exp_smooth_th'].astype(object)

    for i, row in result.iterrows():
        if i == 0 or result.at[i, 'seq'] != result.at[i - 1, 'seq']:
            # new sequence
            result.at[i, 'exp_smooth'] = result.at[i, 'pred']
            result.at[i, 'cum_exp_smooth'] = result.at[i, 'pred']
            # new sequence
            result.at[i, 'exp_smooth_th'] = result.at[i, 'pred_th']
            result.at[i, 'cum_exp_smooth_th'] = result.at[i, 'pred_th']
            continue
        result.at[i, 'exp_smooth'] = exp_alpha * result.at[i, 'pred'] + (1 - exp_alpha) * result.at[i - 1, 'exp_smooth']
        result.at[i, 'cum_exp_smooth'] = result.at[i, 'exp_smooth'] if np.max(result.at[i, 'exp_smooth']) > np.max(result.at[i - 1, 'cum_exp_smooth']) else result.at[i - 1, 'cum_exp_smooth']
        result.at[i, 'exp_smooth_th'] = exp_alpha * result.at[i, 'pred_th'] + (1 - exp_alpha) * result.at[i - 1, 'exp_smooth_th']
        result.at[i, 'cum_exp_smooth_th'] = result.at[i, 'exp_smooth_th'] if np.max(result.at[i, 'exp_smooth_th']) > np.max(
            result.at[i - 1, 'cum_exp_smooth_th']) else result.at[i - 1, 'cum_exp_smooth_th']

    result.to_excel(os.path.join(output_fn+".xlsx"), columns=results_struct)
    result.to_pickle(os.path.join(output_fn+ ".pkl"))


def eval_tfdataset():
    model = tf.keras.models.load_model(os.path.join('data/snapshots/', model_name + '_best.hdf5'))
    model.summary()

    nb_inputs = int(re.search(r'\d{1,}in', model_name).group()[:-2])
    input_size = tuple([int(x) for x in re.search(r'_\d{1,}x\d{1,}_', model_name).group()[1:-1].split('x')])
    ds = Dataset(imageset_dir, batch_size=args.batch_size, target_size=input_size, loading_size=loading_size, augmentations=None, nb_inputs=nb_inputs, shuffle=True, repeat=True, do_cropping=True, balanced=args.balanced, selected_labels=('0', '1'))

    result = model.evaluate(ds.tfdataset, steps=150)
    print(dict(zip(model.metrics_names, result)))


model_names = ['hypVSall_HDall2022_efficientnet_256x256_1in_nf64_bnTrue_fcdo0.5_convdo0.0_lossls0.1_lossDropout0.75_fold0',
               'adnVSall_HDall2022_efficientnet_256x256_1in_nf64_bnTrue_fcdo0.5_convdo0.0_lossls0.1_lossDropout0.75_fold0',
               'sspVSall_HDall2022_efficientnet_256x256_1in_nf64_bnTrue_fcdo0.5_convdo0.0_lossls0.1_lossDropout0.75_fold0']

selected_labels = [
    # ("0","1","2"),
    # ("0","1","2"),
    # ("0","1","2"),
    # ("0","1","2"),
    ("0",("1","2")),
    ("1",("0","2")),
    ("2",("0","1")),
    # ("0","1"),
    # ("1","2"),
    # ("0","2")
]


def vote(df, types, labels, nb_classes=3, confidence_level=None, do_exp_smooth=False):

    df = df.copy()

    pc = 'pred_class_'
    conf = 'confidence_'
    hc = 'hc_'
    vc = 'vote_cls'

    votes = [vc+str(i) for i in range(nb_classes)]
    df[votes] = 0.

    if confidence_level is not None:
        hc_votes = [hc + vc + str(i) for i in range(nb_classes)]
        df[hc_votes] = 0.

    list_of_confidences = []

    for i, t in enumerate(types):
        list_of_confidences.append(conf+t)

        for j, label in enumerate(labels[i]):
            for label_cls in label:
                df.loc[df[pc + t] == j, vc+label_cls] = df.loc[df[pc + t] == j, vc+label_cls] + 1.0/len(label)

                if confidence_level is not None:
                    df.loc[(df[pc + t] == j) & (df[conf+t] > confidence_level), hc+vc + label_cls] = \
                        df.loc[(df[pc + t] == j) & (df[conf+t] > confidence_level), hc+vc + label_cls] \
                        + 1.0 / len(label)

    df['vote_confidence'] = df[list_of_confidences].mean(axis=1, skipna=True)

    df['vote'] = df[votes].copy().idxmax(axis=1).apply(lambda x: int(list(x)[-1]))

    df['match'] = 0
    df.loc[df['vote'] == df['gt_class'], 'match'] = 1

    if confidence_level is not None:
        df[hc+'vote'] = df[hc_votes].copy().idxmax(axis=1).apply(lambda x: int(list(x)[-1]))
        df[hc+'match'] = 0

        df.loc[df[hc+'vote'] == df['gt_class'], hc+'match'] = 1

    if do_exp_smooth:
        es = 'vote_exp_smooth'
        ces = 'vote_cum_exp_smooth'

        preds = [t + '_pred' for t in types]

        prediction = np.zeros_like(range(3))
        for i, row in df.iterrows():
            for j, pred in enumerate(df.loc[i, preds]):
                for k, label in enumerate(labels[j]):
                    for label_cls in label:
                        prediction[int(label_cls)] += pred[k] / len(label)
            df.at[i, 'pred'] = np.max(prediction)

            if i == 0 or df.at[i, 'seq'] != df.at[i - 1, 'seq']:
                # new sequence
                df.at[i, es] = df.at[i, 'pred']
                df.at[i, ces] = df.at[i, 'pred']
                continue
            df.at[i, es] = exp_alpha * df.at[i, 'pred'] + (1 - exp_alpha) * df.at[i - 1, es]
            df.at[i, ces] = df.at[i, es] if np.max(df.at[i, es]) > np.max(df.at[i - 1, ces]) else df.at[i - 1, ces]

    return df




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model_name", action="store", default="hypVSadn_HDall_darknet_256x256_miclass_1in_nf32_bnTrue_fcdo0.5_convdo0.0_lossls0.1", dest="model_name")
    parser.add_argument("-ds", "--dataset", action="store", default="polyps", dest="dataset")
    parser.add_argument("--balanced", action="store_true", dest="balanced", default=True)
    parser.add_argument("--batch_size", action="store", type=int, default=64, dest="batch_size")

    args = parser.parse_args()

    assert args.dataset in ['catsndogs', 'polyps', 'cvc']

    if args.dataset == 'catsndogs':
        from utils.Dataset_catsndogs import Dataset
        labels = {0: 'cat', 1: 'dog'}
        imageset_dir = 'data/catsndogs/val/'
        loading_size = None
    elif args.dataset == 'polyps':
        from utils.Dataset import Dataset
        labels = {0: 'HP', 1: 'AD', 2: 'SSA'}
        imageset_dir = 'data/imagesets_1080p/5bootstrap_all/test.txt'
        # imageset_dir = 'data/imagesets_1080p/all_split_0.1.txt'

        loading_size = image_size
    elif args.dataset == 'cvc':
        from utils.Dataset_cvc import Dataset
        labels = {0: 'ADN', 1: 'nonADN'}
        imageset_dir = 'data/cvc/val/'
        loading_size = image_size

    types = []
    extended_cols = []

    for i, model_name in enumerate(model_names):
        print("######################################")
        print(model_name)
        print("######################################")
        types.append(model_name.split('_')[0])
        # eval_tfdataset()
        extended_cols.append(types[-1]+'_pred')
        extended_cols.append(types[-1] + '_pred_class')
        extended_cols.append(types[-1]+'_confidence')

    results_struct += extended_cols
    # model_name = 'mm_hypVSadn_HDall_efficientnet_256x256_miclass_1in_nf256_bnTrue_fcdo0.5_convdo0.2_lossls0.1'
    # eval_tfdataset()
    # main(model_names, types, selected_labels=selected_labels, output_fn="logs/voting_fold_0")
    # analyse_df(os.path.join("hypVSall_HDall2022_efficientnet_256x256_1in_nf64_bnTrue_fcdo0.5_convdo0.0_lossls0.1_lossDropout0.75_5fold_voted.pkl"), types)
    create_video(os.path.join("results/vote_simple/hypVSall_adnVSall_sspVSall_HDall2022_efficientnet_regularized0.0_256x256_1in_nf64_bnTrue_fcdo0.5_lossls0.1_balancedTrue_h5_5fold4_vote.pkl"), [])
    # hm_creator = HeatmapCreator(tf.keras.models.load_model(os.path.join('data/snapshots/hypVSadn_HDall_efficientnet_256x256_miclass_1in_nf256_bnTrue_fcdo0.5_convdo0.2_lossls0.1_best.hdf5')),
    #                             "data/imagesets_1080p/6bootstrap_all/test_sequences.txt.txt",
    #                             "data/hm/hm_creator",
    #                             (0,1))
    # hm_creator.create_videos()

