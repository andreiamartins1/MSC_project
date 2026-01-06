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
import seaborn as sns

import tensorflow as tf

from utils.Dataset import crop_to_bbox
from utils.losses import DropoutCategoricalCrossEntropy
from utils.networks import vote_net_same_base

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.compat.v1.Session(config=config)

img_dir = '/DATASERVER/MIC/SHARED/ENDOSCOPY/EUROPOL/JPEGImages/1080p/'
ann_dir = '/DATASERVER/MIC/SHARED/ENDOSCOPY/EUROPOL/Annotations/480p/'

# dsc = '09'

# validity_fn = f'/frames_to_use_dsc{dsc}.pickle'
# if dsc == '05':
validity_fn = f'/frames_to_use.pickle'

image_size = (int(1920), int(1080))
BATCH_SIZE = 64
exp_alpha = 0.1

import datetime
my_date = datetime.date.today() # if date is 01/01/2018
year, week_num, day_of_week = my_date.isocalendar()
# logdir = f'logs/{year}_week{week_num}/'
logdir = 'results'
os.makedirs(logdir, exist_ok=True)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

#POLYPS_TO_CLASSIFY = ("0", "1", "2")
POLYPS_TO_CLASSIFY = ("0", "1")

def get_seq_length(seq, correct_seq=False):
    '''
    Returns the number of frames of the given seq
    '''
    if correct_seq:
        return NotImplementedError('automatic sequence name correction is not yet implemented.')

    if not os.path.exists(ann_dir + seq + validity_fn):
        raise ValueError("frames_to_use does not exist.")
    validity = pickle.load(open(ann_dir + seq + validity_fn, 'rb'))
    return len(validity)

def get_seq_harvard(seq, correct_seq=False):
    '''
    Returns the number of frames of the given seq
    '''
    if correct_seq:
        return NotImplementedError('automatic sequence name correction is not yet implemented.')

    if not os.path.exists(ann_dir + seq):
        raise ValueError("sequence does not exist.")
    filenames = next(os.walk(ann_dir + seq), (None, None, []))[2]
    return [k.split(".")[0] for k in filenames]


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
    # if args.dataset == 'harvard_polyps':
    #     img_fn = img_dir + seq + "/" + str(idx) + ".jpg"
    #     ann_fn = ann_dir + seq + "/" + str(idx) + ".xml"

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
        # if args.dataset != 'harvard_polyps':
        if not os.path.exists(ann_dir + seq + validity_fn):
            print(ann_dir + seq + validity_fn)
            raise ValueError("frames_to_use does not exist.")
        result['valid'] = pickle.load(open(ann_dir + seq + validity_fn, 'rb'))[idx]
        # else:
        #     result['valid'] = True
    if check_keyness:
        # Check if key frame
        # if args.dataset != 'harvard_polyps':
        candidates_pattern = glob.glob(img_dir + seq + "/00000_ann*%05d" % (idx + 1) + ".jpg")
        result['key'] = len(candidates_pattern) > 0
        # else:
        #     result['key'] = False
    return result


class Evaluator(object):
    def __init__(self, models, seqs, label, polyp_id, n_samples=20, do_cropping='preloaded', selected_label=None, mn=None):
        self.models = models
        self.seqs = seqs

        self.label = label

        self.translated_label = -1

        if selected_label is not None:
            for l, lbl in enumerate(selected_label):
                if label in lbl:
                    self.translated_label = int(l)

        self.polyp_id = polyp_id
        self.n_samples = n_samples
        self.do_cropping = 'preloaded'
        results_struct = [
                'seq',
                'seqs',  # the seq names that exist for this polyp
                'img_fns',  # list of N original image filenames
                'ann_fns',  # list of N image annotations
                'key',  # list of frame keyness
                'gt_class',
                'exp_smooth',
                'cum_exp_smooth']

        for k in range(len(models)):
            results_struct.append('fold{}_pred'.format(k))
            results_struct.append('fold{}_pred_class'.format(k))
            results_struct.append('fold{}_match'.format(k))

        self.result = pd.DataFrame(columns=results_struct)

        # self.nb_inputs = int(re.search(r'\d{1,}in', mn).group()[:-2])
        self.nb_inputs = 1
        self.input_size = tuple([int(x) for x in re.search(r'_\d{1,}x\d{1,}_', mn).group()[1:-1].split('x')])
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
            # if args.dataset == 'harvard_polyps':
            #     sequences = get_seq_harvard(seq)
            # else:
            sequences = range(get_seq_length(seq))
            for i in sequences:
                try:
                    frame_metadata = get_filename(seq, int(i), return_type='both', check_valid=True, check_keyness=True)
                except AssertionError:
                    continue

                if self.do_cropping == 'preloaded':

                    frame_metadata['img_fn'] = frame_metadata['img_fn'].replace('JPEGImages/1080p/', 'CROPS/256p/')
                    # if args.dataset == 'harvard_polyps':
                    #     frame_metadata['img_fn'] = frame_metadata['img_fn'].replace('Image/', 'CROPS/')


                else:
                    ann = cv2.imread(frame_metadata['ann_fn'], 0)
                    if np.sum(ann) == 0:
                        # print("Skipping - empty annotation file")
                        continue
                    contours, _ = cv2.findContours(ann, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) > 1:
                        # skip frames with multiple bounding boxes
                        continue

                if frame_metadata['valid'] and os.path.exists(frame_metadata['img_fn']):
                    valid_img_fns.append(frame_metadata['img_fn'])
                    valid_ann_fns.append(frame_metadata['ann_fn'])
                    valid_img_keyness.append(frame_metadata['key'])

        # take self.n_samples of self.n_inputs images
        # for idx in range(self.n_samples):
        #     sample_fns, sample_anns, sample_keyness = zip(*random.choices(list(zip(valid_img_fns, valid_ann_fns, valid_img_keyness)), k=self.nb_inputs))
        #     self.result = self.result.append({'seqs': self.seqs, 'img_fns': sample_fns, 'ann_fns': sample_anns, 'key': sample_keyness}, ignore_index=True)

        # self.result = pd.concat([self.result, ])
        for i in range(len(valid_img_fns)):
            self.result = pd.concat([self.result,pd.DataFrame.from_dict({'seqs': [self.seqs], 'img_fns': [[valid_img_fns[i]]], 'ann_fns': [[valid_ann_fns[i]]], 'key': [[valid_img_keyness[i]]]})], ignore_index=True)
        self.valid_img_fns = valid_img_fns

    def run(self):
        input_batch = []
        for idx, row in self.result.iterrows():
            if not row['key']:
                continue

            for img_fn, ann_fn in zip(row['img_fns'], row['ann_fns']):
                img = self._preprocess(img_fn, ann_fn)
                input_batch.append(img)
        if len(input_batch) == 0:
            print("No images to predict")
            return
        preds = self._forward_nn(np.asarray(input_batch))
        self._calculate_metrics(preds)
        # print(self.result)


    def _load_img(self, fn):
        img = tf.io.read_file(fn)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, self.input_size)

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
            img *= 255
            if self.do_cropping == 'preloaded':
                pass
            elif self.do_cropping:
                ann = self._load_ann(ann_file_path)
                ann = tf.image.resize(ann, img.shape[:2])

                padding = 1
                img = tf.convert_to_tensor(crop_to_bbox(img.numpy(), ann.numpy(), padding=padding, output_size=self.input_size))
                img.set_shape(self.input_size + (3,))

            imgs.append(img)

            img_file_path = self._get_filepath_from_same_polyp(img_file_path)
            ann_file_path = img_file_path.replace(img_dir, ann_dir).replace('.jpg', '.png')

        if self.nb_inputs > 1:
            imgs = tf.convert_to_tensor(imgs)
            return imgs
        else:
            return tf.convert_to_tensor(imgs[0])

    def _forward_nn(self, frame):
        result = None
        for i in range(0, len(frame), BATCH_SIZE):
            batch = frame[i:i + BATCH_SIZE]
            # batch = tf.convert_to_tensor(batch, dtype=tf.float32)  # otherwise memory leaks occur when numpy array is fed into model predict
            # pred = [model.predict(batch, batch_size=BATCH_SIZE) for model in self.models]
            pred = np.asarray([model(batch, training=False) for model in self.models])
            if result is None:
                result = pred
            else:
                result = np.append(result, pred, axis=1)
        return result

    def _calculate_metrics(self, preds):
        l = preds.shape[1]
        self.result['gt_class'] = [self.label,] * l
        for k in range(len(self.models)):
            self.result['fold{}_pred'.format(k)] = [list(i) for i in preds[k]]
            self.result['fold{}_pred_class'.format(k)] = self.result['fold{}_pred'.format(k)].apply(lambda x: np.argmax(x))
            self.result['fold{}_match'.format(k)] = self.result['fold{}_pred_class'.format(k)].astype(int) == self.result['gt_class'].astype(int)


        # for idx in range(preds.shape[1]):  # loop over batch dimension
        #     self.result.loc[self.result.index[idx],'gt_class']= self.label
            # for k in range(len(self.models)):
            #     self.result.loc[self.result.index[idx],'fold{}_pred'.format(k)] = preds[k][idx]
            #     self.result.loc[self.result.index[idx], 'fold{}_pred_class'.format(k)] = np.argmax(preds[k][idx])
            #     self.result.loc[self.result.index[idx],'fold{}_match'.format(k)] = int(np.argmax(preds[k][idx])) == int(self.label)

            # self.result.loc[self.result.index[idx]] = row

# TODO: plan standarized evaluation for trained models
def analyse_df(df_fn):
    df = pd.read_pickle(df_fn)
    df['gt_class'] = df['gt_class'].astype(int)
    for k in range(5):
        df['fold{}_pred_class'.format(k)] = df['fold{}_pred_class'.format(k)].astype(int)

    print("### Per frame analysis ###")
    for k in range(5):
        print("FOLD {}".format(k))
        for confidence in np.arange(0.5, 1.0, 0.05):
            df_high_conf = df.loc[df['fold{}_pred'.format(k)].apply(np.max) > confidence]

            tp = ((df_high_conf['gt_class'] == 1) & (df_high_conf['fold{}_pred_class'.format(k)] == 1)).sum()
            fp = ((df_high_conf['gt_class'] == 0) & (df_high_conf['fold{}_pred_class'.format(k)] == 1)).sum()
            fn = ((df_high_conf['gt_class'] == 1) & (df_high_conf['fold{}_pred_class'.format(k)] == 0)).sum()
            tn = ((df_high_conf['gt_class'] == 0) & (df_high_conf['fold{}_pred_class'.format(k)] == 0)).sum()
            print(tp, fp, fn, tn)
            print("At {} % confidence with {} % included".format(100*confidence, 100*len(df_high_conf)/len(df)))
            print("Accuracy    - ", df_high_conf['fold{}_match'.format(k)].sum() / len(df_high_conf))
            print("Sensitivity - ", tp / (tp + fn))
            print("Specificity - ", tn / (tn + fp))
            print("Precision   - ", tp / (tp + fp))

    print("AVERAGE prediction")
    df['gt_class'] = df['gt_class'].astype(int)
    df['avg_pred'] = df[['fold0_pred', 'fold1_pred', 'fold2_pred', 'fold3_pred', 'fold4_pred']].apply(np.mean, axis=1)
    df['avg_pred_class'] = df['avg_pred'].apply(np.argmax)
    df['avg_match'] = df['avg_pred'].apply(np.argmax) == df['gt_class']
    df['avg_TP'] = (df['gt_class'] == 1) & (df['avg_pred_class'] == 1)
    df['avg_FP'] = (df['gt_class'] == 0) & (df['avg_pred_class'] == 1)
    df['avg_FN'] = (df['gt_class'] == 1) & (df['avg_pred_class'] == 0)
    df['avg_TN'] = (df['gt_class'] == 0) & (df['avg_pred_class'] == 0)
    for confidence in np.arange(0.5, 1.0, 0.05):
        df_high_conf = df.loc[df['avg_pred'].apply(np.max) > confidence]
        print("At {} % confidence with {} % included".format(100*confidence, 100*len(df_high_conf)/len(df)))
        print("Accuracy    - ", df_high_conf['avg_match'].sum() / len(df_high_conf))
        print("Sensitivity - ", df_high_conf['avg_TP'].sum() / (df_high_conf['avg_TP'].sum() + df_high_conf['avg_FN'].sum()))
        print("Specificity - ", df_high_conf['avg_TN'].sum() / (df_high_conf['avg_TN'].sum() + df_high_conf['avg_FP'].sum()))
        print("Precision   - ", df_high_conf['avg_TP'].sum() / (df_high_conf['avg_TP'].sum() + df_high_conf['avg_FP'].sum()))
        print("Confusion matrix: ")
        print(df_high_conf['avg_TP'].sum(), " / ", df_high_conf['avg_FP'].sum())
        print("---------------------")
        print(df_high_conf['avg_FN'].sum(), " / ", df_high_conf['avg_TN'].sum())

    print("MAJORITY VOTING prediction")
    df['maj_pred'] = df[['fold0_pred_class', 'fold1_pred_class', 'fold2_pred_class', 'fold3_pred_class', 'fold4_pred_class']].mode(axis=1)
    print("Accuracy: ", (df['maj_pred'] == df['gt_class'].astype(int)).sum() / df['maj_pred'].count())

    print("### Per polyp analysis ###")
    print("sensitivity ", ((df.groupby('polyp_id')['avg_match'].sum() / df.groupby('polyp_id')['avg_match'].count()) > 0.5).sum() / df['polyp_id'].nunique())

    print("sensitivity ", ((df.groupby('polyp_id')['avg_match'].sum() / df.groupby('polyp_id')['avg_match'].count()) > 0.7).sum() / (((df.groupby('polyp_id')['avg_match'].sum() / df.groupby('polyp_id')['avg_match'].count()) > 0.7).sum() + ((df.groupby('polyp_id')['avg_match'].sum() / df.groupby('polyp_id')['avg_match'].count()) < 0.3).sum()))

    for confidence in np.arange(0.5, 1.0, 0.05):
        df_high_conf = df.loc[df['avg_pred'].apply(np.max) > confidence]
        avg_polyp_pred = df_high_conf.groupby('polyp_id')['avg_pred'].apply(np.mean)
        print((avg_polyp_pred.apply(np.argmax) == df_high_conf.groupby('polyp_id')['gt_class'].max()).sum() / (avg_polyp_pred.apply(np.argmax) == df_high_conf.groupby('polyp_id')['gt_class'].max()).count())
    return

    print("# Sensitivity at thresholds")
    for confidence in np.arange(0.5, 1.0, 0.05):
        df_high_conf = df.loc[df['avg_pred'].apply(np.max) > confidence]
        tp = (df_high_conf.groupby('polyp_id')['avg_match'].sum() / df_high_conf.groupby('polyp_id')['avg_match'].count()) > 0.5
        print("sensitivity high conf", ((df_high_conf.groupby('polyp_id')['avg_match'].sum() / df_high_conf.groupby('polyp_id')['avg_match'].count()) > 0.5).sum() / df_high_conf['polyp_id'].nunique(), "with {} % of polyps included.".format(df_high_conf.groupby('polyp_id').ngroups / df.groupby('polyp_id').ngroups))

    print("# Sensitivity at thresholds for sequences longer than 10 frames")
    for confidence in np.arange(0.5, 1.0, 0.05):
        df_high_conf = df.loc[df['avg_pred'].apply(np.max) > confidence]
        df_high_conf_long = df_high_conf.groupby("polyp_id").filter(lambda x: len(x) >= 10)
        print("sensitivity high conf long", ((df_high_conf_long.groupby('polyp_id')['avg_match'].sum() / df_high_conf_long.groupby('polyp_id')['avg_match'].count()) > 0.5).sum() / df_high_conf_long['polyp_id'].nunique(), "with {} % of polyps included.".format(df_high_conf_long.groupby('polyp_id').ngroups / df.groupby('polyp_id').ngroups))

    print("### Per seq analysis ###")
    print("sensitivity ", ((df.groupby('seq')['avg_match'].sum() / df.groupby('seq')['avg_match'].count()) > 0.5).sum() / df['seq'].nunique())
    print("# Sensitivity at thresholds")
    for confidence in np.arange(0.5, 1.0, 0.05):
        df_high_conf = df.loc[df['avg_pred'].apply(np.max) > confidence]
        print("sensitivity high conf", ((df_high_conf.groupby('seq')['avg_match'].sum() / df_high_conf.groupby('seq')['avg_match'].count()) > 0.5).sum() / df_high_conf['seq'].nunique(), "with {} % of seqs included.".format(df_high_conf.groupby('seq').ngroups / df.groupby('seq').ngroups))

    print("# Sensitivity at thresholds for sequences longer than 10 frames")
    for confidence in np.arange(0.5, 1.0, 0.05):
        df_high_conf = df.loc[df['avg_pred'].apply(np.max) > confidence]
        df_high_conf_long = df_high_conf.groupby("seq").filter(lambda x: len(x) >= 10)
        print("sensitivity high conf long", ((df_high_conf_long.groupby('seq')['avg_match'].sum() / df_high_conf_long.groupby('seq')['avg_match'].count()) > 0.5).sum() / df_high_conf_long['seq'].nunique(), "with {} % of seqs included.".format(df_high_conf_long.groupby('seq').ngroups / df.groupby('seq').ngroups))

    # # Calculate exponential smoothing and cumulative exp smoothed value
    df['exp_smooth'] = np.nan
    df['exp_smooth'] = df['exp_smooth'].astype(object)
    df['cum_exp_smooth'] = np.nan
    df['cum_exp_smooth'] = df['cum_exp_smooth'].astype(object)

    for i, row in df.iterrows():
        if i == 0 or df.at[i, 'seq'] != df.at[i - 1, 'seq']:
            # new sequence
            df.at[i, 'exp_smooth'] = exp_alpha * df.at[i, 'avg_pred'] + (1 - exp_alpha) * np.asarray([0.5, 0.5])
            df.at[i, 'cum_exp_smooth'] = df.at[i, 'exp_smooth']
            continue
        df.at[i, 'exp_smooth'] = exp_alpha * df.at[i, 'avg_pred'] + (1 - exp_alpha) * df.at[i - 1, 'exp_smooth']
        df.at[i, 'cum_exp_smooth'] = df.at[i, 'exp_smooth'] if np.max(df.at[i, 'exp_smooth']) > np.max(df.at[i - 1, 'cum_exp_smooth']) else df.at[i - 1, 'cum_exp_smooth']

    print("### Per polyp exp smoothed analysis ###")
    df_final_conf = df.groupby('polyp_id').tail(1)
    for confidence in np.arange(0.5, 1.0, 0.05):
        df_final_conf_high = df_final_conf.loc[df_final_conf['cum_exp_smooth'].apply(np.max) > confidence]
        print("High confidence accuracy {} % at {} % included".format((df_final_conf_high['cum_exp_smooth'].apply(np.argmax) == df_final_conf_high['gt_class'].astype(int)).sum() / (df_final_conf_high['cum_exp_smooth'].apply(np.argmax) == df_final_conf_high['gt_class'].astype(int)).count(), len(df_final_conf_high) / len(df_final_conf)))

    print("### Per seq exp smoothed analysis ###")
    df_final_conf = df.groupby('seq').tail(1)
    for confidence in np.arange(0.5, 1.0, 0.05):
        df_final_conf_high = df_final_conf.loc[df_final_conf['cum_exp_smooth'].apply(np.max) > confidence]
        print("High confidence accuracy {} % at {} % included".format((df_final_conf_high['cum_exp_smooth'].apply(np.argmax) == df_final_conf_high['gt_class'].astype(int)).sum() / (df_final_conf_high['cum_exp_smooth'].apply(np.argmax) == df_final_conf_high['gt_class'].astype(int)).count(), len(df_final_conf_high) / len(df_final_conf)))

    df_final_conf['cum_exp_smooth_pred'] = df_final_conf['cum_exp_smooth'].apply(np.argmax)
    df_final_conf['cum_exp_smooth_conf'] = df_final_conf['cum_exp_smooth'].apply(np.max)
    df_final_conf.to_excel(os.path.join(logdir, args.model_name + "_5{}_finalframe.xlsx".format(split_type)), columns=['seq', 'gt_class', 'cum_exp_smooth', 'cum_exp_smooth_conf', 'cum_exp_smooth_pred'])

    df.to_excel(os.path.join(logdir, args.model_name + "_5{}.xlsx".format(split_type)), columns=results_struct)
    df.to_pickle(os.path.join(logdir, args.model_name + "_5{}.pkl".format(split_type)))


def analyse_df_multiclass(df_fn, nb_classes=2):
    df = pd.read_pickle(df_fn)
    df['gt_class'] = df['gt_class'].astype(int)
    all_pred_classes = [i for i in df.columns if 'fold' in i and 'pred_class' in i]
    if "pred_class" in df.columns:
        all_pred_classes = ["pred_class"]
    df[all_pred_classes] = df[all_pred_classes].astype(int)
    #
    # c = re.compile('.*HDall2022_(.*)_regularized(0\.[0-9]*)_.*_nf([0-9]*).*')
    # m = re.match(c, df_fn)
    # nf = int(m[3])
    # model_name = m[1]

    df_fn = df_fn.replace('.pkl', '')

    names = {0: "hyp", 1: "adn", 2: "ssp"}
    target_names = list(names.items())

    results = {
        'totals': {'pct': [], 'ac':[], "cm":None},
        'cls0': {"cnt_frames":0,"cnt_polyps":0,"cnt_seqs":0,'ac': [], 'sp': [],'se': [],'pr': [], 'npv':[], 'polyps_sp': [], 'polyps_se': [], 'polyps_pr': [], 'polyps_npv':[], 'seq_sp': [], 'seq_se': [], 'seq_pr': [], 'seq_npv':[]},
        'cls1': {"cnt_frames":0,"cnt_polyps":0,"cnt_seqs":0,'ac': [], 'sp': [], 'se': [], 'pr': [], 'npv':[], 'polyps_sp': [], 'polyps_se': [], 'polyps_pr': [], 'polyps_npv':[], 'seq_sp': [], 'seq_se': [], 'seq_pr': [], 'seq_npv':[]},
        'cls2': {"cnt_frames":0,"cnt_polyps":0,"cnt_seqs":0,'ac': [], 'sp': [], 'se': [], 'pr': [], 'npv':[], 'polyps_sp': [], 'polyps_se': [], 'polyps_pr': [], 'polyps_npv':[], 'seq_sp': [], 'seq_se': [], 'seq_pr': [], 'seq_npv':[]},
        'polyps': {'pct': [], 'ac': [], 'total':[], "cm":None},
        'seqs': {'pct': [], 'ac': [], 'total': [],  "cm":None},
    }

    all_preds = [i for i in df.columns if 'fold' in i and 'pred' in i and 'class' not in i]
    if "pred" in df.columns:
        all_preds = ["pred"]

    # print(list(df.columns))
    # print("### Per frame analysis ###")
    for pred in all_preds:
        print("FOLD {}".format(pred))
        # for confidence in np.arange(1.0/nb_classes, 1.0, (nb_classes-1)/(10*nb_classes)):
        df_sizes_original = df.groupby(['polyp_id', 'gt_class']).size().unstack(fill_value=0).idxmax(
                axis=1).fillna(-1).apply(lambda x: int(x))
        # print(f"Number of polyps: {len(df_sizes_original)}")
        df_sizes_original_seqs = df.groupby(['seq', 'gt_class']).size().unstack(fill_value=0).idxmax(
                axis=1).fillna(-1).apply(lambda x: int(x))
        for confidence in np.arange(0.3, 1.0, 0.05):
            df_high_conf = df.loc[(df[pred].apply(np.max) > confidence)]
            # print("At {} % confidence with {} % included".format(100 * confidence, 100 * len(df_high_conf) / len(df)))
            results['totals']['pct'].append(len(df_high_conf) / len(df))

            # cm1 = df_high_conf.groupby(['gt_class', 'pred_class']).size()
            # print(cm1.unstack(fill_value=0).rename(columns=names, index=names), "\n")
            if len(df_high_conf) > 0:
                if pred != "pred":
                    ac = df_high_conf["_".join(pred.split("_")[:-1]) + "_match"].sum() / len(df_high_conf)
                else:
                    ac = df_high_conf["match"].sum() / len(df_high_conf)

                # print(f"Total Accuracy    - ", ac)
                results['totals']['ac'].append(ac)

                n0 = (df_high_conf['gt_class'] == 0).sum()
                n1 = (df_high_conf['gt_class'] == 1).sum()
                n2 = (df_high_conf['gt_class'] == 2).sum()
                
                if confidence < 0.33:
                    cm = confusion_matrix(df_high_conf['gt_class'][df_high_conf[pred + "_class"] != -1], df_high_conf[pred + "_class"][df_high_conf[pred + "_class"] != -1])
                    fig, ax = plt.subplots(figsize=(6, 6))
                    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
                    plt.ylabel('Actual class')
                    plt.xlabel('Predicted class')

                    plt.title(f'N={len(df_high_conf)}, {len(df_high_conf) / len(df):5.3f} of total\n n0={n0}, n1={n1}, n2={n2}')
                    plt.savefig('data/figures/cms/{}_frames_fold{}.png'.format(df_fn.split('/')[-1], pred.split("_")[0][-1]))
                    plt.close()

                    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    fig, ax = plt.subplots(figsize=(6, 6))
                    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names, vmin=0, vmax=1)
                    plt.ylabel('Actual class')
                    plt.xlabel('Predicted class')
                    plt.title(f'N={len(df_high_conf)}, {len(df_high_conf) / len(df):5.3f} of total\n n0={n0}, n1={n1}, n2={n2}')
                    plt.savefig('data/figures/cms/{}_frames_relative_fold{}.png'.format(df_fn.split('/')[-1], pred.split("_")[0][-1]))
                    plt.close()

                for i in range(nb_classes):
                    # print(f"== Class {i}==")

                    tp = ((df_high_conf['gt_class'] == i) & (df_high_conf[pred + "_class"] == i) & (df_high_conf[pred + "_class"] != -1)).sum()
                    fp = ((df_high_conf['gt_class'] != i) & (df_high_conf[pred + "_class"] == i) & (df_high_conf[pred + "_class"] != -1)).sum()
                    fn = ((df_high_conf['gt_class'] == i) & (df_high_conf[pred + "_class"] != i) & (df_high_conf[pred + "_class"] != -1)).sum()
                    tn = ((df_high_conf['gt_class'] != i) & (df_high_conf[pred + "_class"] != i) & (df_high_conf[pred + "_class"] != -1)).sum()
                    # print(f"\n      pred({i})  |pred(not {i})]\n------------------\ngt({i}) |   {tp}   |   {fp} \n------------------\ngt(not {i}) |   {fn}   |   {tn}\n")
                    ac = (tp + tn)/(tp + tn + fp + fn + np.finfo(float).eps)
                    se = tp / (tp + fn+ np.finfo(float).eps)
                    sp = tn / (tn + fp+ np.finfo(float).eps)
                    pr = tp / (tp + fp+ np.finfo(float).eps)
                    npv = tn / (tn + fn+ np.finfo(float).eps)
                    # print(f"Class {i} Accuracy    - ", ac)
                    # print(f"Class {i} Sensitivity - ", se)
                    # print(f"Class {i} Specificity - ", sp)
                    # print(f"Class {i} Precision   - ", pr)

                    results['cls'+str(i)]['ac'].append(ac)
                    results['cls'+str(i)]['se'].append(se)
                    results['cls'+str(i)]['sp'].append(sp)
                    results['cls'+str(i)]['pr'].append(pr)
                    results['cls' + str(i)]['npv'].append(npv)
            else:
                results['totals']['ac'].append(np.nan)
                for i in range(nb_classes):
                    results['cls' + str(i)]['ac'].append(np.nan)
                    results['cls' + str(i)]['se'].append(np.nan)
                    results['cls' + str(i)]['sp'].append(np.nan)
                    results['cls' + str(i)]['pr'].append(np.nan)
                    results['cls' + str(i)]['npv'].append(np.nan)

            df_pred_sizes = df_high_conf.groupby(['polyp_id', 'gt_class']).size().unstack(fill_value=0).idxmax(axis=1).fillna(-1).apply(lambda x: int(x))
            # print(df_pred_sizes)
            df_preds = df_high_conf.groupby(['polyp_id', pred + "_class"]).size()
            prediction = df_preds.unstack(fill_value=0).idxmax(axis=1).fillna(-1).apply(lambda x: int(x))
            # print(df_preds.unstack(fill_value=0).div(df_sizes_original, axis=0).fillna(0))
            # print("\n ## Per polyp analysis ##")
            pct = len(prediction)/ len(df_sizes_original)
            # print("{}% of polyps classified".format(pct * 100))
            ac = (df_pred_sizes == prediction).sum() / len(prediction)
            # print("Accuracy per polyp (based on must voted class): {} \n".format(ac))
            # print(df_pred_sizes.div(df_sizes_original, axis=0))
            if len(prediction) > 0:
                n0 = (df_pred_sizes == 0).sum()
                n1 = (df_pred_sizes == 1).sum()
                n2 = (df_pred_sizes == 2).sum()

                if confidence < 0.33:
                    cm = confusion_matrix(df_pred_sizes[prediction != -1],
                                        prediction[prediction != -1])
                    fig, ax = plt.subplots(figsize=(6, 6))
                    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
                    plt.ylabel('Actual class')
                    plt.xlabel('Predicted class')
                    # plt.show(block=False)
                    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    # disp.plot()

                    plt.title(f'N={len(df_pred_sizes)}, {len(df_pred_sizes) / len(df_sizes_original):5.3f} of total\n n0={n0}, n1={n1}, n2={n2}')
                    plt.savefig('data/figures/cms/{}_polyps_fold{}.png'.format(df_fn.split('/')[-1], pred.split("_")[0][-1]))
                    plt.close()

                    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    fig, ax = plt.subplots(figsize=(6, 6))
                    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names, vmin=0, vmax=1)
                    plt.ylabel('Actual class')
                    plt.xlabel('Predicted class')
                    # plt.show(block=False)
                    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    # disp.plot()

                    plt.title(f'N={len(df_pred_sizes)}, {len(df_pred_sizes) / len(df_sizes_original):5.3f} of total\n n0={n0}, n1={n1}, n2={n2}')
                    plt.savefig('data/figures/cms/{}_polyps_relative_fold{}.png'.format(df_fn.split('/')[-1], pred.split("_")[0][-1]))
                    plt.close()

            for i in range(nb_classes):
                # print(f"== Class {i}==")

                tp = ((df_pred_sizes == i) & (prediction == i) & (
                            prediction != -1)).sum()
                fp = ((df_pred_sizes != i) & (prediction == i) & (
                            prediction != -1)).sum()
                fn = ((df_pred_sizes == i) & (prediction != i) & (
                            prediction != -1)).sum()
                tn = ((df_pred_sizes != i) & (prediction != i) & (
                            prediction != -1)).sum()
                # print(f"\n      pred({i})  |pred(not {i})]\n------------------\ngt({i}) |   {tp}   |   {fp} \n------------------\ngt(not {i}) |   {fn}   |   {tn}\n")
                se = tp / (tp + fn + np.finfo(float).eps)
                sp = tn / (tn + fp + np.finfo(float).eps)
                pr = tp / (tp + fp + np.finfo(float).eps)
                npv = tn / (tn + fn + np.finfo(float).eps)
                # print(f"Class {i} Accuracy    - ", ac)
                # print(f"Class {i} Sensitivity - ", se)
                # print(f"Class {i} Specificity - ", sp)
                # print(f"Class {i} Precision   - ", pr)

                results['cls' + str(i)]['polyps_se'].append(se)
                results['cls' + str(i)]['polyps_sp'].append(sp)
                results['cls' + str(i)]['polyps_pr'].append(pr)
                results['cls' + str(i)]['polyps_npv'].append(npv)

            results['polyps']['pct'].append(pct)
            results['polyps']['ac'].append(ac)
            results['polyps']['total'].append(ac*pct)


            df_pred_sizes = df_high_conf.groupby(['seq', 'gt_class']).size().unstack(fill_value=0).idxmax(
                axis=1).fillna(-1).apply(lambda x: int(x))
            df_preds = df_high_conf.groupby(['seq', pred + "_class"]).size()
            prediction = df_preds.unstack(fill_value=0).idxmax(axis=1).fillna(-1).apply(lambda x: int(x))
            # print(df_preds.unstack(fill_value=0).div(df_sizes_original, axis=0).fillna(0))
            # print("\n ## Per sequence analysis ##")

            pct = len(prediction)/ len(df_sizes_original_seqs)
            # print("{}% of polyps classified".format(pct * 100))
            ac = (df_pred_sizes == prediction).sum() / len(prediction)

            if len(prediction) > 0:
                n0 = (df_pred_sizes == 0).sum()
                n1 = (df_pred_sizes == 1).sum()
                n2 = (df_pred_sizes == 2).sum()

                if confidence < 0.33:
                    cm = confusion_matrix(df_pred_sizes[prediction != -1],
                                        prediction[prediction != -1])
                    fig, ax = plt.subplots(figsize=(6, 6))
                    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
                    plt.ylabel('Actual class')
                    plt.xlabel('Predicted class')
                    # plt.show(block=False)
                    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    # disp.plot()

                    plt.title(f'N={len(df_pred_sizes)}, {len(df_pred_sizes) / len(df_sizes_original_seqs):5.3f} of total\n n0={n0}, n1={n1}, n2={n2}')
                    plt.savefig('data/figures/cms/{}_seqs_fold{}.png'.format(df_fn.split('/')[-1], pred.split("_")[0][-1]))
                    plt.close()

                    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    fig, ax = plt.subplots(figsize=(6, 6))
                    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names, vmin=0, vmax=1)
                    plt.ylabel('Actual class')
                    plt.xlabel('Predicted class')
                    # plt.show(block=False)
                    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    # disp.plot()

                    plt.title(f'N={len(df_pred_sizes)}, {len(df_pred_sizes) / len(df_sizes_original_seqs):5.3f} of total\n n0={n0}, n1={n1}, n2={n2}')
                    plt.savefig(
                        'data/figures/cms/{}_seqs_relative_fold{}.png'.format(df_fn.split('/')[-1], pred.split("_")[0][-1]))
                    plt.close()
            for i in range(nb_classes):
                # print(f"== Class {i}==")

                tp = ((df_pred_sizes == i) & (prediction == i) & (
                            prediction != -1)).sum()
                fp = ((df_pred_sizes != i) & (prediction == i) & (
                            prediction != -1)).sum()
                fn = ((df_pred_sizes == i) & (prediction != i) & (
                            prediction != -1)).sum()
                tn = ((df_pred_sizes != i) & (prediction != i) & (
                            prediction != -1)).sum()
                # print(tp, fp, tn, fn)
                # print(f"\n      pred({i})  |pred(not {i})]\n------------------\ngt({i}) |   {tp}   |   {fp} \n------------------\ngt(not {i}) |   {fn}   |   {tn}\n")
                se = tp / (tp + fn + np.finfo(float).eps)
                sp = tn / (tn + fp + np.finfo(float).eps)
                pr = tp / (tp + fp + np.finfo(float).eps)
                npv = tn / (tn + fn + np.finfo(float).eps)
                # print(f"Class {i} Accuracy    - ", ac)
                # print(f"Class {i} Sensitivity - ", se)
                # print(f"Class {i} Specificity - ", sp)
                # print(f"Class {i} Precision   - ", pr)

                results['cls' + str(i)]['seq_se'].append(se)
                results['cls' + str(i)]['seq_sp'].append(sp)
                results['cls' + str(i)]['seq_pr'].append(pr)
                results['cls' + str(i)]['seq_npv'].append(npv)
            # print("Accuracy per sequence (based on must voted class): {} \n".format(ac))
            results['seqs']['pct'].append(pct)
            results['seqs']['ac'].append(ac)
            results['seqs']['total'].append(ac*pct)





    # print("AVERAGE prediction")
    # df['gt_class'] = df['gt_class'].astype(int)
    # for i in range(nb_classes):
    #     print(f"== Class {i}==")
    #     df['avg_TP'] = (df['gt_class'] == i) & (df['pred_class'] == i) & (df['pred_class'] != -1)
    #     df['avg_FP'] = (df['gt_class'] != i) & (df['pred_class'] == i) & (df['pred_class'] != -1)
    #     df['avg_FN'] = (df['gt_class'] == i) & (df['pred_class'] != i) & (df['pred_class'] != -1)
    #     df['avg_TN'] = (df['gt_class'] != i) & (df['pred_class'] != i) & (df['pred_class'] != -1)
    #     for confidence in np.arange(1.0/nb_classes, 1.0, (nb_classes-1)/(5*nb_classes)):
    #         df_high_conf = df.loc[df['pred'].apply(np.max) > confidence]
    #         print("At {} % confidence with {} % included".format(100*confidence, 100*len(df_high_conf)/len(df)))
    #         if len(df_high_conf) > 0:
    #             print("Accuracy    - ", df_high_conf['match'].sum() / len(df_high_conf))
    #             print("Sensitivity - ", df_high_conf['avg_TP'].sum() / (df_high_conf['avg_TP'].sum() + df_high_conf['avg_FN'].sum()))
    #             print("Specificity - ", df_high_conf['avg_TN'].sum() / (df_high_conf['avg_TN'].sum() + df_high_conf['avg_FP'].sum()))
    #             print("Precision   - ", df_high_conf['avg_TP'].sum() / (df_high_conf['avg_TP'].sum() + df_high_conf['avg_FP'].sum()))
    #             print("Confusion matrix: ")
    #             print(df_high_conf['avg_TP'].sum(), " | ", df_high_conf['avg_FP'].sum())
    #             print("---------------------")
    #             print(df_high_conf['avg_FN'].sum(), " | ", df_high_conf['avg_TN'].sum())

    # print("MAJORITY VOTING prediction")
    # # df['maj_pred'] = df[all_pred_classes].mode(axis=1)
    # print("Accuracy: ", df['vote_match'].sum() / df['vote_match'].count())
    #
    # print("### Per polyp analysis ###")
    # print("Accuracy (more than 50% match) ", ((df.groupby('polyp_id')['match'].sum() / df.groupby('polyp_id')['match'].count()) > 0.5).sum() / df['polyp_id'].nunique())
    #
    # print("Accuracy (more than 70% match) ", ((df.groupby('polyp_id')['match'].sum() / df.groupby('polyp_id')['match'].count()) > 0.7).sum() / (((df.groupby('polyp_id')['match'].sum() / df.groupby('polyp_id')['match'].count()) > 0.7).sum() + ((df.groupby('polyp_id')['match'].sum() / df.groupby('polyp_id')['match'].count()) < 0.3).sum()))

    # for confidence in np.arange(1.0/nb_classes, 1.0, (nb_classes-1)/(5*nb_classes)):
    #     df_high_conf = df.loc[df['pred'].apply(np.max) > confidence]
    #     avg_polyp_pred = df_high_conf.groupby('polyp_id')['pred'].apply(np.mean)
    #     print((avg_polyp_pred.apply(np.argmax) == df_high_conf.groupby('polyp_id')['gt_class'].max()).sum() / (avg_polyp_pred.apply(np.argmax) == df_high_conf.groupby('polyp_id')['gt_class'].max()).count())
    translations = {'ac': 'Accuracy', 'sp': 'Specificity', 'se': 'Sensitivity', 'pr': 'Precision', 'npv': 'Negative Predictive Value', 0: 'Hyperplastic',1:'Adenoma', 2:'Sessile Serrated Polyps', 'polyps_sp': 'Specificity', 'polyps_se': 'Sensitivity', 'polyps_pr': 'Precision', 'polyps_npv': 'Negative Predictive Value', 'seq_sp': 'Specificity', 'seq_se': 'Sensitivity', 'seq_pr': 'Precision', 'seq_npv': 'Negative Predictive Value'}
    EOL = "\n"
    
    s = ""
    s += "\\resizebox{\\textwidth}{!}{ %" + EOL
    s += "\\begin{tabular}{l | | c | c | c | c | c | c | c | c | c | c | c | c | c | c |}"+ EOL
    s += "Threshold & 0.3 & 0.35 & 0.4 & 0.45 & 0.5 & 0.55 & 0.6 & 0.65 & 0.7 & 0.75 & 0.8 & 0.85 & 0.9 & 0.95 \\\\ "+ EOL
    s += "\\hline"+ EOL
    s += '\\multicolumn{15}{l}{\\textbf{Per frame analysis}} \\\\'+ EOL
    s += '\\hline'+ EOL

    s += 'Fraction used '
    for k in results['totals']['pct']:
        s += '& {:5.3f} '.format(k)
    s += '\\\\'+ EOL
    
    s += '\\hline'+ EOL
    s += 'Total Accuracy '
    for k in results['totals']['ac']:
        s += '& {:5.3f} '.format(k)
    s += '\\\\'+ EOL
    
    s += '\\hline'+ EOL


    for i in range(nb_classes):
        s += '\\multicolumn{15}{l}{\\textbf{Class '+str(i)+' ('+translations[i]+')}} \\\\'+ EOL

        s += '\\hline'+ EOL
        start = True
        # for key in ['ac', 'se', 'sp', 'pr', 'npv']:
        for key in ['ac', 'se', 'pr']:
            if not start:
                s += '\\cline{2-15}'+ EOL
            s += translations[key] + ' '
            for k in results['cls'+str(i)][key]:
                s += '& {:5.3f} '.format(k)

            s += '\\\\'+ EOL
            
            start = False
        s += '\\hline'+ EOL

    s += '\\hline'+ EOL
    s += '\\multicolumn{15}{l}{\\textbf{Per polyp analysis}} \\\\'+ EOL
    s += '\\hline'+ EOL
    s += 'Fraction used '
    for k in results['polyps']['pct']:
        s += '& {:5.3f} '.format(k)
    s += '\\\\'+ EOL
    

    s += '\\cline{2-15}'+ EOL
    s += 'Accuracy '
    for k in results['polyps']['ac']:
        s += '& {:5.3f} '.format(k)
    s += '\\\\'+ EOL
    

    s += '\\hline'+ EOL
    s += '\\hline'+ EOL
    s += 'Combined '
    for k in results['polyps']['total']:
        s += '& {:5.3f} '.format(k)
    s += '\\\\'+ EOL
    
    s += '\\hline'+ EOL

    for i in range(nb_classes):
        s += '\\multicolumn{15}{l}{\\textbf{Class '+str(i)+' ('+translations[i]+')}} \\\\'+ EOL

        s += '\\hline'+ EOL
        start = True
        # for key in ['polyps_se', 'polyps_sp', 'polyps_pr', 'polyps_npv']:
        for key in ['polyps_se', 'polyps_pr']:
            if not start:
                s += '\\cline{2-15}'+ EOL
            s += translations[key] + ' '
            for k in results['cls'+str(i)][key]:
                s += '& {:5.3f} '.format(k)

            s += '\\\\'+ EOL
            
            start = False
        s += '\\hline'+ EOL

    s += '\\hline'+ EOL
    s += '\\multicolumn{15}{l}{\\textbf{Per sequence analysis}} \\\\'+ EOL
    s += '\\hline'+ EOL
    s += 'Fraction used '
    for k in results['seqs']['pct']:
        s += '& {:5.3f} '.format(k)
    s += '\\\\'+ EOL


    s += '\\cline{2-15}'+ EOL
    s += 'Accuracy '
    for k in results['seqs']['ac']:
        s += '& {:5.3f} '.format(k)
    s += '\\\\'+ EOL
    

    s += '\\hline'+ EOL
    s += '\\hline'+ EOL
    s += 'Combined '
    for k in results['seqs']['total']:
        s += '& {:5.3f} '.format(k)
    s += '\\\\'+ EOL

    s += '\\hline'+ EOL

    for i in range(nb_classes):
        s += '\\multicolumn{15}{l}{\\textbf{Class '+str(i)+' ('+translations[i]+')}} \\\\'+ EOL

        s += '\\hline'+ EOL
        start = True
        # for key in ['seq_se', 'seq_sp', 'seq_pr', 'seq_npv']:
        for key in ['seq_se', 'seq_pr']:
            if not start:
                s == '\\cline{2-15}'+ EOL
            s += translations[key] + ' '
            for k in results['cls'+str(i)][key]:
                s += '& {:5.3f} '.format(k)

            s += '\\\\'+ EOL

            start = False
        s += '\\hline'+ EOL

    s += "\end{tabular}"+ EOL
    s += "}"+ EOL

    print(s)
    return s

    # print("# Sensitivity at thresholds")
    # for confidence in np.arange(0.5, 1.0, 0.05):
    #     df_high_conf = df.loc[df['avg_pred'].apply(np.max) > confidence]
    #     tp = (df_high_conf.groupby('polyp_id')['avg_match'].sum() / df_high_conf.groupby('polyp_id')['avg_match'].count()) > 0.5
    #     print("sensitivity high conf", ((df_high_conf.groupby('polyp_id')['avg_match'].sum() / df_high_conf.groupby('polyp_id')['avg_match'].count()) > 0.5).sum() / df_high_conf['polyp_id'].nunique(), "with {} % of polyps included.".format(df_high_conf.groupby('polyp_id').ngroups / df.groupby('polyp_id').ngroups))
    #
    # print("# Sensitivity at thresholds for sequences longer than 10 frames")
    # for confidence in np.arange(0.5, 1.0, 0.05):
    #     df_high_conf = df.loc[df['avg_pred'].apply(np.max) > confidence]
    #     df_high_conf_long = df_high_conf.groupby("polyp_id").filter(lambda x: len(x) >= 10)
    #     print("sensitivity high conf long", ((df_high_conf_long.groupby('polyp_id')['avg_match'].sum() / df_high_conf_long.groupby('polyp_id')['avg_match'].count()) > 0.5).sum() / df_high_conf_long['polyp_id'].nunique(), "with {} % of polyps included.".format(df_high_conf_long.groupby('polyp_id').ngroups / df.groupby('polyp_id').ngroups))
    #
    # print("### Per seq analysis ###")
    # print("sensitivity ", ((df.groupby('seq')['avg_match'].sum() / df.groupby('seq')['avg_match'].count()) > 0.5).sum() / df['seq'].nunique())
    # print("# Sensitivity at thresholds")
    # for confidence in np.arange(0.5, 1.0, 0.05):
    #     df_high_conf = df.loc[df['avg_pred'].apply(np.max) > confidence]
    #     print("sensitivity high conf", ((df_high_conf.groupby('seq')['avg_match'].sum() / df_high_conf.groupby('seq')['avg_match'].count()) > 0.5).sum() / df_high_conf['seq'].nunique(), "with {} % of seqs included.".format(df_high_conf.groupby('seq').ngroups / df.groupby('seq').ngroups))
    #
    # print("# Sensitivity at thresholds for sequences longer than 10 frames")
    # for confidence in np.arange(0.5, 1.0, 0.05):
    #     df_high_conf = df.loc[df['avg_pred'].apply(np.max) > confidence]
    #     df_high_conf_long = df_high_conf.groupby("seq").filter(lambda x: len(x) >= 10)
    #     print("sensitivity high conf long", ((df_high_conf_long.groupby('seq')['avg_match'].sum() / df_high_conf_long.groupby('seq')['avg_match'].count()) > 0.5).sum() / df_high_conf_long['seq'].nunique(), "with {} % of seqs included.".format(df_high_conf_long.groupby('seq').ngroups / df.groupby('seq').ngroups))
    #
    # # # Calculate exponential smoothing and cumulative exp smoothed value
    # df['exp_smooth'] = np.nan
    # df['exp_smooth'] = df['exp_smooth'].astype(object)
    # df['cum_exp_smooth'] = np.nan
    # df['cum_exp_smooth'] = df['cum_exp_smooth'].astype(object)
    #
    # for i, row in df.iterrows():
    #     if i == 0 or df.at[i, 'seq'] != df.at[i - 1, 'seq']:
    #         # new sequence
    #         df.at[i, 'exp_smooth'] = exp_alpha * df.at[i, 'avg_pred'] + (1 - exp_alpha) * np.asarray([0.5, 0.5])
    #         df.at[i, 'cum_exp_smooth'] = df.at[i, 'exp_smooth']
    #         continue
    #     df.at[i, 'exp_smooth'] = exp_alpha * df.at[i, 'avg_pred'] + (1 - exp_alpha) * df.at[i - 1, 'exp_smooth']
    #     df.at[i, 'cum_exp_smooth'] = df.at[i, 'exp_smooth'] if np.max(df.at[i, 'exp_smooth']) > np.max(df.at[i - 1, 'cum_exp_smooth']) else df.at[i - 1, 'cum_exp_smooth']
    #
    # print("### Per polyp exp smoothed analysis ###")
    # df_final_conf = df.groupby('polyp_id').tail(1)
    # for confidence in np.arange(0.5, 1.0, 0.05):
    #     df_final_conf_high = df_final_conf.loc[df_final_conf['cum_exp_smooth'].apply(np.max) > confidence]
    #     print("High confidence accuracy {} % at {} % included".format((df_final_conf_high['cum_exp_smooth'].apply(np.argmax) == df_final_conf_high['gt_class'].astype(int)).sum() / (df_final_conf_high['cum_exp_smooth'].apply(np.argmax) == df_final_conf_high['gt_class'].astype(int)).count(), len(df_final_conf_high) / len(df_final_conf)))
    #
    # print("### Per seq exp smoothed analysis ###")
    # df_final_conf = df.groupby('seq').tail(1)
    # for confidence in np.arange(0.5, 1.0, 0.05):
    #     df_final_conf_high = df_final_conf.loc[df_final_conf['cum_exp_smooth'].apply(np.max) > confidence]
    #     print("High confidence accuracy {} % at {} % included".format((df_final_conf_high['cum_exp_smooth'].apply(np.argmax) == df_final_conf_high['gt_class'].astype(int)).sum() / (df_final_conf_high['cum_exp_smooth'].apply(np.argmax) == df_final_conf_high['gt_class'].astype(int)).count(), len(df_final_conf_high) / len(df_final_conf)))
    #
    # df_final_conf['cum_exp_smooth_pred'] = df_final_conf['cum_exp_smooth'].apply(np.argmax)
    # df_final_conf['cum_exp_smooth_conf'] = df_final_conf['cum_exp_smooth'].apply(np.max)
    # df_final_conf.to_excel(os.path.join(logdir, args.model_name + "_5{}_finalframe.xlsx".format(split_type)), columns=['seq', 'gt_class', 'cum_exp_smooth', 'cum_exp_smooth_conf', 'cum_exp_smooth_pred'])
    #
    # df.to_excel(os.path.join(logdir, args.model_name + "_5{}.xlsx".format(split_type)), columns=results_struct)
    # df.to_pickle(os.path.join(logdir, args.model_name + "_5{}.pkl".format(split_type)))

def create_video(df_fn):
    df = pd.read_pickle(df_fn)

    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0,0,0)]
    # print(df.columns)

    video_folder = 'data/inferenced/vote_ensembles'

    os.makedirs(video_folder, exist_ok=True)

    for polyp_id, group in tqdm(df.groupby('polyp_id'), desc="Creating videos"):
        gt_class = int(group.iloc[0]['gt_class'])
        for seq in group['seqs'].iloc[0]:
            # pred_class = int(np.argmax(group.loc[group['seq'] == seq].iloc[-1]['fold0_pred']))
            # confidence = int(100 * np.max(group.loc[group['seq'] == seq].iloc[-1]['cum_exp_smooth']))
            out_name = os.path.join(video_folder, str(seq) + '_{}_--.mp4'.format(gt_class))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_shape = cv2.imread(group.iloc[0]['img_fns'][0].replace("CROPS/256p", "JPEGImages/1080p")).shape[:2][::-1]
            writer = cv2.VideoWriter(out_name, fourcc, 25., video_shape)

            # cum_exp_pred = [0.5, 0.5]
            pred = [0.33, 0.33, 0.33]
            cum_pred = None
            count = 0

            img_fn_df = group['img_fns'].apply(itemgetter(0)).astype(str)
            for frame_idx in trange(get_seq_length(seq), leave=False, desc=seq):
                frame_metadata = get_filename(seq, frame_idx, return_type='both', check_valid=True)

                frame_fn = frame_metadata['img_fn']
                ann_fn = frame_metadata['ann_fn']

                crop_fn = frame_fn.replace("JPEGImages/1080p", "CROPS/256p")
                pred = group.loc[img_fn_df == crop_fn]['fold0_pred'].values[0] if len(group.loc[img_fn_df == crop_fn]['fold0_pred'].values) > 0 else None
                
                # print(group.loc[img_fn_df == frame_fn])
                if pred is not None:
                    if cum_pred is None:
                        cum_pred = [float(i) for i in pred]
                        count = 1
                    else:
                        cum_pred = [(cum_pred[i] * count + float(pred[i]))/(count + 1) for i in range(len(pred))]
                        count += 1
                    pred_class = group.loc[img_fn_df == crop_fn]['fold0_pred_class'].values[0]
                else:
                    pred_class = 3
                    pred = [0.33, 0.33, 0.33]
                
                if cum_pred is not None:
                    cum_pred_class = np.argmax(cum_pred)
                else:
                    cum_pred_class = 3
                
                # print(cum_pred)

                # if len(group.loc[img_fn_df == frame_fn]['exp_smooth'].values) > 0
                # exp_pred = group.loc[img_fn_df == frame_fn]['exp_smooth'].values[0] if len(group.loc[img_fn_df == frame_fn]['exp_smooth'].values) > 0 else None
                # cum_exp_pred = group.loc[img_fn_df == frame_fn]['cum_exp_smooth'].values[0] if len(group.loc[img_fn_df == frame_fn]['cum_exp_smooth'].values) > 0 else cum_exp_pred

                frame = cv2.imread(frame_fn)

                if frame_metadata['valid']:
                    # Draw contours of annotation
                    # print("valid")
                    mask = cv2.imread(ann_fn, 0)
                    mask = cv2.resize(mask, frame.shape[:2][::-1])
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        bbox = cv2.boundingRect(cnt)
                        frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), colors[cum_pred_class], 5)
                        if cum_pred is not None:
                            frame = cv2.putText(frame, str('{:.2f}'.format(np.max(cum_pred))), (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colors[cum_pred_class], thickness=2)

                frame = cv2.putText(frame, 'GT -   {}'.format(gt_class), (int(video_shape[0]*0.7), int(video_shape[1]*0.9)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colors[gt_class], thickness=2)
                if pred is not None:
                    frame = cv2.putText(frame, 'pred - {}'.format(['{:.2f}'.format(p) for p in pred]), (int(video_shape[0]*0.7), int(video_shape[1]*0.9+40)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colors[pred_class], thickness=2)
                if cum_pred is not None:
                    frame = cv2.putText(frame, 'cum pred - {}'.format(['{:.2f}'.format(p) for p in cum_pred]), (int(video_shape[0]*0.7), int(video_shape[1]*0.9+80)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colors[cum_pred_class], thickness=2)
                # print(frame)

                # cv2.imshow('frame',frame)

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                writer.write(frame)
            writer.release()

            os.rename(out_name, out_name.replace("--", str(cum_pred_class)))


# def initialize_modelstate(models):
#     for i, model in enumerate(models):
#         imageset_dir = 'data/imagesets_1080p/5{}_all/{}k_train.txt'.format(split_type, i)
#         ds = Dataset(imageset_dir, batch_size=BATCH_SIZE, target_size=(256, 256), loading_size=(256, 256), augmentations=None, nb_inputs=1, shuffle=True, repeat=True, do_cropping='preloaded', balanced=True, balance_polypids=True, selected_labels=current_label)

#         model.fit(ds.tfdataset, epochs=1, steps_per_epoch=5)  # not sure why, but otherwise model.evaluate doesn't work accurately (see https://github.com/keras-team/keras/issues/6977)


def main(model_name, submodel_names, imageset_dir, output_fn, split_type="fold", num_models=1, dsc=""):
    sequences_fn = imageset_dir.replace('.txt', '_sequences.txt')
    assert os.path.exists(sequences_fn), sequences_fn

    # possible_types = {
    #     "hypVSall": ("0", ("1", "2")),
    #     "adnVSall": ("1", ("0", "2")),
    #     "sspVSall": ("2", ("0", "1")),
    #     "hypVSadn": ("0", "1"),
    #     "hypVSssp": ("0", "2"),
    #     "adnVSssp": ("1", "2"),
    #     "hypVSadnVSssp": ("0", "1", "2"),        
    # }

    # possible_types = {
    #     "0vs12": ("0", ("1", "2")),
    #     "1vs02": ("1", ("0", "2")),
    #     "2vs01": ("2", ("0", "1")),
    #     "0vs1": ("0", "1"),
    #     "0vs2": ("0", "2"),
    #     "1vs2": ("1", "2"),
    #     "0vs1vs2": ("0", "1", "2"),        
    #     "0vs1vs2vs3": ("0", "1", "2", "3"),        
    # }
    
    possible_types = {
    "hypVSall": ("0", ("1", "2")),
    "adnVSall": ("1", ("0", "2")),
    "sspVSall": ("2", ("0", "1")),
    "hypVSadn": ("0", "1"),
    "hypVSssp": ("0", "2"),
    "adnVSssp": ("1", "2"),
    "hypVSadnVSssp": ("0", "1", "2"),        
    "0vs12": ("0", ("1", "2")),
    "1vs02": ("1", ("0", "2")),
    "2vs01": ("2", ("0", "1")),
    "0vs1": ("0", "1"),
    "0vs2": ("0", "2"),
    "1vs2": ("1", "2"),
    "0vs1vs2": ("0", "1", "2"),        
    "0vs1vs2vs3": ("0", "1", "2", "3"),        
}

    

    # global validity_fn
    # validity_fn = f'/frames_to_use_dsc{dsc}.pickle'
    # if dsc == '05':
    #     validity_fn = f'/frames_to_use.pickle'


    test_type = model_name.split("_")[0]
    current_label = possible_types[test_type]


    # Read all sequences from file
    with open(sequences_fn) as f:
        content = f.readlines()
    sequences = [x.strip().split(' ')[0] for x in content]
    labels = [x.strip().split(' ')[1] for x in content]
    polyp_ids = [int(x.strip().split(' ')[2]) for x in content]

    # print(model_name.split("_"))
    models = [tf.keras.models.load_model(os.path.join('data/snapshots/all', submodel), compile=False) for submodel in submodel_names]
    models[0].summary()
    result = None

    # sl = ["0", "1", "2", "3"]
    sl = ["0", "1"]
    # for i in current_label:
    #     sl += list(i)

    for polyp_id in sorted(set(polyp_ids)):
        polyp_seqs = [seq for idx, seq in enumerate(sequences) if polyp_ids[idx] == polyp_id]
        label = labels[polyp_ids.index(polyp_id)]
        if label not in sl:
            continue
        print("Processing polyp {} with label {} and sequences {}.".format(polyp_id, label, polyp_seqs))
        # try:
        evaluator = Evaluator(models, polyp_seqs, label, polyp_id, selected_label=current_label, mn=model_name)
        evaluator.run()
        evaluator.result['polyp_id'] = polyp_id
        evaluator.result = evaluator.result.set_index('polyp_id')
        if result is not None:
            result = pd.concat([result, evaluator.result])
        else:
            result = evaluator.result
        # except ValueError as e:
        #     print(e)  # sequence not complete

        # reset recurrent network state before starting new sequence
        # for m in evaluator.model:
        #     m.reset_states()

        # print(evaluator.result)

    # Get sequence column
    result = result.reset_index()
    result['seq'] = ""
    for i, row in result.iterrows():
        result.at[i, 'seq'] = result.at[i, 'img_fns'][0].split('/')[-2]

    # result.to_excel(os.path.join(logdir, model_name + "_5{}.xlsx".format(split_type)), columns=results_struct)
    result.to_pickle(output_fn)

    del result
    del models


def votenet_main(model_names, model_labels, imageset_dir, output_fn, gather_type, th, output_weights):
    sequences_fn = imageset_dir.replace('.txt', '_sequences.txt')
    assert os.path.exists(sequences_fn), sequences_fn

    # Read all sequences from file
    with open(sequences_fn) as f:
        content = f.readlines()
    sequences = [x.strip().split(' ')[0] for x in content]
    labels = [x.strip().split(' ')[1] for x in content]
    polyp_ids = [int(x.strip().split(' ')[2]) for x in content]

    print(model_names)
    # print(model_name.split("_"))
    models = [vote_net_same_base(model_names, image_dimensions=(256, 256, 3), labels=model_labels, gather_type=gather_type, voting_th=th, output_weights=output_weights)]

    
    # models[0].summary()
    # initialize_modelstate(models)
    model_name = output_fn.split("/")[-1].replace(".pkl", "")
    result = None

    models[0].save('data/snapshots/votenet_' + model_name)
    sl = []
    for current_label in model_labels:
        for i in current_label:
            sl += list(i)


    for polyp_id in sorted(set(polyp_ids)):
        polyp_seqs = [seq for idx, seq in enumerate(sequences) if polyp_ids[idx] == polyp_id]
        label = labels[polyp_ids.index(polyp_id)]
        if label not in sl:
            continue
        print("Processing polyp {} with label {} and sequences {}.".format(polyp_id, label, polyp_seqs))
        try:
            evaluator = Evaluator(models, polyp_seqs, label, polyp_id, selected_label=sl, mn=model_name)
            evaluator.run()
            evaluator.result['polyp_id'] = polyp_id
            evaluator.result = evaluator.result.set_index('polyp_id')
            if result is not None:
                result = pd.concat([result, evaluator.result])
            else:
                result = evaluator.result
        except ValueError as e:
            print(e)  # sequence not complete

        # reset recurrent network state before starting new sequence
        # evaluator.model.reset_states()

        # print(evaluator.result)

    # Get sequence column
    result = result.reset_index()
    result['seq'] = ""
    for i, row in result.iterrows():
        result.at[i, 'seq'] = result.at[i, 'img_fns'][0].split('/')[-2]

    # result.to_excel(os.path.join(logdir, model_name + "_5{}.xlsx".format(split_type)), columns=results_struct)
    model_type = model_name.split("_")[0]
    result.to_pickle(output_fn)

def eval_tfdataset(model_name):
    options = glob.glob(os.path.join('data/snapshots/', model_name, 'best-*'))
    best_model_fn = sorted(options, key=lambda fn: int(os.path.basename(fn).split('-')[1]))[-1]
    print("Loading best model {}".format(best_model_fn))
    model = tf.keras.models.load_model(best_model_fn)
    model.summary()

    nb_inputs = int(re.search(r'\d{1,}in', model_name).group()[:-2])
    input_size = tuple([int(x) for x in re.search(r'_\d{1,}x\d{1,}_', model_name).group()[1:-1].split('x')])
    ds = Dataset(imageset_dir.replace('val', 'train'), batch_size=BATCH_SIZE, target_size=input_size, loading_size=input_size, augmentations=None, nb_inputs=nb_inputs, shuffle=True, repeat=True, do_cropping='preloaded', balanced=True, balance_polypids=True)
    model.fit(ds.tfdataset, epochs=1, steps_per_epoch=5)  # not sure why, but otherwise model.evaluate doesn't work accurately (see https://github.com/keras-team/keras/issues/6977)

    metrics = ['categorical_accuracy', tf.keras.metrics.Recall(class_id=0, name='recall_0'), tf.keras.metrics.Recall(class_id=1, name='recall_1'), tf.keras.metrics.Precision(class_id=0, name='prec_0'), tf.keras.metrics.Precision(class_id=1, name='prec_1')]
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=metrics)
    ds = Dataset('data/imagesets_characterisation/all_split_0.2_balanced.txt'.format(split_type), batch_size=BATCH_SIZE, target_size=input_size, loading_size=input_size, augmentations=None, nb_inputs=nb_inputs, shuffle=True, repeat=True, do_cropping='preloaded', balanced=args.balanced, balance_polypids=args.balanced)
    result = model.evaluate(ds.tfdataset, steps=50)
    print(dict(zip(model.metrics_names, result)))


def vote(model_names, labels,fn, nb_classes=3, confidence_level=None, mv_avg_len=20):

    df = None
    pc = 'pred_class_'
    conf = 'confidence_'
    hc = 'hc_'
    vc = 'vote_cls'

    all_pred_cols = []
    all_conf_cols = []
    lenths_per_type = []

    def translate_conf(conf, label):
        new_conf = [conf[0] if str(i) in label[0] else conf[1] if str(i) in label[1] else 0. for i in range(nb_classes)]

    # for i, model_name in enumerate(model_names):
    l = labels[0]
    df_fn = fn
    df_new = pd.read_pickle(df_fn)
    t = model_name.split("_")[0]
    mapper = {i: t+"_"+i for i in df_new.columns if "fold" in i}
    df_new = df_new.rename(columns=mapper)
    pred_cols = [i for i in df_new.columns if "pred_class" in i]
    conf_cols = [i for i in df_new.columns if "pred" in i and 'class' not in i]
    all_pred_cols += pred_cols
    all_conf_cols += conf_cols
    lenths_per_type.append(len(pred_cols))
    # if len(l[1]) > 1:
    #     label_translation = [int(l[0]), nb_classes+int(l[0])]

    # else:
    #     label_translation = [int(l[0]), int(l[1])]

    # print(df_new)
    # df_new[pred_cols] = df_new[pred_cols].applymap(lambda x: label_translation[int(x)])
    df_new[['img_fns']] = df_new[['img_fns']].applymap(lambda x: str(x))

    # df_new[conf_cols] = df_new[conf_cols].applymap(lambda x: [x[0] if str(i) in l[0] else x[1] if str(i) in l[1] else 0. for i in range(nb_classes)])
    if df is None:
        df = df_new.copy()
    else:
        df = df.merge(df_new[list(mapper.values()) + ['img_fns']], how='left', left_on='img_fns', right_on='img_fns')
    df = df.fillna(-1.0)
    df[conf_cols] = df[conf_cols].replace(-1.0, '000')
    df[conf_cols] = df[conf_cols].applymap(lambda x: [float(i) for i in x])


    votes = [vc+str(i) for i in range(nb_classes)]
    df[votes] = 0.

    if confidence_level is not None:
        hc_votes = [hc + vc + str(i) for i in range(nb_classes)]
        df[hc_votes] = 0.

    doubles = 1 # 2 if count negatives
    for i in range(nb_classes * doubles):
        if i < nb_classes:
            df[votes[i]] = df[[votes[i]]].merge((df[all_pred_cols]==i), left_index=True, right_index=True).sum(skipna=True, axis=1)
        else:
            df[votes] += len(all_pred_cols)
            df[votes[i % nb_classes]] = df[[votes[i % nb_classes],]].merge(-1*(df[all_pred_cols]==i), left_index=True, right_index=True).sum(skipna=True, axis=1)
    df_test = df.copy().explode(all_conf_cols)
    df_test['pred'] = df_test[all_conf_cols].mean(axis=1)
    df_test = df_test.groupby('img_fns').aggregate({i: lambda x: list(x) for i in all_conf_cols + ['pred']})
    df = df.merge(df_test['pred'], on='img_fns')
    df['pred'] = df[['pred']].applymap(lambda x: np.array(x) / np.sum(x))
    df['gt_class'] = df[['gt_class']].applymap(lambda x: int(x))
    df['pred_class'] = df[['pred']].applymap(lambda x: np.argmax(x))
    df['conf'] = df[['pred']].applymap(lambda x: np.max(x))
    df['vote'] = 0
    df['vote'] = df[votes].copy().idxmax(axis=1).apply(lambda x: int(list(x)[-1]))
    print(df['vote'] == df['gt_class'])
    df['vote_match'] = (df['vote'] == df['gt_class'])
    df['match'] = 0
    df['match'] = (df['pred_class'] == df['gt_class'])
    print(df['pred_class'] == df['gt_class'])
    print(df.columns)
    print(df[['gt_class', 'vote', 'vote_match', 'match', 'pred', 'pred_class']])


    df['mv_avg'] = None

    # if confidence_level is not None:
    #     df[hc+'vote'] = df[hc_votes].copy().idxmax(axis=1).apply(lambda x: int(list(x)[-1]))
    #     df[hc+'match'] = 0
    #
    #     df.loc[df[hc+'vote'] == df['gt_class'], hc+'match'] = 1
    #
    es = 'exp_smooth'
    ces = 'cum_exp_smooth'
    df[es] = None
    df[ces] = None
    #
    # preds = [t + '_pred' for t in types]
    #
    current_pred_seq = None
    prediction = np.ones_like(range(nb_classes)) * 1.0 / nb_classes
    for i, row in df.iterrows():

        if i == 0 or df.at[i, 'seq'] != df.at[i - 1, 'seq']:
            # new sequence
            # df.at[i, es] = df.at[i, 'pred']
            # df.at[i, ces] = df.at[i, 'pred']
            df.at[i, 'mv_avg'] = df.at[i, 'pred']
            current_pred_seq = np.array(df.at[i, 'pred'])
            continue
        # df.at[i, es] = exp_alpha * df.at[i, 'pred'] + (1 - exp_alpha) * df.at[i - 1, es]
        # df.at[i, ces] = df.at[i, es] if np.max(df.at[i, es]) > np.max(df.at[i - 1, ces]) else df.at[i - 1, ces]
        current_pred_seq = np.vstack([current_pred_seq, np.array(df.at[i, 'pred'])])
        df.at[i, 'mv_avg'] = np.mean(current_pred_seq[-mv_avg_len:, :], axis=0)
        # print(df.at[i, 'mv_avg'])
    # df.to_excel(os.path.join("logs/", args.model_name + "_5{}_voted.xlsx".format(split_type)), columns=results_struct)

    print(df['match'].sum() / len(df))

    df.to_pickle(fn.replace(".pkl", "_voted.pkl"))
    return df


if __name__ == '__main__':
    pass
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", action="store", default="hypVSadn_HDall2023_efficientnet_0_regularized0.0_256x256_1in_nf64_bnTrue_fcdo0.0_balancedTrue_loss_fl_gamma1.0_sgd_5fold0_best", 
                    dest="model_name")
    parser.add_argument("-ds", "--dataset", action="store", default="polyps", dest="dataset")
    parser.add_argument("--balanced", action="store_true", dest="balanced")
    args = parser.parse_args()

    split_type = 'fold'
    assert split_type in ['bootstrap', 'fold']
    assert args.dataset in ['catsndogs', 'polyps', 'cvc']
    num_submodels = 1
    # dsc = '07'

    if args.dataset == 'catsndogs':
        from utils.Dataset_catsndogs import Dataset
        labels = {0: 'cat', 1: 'dog'}
        imageset_dir = 'data/catsndogs/val/'
        loading_size = None
    elif args.dataset == 'polyps':
        from utils.Dataset import Dataset
        #labels = {0: 'HP', 1: 'AD', 2: 'SSA'}
        labels = {0: 'HP', 1: 'AD'}
        # imageset_dir = 'data/imagesets_characterisation_1080p/{}{}_all{}/test.txt'.format(num_submodels, split_type, dsc)
        # imageset_dir = 'data/imagesets_1080p/{}{}_{}/test.txt'.format(5, split_type, 'all')
        # imageset_dir = 'data/imagesets_filtered_size_smaller_th_0.05/{}{}_{}/test.txt'.format(5, split_type, 'all')
        # imageset_dir = 'data/imagesets_1080p/5fold_crop_brisque_all/{}{}_{}/test.txt'.format(5, split_type, 'all')
        imageset_dir = "/DATASERVER/MIC/GENERAL/STUDENTS/amartic/mscthesis/polypclassificationmi/data/imagesets_characterisation/all_split_0.2_balanced.txt"

        loading_size = image_size
    elif args.dataset == 'harvard_polyps':
        from utils.Dataset import Dataset
        labels = {0: 'HP', 1: 'AD', 2: 'SSA'}
        imageset_dir = 'data/harvard_imagesets/test2019.txt'
        img_dir = '/DATASERVER/MIC/SHARED/ENDOSCOPY/HARVARD_POLYPS/test2019/Image/'
        ann_dir = '/DATASERVER/MIC/SHARED/ENDOSCOPY/HARVARD_POLYPS/test2019/Annotation/'

        loading_size = image_size
    elif args.dataset == 'cvc':
        from utils.Dataset_cvc import Dataset
        labels = {0: 'ADN', 1: 'nonADN'}
        imageset_dir = 'data/cvc/val/'
        loading_size = image_size

    # for k in range(5):
    #     imageset_dir = 'data/imagesets_1080p+artipod_long/5{}_all/{}k_val.txt'.format(split_type, k)
    #     eval_tfdataset(args.model_name + '_{}{}'.format(split_type, k))
    list_of_nf = [64]
    # nf = 64
    output_weigths = [1.0, 1.0, 1.0]
    fold_select = 0
    opts = [1,5,50]
    t = "enhancedContrast"

    # "hypVSadnVSssp_largePolyps01_HDall2023_efficientnet_0_regularized0.0_256x256_1in_nf64_bnTrue_fcdo0.0_balancedTrue_loss_fl_gamma1.0_sgd_5fold1_final"
    # model_name = "{t}_HDall2024_efficientnet_0_brisque{n}_regularized0.0_256x256_1in_nf{nf}_bnTrue_fcdo0.0_balancedTrue_loss_fl_gamma1.0_sgd"
    model_name = "hypVSadn_HDall2023_efficientnet_0_regularized0.0_256x256_1in_nf64_bnTrue_fcdo0.0_balancedTrue_loss_fl_gamma1.0_sgd_5fold0_best"
    # model_name = "_effnet_nf{nf}_reg0.001_"
    model_filename = args.model_name + ".h5"
    model_files = [model_filename]

    output_fn = os.path.join(logdir, "evaluation_output_.pkl")

    try:
        print("Avaliando o modelo:", model_filename)
        
        main(model_files[0], model_files, imageset_dir, output_fn)
    except OSError:
        print("No file found!")
    # for opt in opts:
    #     # for reduce_type in ['vote', 'sum']:
    # # for reduce_type in []:
    # #     for th_i in range(0, 5):
    # #         th = th_i / 10
    #         th = 0.0
    #         for nf in list_of_nf:
    #             # print(reduce_type, nf, th)
    #             temp_name = model_name.replace("{nf}", str(nf)).replace("{t}", t).replace("{n}", str(opt))
    #             model_names = [
    #                 # f'hypVSadnVSssp_{temp_name}',
    #                 # f'hypVSall_{temp_name}',
    #                 # f'adnVSall_{temp_name}',
    #                 # f'sspVSall_{temp_name}',
    #                  f'hypVSadn_{temp_name}'
    #                 # f'hypVSssp_{temp_name}',
    #                 # f'adnVSssp_{temp_name}',
    #             ]
    #             # model_names = [
    #             #     f'hypVSadn_{temp_name}',
    #             #     # f'hypVSssp_{temp_name}',
    #             #     # f'adnVSssp_{temp_name}',
    #             # ]

    #             label_names = [
    #                 # ('0', '1', '2'),
    #                 # ('0', ('1', '2')),
    #                 # ('1', ('0', '2')),
    #                 # ('2', ('0', '1')),
    #                  ('0', '1'),
    #                 # ('0', '2'),
    #                 # ('1', '2'),
    #             ]
    #             current_label = None
    #             # for l, model_name in enumerate(model_names):
    #             #     current_label = label_names[l]
    #             all_names = []
    #             all_labels = []
    #             for i, name in enumerate(model_names):
    #                 all_names += [name + f'_{num_submodels}{split_type}{k}_best.h5' for k in range(num_submodels)]
    #                 all_labels += [label_names[i] for _ in range(num_submodels)]
    #             if args.dataset == 'harvard_polyps':
    #                 output_fn = os.path.join(logdir, "harvard_test_hypVSadn_{}_submodel_{}_votenet_reduce_{}_threshold{}_{}{}_dsc{}.pkl".format(num_submodels, temp_name, reduce_type, th, num_submodels,split_type, dsc))
    #             else:
    #                 # output_fn = os.path.join(logdir,
    #                 #                          "{}_brisque{}_hpVSadVSss_vote_{}_submodel_{}_votenet_reduce_{}_threshold{}_{}{}_dsc{}.pkl".format(
    #                 #                              t, str(opt) ,num_submodels, temp_name, reduce_type, th, num_submodels, split_type,
    #                 #                              dsc))
    #                 output_fn = os.path.join(logdir,
    #                                          "hypVSadn.pkl")
    #             # votenet_main(all_names, all_labels, imageset_dir,output_fn, reduce_type, th, output_weigths)
    #             try:
    #                 print(output_fn)
                    
    #                 main(all_names[0],all_names, imageset_dir,output_fn)
    #             except OSError:
    #                 print("No file found")

    # vote_net_same_base(model_names, 3, (256, 256, 3), labels=label_names)
                # vote(model_names, label_names, output_fn)
                # if os.path.exists(output_fn):
                #     analyse_df_multiclass(output_fn, 3)
                    # analyse_df_multiclass(output_fn.replace(".pkl", "_voted.pkl"), 3)
                    # except FileNotFoundError:
                    #     print("No file found")
    # create_video(os.path.join("results/vote_simple/hypVSall_adnVSall_sspVSall_HDall2022_efficientnet_regularized0.0_256x256_1in_nf64_bnTrue_fcdo0.5_lossls0.1_balancedTrue_h5_5fold4_vote.pkl"))
