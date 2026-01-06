import os
import sys
sys.path.append('.')
sys.path.append('..')

import pandas as pd
import numpy as np
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import random
import cv2
from operator import itemgetter
from tqdm import tqdm, trange
import warnings
warnings.filterwarnings("always")

problem_translation = {
    "hpVSadVSss": ("0","1","2"),
    "hpVSall": ("0",("1","2")),
    "adVSall": ("1",("0","2")),
    "ssVSall": ("2",("0","1")),
    "hpVSad": ("0","1"),
    "hpVSss": ("0","2"),
    "adVSss": ("1","2"),
}

problem_inverse_translation = {
    "hpVSadVSss": (0,1,2),
    "hpVSall": (0,1, 1),
    "adVSall": (1,0, 1),
    "ssVSall": (1,1, 0),
    "hpVSad": (0,1, -1),
    "hpVSss": (0,-1,1),
    "adVSss": (-1, 0,1),
}

file_name_base = "enhancedContrast_brisque{n}_{vs}_threshold0.0_5fold.pkl"
logdir = 'results'
imageset_dir = 'data/imagesets_1080p/5fold_crop_brisque_all/5fold_all/test.txt'
brisque_options = [1,5,50]
problems = ["hpVSadVSss","hpVSall", "adVSall", "ssVSall", "hpVSad", "hpVSss", "adVSss"]


def load_brisque_values():
    lookup = dict()
    with open(imageset_dir, "r") as f:
        lines = f.readlines()
    for line in lines:
        img_fn, ann_fn, label, polyp_id, brisque = line.replace("\n", "").split(" ")
        lookup.update({img_fn.replace('JPEGImages/1080p/', 'CROPS/256p/'): float(brisque), 
                       img_fn.replace('JPEGImages/1080p/', 'CROPS/256p_contrast_enhanced/'): float(brisque)})
    return lookup


def clean_df(df ,lookup, df_name, new_df=None):
    if new_df is None:
        df["img_fns"] = df["img_fns"].apply(lambda x: x[0])
        df["ann_fns"] = df["ann_fns"].apply(lambda x: x[0])
        df["brisque_value"] = df["img_fns"].apply(lambda x: lookup[x])
        new_df = df.copy()
        df = df.drop(columns=["exp_smooth", "cum_exp_smooth", "seqs", "key"] + [f"fold{i}_pred" for i in range(5)]+[f"fold{i}_pred_class" for i in range(5)]+[f"fold{i}_match" for i in range(5)])
    if "img_fns" in df.columns:
        df.set_index("img_fns")
    new_df.set_index("img_fns")
    problem = problem_translation[df_name.split("_")[0]]
    new_df=new_df[[f"fold{i}_pred" for i in range(5)]]
    
    # df = df.reset_index()
    # print(df.columns)
    for n in range(5):
        for i, clss in enumerate(problem):
        # print( df[f"fold{n}_pred"].tolist())
        # print(np.shape(df[f"fold{n}_pred"].tolist()))
            for j in clss:
                new_df[f"{df_name}_fold{n}_pred_{j}"] = new_df[f"fold{n}_pred"].apply(lambda x: x[i])
        # for i in new_df[f"fold{n}_pred"]:
        #     print(i)
        # print(new_df[[f"fold{n}_pred"]])
        new_df[f"{df_name}_fold{n}_pred_cls"] = new_df[f"fold{n}_pred"].apply(lambda x: np.argmax(x))
        new_df[f"{df_name}_fold{n}_pred"] = new_df[f"fold{n}_pred"].apply(lambda x: np.max(x))
    new_df = new_df.drop(columns=[f"fold{n}_pred" for n in range(5) if f"fold{n}_pred" in new_df.columns])
    df = df.join(new_df, how="outer")
    return df.copy()


def analyse_all_per_frame(df):
    gt = df["gt_class"].copy()

    prediction_problems = sorted(set(["_".join(col.split("_")[:3]) for col in df.columns if "pred" in col]))
    print(prediction_problems)
    results = {}
    for problem in prediction_problems:
        inverse_translation =  problem_inverse_translation[problem.split("_")[0]]
        translated_gt = gt.apply(lambda x: inverse_translation[int(x)])
        print(problem)
        # cm = confusion_matrix(translated_gt, df[f"{problem}_pred_cls"].fillna(-1), labels=range(len(problem_translation[problem.split("_")[0]])))
        results.update({problem:classification_report(translated_gt, df[f"{problem}_pred_cls"].fillna(-1), labels=range(len(problem_translation[problem.split("_")[0]])), output_dict=True)})
        # print(multilabel_confusion_matrix(translated_gt, df[f"{problem}_pred_cls"].fillna(-1), labels=range(len(problem_translation[problem.split("_")[0]]))))
    print(results)
    
    # for col in prediction_cols:
    #     s = col.split("_")
    #     problem = s[0]
    #     fold = s[2]
    #     cls = s[-1]
    #     if cls == "pred":
    #         continue
        

def analyse_per_polyp(df, create_new=False):
    gt = df["gt_class"].copy()

    out_path = os.path.join(logdir, "enhanced_contrast_metrics_perpolyp.pkl")

    if create_new or not os.path.exists(out_path):

        prediction_problems = sorted(set(["_".join(col.split("_")[:3]) for col in df.columns if "pred" in col]))
        print(prediction_problems)
        results = {}
        for problem in prediction_problems:
            print(problem)
            inverse_translation =  problem_inverse_translation[problem.split("_")[0]]
            translated_gt = gt.apply(lambda x: inverse_translation[int(x)])
            for polyp_id in pd.unique(df["polyp_id"]):
                # print(polyp_id)
                sub_gt = translated_gt[df["polyp_id"] == polyp_id]
                current_gt = sub_gt.iloc[0]
                
                # print(sub_gt)
                if len(sub_gt) > 0 and current_gt != "-1":
                    if polyp_id not in results:
                        results[polyp_id] = {}
                    # cm = confusion_matrix(translated_gt, df[f"{problem}_pred_cls"].fillna(-1), labels=range(len(problem_translation[problem.split("_")[0]])))
                    # print(sub_gt)
                    # print(df[f"{problem}_pred_cls"][df["polyp_id"] == polyp_id].fillna(-1))
                    results[polyp_id].update({problem:classification_report(sub_gt, df[f"{problem}_pred_cls"][df["polyp_id"] == polyp_id].fillna(-1), labels=[current_gt], output_dict=True,zero_division=0)["weighted avg"]})
                    # print(multilabel_confusion_matrix(translated_gt, df[f"{problem}_pred_cls"].fillna(-1), labels=range(len(problem_translation[problem.split("_")[0]]))))
        reform = {key:{("_".join(problem.split("_")[:2]),problem.split("_")[2], metric):val for problem, innervalue in value.items() for metric, val in innervalue.items()} for key, value in results.items()}    

        # for polyp_id in sorted(results.keys()):
        #     print(polyp_id)
        #     for problem in sorted(results[polyp_id].keys()):
        #         print(problem)
        #         print(results[polyp_id][problem])
        
        results_df = pd.DataFrame().from_dict(reform)
    
        results_df.to_pickle(out_path)
    else:
        results_df = pd.read_pickle(out_path)
    print(results_df)
    mask = results_df.loc[(slice(None), slice(None), "support"), :].mean(axis=0) > 25
    filtered_results_df = results_df.loc[:, mask]
    # print(mask)
    # print(results_df.mean(axis=1).to_string())
    print(results_df.mean(axis=1).unstack(-1).groupby(level=[0]).mean().to_string())
    print(results_df.mean(axis=1).unstack(-1).groupby(level=[1]).mean().to_string())

    # print(filtered_results_df.mean(axis=1).to_string())
    print(filtered_results_df.mean(axis=1).unstack(-1).groupby(level=[0]).mean().to_string())
    print(filtered_results_df.mean(axis=1).unstack(-1).groupby(level=[1]).mean().to_string())
    




def main():
    
    lookup = load_brisque_values()
    output_name = os.path.join(logdir, "enhancedContrast_quality_all.pkl")
    if os.path.exists(output_name):
        base_df = pd.read_pickle(output_name)
    else:
        base_df = None
        for current_problem in problems:
            for brisque in brisque_options:
                df_fn = os.path.join(logdir,file_name_base.replace("{n}", str(brisque)).replace("{vs}", current_problem))
                if not os.path.exists(df_fn):
                    print("Can't find file", df_fn)
                df = pd.read_pickle(df_fn)

                if base_df is None:
                    base_df = clean_df(df, lookup, f"{current_problem}_brisque{brisque}")
                else:
                    base_df = clean_df(base_df, lookup, f"{current_problem}_brisque{brisque}", df)

        base_df['gt_class'] = base_df['gt_class'].astype(int)
        base_df.to_pickle(output_name)


    # analyse_all_per_frame(base_df)
    analyse_per_polyp(base_df)
    # col_names = [col for col in base_df.columns]
    # print(col_names)
    # gt_classes = base_df["gt_class"].unique()
    # print(f"Ground truth classes: {gt_classes}")

if __name__=="__main__":
    main()