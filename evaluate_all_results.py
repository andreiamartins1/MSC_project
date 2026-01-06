import os
import pandas as pd
import re
from eval_NN_kfold import main, votenet_main, analyse_df_multiclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf


# model_name_regex = "(.*)_HDall2022_(.*)_(regularized)?(.*)(\d*)x(\d*)_(\d*)in_nf(\d*)_bn(.*)_fcdo(.*)_convdo(.*)_lossls(.*)_lossDropout(\d\.\d+)_(shuffle)?_?(dsc0\d)?_?(reflect)?_?(\d?(fold|bootstrap)\d)_(\w*)\.?(h5|hdf5)?"
#model_name_regex = "\d*_(.*)_(manualTrue_)?efficientnet_(\d*)x(\d*)_regularized(.*)_bn(.*)_fcdo(.*)_convdo(.*)_loss_gc(.*)_sgd_(\d?(fold|bootstrap)\d)_(\w*)\.?(h5|hdf5)?"pp
POSSIBLE_TYPES = {
        "0vs12": ("0", ("1", "2")),
        "1vs02": ("1", ("0", "2")),
        "2vs01": ("2", ("0", "1")),
        "0vs1": ("0", "1"),
        "0vs2": ("0", "2"),
        "1vs2": ("1", "2"),
        "0vs1vs2": ("0", "1", "2"),        
        "0vs1vs2vs3": ("0", "1", "2", "3"),        
    }

def find_all_models(model_dir="data/snapshots/all"):
    
    walker = os.walk(model_dir)
    curr_dir, folders, files = next(walker)
    print(len(folders), len(files))
    regex =re.compile(model_name_regex)
    all = []
    for folder in folders:
        if folder == ".AppleDouble":
            continue
        if "best" in folder:
            new_folder = folder.replace("best", "final")
        else:
            new_folder = folder
            folder = folder.replace("final", "best")

        if folder in all:
                continue
        
        if new_folder in folders and folder in folders:
            all.append(folder.replace("final", "best"))

    for file in files:
        if file == ".AppleDouble":
            continue
        if "best" in file:
            new_file = file.replace("best", "final")
        else:
            new_file = file
            file = file.replace("final", "best")
        if file in all:
                continue
        if new_file in files and file in files:
            all.append(file.replace("final", "best"))
        # print(new_file)

    grouped = {}
    for name in all:
        if "fold" not in name and "bootstrap" not in name:
            continue
        if "cat_xe" not in name:
            continue
        # print(name)
        splits = name.split(".")
        file_extension = splits[-1]
        # print(file_extension)
        if file_extension[0] == "h":
            file_extension = "_" + file_extension
        else:
            file_extension = ""

        splits = name.split("_")
        # print(splits)
        base_name = "_".join(splits[1:-2]) + file_extension
        if base_name not in grouped:
            grouped[base_name] = []
        grouped[base_name].append(name)
        # num = splits[-2][0]
        # if num == "f" or num == "b":
        #     num = "5"
        # last_num = str(int(splits[-2][-1]) + 1)

        # if num == last_num:
        #     grouped.update({base_name: [base_name.replace(file_extension, "")[:-1] + str(i) + "_" + splits[-1] for i in range(int(num))]})

    print(len(all), len(grouped))
    # print(grouped)
    return all, grouped

def get_all_pkl_files(data_dir="logs"):
    walker = os.walk(data_dir)
    pkl_files = []
    for c, fo, fi in walker:
        pkl_files += [c + "/" + f for f in fi if ".pkl" in f]
    return pkl_files    



def create_all_results(models, grouped, output_location):
    os.makedirs(output_location, exist_ok=True)
    walker = os.walk(output_location)
    _, _, files = next(walker)
    print(files)
    regex = re.compile(".*dsc(\d*).*")


    for model_type, models in grouped.items():
        new_batch_size = 128
        output_name = f"{model_type}.pkl"
        print("#################################")
        print(output_name)
        print("#################################")
        if output_name in files:
            continue
        if "vgg" in output_name:
            continue
        # if "1in" not in output_name:
        #     continue
        print(models)
        output_name = os.path.join(output_location, output_name)
        split_type = "fold"
        model_name = model_type
        # splits = model_type.split("_")
        # model_name = "_".join(splits[:-1])
        # split_type = splits[-1]
        # num_models = split_type[-1]
        # split_type = split_type.replace(num_models, "")
        num_models = 5
        # split_type = split_type.replace(str(num_models), "")
                
        # m = re.match(regex, model_type)
        # if m is not None:
        #     imageset_dir = 'data/imagesets_characterisation_1080p/{}{}_all{}/test.txt'.format(num_models, split_type, m[1])
        # else:
        imageset_dir = 'data/imagesets_1080p/{}{}_all/test.txt'.format(num_models, split_type)

        main(model_name, models, imageset_dir, output_name, split_type, num_models)

def analyse_single_results(results_location):

    if not os.path.exists("all_results.pkl"):

        walker = os.walk(results_location)
        _, _, pickle_files = next(walker)

        print(len(pickle_files))

        df = None
        # pc = 'pred_class_'
        # conf = 'confidence_'
        # hc = 'hc_'
        # vc = 'vote_cls'

        all_pred_cols = []
        all_conf_cols = []
        all_match_cols = []
        lenths_per_type = []

        for pickle_file in tqdm(pickle_files):
            # print(pickle_file)
            df_fn = os.path.join(results_location, pickle_file)

            begin = pickle_file.split("_")[0]
            # print(begin)
            # if begin != "adnVSall":
            #     continue

            l = POSSIBLE_TYPES[begin]
            nb_classes = len(l)
            
            df_new = pd.read_pickle(df_fn)

            t = ".".join(pickle_file.split(".")[:-1])

            mapper = {i: t+"_"+i for i in df_new.columns if "fold" in i}
            df_new = df_new.rename(columns=mapper)
            # df_new - df_new.fillna([0.,0.,0.])
            pred_cols = [i for i in df_new.columns if "pred_class" in i]
            conf_cols = [i for i in df_new.columns if "pred" in i and 'class' not in i]
            match_cols = [i for i in df_new.columns if "match" in i]
            # print(conf_cols)
            all_pred_cols += pred_cols
            all_conf_cols += conf_cols
            all_match_cols += match_cols
            lenths_per_type.append(len(pred_cols))


            # print(df_new)
            df_new[['img_fns']] = df_new[['img_fns']].applymap(lambda x: str(x))

            df_new[conf_cols] = df_new[conf_cols].applymap(lambda x: [x[0] if str(i) in l[0] else x[1] if str(i) in l[1] else 0. for i in range(nb_classes)] if isinstance(x, list) else 0.)
            if df is None:
                df = df_new.copy()
            else:
                df = df.merge(df_new[list(mapper.values()) + ['img_fns']], how='left', left_on='img_fns', right_on='img_fns')
        # df = df.fillna(0.0)
        # df[conf_cols] = df[conf_cols].replace(0.0, '000')
        # df[conf_cols] = df[conf_cols].applymap(lambda x: [float(i) for i in x])
        print(df.columns)
        df.to_pickle("all_results.pkl")
    else:
        df = pd.read_pickle("all_results.pkl")
        all_pred_cols = [i for i in df.columns if "pred_class" in i]
        all_conf_cols = [i for i in df.columns if "pred" in i and 'class' not in i]
        all_match_cols = [i for i in df.columns if "match" in i]
        print("Loaded df")


    # Calculate the correctness per image and display the difficult and easy images
    df['avg_match'] = df[[i for i in all_match_cols if "5fold" in i]].mean(axis=1)

    for t in POSSIBLE_TYPES.keys():
        df[t +"_avg_match"] = df[[i for i in all_match_cols if t in i and "5fold" in i]].mean(axis=1)
    # new_df = df[['avg_match'] + [t + "_avg_match" for t in POSSIBLE_TYPES.keys()]]
    avg_matchs = ['avg_match'] + [t + "_avg_match" for t in POSSIBLE_TYPES.keys()]

    new_df = df[["seq", "polyp_id"] + avg_matchs].groupby(by="seq")

    for name, grouped in new_df:
        grouped[avg_matchs[:4]].plot.line(subplots=True, ylim=(0., 1.0))

        plt.savefig(f"data/figures/total_results/avg_results_polyp{name}.png")
        plt.close()



    # Calculate the correctness per sequence and polyp and show the difficult and easy ones


    # Calculate the metric per hyperparameter and see the different effects

    pass

def create_vote_results(models, grouped, output_location, fold=None, input_subfolder=""):
    os.makedirs(output_location, exist_ok=True)
    walker = os.walk(output_location)
    _, _, files = next(walker)
    # print(files)
    regex = re.compile(".*dsc(\d*).*")
    # print(grouped)

    used_models = []

    list_of_used_types = [
        # ["0vs12"],
        # ["1vs02"],
        # ["2vs01"],
        ["1vs02", "2vs01"],
        ["0vs12", "2vs01"],
        ["0vs12", "1vs02"],
        ["0vs12", "1vs02", "2vs01"],
        ["0vs12", "1vs02", "2vs01", "0vs1", "1vs2", "0vs2"],
        ["0vs1", "1vs2", "0vs2"],
        # ["0vs1vs2"]
    ]
    # used_types = ["hypVSall"]
    # used_types = ["adnVSall"]
    # used_types = ["sspVSall"]
    # used_types = ["adnVSall", "sspVSall"]
    # used_types = ["hypVSall", "sspVSall"]
    # used_types = ["hypVSall", "adnVSall"]
    # used_types = ["hypVSall", "adnVSall", "sspVSall"]
    # used_types = ["hypVSall", "adnVSall", "sspVSall", "hypVSadn", "adnVSssp", "hypVSssp"]
    # used_types = ["hypVSadn", "adnVSssp", "hypVSssp"]
    # used_types = ["hypVSadnVSssp"]
    model_weights = [1.0, 1.0, 1.0]

    list_of_used_types = [["0vs12", "1vs2"],]
    model_weights = [1.0, 0.5, 0.5]
    used_types = [["1vs02", "0vs2"],]
    model_weights = [0.5, 1.0, 0.5]

    used_types = [["2vs01", "0vs1"],]
    model_weights = [0.5, 0.5, 1.0]

    for used_types in list_of_used_types:

        for model_type in grouped.keys():
            print(model_type, grouped[model_type])
            if model_type in used_models:
                continue

            new_batch_size = 128
            output_name = f"{model_type}_vote.pkl"
            
            if "efficientnet" not in output_name:
                continue
            # if "1in" not in output_name:
            #     continue

            # if "3fold2" in output_name:
            #     continue

            # if "bootstrap" in output_name:
            #     continue

            # if "pretuned" in output_name:
            #     continue
            output_name = os.path.join(output_location, output_name)
            splits = model_type.split("_")
            test_type = splits[0]
            print(test_type)
            model_name = "_".join(splits[:-1])
            split_type = splits[-1]
            # num_models = split_type[-1]
            num_models = 5
            # split_type = split_type.replace(num_models, "")
            split_type = "fold"
            # num_models = int(num_models) + 1
            # split_type = split_type.replace(str(num_models), "")

            
            if test_type in used_types:
                not_usable = False
                local_used_types = []
                models = []
                for tt in used_types:
                    other_model_type = model_type.replace(test_type, tt)
                    local_used_types.append(other_model_type)
                    
                    if other_model_type not in grouped.keys():
                        not_usable = True
                        break
                    models += grouped[other_model_type]
                if not_usable:
                    continue
                output_name = output_name.replace(test_type, "_".join(used_types))
            else:
                continue
            
            models = [input_subfolder+m for m in models]
            
            if fold is not None:
                models = [m for m in models if "fold"+str(fold) in m]
                output_name = output_name.replace("vote.pkl", "1fold"+str(fold)+"_vote.pkl")

            if len(models) <= 1:
                continue

            print("#################################")
            print(output_name)
            print("#################################")
            if output_name.split("/")[-1] in files:
                continue
            
            
            

            # models = grouped[model_type]
            # print(models)
            labels = [POSSIBLE_TYPES[i.split("_")[1]] for i in models]
            print(models, labels)
                    
            m = re.match(regex, model_type)
            if m is not None:
                imageset_dir = 'data/imagesets_characterisation_1080p/{}{}_all{}/test.txt'.format(num_models, split_type, m[1])
            else:
                imageset_dir = 'data/imagesets_1080p/{}{}_all/test.txt'.format(num_models, split_type)

            votenet_main(models, labels, imageset_dir,  output_name, "vote", 0.5, model_weights)
        #     main(model_name, models, imageset_dir, output_name, split_type, num_models)

            used_models += local_used_types
            # used_models += [model_type]
            tf.keras.backend.clear_session()

def analyse_vote_results(results_location, nb_classes=3):
    walker = os.walk(results_location)
    folder, _, files = next(walker)
    print(files)
    total_string = ""

    for file in sorted(files):
        if "test" in file:
            continue
        if os.path.exists('data/figures/cms/{}_cm_frames_0.30.png'.format(file.replace(".pkl", ""))):
            continue
        total_string += "\\section*{"+file+"}\n"
        total_string += "%#################################\n"
        print("#################################")
        print(file)
        print("#################################")
        total_string += analyse_df_multiclass(os.path.join(folder, file), nb_classes)
    with open(os.path.join(results_location,"test_output"), "w") as f:        
        f.write(total_string)








if __name__ == "__main__":
    models, grouped = find_all_models()
    # pkl_files = get_all_pkl_files()
    # # print(grouped)
    # create_all_results(models, grouped, 'results/all_gc')
    # # # analyse_single_results('results')
    # for fold in range(5):
    create_vote_results(models, grouped, f'results/vote_gc', fold=None, input_subfolder='all/')
    # analyse_vote_results('results/vote_simple_latest', 3)