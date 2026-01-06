import cv2
import numpy as np
from tqdm import tqdm
import os
from libsvm import svmutil
from brisque import calculate_brisque_features
import pickle


input_folder = "data/imagesets_1080p/"
output_folder = "data/imagesets_1080p/5fold_crop_brisque_all"

input_file = "total_all.txt"


def laplacian_blur(image):
    l = cv2.Laplacian(image, cv2.CV_64F)

    mu, sigma = cv2.meanStdDev(l)
    return sigma[0][0] * sigma[0][0]


def scale_features(features):
    with open('normalize.pickle', 'rb') as handle:
        scale_params = pickle.load(handle)
    min_ = np.array(scale_params['min_'])
    max_ = np.array(scale_params['max_'])
    return -1 + (2.0 / (max_ - min_) * (features - min_))

def calculate_image_quality_score(brisque_features):
    model = svmutil.svm_load_model('brisque_svm.txt')
    scaled_brisque_features = scale_features(brisque_features)
    
    x, idx = svmutil.gen_svm_nodearray(
        scaled_brisque_features,
        isKernel=(model.param.kernel_type == svmutil.PRECOMPUTED))
    
    nr_classifier = 1
    prob_estimates = (svmutil.c_double * nr_classifier)()
    
    return svmutil.libsvm.svm_predict_probability(model, x, prob_estimates)
    
class Brisque(object):

    def __init__(self) -> None:
        self.classifier = svmutil.svm_load_model('allmodel.txt')

        self.feature_stash = []
        self.current_seq = ""
        self.lines = []
        self.out_text_counter = 0
        self.intermediate_output_folder = output_folder + "/intermediate"

        os.makedirs(self.intermediate_output_folder, exist_ok=True)
    
    def calculate_image_quality_score(self, brisque_features):
        # scaled_brisque_features = scale_features(brisque_features)
        
        x, idx = svmutil.gen_svm_nodearray(
            brisque_features,
            isKernel=(self.classifier.param.kernel_type == svmutil.PRECOMPUTED))
        
        nr_classifier = 1
        prob_estimates = (svmutil.c_double * nr_classifier)()
    
        return svmutil.libsvm.svm_predict_probability(self.classifier, x, prob_estimates)

    def write_and_reset(self):
        features = np.array(self.feature_stash)
        max_ = np.max(features, axis=0)
        min_ = np.min(features, axis=0)
        scaled_features = -1 + (2.0 / (max_ - min_) * (features - min_))
        out_text = ""
        
        for i, line in enumerate(self.lines):
            score = self.calculate_image_quality_score(scaled_features[i,:])
            line += [str(score), ]
            out_text += " ".join(line) + "\n"

        with open(os.path.join(self.intermediate_output_folder, str(self.out_text_counter) + input_file), "w") as f:
            f.write(out_text)

        self.lines = []
        self.feature_stash = []
        self.out_text_counter += 1


    
    def add_image(self, image, line):
        if line[0].split("/")[-2] != self.current_seq and len(self.feature_stash) > 1:
            
            self.write_and_reset()
        self.current_seq = line[0].split("/")[-2]
        brisque_features = calculate_brisque_features(image)
        downscaled_image = cv2.resize(image, None, fx=1/2, fy=1/2, interpolation = cv2.INTER_CUBIC)
        downscale_brisque_features = calculate_brisque_features(downscaled_image, kernel_size=7, sigma=7/6)
        features = np.concatenate((brisque_features, downscale_brisque_features))
        self.lines.append(line)
        self.feature_stash.append(features)
    
    def get_text(self):
        return self.out_text

        




def brisque(image):
    features = calculate_brisque_features(image)

    downscaled_image = cv2.resize(image, None, fx=1/2, fy=1/2, interpolation = cv2.INTER_CUBIC)
    downscale_brisque_features = calculate_brisque_features(downscaled_image, kernel_size=7, sigma=7/6)

    features = np.concatenate((features, downscale_brisque_features))
    features = scale_features(features)
    return calculate_image_quality_score(features)



# def fft_blur(image):
#     f = np.fft.fft2(image)

def main():
    os.makedirs(output_folder, exist_ok=True)

    # w = os.walk(input_folder)
    # root, dirs, files = next(w)
    
    # for input_file in files:
    #     if input_file




    with open(os.path.join(input_folder, input_file), "r") as f:
        lines = f.read()
    
    lines = lines.split("\n")
    
    # Limitar para as primeiras N linhas (exemplo: 1000 linhas)
    N = 1000
    lines = lines[:N]

    new_lines = ""

    current_sequence = ""
    best_line = []
    best_var = 0

    brisque = Brisque()

    for line in tqdm(lines):
        if line != "":
            line = line.split(" ")
            polyp_type = line[2]
            # print(polyp_type)
            if polyp_type != "-1":
                image_fn = line[0]
                # image_fn.replace('JPEGImages/1080p/', 'CROPS/256p/')

                # sequence = image_fn.split("/")[-2]
                # if sequence != current_sequence:
                #     new_lines += " ".join(best_line) +"\n"
                #     current_sequence = sequence
                #     best_line = []
                #     best_var = 0

                
                image = cv2.imread(image_fn)
                if image is not None:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    brisque.add_image(gray, line)
                    # print(image_fn.split("/")[-2],laplacian_blur(gray))
                    # var = brisque(gray)
                    # print(var)
                    # line += [str(var),]
                    # new_lines += " ".join(line) +"\n"
                # if var > best_var:
                #     best_var = var
                #     best_line = line + [str(var),]
            else:
                var = 0
    brisque.write_and_reset()


def stitch_all():
    intermediate_folder = output_folder + "/intermediate"

    w = os.walk(intermediate_folder)
    base, dirs, files = next(w)
    
    new_file = "total_all.txt"
    
    s = ""
    for i in range(len(files)):
        with open(os.path.join(intermediate_folder, f"{i}{new_file}"), "r") as f:
            s += f.read()
    
    with open(os.path.join(output_folder, new_file), "w") as f:
        f.write(s)






if __name__ == "__main__":
    main()
    stitch_all()