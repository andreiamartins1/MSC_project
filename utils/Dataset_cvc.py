import os
import sys
sys.path.append('.')
sys.path.append('..')

import csv
from utils.Dataset import Dataset as BaseDataset
from utils.Dataset import TripletDataset as BaseTripletDataset


class Dataset(BaseDataset):
    def __populate_attrs__(self):
        with open(os.path.join(self.image_set, 'labels.csv')) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            # This skips the first row of the CSV file.
            next(reader)

            for row in reader:
                img_fn = os.path.join(self.image_set, 'images', '{}.png'.format(row[0]))
                mask_fn = os.path.join(self.image_set, 'masks', '{}.png'.format(row[0]))

                # for local visualizations
                # print("WARNING: Loading from local paths.")
                # img_fn = img_fn.replace('/DATASERVER/MIC/SHARED/ENDOSCOPY/EUROPOL/', '/Users/teelbo0/apollo_shared/')
                # mask_fn = mask_fn.replace('/DATASERVER/MIC/DATA/STAFF/teelbo0/tmp/', '/Users/teelbo0/apollo_tmp/')

                self.filenames.append(img_fn)
                self.ann_filenames.append(mask_fn)
                self.labels.append(int(row[1]))

        self.class_names = {0: "HP", 1: "ADN"}
        self.nb_classes = 2

        # # include non polyp images
        # non_polyp_dir = '/DATASERVER/MIC/DATA/STAFF/teelbo0/tmp/PolypClassificationMI/data/neg_frames'
        # if 'val' in self.image_set:
        #     non_polyp_dir = non_polyp_dir + '_val'
        # for fn in os.listdir(non_polyp_dir):
        #     if fn.startswith("._"):
        #         continue
        #     if fn == "empty_mask.png":
        #         continue
        #     if os.path.getsize(os.path.join(non_polyp_dir, fn)) == 0:
        #         continue  # there are some erroneous zero-byte files in this folder. No clue why
        #     self.filenames.append(os.path.join(non_polyp_dir, fn))
        #     self.ann_filenames.append(os.path.join(non_polyp_dir, 'empty_mask.png'))
        #     self.labels.append(2)

        # self.class_names = {0: "HP", 1: "ADN", 2: "non-polyp"}
        # self.nb_classes = 3

    def get_filepath_from_same_polyp(self, file_path):
        '''
        Only one image from each polyp
        '''
        return file_path


class TripletDataset(BaseTripletDataset):
    def __populate_attrs__(self):
        with open(os.path.join(self.image_set, 'labels.csv')) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            # This skips the first row of the CSV file.
            next(reader)

            for row in reader:
                img_fn = os.path.join(self.image_set, 'images', '{}.png'.format(row[0]))
                mask_fn = os.path.join(self.image_set, 'masks', '{}.png'.format(row[0]))

                # for local visualizations
                # print("WARNING: Loading from local paths.")
                # img_fn = img_fn.replace('/DATASERVER/MIC/SHARED/ENDOSCOPY/EUROPOL/', '/Users/teelbo0/apollo_shared/')
                # mask_fn = mask_fn.replace('/DATASERVER/MIC/DATA/STAFF/teelbo0/tmp/', '/Users/teelbo0/apollo_tmp/')

                self.filenames.append(img_fn)
                self.ann_filenames.append(mask_fn)
                self.labels.append(int(row[1]))

        self.class_names = {0: "HP", 1: "ADN"}
        self.nb_classes = 2

        # # include non polyp images
        # non_polyp_dir = '/DATASERVER/MIC/DATA/STAFF/teelbo0/tmp/PolypClassificationMI/data/neg_frames'
        # if 'val' in self.image_set:
        #     non_polyp_dir = non_polyp_dir + '_val'
        # for fn in os.listdir(non_polyp_dir):
        #     if fn.startswith("._"):
        #         continue
        #     if fn == "empty_mask.png":
        #         continue
        #     if os.path.getsize(os.path.join(non_polyp_dir, fn)) == 0:
        #         continue  # there are some erroneous zero-byte files in this folder. No clue why
        #     self.filenames.append(os.path.join(non_polyp_dir, fn))
        #     self.ann_filenames.append(os.path.join(non_polyp_dir, 'empty_mask.png'))
        #     self.labels.append(2)

        # self.class_names = {0: "HP", 1: "ADN", 2: "non-polyp"}
        # self.nb_classes = 3

    def get_filepath_from_same_polyp(self, file_path):
        '''
        Only one image from each polyp
        '''
        return file_path


def main():
    import numpy as np
    target_size = (64, 64)
    image_size = (int(1920), int(1080))
    augmentations = {
        "rotate": 90.,
        "flip_horizontal": True,
        "flip_vertical": True,
        "brightness": 0.2}
    ds = Dataset('data/cvc/train/', target_size=target_size, loading_size=image_size, augmentations=augmentations)
    print(ds.tfdataset.element_spec)
    for img, label in ds.tfdataset.take(3):
        print("img shape", np.asarray(img).shape, " label ", label.shape)


if __name__ == '__main__':
    main()
