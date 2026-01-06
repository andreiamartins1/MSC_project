import os
import sys
sys.path.append('.')
sys.path.append('..')

import random
import tensorflow as tf
from utils.Dataset import Dataset as BaseDataset


class Dataset(BaseDataset):
    def __populate_attrs__(self):
        for fn in os.listdir(self.image_set):
            if fn[:3] == 'cat':
                if self.labels.count(0) == 1000:
                    continue
                self.labels.append(0)
            else:
                if self.labels.count(1) == 1000:
                    continue
                self.labels.append(1)
            self.filenames.append(os.path.join(self.image_set, fn))
        print("WARNING: Manually limited number of samples.")
        self.class_names = {0: "cat", 1: "dog"}

    def combine_images_labels(self, file_path: tf.Tensor):
        imgs = []
        for i in range(self.nb_inputs):
            img = self.load_img(file_path)
            img.set_shape(self.image_shape)
            imgs.append(img)
        label = tf.convert_to_tensor(self.get_label(file_path), dtype=tf.int64)
        if self.nb_inputs > 1:
            return imgs, label
        else:
            return imgs[0], label


class TripletDataset(Dataset):
    def __init__(self, image_set, batch_size=64, target_size=(256, 256), loading_size=None, nb_inputs=1, nb_classes=2, shuffle=True, augmentations={}, balanced=False, repeat=True, do_cropping=None):
        super().__init__(image_set, batch_size=batch_size, target_size=target_size, loading_size=loading_size, nb_inputs=nb_inputs, nb_classes=nb_classes, shuffle=shuffle, augmentations=augmentations, balanced=balanced)
        self.do_cropping = do_cropping  # inert argument just for API conformity with polyp datasets

        self.filenames_per_class = {}
        for class_id in self.class_names.keys():
            self.filenames_per_class[class_id] = [self.filenames[i] for i, x in enumerate(self.labels) if x == class_id]

        # overwrite tf dataset object from parent class
        self.tfdataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        if shuffle:
            self.tfdataset = self.tfdataset.shuffle(buffer_size=len(self.filenames), reshuffle_each_iteration=True)
        if repeat:
            self.tfdataset = self.tfdataset.repeat()

        self.tfdataset = self.tfdataset.map(lambda x: tf.py_function(func=self.load_triplet, inp=[x], Tout=(tf.float32, tf.float32, tf.float32, tf.string, tf.string, tf.string, tf.float32, tf.float32, tf.float32)), num_parallel_calls=12, deterministic=False)
        self.tfdataset = self.tfdataset.map(lambda anchor, positive, negative, anchor_fn, pos_fn, neg_fn, anchor_label, pos_label, neg_label: tf.py_function(func=self.augment_triplet, inp=[anchor, positive, negative, anchor_fn, pos_fn, neg_fn, anchor_label, pos_label, neg_label], Tout=(tf.float32, tf.float32, tf.float32, tf.string, tf.string, tf.string, tf.float32, tf.float32, tf.float32)), num_parallel_calls=12, deterministic=False)
        self.tfdataset = self.tfdataset.map(lambda anchor, positive, negative, anchor_fn, pos_fn, neg_fn, anchor_label, pos_label, neg_label: {'anchor_input': anchor, 'positive_input': positive, 'negative_input': negative, 'anchor_fn': anchor_fn, 'pos_fn': pos_fn, 'neg_fn': neg_fn, 'anchor_label': anchor_label, 'pos_label': pos_label, 'neg_label': neg_label}, num_parallel_calls=12, deterministic=False)  # rename for multi-input network
        self.tfdataset = self.tfdataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def load_anchor(self, file_path: tf.Tensor):
        '''Load anchor image'''
        label = self.get_label(file_path)
        anchor = self.load_img(file_path)
        return anchor, file_path.numpy(), label

    def load_positive(self, file_path: tf.Tensor):
        '''Load random image with same label'''
        label = self.get_label(file_path)
        filename = random.choice(self.filenames_per_class[tf.math.argmax(label).numpy()])
        return self.load_img(filename), filename, label

    def load_negative(self, file_path: tf.Tensor):
        '''Load random image with different label'''
        label = self.get_label(file_path)
        filename = random.choice(self.filenames_per_class[1 - tf.math.argmax(label).numpy()])
        return self.load_img(filename), filename, label

    def load_triplet(self, file_path: tf.Tensor):
        anchor, anchor_fn, anchor_label = self.load_anchor(file_path)
        positive, pos_fn, pos_label = self.load_positive(file_path)
        negative, neg_fn, neg_label = self.load_negative(file_path)
        return anchor, positive, negative, anchor_fn, pos_fn, neg_fn, anchor_label, pos_label, neg_label

    def augment_triplet(self, anchor, positive, negative, anchor_fn, pos_fn, neg_fn, anchor_label, pos_label, neg_label):
        for f in self.augmenter.augmenters:
            anchor, _ = f(anchor, None)
            positive, _ = f(positive, None)
            negative, _ = f(negative, None)
        return anchor, positive, negative, anchor_fn, pos_fn, neg_fn, anchor_label, pos_label, neg_label


def main():
    import numpy as np
    target_size = (128, 128)
    augmentations = {
        "rotate": 90.,
        "flip_horizontal": True,
        "flip_vertical": True,
        "brightness": 0.2}

    ds = Dataset('data/catsndogs/train/', target_size=target_size, augmentations=augmentations, nb_inputs=1)
    print(ds.tfdataset.element_spec)
    for img, label in ds.tfdataset.take(3):
        print("img shape", np.asarray(img).shape, " label ", np.asarray(label).shape)


if __name__ == '__main__':
    main()
