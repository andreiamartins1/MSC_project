import os
import sys
sys.path.append('.')
sys.path.append('..')

import cv2
import numpy as np
import tensorflow as tf
import random
random.seed(1994)
from matplotlib import pyplot as plt

from utils.augmentations import Augmenter, preprocess_vgg16, preprocess_input

MAX_POSSIBLE_CLASSES = 3

def pad_ragged_list(l, fill_value=-1):
    '''
    For a given list of lists, pad each sublist to have the same length as the
    longest one.
    '''
    max_len = 0
    for subl in l:
        if len(subl) > max_len:
            max_len = len(subl)
    rectangle = np.full((len(l), max_len), fill_value=str(fill_value), dtype=object)
    # rectangle = [[]*max_len]*len(l)
    for i in range(len(l)):
        rectangle[i:i+1, 0:len(l[i])] = l[i]
    return rectangle


def get_bbox_mask(ann, padding=0.):
    '''
    Given a grayscalen segmentation result, the bounding box for the detection
    is returned after threshold. Its dimensions have been extended in every direction
    with the given padding.
    Returns a binary image with 1 inside the bboxes and 0 for the background
    '''
    res = np.zeros(ann.shape, dtype='uint8')
    contours, _ = cv2.findContours(ann.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        bbox = cv2.boundingRect(cnt)
        # res = cv2.rectangle(res, (np.clip(int(bbox[0] * (1 - padding)), 0, None), np.clip(int(bbox[1] * (1 - padding)), 0, None)), (np.clip(int((bbox[0]+bbox[2]) * (1 + padding)), 0, res.shape[1]-1), np.clip(int((bbox[1]+bbox[3]) * (1 + padding)), 0, res.shape[0]-1)), (255, 255, 255), -1)
        res = cv2.rectangle(res, (np.clip(int(bbox[0] - padding * bbox[2]), 0, None), np.clip(int(bbox[1] - padding * bbox[3]), 0, None)), (np.clip(int((bbox[0]+bbox[2]) + padding * bbox[2]), 0, res.shape[1]-1), np.clip(int((bbox[1]+bbox[3]) + padding * bbox[3]), 0, res.shape[0]-1)), (255, 255, 255), -1)
        # res = cv2.rectangle(res, (np.clip(int(bbox[0]), 0, None), np.clip(int(bbox[1]), 0, None)), (np.clip(int((bbox[0]+bbox[2])), 0, res.shape[1]-1), np.clip(int((bbox[1]+bbox[3])), 0, res.shape[0]-1)), (255, 255, 255), -1)

    return res > 0


def mask_img_w_bbox(img, ann, padding=0.):
    '''
    Given an image and a segmentation mask, the image will be masked with the
    bounding boxes for the given segmentation with the given padding around the box.
    '''
    mask = get_bbox_mask(ann, padding=padding)
    if np.sum(mask) <= 10:
        return img
    else:
        return (img * mask).astype('uint8')


def resize_image(img, size=(28,28)):

    h, w, c = img.shape

    # if h == w:
    #     return cv2.resize(img, size, cv2.INTER_CUBIC)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC
    # interpolation = cv2.INTER_CUBIC
    # interpolation = cv2.INTER_AREA

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)


def crop_to_rotated_bbox(img, ann, padding=0., output_size=(256,256)):
    '''
    Given an image and a segmentation mask, the image will be cropped to the
    rotated bounding box for the given segmentation with the given padding
    around the box.
    '''
    # Find largest contour in segmentation mask
    contours, _ = cv2.findContours(ann.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key = cv2.contourArea)

    # find rotated rectangle that fits
    bbox = cv2.minAreaRect(cnt)  # returns format ((center_x, center_y), (width, height), angle)

    # Get the square that contains this rectangle
    bbox = [list(x) if type(x) is tuple else x for x in bbox ]  # make mutable
    bbox[1][0] = bbox[1][1] = np.maximum(bbox[1][0], bbox[1][1]) #* (1+padding)
    bbox = tuple([tuple(x) if type(x) is list else x for x in bbox]) # make unmutable

    # Get real coordinates of box points
    box = cv2.boxPoints(bbox) # order: (top left, top right, bottom right, bottom left)
    box = np.int0(box)

    # Get transformation params
    width = int(bbox[1][0])
    height = int(bbox[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")     # coordinate of the points in box points after the rectangle has been straightened

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))

    # return resize_image(warped, output_size)
    return cv2.resize(warped, output_size)


def crop_to_bbox(img, ann, padding=0., output_size=(256, 256)):
    '''
    Given an image and a segmentation mask, the image will be cropped to the
    bounding box for the given segmentation with the given padding around the box.
    '''
    # step 1 - get biggest bounding box
    # cv2.imwrite('mask_cnt.png', cv2.cvtColor((255*ann).astype('uint8'), cv2.COLOR_GRAY2BGR))

    mask = get_bbox_mask(ann, padding=padding)

    # plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.imshow(img)
    # plt.subplot(1, 3, 2)
    # plt.imshow(ann)
    # plt.subplot(1, 3, 3)
    # plt.imshow(mask)
    # plt.savefig('mask.png')

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask.astype('uint8'), connectivity=4)
    sizes = stats[:, -1]

    if nb_components <= 1:
        return np.zeros(output_size[::-1] + (3,), dtype=np.float32)

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    mask = np.zeros(output.shape)
    mask[output == max_label] = 255

    # step 2 - crop to content of bbox
    (y, x) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = img[topy:bottomy+1, topx:bottomx+1]
    # print("Box size is W ", np.abs(topy - topx), " H ", np.abs(bottomy - topy))

    # step 3 - resize to fixed input size
    # return cv2.resize(out, output_size)
    # return resize_image(out, output_size)
    return cv2.resize(out, output_size)


def balance_imageset(img_fns, ann_fns, labels, polyp_ids, balance={"0": 0.5, "1": 0.5}, method='oversample', label_map=None):
    '''
    Balance the given imageset_fn for the classes provided in balance.

    balance example - {"0": 0.5, "1": 0.5}
    '''
    assert method == 'oversample', "balancing method not implemented"

    img_fns = np.array(img_fns)
    ann_fns = np.array(ann_fns)
    labels = np.array(labels)
    polyp_ids = np.array(polyp_ids)

    nb_samples = [np.count_nonzero(np.where(labels == int(x))) for x in balance.keys()]

    result_img_fns = []
    result_ann_fns = []
    result_labels = []
    result_polyp_ids = []

    # feed with dominant class
    dominant_class = np.argmax(nb_samples)

    for i, key in enumerate(balance.keys()):
        key = int(key)
        result_img_fns.extend(img_fns[np.where(labels == key)[0]])
        if len(ann_fns) > 0:
            result_ann_fns.extend(ann_fns[np.where(labels == key)[0]])
        result_labels.extend(labels[np.where(labels == key)[0]])
        if len(polyp_ids) > 0:
            result_polyp_ids.extend(polyp_ids[np.where(labels == key)[0]])

        if key != int(list(balance.keys())[dominant_class]):
            # Calculate number to oversample
            frac = balance[str(key)] / balance[list(balance.keys())[dominant_class]]
            nb = int(nb_samples[dominant_class]*frac) - nb_samples[i]

            # do oversampling
            indices = random.choices(np.where(labels == key)[0], k=nb)
            result_img_fns.extend(img_fns[indices])
            if len(ann_fns) > 0:
                result_ann_fns.extend(ann_fns[indices])
            result_labels.extend(labels[indices])
            if len(polyp_ids) > 0:
                result_polyp_ids.extend(polyp_ids[indices])


    return result_img_fns, result_ann_fns, result_labels, result_polyp_ids


def balance_imageset_perpolyp(img_fns, ann_fns, labels, polyp_ids, method='oversample'):
    '''
    Balance the given imageset_fn for the classes provided in balance.

    balance example - {"0": 0.5, "1": 0.5}
    '''
    assert method == 'oversample', "balancing method {} not implemented".format(method)

    nb_samples = [polyp_ids.count(x) for x in set(polyp_ids)]

    img_fns = np.array(img_fns)
    ann_fns = np.array(ann_fns)
    labels = np.array(labels)
    polyp_ids = np.array(polyp_ids)

    result_img_fns = []
    result_ann_fns = []
    result_labels = []
    result_polyp_ids = []

    # feed with dominant class
    dominant_class = np.argmax(nb_samples)

    for i, key in enumerate(set(polyp_ids)):
        # key = int(key)
        result_img_fns.extend(img_fns[np.where(polyp_ids == key)[0]])
        if len(ann_fns) > 0:
            result_ann_fns.extend(ann_fns[np.where(polyp_ids == key)[0]])
        result_labels.extend(labels[np.where(polyp_ids == key)[0]])
        if len(polyp_ids) > 0:
            result_polyp_ids.extend(polyp_ids[np.where(polyp_ids == key)[0]])

        if i != dominant_class:
            # do oversampling
            indices = random.choices(np.where(polyp_ids == key)[0], k=nb_samples[dominant_class] - nb_samples[i])
            result_img_fns.extend(img_fns[indices])
            if len(ann_fns) > 0:
                result_ann_fns.extend(ann_fns[indices])
            result_labels.extend(labels[indices])
            if len(polyp_ids) > 0:
                result_polyp_ids.extend(polyp_ids[indices])

    return result_img_fns, result_ann_fns, result_labels, result_polyp_ids


class Dataset(object):
    def __init__(self, image_set, batch_size=64, target_size=(256, 256), loading_size=None, nb_inputs=1, nb_classes=2, shuffle=True, repeat=True, augmentations={}, balanced=False, do_cropping=True, balance_polypids=False, selected_labels=None, oneVSall_outputs=False, top_n_frames=None):
        """Init."""
        self.target_size = tuple(target_size)
        self.image_shape = self.target_size + (3,)  # rgb
        if not loading_size:
            self.loading_size = self.target_size
        else:
            self.loading_size = tuple(loading_size)

        if not os.path.exists(image_set):
            raise IOError('Image set {} does not exist. Please provide a'
                          'valid file.'.format(image_set))
        self.image_set = image_set
        self.batch_size = batch_size
        self.nb_inputs = nb_inputs
        self.nb_classes = nb_classes
        self.do_cropping = do_cropping
        self.top_n_frames = top_n_frames

        if not selected_labels:
            self.selected_labels = [str(i) for i in range(self.nb_classes)]
            self.inverse_label_map = np.array(range(self.nb_classes), dtype=np.int64)
            self.label_map = self.inverse_label_map
            print("No selected_labels")
        else:
            inv_lm = []
            sl = []
            lm = np.zeros_like(range(MAX_POSSIBLE_CLASSES), dtype=np.int64)
            for k,j in enumerate(selected_labels):
                for i in j:
                    sl.append(i)
                    inv_lm.append(int(i))
                    lm[int(i)] = k
            self.selected_labels = sl
            self.inverse_label_map = np.array(inv_lm, dtype=np.int64)
            self.label_map = lm

        self.filenames = []
        self.ann_filenames = []
        self.labels = []
        self.polyp_ids = []

        self.__populate_attrs__()
        self.__print_statistics__()

        

        self.balanced = balanced
        self.balance_polypids = balance_polypids
        if self.balance_polypids:
            self.filenames, self.ann_filenames, self.labels, self.polyp_ids = balance_imageset_perpolyp(self.filenames, self.ann_filenames, self.labels, self.polyp_ids)
            print("AFTER BALANCING PER POLYP:")
            self.__print_statistics__()

        if self.balanced:
            balance = {}
            nb_selected = len(selected_labels)
            for labels in selected_labels:
                nb_in_class = len(labels)
                for j in labels:
                    balance[j] = 1. / (nb_selected * nb_in_class)
            self.filenames, self.ann_filenames, self.labels, self.polyp_ids = balance_imageset(self.filenames, self.ann_filenames, self.labels, self.polyp_ids, balance=balance)
            print("AFTER BALANCING CLASSES:")

        if self.label_map is not None:
            self.labels = self.label_map[self.labels].tolist()

        new_class_names = {}

        for i, label in enumerate(selected_labels):
            for j in label:
                label_id = int(i)
                if label_id not in new_class_names.keys():
                    new_class_names[i] = self.class_names[int(j)]
                else:
                    new_class_names[i] += '+'+self.class_names[int(j)]
        self.class_names = new_class_names
        self.__print_statistics__()

        # Use lookup table for annotation fns for fast finding
        self.ann_filenames_lot = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=self.filenames,values=self.ann_filenames),
            default_value=tf.constant(""),
            name="ann_filenames") if len(self.ann_filenames) > 0 else None

        # Use lookup table for labels for fast label finding
        self.labels_lot = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=self.filenames,values=self.inverse_label_map[self.labels].tolist()),
            default_value=tf.constant(-1),
            name="labels") if len(self.labels) > 0 else None

        # Use lookup table for labels for fast polyp id finding
        self.polyp_ids_lot = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=self.filenames,values=self.polyp_ids),
            default_value=tf.constant(-1),
            name="polyp_ids") if len(self.polyp_ids) > 0 else None

        # Use lookup table for labels for fast finding of filenames with polyp_id
        self.filenames_per_polyp_id = {}
        if len(self.polyp_ids) > 0:
            for i, idx in enumerate(self.polyp_ids):
                if idx in self.filenames_per_polyp_id.keys():
                    self.filenames_per_polyp_id[idx].append(self.filenames[i])
                else:
                    self.filenames_per_polyp_id[idx] = [self.filenames[i]]

        self.augmenter = Augmenter(augmentations, target_size=self.target_size, nb_inputs=self.nb_inputs)

        self.tfdataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        if shuffle:
            self.tfdataset = self.tfdataset.shuffle(buffer_size=len(self.filenames), reshuffle_each_iteration=True)
        if repeat:
            self.tfdataset = self.tfdataset.repeat()
        self.oneVSall_outputs = oneVSall_outputs

        AUTO = tf.data.experimental.AUTOTUNE
        self.tfdataset = self.tfdataset.map(lambda x: tf.py_function(func=self.combine_images_labels, inp=[x], Tout=(tf.float32, tf.int64)), num_parallel_calls=AUTO, deterministic=False)

        for f in self.augmenter.augmenters:
            self.tfdataset = self.tfdataset.map(f, num_parallel_calls=AUTO, deterministic=False)

        self.tfdataset = self.tfdataset.batch(self.batch_size)
        # TODO: Preprocess VGG is here
        # self.tfdataset = self.tfdataset.map(preprocess_vgg16, num_parallel_calls=AUTO, deterministic=False)
        self.tfdataset = self.tfdataset.map(preprocess_input, num_parallel_calls=AUTO, deterministic=False)
        self.tfdataset = self.tfdataset.prefetch(buffer_size=AUTO)


    def __populate_attrs__(self):
        with open(self.image_set) as f:
            content = f.readlines()
        content = [x.strip() for x in content] # get rid of newline statements
        temp_rows = []
        prev_seq = ""
        prev_polyp = ""
        for row in content:
            img_path, ann_path, label, polyp_id, quality_metric = row.split(" ")

            if self.do_cropping == 'preloaded':
                img_path = img_path.replace('JPEGImages/1080p/', 'CROPS/256p_contrast_enhanced/')
            
            seq = img_path.split("/")[-2]

            if "-1" in label:
                continue
            elif label in self.selected_labels and os.path.exists(img_path):
                if self.top_n_frames is None:
                    self.labels.append(int(label))
                    self.filenames.append(img_path)
                    self.ann_filenames.append(ann_path)
                    self.polyp_ids.append(int(polyp_id))
                else:
                    if polyp_id != prev_polyp or seq != prev_seq:

                        s = sorted(temp_rows, key=lambda row: float(row[-1]))
                        i = 0
                        for row in s:
                            if i == self.top_n_frames:
                                break
                            self.labels.append(int(row[2]))
                            self.filenames.append(row[0])
                            self.ann_filenames.append(row[1])
                            self.polyp_ids.append(int(row[3]))
                            i += 1
                        temp_rows = []
                    temp_rows.append((img_path, ann_path, label, polyp_id, quality_metric))
                    
            else:
                continue

            prev_polyp = polyp_id
            prev_seq = seq

        self.class_names = {0: "HP", 1: "ADN", 2: "SSP"}

    def __print_statistics__(self):
        def most_common(lst):
            return max(set(lst), key=lst.count)

        def least_common(lst):
            return min(set(lst), key=lst.count)

        def median_common(lst):
            frequency = [lst.count(x) for x in set(lst)]
            return list(set(lst))[np.argsort(frequency)[len(frequency)//2]]

        print("-"*50)
        print("### Dataset statistics ###")
        print("Loaded from {}".format(self.image_set))
        print("Total number of {} images.".format(len(self.filenames)))
        print("Total number of {} polyps. \n".format(len(list(set(self.polyp_ids)))))

        unique_labels = list(set(self.labels))
        print("Number of classes {}, i.e. {}".format(len(unique_labels), unique_labels))
        print("Number of images for each class: ")
        for i in range(len(unique_labels)):
            classname = unique_labels[i]
            count = self.labels.count(classname)
            percentage = round(100 * count / len(self.labels), 2)
            nunique_polyps = len(list(set([self.polyp_ids[k] for k in np.where(np.asarray(self.labels) == classname)[0]])))
            percentage_polyps = round(100 * nunique_polyps / len(list(set(self.polyp_ids))))
            print("\t Class {}({}) = {} images, {} % of total. - {} polyps, {} % of total".format(classname, self.class_names[int(classname)], count, percentage, nunique_polyps, percentage_polyps))

        print("\n")
        frequency = [self.polyp_ids.count(x) for x in set(self.polyp_ids)]
        print("Number of images per polyp: MIN {} - MED {} - MAX {}".format(min(frequency), np.median(frequency), max(frequency)))
        print("Respective polyp ids for  : MIN {} - MED {} - MAX {}".format(least_common(self.polyp_ids), median_common(self.polyp_ids), most_common(self.polyp_ids)))

        print("-"*50)

    def __len__(self):
        return len(self.filenames)

    def get_filepath_from_same_polyp(self, file_path):
        polyp_id = self.polyp_ids_lot.lookup(file_path)
        return tf.convert_to_tensor(random.choice(self.filenames_per_polyp_id[polyp_id.numpy()]), dtype=tf.string)

    def get_label(self, fn):
        y_hot = tf.keras.utils.to_categorical(self.label_map[self.labels_lot.lookup(fn)], self.nb_classes, dtype=np.int64)
        return y_hot

    def load_img(self, fn):
        try:
            img = tf.io.read_file(fn)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(img, self.loading_size)
        except:
            print("Could not load image {}".format(fn))

    def load_ann(self, fn):
        ann_fn = self.ann_filenames_lot.lookup(fn)
        ann = tf.io.read_file(ann_fn)
        ann = tf.image.decode_jpeg(ann, channels=1)
        ann = tf.image.convert_image_dtype(ann, tf.float32)
        return ann

    def combine_images_labels(self, file_path: tf.Tensor):
        imgs = []
        for i in range(self.nb_inputs):
            img = self.load_img(file_path)

            if self.do_cropping == 'preloaded':
                pass  # image already loaded in correct size and format
            elif self.do_cropping:
                ann = self.load_ann(file_path)
                ann = tf.image.resize(ann, img.shape[:2])

                padding = 0.2 if self.augmenter.augment_cropbox is None else random.uniform(self.augmenter.augment_cropbox[0], self.augmenter.augment_cropbox[1])
                img = tf.convert_to_tensor(crop_to_bbox(img.numpy(), ann.numpy(), padding=padding, output_size=self.target_size))
                img.set_shape(tf.TensorShape(self.image_shape))
            else:
                img = tf.image.resize(img, self.target_size)

            file_path = self.get_filepath_from_same_polyp(file_path)
            imgs.append(img)


        if self.oneVSall_outputs:
            label = np.zeros((2, self.nb_classes), dtype=int)
            label[0, :] = self.get_label(file_path)
            label[1, :] = 1-label[0, :]
            label = tf.convert_to_tensor(label, dtype=tf.int64)
        else:
            label = tf.convert_to_tensor(self.get_label(file_path), dtype=tf.int64)
        # label.set_shape(tf.TensorShape([2]))
        if self.nb_inputs > 1:
            imgs = tf.convert_to_tensor(imgs)
            return imgs, label
        else:
            return tf.convert_to_tensor(imgs[0]), label


class TripletDataset(Dataset):
    def __init__(self, image_set, batch_size=64, target_size=(256, 256), loading_size=None, nb_inputs=1, nb_classes=2, shuffle=True, repeat=True, augmentations={}, balanced=False, do_cropping=True):
        super().__init__(image_set, batch_size=batch_size, target_size=target_size, loading_size=loading_size, nb_inputs=nb_inputs, nb_classes=nb_classes, shuffle=shuffle, repeat=repeat, augmentations=augmentations, balanced=balanced, do_cropping=do_cropping)

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
        anchor = self.load_img(file_path)
        label = self.get_label(file_path)

        if self.do_cropping:
            ann = self.load_ann(file_path)
            ann = tf.image.resize(ann, anchor.shape[:2])

            padding = 0.2
            padding = 0.2 if self.augmenter.augment_cropbox is None else random.uniform(self.augmenter.augment_cropbox[0], self.augmenter.augment_cropbox[1])
            anchor = tf.convert_to_tensor(crop_to_bbox(anchor.numpy(), ann.numpy(), padding=padding, output_size=self.target_size))
            anchor.set_shape(self.image_shape)
        else:
            anchor = tf.image.resize(anchor, self.target_size)
        return anchor, file_path.numpy(), tf.math.argmax(label).numpy()

    def load_positive(self, file_path: tf.Tensor):
        '''Load random image with same label'''
        label = self.get_label(file_path)
        filename = random.choice(self.filenames_per_class[tf.math.argmax(label).numpy()])
        positive = self.load_img(filename)

        if self.do_cropping:
            ann = self.load_ann(tf.convert_to_tensor(filename))
            ann = tf.image.resize(ann, positive.shape[:2])

            padding = 0.2
            padding = 0.2 if self.augmenter.augment_cropbox is None else random.uniform(self.augmenter.augment_cropbox[0], self.augmenter.augment_cropbox[1])
            positive = tf.convert_to_tensor(crop_to_bbox(positive.numpy(), ann.numpy(), padding=padding, output_size=self.target_size))
            positive.set_shape(self.image_shape)
        else:
            positive = tf.image.resize(positive, self.target_size)
        return positive, filename, tf.math.argmax(label).numpy()

    def load_negative(self, file_path: tf.Tensor):
        '''Load random image with different label'''
        label = self.get_label(file_path)

        other_label = random.choice([n for n in list(self.class_names.keys()) if n != tf.math.argmax(label).numpy()])

        filename = random.choice(self.filenames_per_class[other_label])
        negative = self.load_img(filename)

        if self.do_cropping:
            ann = self.load_ann(tf.convert_to_tensor(filename))
            ann = tf.image.resize(ann, negative.shape[:2])

            padding = 0.2
            padding = 0.2 if self.augmenter.augment_cropbox is None else random.uniform(self.augmenter.augment_cropbox[0], self.augmenter.augment_cropbox[1])
            negative = tf.convert_to_tensor(crop_to_bbox(negative.numpy(), ann.numpy(), padding=padding, output_size=self.target_size))
            negative.set_shape(self.image_shape)
        else:
            negative = tf.image.resize(negative, self.target_size)
        return negative, filename, other_label

    def load_triplet(self, file_path: tf.Tensor):
        anchor, anchor_fn, anchor_label = self.load_anchor(file_path)
        positive, pos_fn, pos_label = self.load_positive(file_path)
        negative, neg_fn, neg_label = self.load_negative(file_path)

        # vgg16 preprocessing
        anchor = tf.keras.applications.vgg16.preprocess_input((255 * anchor))
        positive = tf.keras.applications.vgg16.preprocess_input((255 * positive))
        negative = tf.keras.applications.vgg16.preprocess_input((255 * negative))

        return anchor, positive, negative, anchor_fn, pos_fn, neg_fn, anchor_label, pos_label, neg_label

    def augment_triplet(self, anchor, positive, negative, anchor_fn, pos_fn, neg_fn, anchor_label, pos_label, neg_label):
        for f in self.augmenter.augmenters:
            anchor, _ = f(anchor, None)
            positive, _ = f(positive, None)
            negative, _ = f(negative, None)
        return anchor, positive, negative, anchor_fn, pos_fn, neg_fn, anchor_label, pos_label, neg_label


def main():
    target_size = (64, 64)
    image_size = (int(1920), int(1080))
    augmentations = {
        "rotate": 90.,
        "flip_horizontal": True,
        "flip_vertical": True,
        "brightness": 0.2}
    nb_inputs = 1
    ds = Dataset('data/imagesets_1080p/all_split_0.1.txt', nb_inputs=nb_inputs, target_size=target_size, loading_size=image_size, augmentations=augmentations, balanced=True, selected_labels=('0', ('1', '2')), nb_classes=2)
    print(ds.tfdataset.element_spec)
    # for img, label in ds.tfdataset.take(3):
    #     print("img shape ", img.shape, " label ", label.shape)


if __name__ == '__main__':
    main()
