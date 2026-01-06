import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import sys
from utils.tensorflow_addons_filters import gaussian_filter2d


fill_mode_selection = "reflect"

def rescale(factor):
    def rescale(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """Rescale augmentation

        Args:
            x: Image

        Returns:
            Augmented image
        """
        return x * factor, y
    return rescale


def rotate(max_angle, nb_inputs):
    def rotate(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """Rotation augmentation

        Args:
            x: Image

        Returns:
            Augmented image
        """
        if nb_inputs > 1:
            intermediate = [tfa.image.rotate(t, tf.random.uniform(shape=[], minval=0, maxval=int(max_angle)), interpolation='bilinear', fill_mode=fill_mode_selection) for t in tf.unstack(x, num=nb_inputs)]
            x = tf.stack(intermediate)
        else:
            x = tfa.image.rotate(x, tf.random.uniform(shape=[], minval=0, maxval=int(max_angle)), interpolation='bilinear', fill_mode=fill_mode_selection)
        return x, y

    return rotate


def flip_horizontal(nb_inputs):
    def flip_horizontal(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """Flip augmentation

        Args:
            x: Image to flip

        Returns:
            Augmented image
        """
        if nb_inputs > 1:
            intermediate = [tf.image.random_flip_left_right(t) for t in tf.unstack(x, num=nb_inputs)]
            x = tf.stack(intermediate)
        else:
            x = tf.image.random_flip_left_right(x)
        return x, y
    return flip_horizontal


def flip_vertical(nb_inputs):
    def flip_vertical(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """Flip augmentation

        Args:
            x: Image to flip

        Returns:
            Augmented image
        """
        if nb_inputs > 1:
            intermediate = [tf.image.random_flip_up_down(t) for t in tf.unstack(x, num=nb_inputs)]
            x = tf.stack(intermediate)
        else:
            x = tf.image.random_flip_up_down(x)
        return x, y
    return flip_vertical


def color(max_delta):
    def color(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """Color augmentation

        Args:
            x: Image

        Returns:
            Augmented image
        """
        x = tf.image.random_hue(x, max_delta / 2)
        x = tf.image.random_saturation(x, 1. - max_delta, 1. + max_delta)
        x = tf.image.random_brightness(x, max_delta)
        x = tf.image.random_contrast(x, 1. - max_delta * 2, 1. + max_delta * 2)
        x = tf.clip_by_value(x, 0., 1.)

        return x, y
    return color


def saturation(max_delta):
    def saturation(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        x = tf.image.random_saturation(x, 1. - max_delta, 1. + max_delta)
        return x, y
    return saturation


def contrast(max_delta):
    def contrast(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        x = tf.image.random_contrast(x, 1. - max_delta, 1. + max_delta)
        return x, y
    return contrast


def brightness(max_delta):
    def brightness(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        x = tf.image.random_brightness(x, max_delta)
        return x, y
    return brightness


def shear(max_delta, nb_inputs):
    def shear(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        if nb_inputs > 1:
            intermediate = [tfa.image.transform(t, [1.0, max_delta * tf.random.uniform(shape=[], minval=-1, maxval=1), 0.0, max_delta * tf.random.uniform(shape=[], minval=-1, maxval=1), 1.0, 0.0, 0.0, 0.0], fill_mode=fill_mode_selection) for t in tf.unstack(x, num=nb_inputs)]
            x = tf.stack(intermediate)
        else:
            x = tfa.image.transform(x, [1.0, max_delta * tf.random.uniform(shape=[], minval=-1, maxval=1), 0.0, max_delta * tf.random.uniform(shape=[], minval=-1, maxval=1), 1.0, 0.0, 0.0, 0.0], fill_mode=fill_mode_selection)
        return x, y
    return shear


def blur(max_sigma, nb_inputs):
    def blur(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        if nb_inputs > 1:
            intermediate = [gaussian_filter2d(t, filter_shape=3, sigma=tf.random.uniform(shape=[], minval=1., maxval=max_sigma)) for t in tf.unstack(x, num=nb_inputs)]
            x = tf.stack(intermediate)
        else:
            x = gaussian_filter2d(x, filter_shape=3, sigma=tf.random.uniform(shape=[], minval=1., maxval=max_sigma))
        return x, y
    return blur


def sharpen(max_delta, nb_inputs):
    def sharpen(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        if nb_inputs > 1:
            intermediate = [tfa.image.sharpness(255 * t, tf.random.uniform(shape=[], minval=0., maxval=max_delta)) / 255. for t in tf.unstack(x, num=nb_inputs)]
            x = tf.stack(intermediate)
        else:
            x = tfa.image.sharpness(255 * x, tf.random.uniform(shape=[], minval=0., maxval=max_delta)) / 255.
        return x, y
    return sharpen


def cutout(nb_cutouts, nb_inputs):
    def cutout(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        if nb_inputs > 1:
            intermediate = [tfa.image.random_cutout(tf.expand_dims(t, axis=0), mask_size=20, constant_values=0) for t in tf.unstack(x, num=nb_inputs)]
            for _ in range(nb_cutouts):
                intermediate = [tfa.image.random_cutout(t, mask_size=20, constant_values=0) for t in intermediate]
            intermediate = [tf.squeeze(t) for t in intermediate]
            x = tf.stack(intermediate)
        else:
            x = tfa.image.random_cutout(tf.expand_dims(x, axis=0), mask_size=20, constant_values=0)
            for _ in range(nb_cutouts):
                x = tfa.image.random_cutout(x, mask_size=20, constant_values=0)
            x = tf.squeeze(x)
        return x, y
    return cutout


def preprocess_vgg16(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    x = tf.keras.applications.vgg16.preprocess_input((255 * x))
    return x, y


def preprocess_input(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    x = (255 * x)
    return x, y

def zoom(max_delta, crop_size=(64,64)):
    def zoom(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """Zoom augmentation

        Args:
            x: Image

        Returns:
            Augmented image
        """

        # Generate 20 crop settings, ranging from a 1% to 20% crop.
        scales = list(np.arange(1. - max_delta, 1.0, 0.01))
        boxes = np.zeros((len(scales), 4))

        for i, scale in enumerate(scales):
            x1 = y1 = 0.5 - (0.5 * scale)
            x2 = y2 = 0.5 + (0.5 * scale)
            boxes[i] = [x1, y1, x2, y2]

        def random_crop(img):
            # Create different crops for an image
            crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=crop_size)
            # Return a random crop
            return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

        # Only apply cropping 50% of the time
        return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x)), y
    return zoom


def jpeg_quality(max_delta, nb_inputs):
    raise NotImplementedError("jpeg quality augmentation not yet implemented due to unknown shape error.")

    def jpeg_quality(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        if nb_inputs > 1:
            intermediate = [tf.image.random_jpeg_quality(t, min_jpeg_quality=int(max_delta * 100), max_jpeg_quality=100) for t in tf.unstack(x, num=nb_inputs)]
            x = tf.stack(intermediate)
        else:
            x = tf.image.random_jpeg_quality(x, min_jpeg_quality=int(max_delta * 100), max_jpeg_quality=100)
        return x, y
    return jpeg_quality


def translate(SHIFT, nb_inputs):
    '''
    SHIFT is the maximal number of pixels to translate
    '''
    def translate(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        if nb_inputs > 1:
            intermediate = [tfa.image.translate(t, [SHIFT * tf.random.uniform(shape=[], minval=-1., maxval=1.), SHIFT * tf.random.uniform(shape=[], minval=-1., maxval=1.)], fill_mode=fill_mode_selection) for t in tf.unstack(x, num=nb_inputs)]
            x = tf.stack(intermediate)
        else:
            x = tfa.image.translate(x, [SHIFT * tf.random.uniform(shape=[], minval=-1., maxval=1.), SHIFT * tf.random.uniform(shape=[], minval=-1., maxval=1.)], fill_mode=fill_mode_selection)
        return x, y
    return translate


class Augmenter(object):
    def __init__(self, augmentations, target_size=(64, 64), nb_inputs=1):
        self.augmenters = []
        self.target_size = target_size
        self.nb_inputs = nb_inputs

        if augmentations is not None:
            assert type(augmentations) == dict, "Augmentations must be in dictionary format."
            for key in augmentations:
                if key == "rescale" and augmentations["rescale"] is not None:
                    self.augmenters.append(rescale(augmentations["rescale"]))
                if key == "rotate" and augmentations["rotate"] is not None:
                    self.augmenters.append(rotate(augmentations["rotate"], nb_inputs=self.nb_inputs))
                if key == "flip_horizontal" and augmentations["flip_horizontal"] is not None:
                    self.augmenters.append(flip_horizontal(nb_inputs=self.nb_inputs))
                if key == "flip_vertical" and augmentations["flip_vertical"] is not None:
                    self.augmenters.append(flip_vertical(nb_inputs=self.nb_inputs))
                if key == "color" and augmentations["color"] is not None:
                    self.augmenters.append(color(augmentations["color"]))
                if key == "brightness" and augmentations["brightness"] is not None:
                    self.augmenters.append(brightness(augmentations["brightness"]))
                if key == "zoom" and augmentations["zoom"] is not None:
                    self.augmenters.append(zoom(augmentations["zoom"], self.target_size))
                if key == "saturation" and augmentations["saturation"] is not None:
                    self.augmenters.append(saturation(augmentations["saturation"]))
                if key == "contrast" and augmentations["contrast"] is not None:
                    self.augmenters.append(contrast(augmentations["contrast"]))
                if key == "cropbox" and augmentations["cropbox"] is not None:
                    '''
                    Augmentation of the cropping box that is obtained from the annotation file.
                    A number between 0.0->1.0 that denotes the percentage/100 by which the number
                    of pixels that the cropbox is padded with is maximally augmented.

                    E.g. by default, the bounding box that is obtained from the segmentation mask
                    in the annotation file, is padded with 20 pixels in each direction.
                    If augmentation is used, this number '20' is randomly changed to be
                    in the range [20 - 20*augmentations["cropbox"], 20 + 20*augmentations["cropbox"]]

                    Since this is a static method, the augmentation is handled in Dataset.combine_images_labels()
                    and not as a standard augmentation method.
                    '''
                    self.augment_cropbox = augmentations["cropbox"]
                else:
                    self.augment_cropbox = None
                if key == "shear" and augmentations["shear"] is not None:
                    self.augmenters.append(shear(augmentations["shear"], nb_inputs=self.nb_inputs))
                if key == "blur" and augmentations["blur"] is not None:
                    self.augmenters.append(blur(augmentations["blur"], nb_inputs=self.nb_inputs))
                if key == "sharpen" and augmentations["sharpen"] is not None:
                    self.augmenters.append(sharpen(augmentations["sharpen"], nb_inputs=self.nb_inputs))
                if key == "cutout" and augmentations["cutout"] is not None:
                    self.augmenters.append(cutout(augmentations["cutout"], nb_inputs=self.nb_inputs))
                if key == "jpeg_quality" and augmentations["jpeg_quality"] is not None:
                    self.augmenters.append(jpeg_quality(augmentations["jpeg_quality"], nb_inputs=self.nb_inputs))
                if key == "translate" and augmentations["translate"] is not None:
                    self.augmenters.append(translate(augmentations["translate"], nb_inputs=self.nb_inputs))
        else:
            self.augment_cropbox = None
        print("created augmenter with augmentations: ", augmentations)

def main():
    augmentations = {
        "rescale": 1. / 255.,
        "rotate": 90.,
        "flip_horizontal": True,
        "flip_vertical": True,
        "brightness": 0.2,
        }
    a = Augmenter(augmentations)
    print(a.augmenters)

if __name__ == '__main__':
    main()
