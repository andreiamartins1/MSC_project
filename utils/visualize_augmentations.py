'''
Demo file to show what the augmentations in DataGenerator have as effect on the images.
'''
import sys
import numpy as np
import time
from matplotlib import pyplot as plt

sys.path.append('.')
sys.path.append('..')


def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time()  # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti # returns the time difference


TicToc = TicTocGenerator()  # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" % tempTimeInterval)


def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


dataset = 'polypsHD'
assert dataset in ['catsndogs', 'polyps', 'cvc', 'catsndogs_triplet', 'polyps_triplet', 'cvc_triplet', 'polypsHD_triplet', 'polypsHD']

image_size = (int(1920),int(1080))
target_size = (int(256), int(256))
batch_size = 16

nb_inputs = 5
assert np.sqrt(batch_size) - np.floor(np.sqrt(batch_size)) == 0, "Batch size must be square of integer."

if dataset == 'polyps':
    from utils.Dataset import Dataset
    labels = {0: 'HP', 1: 'AD', 2: 'SSA'}
    imageset_dir = 'data/imagesets/all_split_0.1.txt'
    loading_size = image_size
elif dataset == 'polypsHD':
    from utils.Dataset import Dataset
    labels = {0: 'HP', 1: 'AD', 2: 'SSA'}
    imageset_dir = 'data/imagesets_1080p/5bootstrap_all/test.txt'
    loading_size = image_size
elif dataset == 'catsndogs':
    from utils.Dataset_catsndogs import Dataset
    labels = {0: 'cat', 1: 'dog'}
    imageset_dir = 'data/catsndogs/train/'
    loading_size = None
elif dataset == 'cvc':
    from utils.Dataset_cvc import Dataset
    labels = {0: 'ADN', 1: 'nonADN'}
    imageset_dir = 'data/cvc/train/'
    loading_size = image_size
elif dataset == 'catsndogs_triplet':
    from utils.Dataset_catsndogs import TripletDataset as Dataset
    labels = {0: 'cat', 1: 'dog'}
    imageset_dir = 'data/catsndogs/train/'
    loading_size = None
elif dataset == 'polyps_triplet':
    from utils.Dataset import TripletDataset as Dataset
    labels = {0: 'HP', 1: 'AD', 2: 'SSA'}
    imageset_dir = 'data/imagesets/all_split_0.1.txt'
    loading_size = image_size
elif dataset == 'polypsHD_triplet':
    from utils.Dataset import TripletDataset as Dataset
    labels = {0: 'HP', 1: 'AD', 2: 'SSA'}
    imageset_dir = 'data/imagesets_1080p/all_split_0.1.txt'
    loading_size = image_size
elif dataset == 'cvc_triplet':
    from utils.Dataset_cvc import TripletDataset as Dataset
    labels = {0: 'ADN', 1: 'nonADN'}
    imageset_dir = 'data/cvc/train/'
    loading_size = image_size

# augmentations = {
#     "rotate": 10.,
#     "flip_horizontal": True,
#     "flip_vertical": True,
#     # "brightness": 0.2,
#     # "saturation": 0.1,
#     # "contrast": 0.1,
#     "color": 0.2,
#     "zoom": 0.1,
#     "cropbox": 0.5 # augment the cropbox padding by +-50%
#     }
augmentations = {
        "flip_horizontal": True,
        "flip_vertical": True,
        "cutout": 10,
        "shear": 0.2,
        "rotate": 90.,
        "translate": 20,  # max number of pixels
        "color": 0.2,
        "blur": 1.3
        }

tic()
ds = Dataset(imageset_dir, target_size=target_size, loading_size=target_size, augmentations=augmentations, batch_size=batch_size, nb_inputs=nb_inputs, balanced=True, shuffle=True, do_cropping='preloaded', balance_polypids=True)
toc()
print(ds.tfdataset.element_spec)

ds.tfdataset.take(1)  # to eliminate startup costs
tic()
t = ds.tfdataset.take(3)
batch = list(t.as_numpy_iterator())
toc()

print("minmax: ", np.min(batch[0][0]), np.max(batch[0][0]))

batch_idx = 0
image_idx = 1
n_rows = 8
mean = [103.939, 116.779, 123.68]

if 'triplet' in dataset:
    print(batch[batch_idx]['anchor_input'].shape)
    f, axarr = plt.subplots(n_rows, 3)
    print(np.min(batch[batch_idx]['anchor_input'][image_idx]), np.max(batch[batch_idx]['anchor_input'][image_idx]))
    for row_idx in range(n_rows):
        axarr[row_idx, 0].imshow(batch[batch_idx]['anchor_input'][image_idx + row_idx])
        axarr[row_idx, 1].imshow(batch[batch_idx]['positive_input'][image_idx + row_idx])
        axarr[row_idx, 2].imshow(batch[batch_idx]['negative_input'][image_idx + row_idx])
        print(batch[batch_idx]['anchor_label'][image_idx + row_idx])
    plt.savefig('triplet.png')

    plt.figure()
    plt.imshow(batch[batch_idx]['anchor_input'][image_idx])
    plt.savefig('anchor.png')
    plt.figure()
    plt.imshow(batch[batch_idx]['positive_input'][image_idx])
    plt.savefig('positive.png')
    plt.figure()
    plt.imshow(batch[batch_idx]['negative_input'][image_idx])
    plt.savefig('negative.png')
else:
    for b in range(1):
        batch_x, batch_y = batch[b]
        print(batch_x.shape, batch_y.shape)
        for l in range(nb_inputs):
            if len(batch_x.shape) == 5:
                batch_xx = batch_x[:, l, :, :, :].copy()
            else:
                batch_xx = batch_x.copy()
            # batch_x += 1.0
            # batch_x *= 127.5
            # batch_xx *= 255.

            # reverse preprocessing vgg16
            batch_xx = batch_xx[..., ::-1]
            batch_xx[..., 0] += mean[0]
            batch_xx[..., 1] += mean[1]
            batch_xx[..., 2] += mean[2]
            batch_xx /= 255.
            batch_xx = np.clip(batch_xx, 0., 1.)
            print("minmax after reverse preprocessing: ", np.min(batch_xx), np.max(batch_xx))

            onedim = int(np.sqrt(batch_size))
            f, axarr = plt.subplots(onedim,onedim)
            for i in range(batch_size):
                axarr[int(np.floor(i/onedim)),i%onedim].imshow(batch_xx[i])
                axarr[int(np.floor(i/onedim)),i%onedim].set_title(labels[np.argmax(batch_y[i])])

            plt.savefig('input{}.png'.format(l))
