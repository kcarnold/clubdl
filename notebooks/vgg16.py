import os
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
weights_path = BASE_PATH + '/../vgg16_weights.h5'
synsets = [line.strip().split(' ', 1) for line in open(BASE_PATH + '/../synset_words.txt')]

import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout

img_width, img_height = 224, 224

# build the VGG16 network
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool_1'))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool_2'))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool_3'))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool_4'))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool_5'))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# Load weights
import h5py
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')


VGG_MEAN_PIXEL = np.array([103.939, 116.779, 123.68])
def img_to_vgg_input(img):
    return imgs_to_vgg_input(np.expand_dims(img, axis=0))

def imgs_to_vgg_input(imgs):
    assert imgs.shape[-1] == 3
    imgs = imgs[..., [2,1,0]]
    imgs = imgs - VGG_MEAN_PIXEL
    return imgs.transpose((0, 3, 1, 2))


from nltk.corpus import wordnet
def get_synset(imagenet_synset_id):
    return wordnet.of2ss(imagenet_synset_id[1:] + 'n')
def is_a(leaf, root):
    return any(root in path for path in leaf.hypernym_paths())
def get_all_leaf_indices(root):
    return [i for i, (ssid, words) in enumerate(synsets) if is_a(get_synset(ssid), root)]


all_dogs = get_all_leaf_indices(wordnet.synset('dog.n.01'))
all_cats = get_all_leaf_indices(wordnet.synset('cat.n.01'))
