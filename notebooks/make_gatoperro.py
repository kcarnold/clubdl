#!/usr/bin/env python
import os
import numpy as np
from scipy.misc import imsave, fromimage
from PIL import Image, ImageOps
from tqdm import tqdm, trange

if os.path.exists('/data/gatos'):
    BASE = '/data/gatos'
else:
    BASE = '../data/gatos'

WIDTH = HEIGHT = 224
def load_and_crop_image(filename, target_size):
    return ImageOps.fit(Image.open(filename), target_size)
gatos = np.array([fromimage(load_and_crop_image(BASE+'/cat.{:04}.jpg'.format(i), (WIDTH, HEIGHT))) for i in trange(100, desc='cargar gatos')])
perros = np.array([fromimage(load_and_crop_image(BASE+'/dog.{:04}.jpg'.format(i), (WIDTH, HEIGHT))) for i in trange(100, desc='cargar perros')])

imagenes = np.concatenate((gatos, perros), axis=0)
labels = np.zeros(len(gatos) + len(perros), dtype=int)
labels[:len(gatos)] = 1
brightness = np.mean(imagenes, axis=(1,2,3))
bright_dogs = np.flatnonzero((brightness > 150) & (labels == 0))
dark_cats = np.flatnonzero((brightness < 100) & (labels == 1))

np.random.seed(3)
dog_images = np.random.choice(bright_dogs, 3, replace=False)
cat_images = np.random.choice(dark_cats, 7, replace=False)
first_set_of_images = np.concatenate((dog_images, cat_images))
np.random.shuffle(first_set_of_images)
np.random.shuffle(first_set_of_images)

from vgg16 import model, img_to_vgg_input, all_cats
from keras import backend as K

predictor = K.function([model.input] + [K.learning_phase()], [model.layers[-1].output])
predictions = np.array([predictor([img_to_vgg_input(imagen), False])[0][0] for imagen in tqdm(imagenes, desc='caracteristicas vgg')])
cat_probs = np.sum(predictions[:, all_cats], axis=1)

np.save('vgg_output.npy', predictions)

all_feats = np.c_[np.ones(len(imagenes)), cat_probs, brightness]
np.save('all_feats.npy', all_feats)
np.save('train_indices.npy', first_set_of_images)

def save_imgs(indices, fmt, filename):
    feats = []
    true = []
    for i, idx in enumerate(indices):
        imsave(fmt.format(i), imagenes[idx])
        # labels[idx] - .5 + np.random.standard_normal()*.1,
        feats.append([1, cat_probs[idx], brightness[idx]])
        true.append(labels[idx])
    true = np.array(true)
    feats = np.array(feats)

    with open(filename,'w') as f:
        print('''<style>
            div { page-break-after: always; }
            table { border-collapse: collapse; margin: 5px; }
            td { border: 1px solid black; padding: 5px; }
        </style>''', file=f)
        for i, feat in enumerate(feats):
            print('<div>', file=f)
            print('<h1>{}</h1>'.format(i+1), file=f)
            print('<img src="{}">'.format(fmt.format(i)), file=f)
            print(
                '<table><tr>',
                ''.join('<td>{:.02f}</td>'.format(f) for f in feat),
                '</tr><tr>',
                ''.join(['<td>&nbsp;</td>'] * len(feat)),
                '</tr><tr>',
                ''.join(['<td>&nbsp;</td>'] * len(feat)),
                '</tr></table>', file=f)
            print('</div>', file=f)

save_imgs(first_set_of_images, 'img_{:02}.jpg', 'page.html')

unused = np.ones(len(imagenes), dtype=bool)
unused[first_set_of_images] = 0
second_set = np.random.choice(np.flatnonzero(unused), 20, replace=False)
save_imgs(second_set, 'img2_{:02}.jpg', 'page2.html')
