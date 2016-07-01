import os
for i in range(1000): os.rename('other/cat.{}.jpg'.format(i), 'train/cats/cat.{:04}.jpg'.format(i))
for i in range(1000): os.rename('other/dog.{}.jpg'.format(i), 'train/dogs/dog.{:04}.jpg'.format(i))
for i in range(1000, 2000): os.rename('other/dog.{}.jpg'.format(i), 'validation/dogs/dog.{:04}.jpg'.format(i))
for i in range(1000, 2000): os.rename('other/cat.{}.jpg'.format(i), 'validation/cats/cat.{:04}.jpg'.format(i))
