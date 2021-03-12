import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img

class DatasetGenerator(Sequence):
    def __init__(self, path, batch_size=32, flip=True, shuffle=False):
        self.path = path
        self.files = sorted(tf.io.gfile.listdir(path))
        self.length = len(self.files)
        self.batch_size = batch_size
        self.flip = flip
        if shuffle:
            random.shuffle(self.files)
        

    def __len__(self):
        return int(tf.math.floor(self.length / self.batch_size))

    def __getitem__(self, index):
        #indexes = range(index * self.batch_size, (index+1) * self.batch_size)
        #return __load_and_process_files__(self, self.files[index * self.batch_size:(index+1) * self.batch_size])
        files = self.files[index * self.batch_size:(index+1) * self.batch_size]
        return self.__load_and_process_files__(files)

    def on_epoch_end(self):
        return 0

    def __load_and_process_files__(self, files):
        X = []
        for file in files:
            img = np.asarray(load_img(self.path+file)).astype(np.float32)
            if self.flip and random.randint(0, 1):
                img = np.flip(img, axis=1)
            crop = random.randint(0, 120)
            img = img[:,crop:crop+360,:]
            X.append(img)
        return(X)

    