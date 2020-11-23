import numpy as np
import os
import multiprocessing
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

class DataSet(object):
    """
    Create a data set.
    """

    def __init__(self,
                 data,
                 image_dims,
                 use_distortion,
                 shuffle,
                 repeat,
                 nThreads):

        self.image_height = image_dims[0]
        self.image_width = image_dims[1]
        self.image_depth = image_dims[2]
        self.use_distortion = use_distortion
        self.data = data
        self.shuffle = shuffle
        self.repeat = repeat
        if nThreads:
            self.nrof_threads = nThreads
        else:
            self.nrof_threads = multiprocessing.cpu_count()

    def _map_fn(self, data_example):
        pass

    def make_batch(self,
                   batch_size,
                   map_fn = None,
                   filter_fn = None):
        """
        Make data batches
        """
        # Extract data
        dataset = tf.data.Dataset.from_tensor_slices(self.data)

        #Shuffle the data before
        if self.shuffle:
            dataset = dataset.shuffle(128*batch_size)

        if self.repeat:
            dataset = dataset.repeat(self.repeat)

        # Transform data

        if filter_fn:
            dataset = dataset.filter(filter_fn)

        dataset = dataset.map(self._map_fn,
                              num_parallel_calls=self.nrof_threads)

        # Batch it up.
        dataset = dataset.batch(batch_size)
        
        dataset = dataset.prefetch(batch_size)

        return dataset

class Cifar10DataSet(DataSet):
    def __init__(self,
                image_dims = (32,32,3),
                subset='train', 
                use_distortion=True,
                shuffle=False,
                repeat=1,
                nThreads=None):

        train_data, test_data = tf.keras.datasets.cifar10.load_data()
        indexes = np.arange(len(train_data[0]))

        if subset == "train":
            train_indexes = indexes[:45000]
            train_images = train_data[0][train_indexes,:,:,:]
            train_labels = train_data[1][train_indexes,:]
            data = (train_images, train_labels)
            self.num_samples = len(train_images)
        elif subset == "valid":
            val_indexes = indexes[45000:]
            val_images = train_data[0][val_indexes,:,:,:]
            val_labels = train_data[1][val_indexes,:]
            data = (val_images, val_labels)
            self.num_samples = len(val_images)
        elif subset == "test":
            data = test_data

        self.subset = subset

        super(Cifar10DataSet, self).__init__(data,
                                             image_dims,
                                             use_distortion,
                                             shuffle,
                                             repeat,
                                             nThreads)
    def _map_fn(self, image, label):
        """
        Apply transformations on the data
        """
        image, label = self.preprocess(image, label)
        return image, label

    def preprocess(self, image, label):
        """Preprocess a single image in [height, width, depth] layout."""
        image = tf.cast(image, tf.float32)/255.0
        label = tf.cast(label, tf.int64)

        return image, label
    
def generate_split(X_train, X_validate, y_train, y_validate, k_folds, fold, subset):
    images = np.concatenate([X_train, X_validate], axis=0)
    labels = np.concatenate([y_train, y_validate])
    skfolds = StratifiedKFold(n_splits=k_folds, random_state=1, shuffle=True)
    counter = 0
    for train_indices, validate_indices in skfolds.split(images, labels):
        if subset == "train":
            (train_images, train_labels) = (images[train_indices], labels[train_indices])
        else:
            (train_images, train_labels) = (images[validate_indices], labels[validate_indices])
        if counter == fold:
            print("train_indices: ")
            print(train_indices)
            print("validate_indices: ")
            print(validate_indices)
            break
        counter = counter + 1
    return (train_images, train_labels)
    
class MnistDataSet(DataSet):
    def __init__(self,
                image_dims = (28,28,1),
                subset='train',
                use_distortion=True,
                shuffle=False,
                repeat=1,
                nThreads=None,
                k_folds=0,
                fold=0,
                fashion=False):
        raw_data = 0
        if fashion:
            raw_data = tf.keras.datasets.fashion_mnist.load_data()
        else:
            raw_data = tf.keras.datasets.mnist.load_data()
        
        (train_images, train_labels), (test_images, test_labels) = raw_data
        
        self.subset = subset
        if self.subset == "train":
            if k_folds > 1:
                (train_images, train_labels) = generate_split(train_images, test_images, train_labels, 
                                                              test_labels, k_folds, fold, subset)

            data = (train_images, train_labels)
            self.num_samples = train_images.shape[0]
        else:
            if k_folds > 1:
                (test_images, test_labels) = generate_split(train_images, test_images, train_labels, 
                                                              test_labels, k_folds, fold, subset)
            data = (test_images, test_labels)
            self.num_samples = test_images.shape[0]
          
        
        test_images, test_labels = data
        
        print(self.subset + ":")
        print("test_images.shape: " + str(test_images.shape))
        print("test_images.shape: " + str(test_labels.shape))

        super(MnistDataSet, self).__init__(data,
                                           image_dims,
                                           use_distortion,
                                           shuffle,
                                           repeat,
                                           nThreads)
    def _map_fn(self, image, label):
        return self.preprocess(image, label)

    def preprocess(self, image, label):
        image = tf.cast(image, tf.float32)/255.0
        label = tf.cast(label, tf.int64)
        return image, label

def get_dataset(datasetname, batch_size, subset="train", shuffle=True, repeat=1, use_distortion=False, k_folds=0, fold=0):
    if datasetname=='mnist':
        mnistdataset = MnistDataSet(subset= subset,
                                            shuffle=shuffle,
                                            repeat=repeat,
                                            use_distortion=use_distortion,
                                            k_folds=k_folds,
                                            fold=fold,)
        dataset = mnistdataset.make_batch(batch_size)
        nrof_samples = mnistdataset.num_samples
        return dataset, nrof_samples
    
    if datasetname=='fashion_mnist':
        fashionmnistdataset = MnistDataSet(subset=subset,
                                            shuffle=shuffle,
                                            repeat=repeat,
                                            use_distortion=use_distortion,
                                            k_folds=k_folds,
                                            fold=fold,
                                            fashion=True)
        dataset = fashionmnistdataset.make_batch(batch_size)
        nrof_samples = fashionmnistdataset.num_samples
        return dataset, nrof_samples

    if datasetname=='cifar10':
        cifardataset = Cifar10DataSet(subset=subset,
                                               shuffle=shuffle,
                                              repeat=repeat,
                                              use_distortion=use_distortion)
        dataset = cifardataset.make_batch(batch_size)
        nrof_samples = train_data.num_samples

        return dataset, nrof_samples
                                                                              
        
