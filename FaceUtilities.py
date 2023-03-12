import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mtcnn.mtcnn import MTCNN
from PIL import Image
import cv2
from scipy.spatial.distance import cosine
from tensorflow import keras
from keras_vggface.vggface import VGGFace
from scipy.integrate import simpson

class FaceUtilities:
    '''
    Complete set of tools to prepare and compare models for face comparison.
    '''
    def __init__(self, **kwargs):
        self.mtcnn_detector = None
        self.data_directory = kwargs.get("data_directory", "data/CelebA")
        self.img_height = kwargs.get("img_height", 224)
        self.img_width = kwargs.get("img_width", 224)
        # Model params
        self.model = None
        self.batch_size = kwargs.get("batch_size", 16)

    def extract_face(self, filename, required_size=(224, 224)):
        '''
        Given an arbitrary image (with a single assumed face), locate
        the face and stretch it to `required_size`.
        '''
        if self.mtcnn_detector is None:
            self.mtcnn_detector = MTCNN()
        pixels = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        results = self.mtcnn_detector.detect_faces(pixels)
        face = pixels
        if len(results) > 0:
            x1, y1, width, height = results[0]['box']
            x2, y2 = x1 + width, y1+height
            face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        return image

    def set_directory(self, directory):
        '''
        Create the dataframe that matches the specific image filenames to
        their identities and attributes.
        '''
        identities = pd.read_csv(f'data/CelebA/Anno/identity_CelebA.txt', names=["filename", "pid"], delimiter=' ')
        identities.filename = identities.filename.str.replace('.jpg','.png', regex=True)

        attributes = pd.read_csv(f'data/CelebA/Anno/list_attr_celeba.txt',
                                 skiprows=1, delimiter=' ')
        attributes.filename = attributes.filename.str.replace('.jpg', '.png', regex=True)

        labels = pd.merge(attributes, identities, on='filename')
        labels.filename = f'{directory}' + labels.filename.astype(str)
        labels = labels[[os.path.exists(fname) for fname in labels.filename]]
        labels = labels.reset_index()
        self.labels = labels

    def load_vggface(self):
        '''
        For face feature extraction, resnet50 is pretrained to produce ~2k attributes
        '''
        self.model = VGGFace(model='resnet50', include_top=False, pooling='avg',
                             input_shape=(self.img_height, self.img_width, 3))

    def _load_image(self, filename):
        return keras.utils.img_to_array(keras.utils.load_img(filename))

    def _get_feature(self, filename):
        return self.model(np.array([self._load_image(filename)])).numpy()[0]

    def _get_unprocessed_feature(self, filename):
        '''
        Unprocessed images need to be resized and face searched with mtcnn
        '''
        face = np.array(self.extract_face(filename))
        return self.model(np.array([face])).numpy()[0]

    def load_features(self, **kwargs):
        useCache = kwargs.get('use_cache', False)
        cacheFrame = kwargs.get('cache_file', 'cachedFeatures.h5')
        if useCache and os.path.exists(cacheFrame):
            print(f"Loading from cache: {cacheFrame}")
            self.labels = pd.read_hdf(cacheFrame, 'df')
            self.labels.features = self.labels.features.apply(lambda x: np.array(x))
        else:
            print(f"Creating new feature cache")
            self.labels['features'] = [self._get_feature(fname) for fname in self.labels.filename]
            self.labels.to_hdf(cacheFrame, key='df')

    def create_feature_metrics(self, metric_function, **kwargs):
        '''
        A comparison of matching faces is made on all possible pairs.
        For the negative comparison, random sets are drawn, removing
        any entries where the PID match.

        `metric_function` must be provided as metric_function(a, b) where
        a and b are vectors of equal length.
        '''
        random_sets = kwargs.get("random_sets", 10)
        feature_results = []
        random_results = []
        ## Positive comparison
        for name,group in self.labels.groupby('pid'):
            img_count = group.shape[0]
            if img_count > 1:
                feature_set = group.features
                comparison = np.concatenate([[metric_function(fa, fb) for fb in feature_set 
                                              if metric_function(fa,fb) != 0] for fa in feature_set])
                feature_results.append(comparison)
        ## Negative comparison
        base_features = self.labels.features.to_numpy().copy()
        base_pid = self.labels.pid.to_numpy().copy()
        test_features = self.labels.features.to_numpy().copy()
        test_pid = self.labels.pid.to_numpy().copy()
        pid_count = len(test_pid)
        assert len(test_pid) == len(base_pid)
        for rset in range(random_sets):
            select = np.random.permutation(len(test_pid))
            test_features = test_features[select]
            test_pid = test_pid[select]
            ## Ignore matching pid
            delta = (test_pid != base_pid)
            random_results.append(np.array([metric_function(bf, tf) for bf,tf in
                                            zip(base_features[delta], test_features[delta])]))
        random_results = np.concatenate(random_results)
        feature_results = np.concatenate(feature_results)
        return feature_results, random_results

    def doppleganger(self, sample_image_filename, feature_function, **kwargs):
        '''
        Given an input image, try to find the closest matches in the celebrity database.
        The images up to `max_images` will be displayed and returned in a dataframe.
        '''
        max_images = kwargs.get('max_images', 5)
        process_image = kwargs.get('process_image', True)
        draw = kwargs.get('draw', True)
        sample_features = None
        if process_image:
            sample_features = self._get_unprocessed_feature(sample_image_filename)
        else:
            sample_features = self._get_feature(sample_image_filename)
        returndf = self.labels.copy()
        returndf['closeness'] = [feature_function(feature, sample_features) for feature in returndf.features]
        returndf = returndf.sort_values('closeness')
        returndf = returndf.reset_index()
        if not draw:
            return returndf[:max_images]

        def show_img(ax, img, value=None):
            ax.imshow(img)
            if value is not None:
                ax.set_title(f'Vector: {value:0.2f}')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        ## Display the original photo -> transformation, then a row of max_images
        fig, axs = plt.subplots(1, 2, figsize=(6,5))
        show_img(axs[0], keras.utils.load_img(sample_image_filename))
        show_img(axs[1], self.extract_face(sample_image_filename))
        ## Display the "matches"
        sliced_df = returndf[0:max_images].copy()
        sliced_df.filename = sliced_df.filename.str.replace('TestMTC', 'TestImg', regex=True)
        sliced_df.filename = sliced_df.filename.str.replace('ImgMTC', 'Img', regex=True)
        fig, axs = plt.subplots(1, max_images, figsize=(12,5))
        for idx, (fname, closeness) in enumerate(zip(sliced_df.filename, sliced_df.closeness)):
            show_img(axs[idx], keras.utils.load_img(fname), closeness)
        plt.show()
        return sliced_df
