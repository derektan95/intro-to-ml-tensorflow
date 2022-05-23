# TODO: Make all necessary imports.
import warnings
warnings.filterwarnings('ignore')

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import time
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import argparse
import os


class DL_Utils():
    """
    Simple Deep Learning Utilities for inference
    """
    
    def __init__(self, top_k, class_names_path):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            total_state_size (int): dimension of each state
            action_size (int): dimension of each action
        """

        self.top_k = top_k
        self.class_names_path = class_names_path
        self.image_size = 224

        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)



    def process_image(self, image):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (self.image_size, self.image_size))
        image /= 255
        return image.numpy()

    def predict(self, img_path, model):
        img = np.asarray(Image.open(img_path))
        img = np.expand_dims(self.process_image(img), axis=0)
        all_probs = np.array(model.predict(img)[0])
        top_indices = (-all_probs).argsort()[:self.top_k]
        return all_probs[top_indices], (top_indices+1)

    def print_img(self, probs, classes):
        img = np.asarray(Image.open(self.img_path))
        top_classes = []
        for cls in classes:
            top_classes.append(self.class_names.get(str(cls)))
        
        fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
        ax1.imshow(img, cmap = plt.cm.binary)
        ax1.axis('off')
        ax2.barh(np.arange(self.top_k), probs)
        ax2.set_aspect(0.1)
        ax2.set_yticks(np.arange(self.top_k))
        ax2.set_yticklabels(top_classes, size='small');
        ax2.set_title('Class Probability')
        ax2.set_xlim(0, 1.1)
        plt.tight_layout()

    def print_prediction_results(self, probs, classes, image_path):
        
        top_classes = []
        for cls in classes:
            top_classes.append(self.class_names.get(str(cls)))
        
        print("\n===============================")
        print("Prediction for {} (top_k={})".format(image_path, self.top_k))
        print("Classes: ", top_classes)
        print("Probs: ", probs)


# MAIN

# Helpful print msgs
print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')

# Parse Inputs
parser = argparse.ArgumentParser(description='Deep Learning for flower class prediction.')
parser.add_argument('img_path', action="store", type=str,
                    help='Path to image used during inference.')
parser.add_argument('model_path', action="store", type=str,
                    help='Path to model used during inference')
parser.add_argument('--top_k', action="store", dest='top_k', type=int, default=5,
                    help='Display K number of top predictions from model inference')
parser.add_argument('--category_names', action="store", dest='category_names_filepath', type=str, default="label_map.json",  
                    help='File that maps labels to flower names')

args = parser.parse_args()
print("Input Arguments: ", args)

# NOTE: CHANGE ME! (RESOLVE BUG WHERE LOCAL PATH IS SOMEHOW ".../AppData/Local/Temp..."")
os.environ["TFHUB_CACHE_DIR"] = "C:/Users/derek/Desktop/intro_to_ml_tensorflow/projects/p2_image_classifier"

# Load Keras Model
print("Model loading from: ", args.model_path)
reloaded_keras_model = tf.keras.models.load_model((args.model_path),custom_objects={'KerasLayer':hub.KerasLayer})
print(reloaded_keras_model.summary())
print("Model loaded: ", args.model_path)

# Perform inference
utils = DL_Utils(args.top_k, args.category_names_filepath)
probs, classes = utils.predict(args.img_path, reloaded_keras_model)
utils.print_prediction_results(probs, classes, args.img_path)


