# =============================================================================
# File Description:
# ------------------
# This file contains functions for KMeans segmentation
# ======================================================================== #

# =================== Imports =================== #
from sklearn.cluster import KMeans
import cv2
import pydicom
import time
import matplotlib.pyplot as plt
import numpy as np
import os

# =================== KMeans =================== #
def kmeans_segmentation(image, num_class=3, n_init=30, max_iter=500):
    reshape = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

    kmeans = KMeans(n_clusters=num_class, n_init=n_init, max_iter=max_iter).fit(reshape)
    cluster = np.reshape(np.array(kmeans.labels_, dtype=np.uint8), (image.shape[0], image.shape[1]))
    sort_labels = sorted([n for n in range(num_class)], key=lambda x:-np.sum(cluster == x))
    colors = np.linspace(0, 255, num_class+1).astype(np.uint8)
    result = np.zeros(image.shape[:2], dtype=np.uint8)
    for k, label in enumerate(sort_labels):
        result[cluster==label] = colors[k]
    return result

# =================== Add information =================== #
def add_jitter(image, reshape, iter=5):
    for i in range(iter):
        jitter = cv2.convertScaleAbs(image, alpha=np.random.randint(1, 4), beta=np.random.randint(10, 20))
        jitter = cv2.cvtColor(jitter, cv2.COLOR_RGB2GRAY)
        reshape = np.concatenate((reshape, jitter.reshape(image.shape[0] * image.shape[1], 1)), axis=1)
    return image, reshape

def add_gaussian(image, reshape, iter=5):
    for i in range(iter):
        gaussian = cv2.GaussianBlur(image, (np.random.randint(3, 5), np.random.randint(3, 5)), 0)
        gaussian = cv2.cvtColor(gaussian, cv2.COLOR_RGB2GRAY)
        reshape = np.concatenate((reshape, gaussian.reshape(image.shape[0] * image.shape[1], 1)), axis=1)
    return image, reshape

def add_equalize(image, reshape):
    equalize = cv2.equalizeHist(image)
    reshape = np.concatenate((reshape, equalize.reshape(image.shape[0] * image.shape[1], 1)), axis=1)
    return image, reshape
    
def add_canny(image, reshape, iter=2):
    for i in range(iter):
        edge = cv2.Canny(image, np.random.randint(0, 100), np.random.randint(150, 200))
        reshape = np.concatenate((reshape, edge.reshape(image.shape[0] * image.shape[1], 1)), axis=1)
    return image, reshape

def add_sobel(image, reshape, iter=2):
    for i in range(iter):
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)
        reshape = np.concatenate((reshape, sobel.reshape(image.shape[0] * image.shape[1], 1)), axis=1)
    return image, reshape

def add_scharr(image, reshape, iter=2):
    for i in range(iter):
        scharr = cv2.Scharr(image, cv2.CV_64F, 1, 1)
        reshape = np.concatenate((reshape, scharr.reshape(image.shape[0] * image.shape[1], 1)), axis=1)
    return image, reshape

def add_coord(image, reshape):
    x = np.linspace(0, 1, image.shape[0])
    y = np.linspace(0, 1, image.shape[1])
    xx, yy = np.meshgrid(x, y)
    coord = np.concatenate((xx.reshape(image.shape[0] * image.shape[1], 1), yy.reshape(image.shape[0] * image.shape[1], 1)), axis=1)
    coord = cv2.normalize(coord, None, 0, 255, cv2.NORM_MINMAX)
    reshape = np.concatenate((reshape, coord), axis=1)
    return image, reshape

def add_mask(image, reshape, mask_folder):
    for mask in os.listdir(mask_folder):
        mask = cv2.imread(os.path.join(mask_folder, mask), 0)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        reshape = np.concatenate((reshape, mask.reshape(image.shape[0] * image.shape[1], 1)), axis=1)
    return image, reshape

def add_dist(image, reshape, transducers):
    for transducer in transducers:
        dist = np.sqrt((image.shape[0] / 2 - transducer[0]) ** 2 + (image.shape[1] / 2 - transducer[1]) ** 2)
        dist = np.ones((image.shape[0], image.shape[1])) * dist
        reshape = np.concatenate((reshape, dist.reshape(image.shape[0] * image.shape[1], 1)), axis=1)
    return image, reshape
