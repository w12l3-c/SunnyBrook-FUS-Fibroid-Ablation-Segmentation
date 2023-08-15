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
