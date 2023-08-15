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
    """
    Perform K-means image segmentation.

    Args:
        image (numpy.ndarray): Input image.
        num_class (int, optional): Number of clusters. Default is 3.
        n_init (int, optional): Number of time the k-means algorithm will be run with different centroid seeds.
        max_iter (int, optional): Maximum number of iterations of the k-means algorithm.

    Returns:
        numpy.ndarray: Segmented image.
    """
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
class AddInfo:
    def __init__(self):
        pass
    
    def add_jitter(image, reshape, iter=5):
        """
        Add jitter effect to the image and reshape data array.

        Args:
            image (numpy.ndarray): Input image.
            reshape (numpy.ndarray): Data array to be reshaped and modified.
            iter (int, optional): Number of iterations. Default is 5.

        Returns:
            numpy.ndarray: Modified image.
            numpy.ndarray: Reshaped and modified data array.
        """
        for i in range(iter):
            jitter = cv2.convertScaleAbs(image, alpha=np.random.randint(1, 4), beta=np.random.randint(10, 20))
            jitter = cv2.cvtColor(jitter, cv2.COLOR_RGB2GRAY)
            reshape = np.concatenate((reshape, jitter.reshape(image.shape[0] * image.shape[1], 1)), axis=1)
        return image, reshape

    def add_gaussian(image, reshape, iter=5):
        """
        Add Gaussian blur effect to the image and reshape data array.

        Args:
            image (numpy.ndarray): Input image.
            reshape (numpy.ndarray): Data array to be reshaped and modified.
            iter (int, optional): Number of iterations. Default is 5.

        Returns:
            numpy.ndarray: Modified image.
            numpy.ndarray: Reshaped and modified data array.
        """
        for i in range(iter):
            gaussian = cv2.GaussianBlur(image, (np.random.randint(3, 5), np.random.randint(3, 5)), 0)
            gaussian = cv2.cvtColor(gaussian, cv2.COLOR_RGB2GRAY)
            reshape = np.concatenate((reshape, gaussian.reshape(image.shape[0] * image.shape[1], 1)), axis=1)
        return image, reshape

    def add_equalize(image, reshape):
        """
        Add Histogram Equalize to the image and reshape data array.

        Args:
            image (numpy.ndarray): Input image.
            reshape (numpy.ndarray): Data array to be reshaped and modified.

        Returns:
            numpy.ndarray: Modified image.
            numpy.ndarray: Reshaped and modified data array.
        """
        equalize = cv2.equalizeHist(image)
        reshape = np.concatenate((reshape, equalize.reshape(image.shape[0] * image.shape[1], 1)), axis=1)
        return image, reshape
        
    def add_canny(image, reshape, iter=2):
        """
        Add Canny edge to the image and reshape data array.

        Args:
            image (numpy.ndarray): Input image.
            reshape (numpy.ndarray): Data array to be reshaped and modified.
            iter (int, optional): Number of iterations. Default is 5.

        Returns:
            numpy.ndarray: Modified image.
            numpy.ndarray: Reshaped and modified data array.
        """
        for i in range(iter):
            edge = cv2.Canny(image, np.random.randint(0, 100), np.random.randint(150, 200))
            reshape = np.concatenate((reshape, edge.reshape(image.shape[0] * image.shape[1], 1)), axis=1)
        return image, reshape

    def add_sobel(image, reshape, iter=2):
        """
        Add Sobel edge to the image and reshape data array.

        Args:
            image (numpy.ndarray): Input image.
            reshape (numpy.ndarray): Data array to be reshaped and modified.
            iter (int, optional): Number of iterations. Default is 5.

        Returns:
            numpy.ndarray: Modified image.
            numpy.ndarray: Reshaped and modified data array.
        """
        for i in range(iter):
            sobel = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)
            reshape = np.concatenate((reshape, sobel.reshape(image.shape[0] * image.shape[1], 1)), axis=1)
        return image, reshape

    def add_scharr(image, reshape, iter=2):
        """
        Add Scharr effect to the image and reshape data array.

        Args:
            image (numpy.ndarray): Input image.
            reshape (numpy.ndarray): Data array to be reshaped and modified.
            iter (int, optional): Number of iterations. Default is 5.

        Returns:
            numpy.ndarray: Modified image.
            numpy.ndarray: Reshaped and modified data array.
        """
        for i in range(iter):
            scharr = cv2.Scharr(image, cv2.CV_64F, 1, 1)
            reshape = np.concatenate((reshape, scharr.reshape(image.shape[0] * image.shape[1], 1)), axis=1)
        return image, reshape

    def add_coord(image, reshape):
        """
        Add pixel coordinate system to the image and reshape data array.

        Args:
            image (numpy.ndarray): Input image.
            reshape (numpy.ndarray): Data array to be reshaped and modified.

        Returns:
            numpy.ndarray: Modified image.
            numpy.ndarray: Reshaped and modified data array.
        """
        x = np.linspace(0, 1, image.shape[0])
        y = np.linspace(0, 1, image.shape[1])
        xx, yy = np.meshgrid(x, y)
        coord = np.concatenate((xx.reshape(image.shape[0] * image.shape[1], 1), yy.reshape(image.shape[0] * image.shape[1], 1)), axis=1)
        coord = cv2.normalize(coord, None, 0, 255, cv2.NORM_MINMAX)
        reshape = np.concatenate((reshape, coord), axis=1)
        return image, reshape

    def add_mask(image, reshape, mask_folder):
        """
        Add predicted masks from a trained ML model like Unet to the image and reshape data array.

        Args:
            image (numpy.ndarray): Input image.
            reshape (numpy.ndarray): Data array to be reshaped and modified.
            mask_folder (str): Path to the folder containing mask images.

        Returns:
            numpy.ndarray: Modified image.
            numpy.ndarray: Reshaped and modified data array.
        """
        for mask in os.listdir(mask_folder):
            mask = cv2.imread(os.path.join(mask_folder, mask), 0)
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            reshape = np.concatenate((reshape, mask.reshape(image.shape[0] * image.shape[1], 1)), axis=1)
        return image, reshape

    def add_dist(image, reshape, transducers):
        """
        Add distances from transducers to the image and reshape data array.

        Args:
            image (numpy.ndarray): Input image.
            reshape (numpy.ndarray): Data array to be reshaped and modified.
            transducers (list): List of transducer coordinates as tuples (x, y).

        Returns:
            numpy.ndarray: Modified image.
            numpy.ndarray: Reshaped and modified data array.
        """
        for transducer in transducers:
            dist = np.sqrt((image.shape[0] / 2 - transducer[0]) ** 2 + (image.shape[1] / 2 - transducer[1]) ** 2)
            dist = np.ones((image.shape[0], image.shape[1])) * dist
            reshape = np.concatenate((reshape, dist.reshape(image.shape[0] * image.shape[1], 1)), axis=1)
        return image, reshape
