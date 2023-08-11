# =============================================================================
# File Description:
# ------------------
# (Deprecated)
# This file contains the functions for KNN segmentation
# ======================================================================== #

# =================== Imports =================== #
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import greycomatrix, greycoprops
from skimage.segmentation import mark_boundaries

# =================== KNN =================== #
def texture_features(image):
    # Calculate texture features using Grey-Level Co-occurrence Matrix (GLCM)
    glcm = greycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast').mean()
    homogeneity = greycoprops(glcm, 'homogeneity').mean()
    correlation = greycoprops(glcm, 'correlation').mean()
    energy = greycoprops(glcm, 'energy').mean()
    return np.array([contrast, homogeneity, correlation, energy])


def knn_segmentation(image, labeled_image, k):
    # Prepare feature vectors and labels for KNN
    h, w = image.shape
    feature_vectors = []
    labels = []
    for y in range(h):
        for x in range(w):
            if labeled_image[y, x] != 0:
                labels.append(labeled_image[y, x])
                patch = image[max(0, y-k):min(h, y+k+1), max(0, x-k):min(w, x+k+1)]
                texture_feature = texture_features(patch)
                feature_vectors.append(texture_feature)

    # Convert to numpy arrays for scikit-learn
    feature_vectors = np.array(feature_vectors)
    labels = np.array(labels)

    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier with the feature vectors and labels
    knn.fit(feature_vectors, labels)

    # Segment the image using the trained classifier
    segmentation = np.zeros_like(image, dtype=int)
    for y in range(h):
        for x in range(w):
            if labeled_image[y, x] != 0:
                continue
            patch = image[max(0, y-k):min(h, y+k+1), max(0, x-k):min(w, x+k+1)]
            texture_feature = texture_features(patch)
            predicted_label = knn.predict([texture_feature])
            segmentation[y, x] = predicted_label

    return segmentation