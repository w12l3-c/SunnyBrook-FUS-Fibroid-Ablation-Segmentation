from sklearn.cluster import KMeans
import cv2
import pydicom
import time
import matplotlib.pyplot as plt
import numpy as np

def display(image, result):
    plt.imshow(image, alpha=0.9)
    plt.imshow(result, alpha=0.5)
    plt.show()
    
# Kmeans is unsupervised learning algorithm
def inference(filename, num_classes):
    image = pydicom.dcmread(filename).pixel_array
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    reshape = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

    kmeans = KMeans(n_clusters=num_classes, n_init=10, max_iter=500).fit(reshape)
    
    cluster = np.reshape(np.array(kmeans.labels_, dtype=np.uint8), (image.shape[0], image.shape[1]))
    
    sort_labels = sorted([n for n in range(num_classes)], key=lambda x:-np.sum(cluster == x))
    
    palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = np.array([i for i in range(num_classes)]) * palette
    colors = (colors % 255).astype("uint8")
    
    result = np.zeros(image.shape[:2], dtype=np.uint8)
    for i, label in enumerate(sort_labels):
        result[cluster==label] = palette[i]
        
    display(image, result)
    
    save_name = f"{filename}_mask.jpg"
    cv2.imwrite(save_name, result)