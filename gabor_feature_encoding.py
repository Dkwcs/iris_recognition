import cv2
import numpy as np

def feature_encoding(image):
    g_kernel = cv2.getGaborKernel((27, 27), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(image, cv2.CV_8UC3, g_kernel)

    # h, w = g_kernel.shape[:2]
    # g_kernel = cv2.resize(g_kernel, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
    return filtered_img