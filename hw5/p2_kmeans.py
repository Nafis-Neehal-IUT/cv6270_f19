# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import os
import cv2
import sys
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    file = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            file.append(filename)
    return images, file

'''
This function will generate the feature matrix for the input image.
Features - Location, Color Intensity, SD of Color Intensity
'''
def build_feature_matrix(img):
    
    #split color channels
    b,g,r = cv2.split(img)
    
    #create meshgrid for indices
    shape_x, shape_y = img.shape[0], img.shape[1]
    row, col = np.meshgrid(np.arange(shape_y), np.arange(shape_x))
    
    #flatten all indices (0->0...0->359...239->359), rows 240, cols 360 --> SPATIAL features
    row = np.ravel(row)
    col = np.ravel(col)
    
    #flatten b,g and r channels for concatenation --> COLOR features
    b = np.ravel(b)
    g = np.ravel(g)
    r = np.ravel(r)
    
    #calculate SD for each channel in 3x3 nbrhood --> TEXTURE feature
    blur_im_sqrd = cv2.blur(img**2,(3,3))
    im_blur_sqrd = (cv2.blur(img,(3,3)))**2
    sd = np.sqrt(blur_im_sqrd-im_blur_sqrd)
    
    #split SD for each channel
    b_sd = sd[:,:,0]
    g_sd = sd[:,:,1]
    r_sd = sd[:,:,2]
    
    #flatten b,g,r channel SD
    b_sd = np.ravel(b_sd)
    g_sd = np.ravel(g_sd)
    r_sd = np.ravel(r_sd)
    
    #concatenate
    feature_matrix = np.vstack((col, row, b, g, r, b_sd, g_sd, r_sd)).astype(np.float32)
    
    return feature_matrix.T

''''
init method options - cv2.KMEANS_PP_CENTERS / cv2.KMEANS_RANDOM_CENTERS
criteria - is the stop condition; eps - target accuracy, maxiter - maximum num of iters
number of reinitialization - number of attempts algo has taken using different initial labels
PP_Centers - Kmeans++ process to initialize centers
RANDOM_Centers - trivial process
'''
def k_means_clustering(feature_matrix, num_clusters, eps, maxiter, num_reinit, init_method):
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, maxiter, eps) 
    ret, label, center = cv2.kmeans(feature_matrix, num_clusters, None, criteria,
                                    num_reinit, init_method)
    
    return ret, label, center

'''
This function normalizes a matrix from 0 to 255
'''
def mapz2tff(z):
    z = cv2.normalize(z, None, alpha = 0, beta = 255, 
                               norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    return z

'''
This function is to show figures using Matplotlib.
'''
def show_image(im):
    plt.figure()
    plt.imshow(im.astype(np.uint8)) 
    plt.show()


if __name__ == "__main__":
   
    #load input image
    args = sys.argv
    img_name = args[1]
    img = cv2.imread(img_name)
    
    #feature matrix build (only location and bgr)
    feature_matrix = build_feature_matrix(img)

    #weight maps for different features
    '''
    col 1,2 -> pixel locations
    col 3,4,5 -> color channels b,g,r
    col 6,7,8 -> standard deviation of b,g,r
    '''
    weight = np.array([1.5, 1.5, 2, 2, 2, 2, 2, 2])
    feature_matrix *= weight
    
    #apply k-means
    ret, label, center = k_means_clustering(feature_matrix, 9, 1.0, 
                                            100, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    #produce color quantized image
    center[:,2:5] = mapz2tff(center[:,2:5])
    result = center[label.flatten()]
    result = result[:,2:5].reshape((img.shape))
    
    #show image
    show_image(result)
    
    #write image
    cv2.imwrite("output_kmeans.jpg", result)
    
    
    
    
    
    
    
    
    
    