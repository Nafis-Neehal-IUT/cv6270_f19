# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:28:57 2019

@author: nafis
"""

import numpy as np
import cv2
import sys

def calculate_distance(X,Y):
    return np.sqrt(np.sum((X-Y)**2, axis=1))

def calculate_manhattan_distance(X,Y):
    return np.sum(np.abs(X-Y), axis=1)

if __name__=="__main__":
    
    args = sys.argv
    sigma = int(args[1])
    img = args[2]
    
    ''' import image as grayscale'''
    
    im_name = img
    im = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)
    
    ''' gaussian smoothing'''
   
    ksize = (4*sigma+1,4*sigma+1) 
    im_s = cv2.GaussianBlur(im.astype(np.float32), ksize, sigma)
    
    '''  Derivative kernels '''
    kx,ky = cv2.getDerivKernels(1,1,3) 
    kx = np.transpose(kx/2)
    ky = ky/2
    
    '''  Derivatives '''
    im_dx = cv2.filter2D(im_s,-1,kx)
    im_dy = cv2.filter2D(im_s,-1,ky)
    
    ''' Components of the outer product '''
    im_dx_sq = im_dx * im_dx
    im_dy_sq = im_dy * im_dy
    im_dx_dy = im_dx * im_dy
    
    ''' Convolution of the outer product with the Gaussian kernel
        gives the summed values desired '''
    h_sigma = 2*sigma 
    h_ksize = (4*h_sigma+1,4*h_sigma+1) 
    im_dx_sq = cv2.GaussianBlur(im_dx_sq, h_ksize, h_sigma)
    im_dy_sq = cv2.GaussianBlur(im_dy_sq, h_ksize, h_sigma)
    im_dx_dy = cv2.GaussianBlur(im_dx_dy, h_ksize, h_sigma)
    
    ''' Compute the Harris measure '''
    kappa = 0.004
    im_det = im_dx_sq * im_dy_sq - im_dx_dy * im_dx_dy
    im_trace = im_dx_sq + im_dy_sq
    im_harris = im_det - kappa * im_trace*im_trace

    '''------------------------------------HARRIS--------------------------------------'''
    
    
    ''' Renormalize the intensities into the 0..255 range '''
    i_min = np.min(im_harris)
    i_max = np.max(im_harris)
    im_harris = 255 * (im_harris - i_min) / (i_max-i_min)
    
    '''
    Apply non-maximum thresholding using dilation, which requires the image
    to be uint8.  Comparing the dilated image to the Harris image will preserve
    only those locations that are peaks.
    '''
    max_dist = 2*sigma
    kernel = np.ones((2*max_dist+1, 2*max_dist+1), np.uint8)
    im_harris_dilate = cv2.dilate(im_harris, kernel)
    im_harris[np.where(im_harris < im_harris_dilate)] = 0

    '''
    Get the normalized Harris measures of the peaks
    '''
    peak_values = im_harris[np.where(im_harris>0)]
    peak_values = peak_values[np.argsort(-peak_values, axis=None)]
    top_two_hundred_harris = []
    top_two_hundred_orb = []
    
    ''' TOP 10 Keypoints for Harris '''
    print("\nTop 10 Harris keypoints:")
    for i in range(200):
        X, Y = np.where(im_harris==peak_values[i])
        if(i<10):
            print("%d: (%.1f, %.1f) %.4f"%(i, Y[0], X[0], peak_values[i]))
        top_two_hundred_harris.append((Y[0], X[0]))    
        
        
    '''-----------------------------------ORB----------------------------------------'''
    
    num_features = 1000
    orb = cv2.ORB_create(num_features)       
    kp, des = orb.detectAndCompute(im, None)  
    
    '''sort peak values in descending order, base is response value'''
    peak_orb = np.array([k.response for k in kp])
    peak_orb = peak_orb[np.argsort(-peak_orb, axis=None)]
    
    '''empty container'''
    keypoints_list = []
    response_list = []
    size_list = []
    
    '''unraveling'''
    for k in kp:
        keypoints_list.append(k.pt)
        response_list.append(k.response)
        size_list.append(k.size)
            
    '''splice and print'''
    count = 0
    it = 0
    print("\nTop 10 ORB keypoints:")
    for i in range(num_features):
        if (size_list[i]<=45):      
            count = count + 1
            index = np.where(response_list==peak_orb[i])
            if(it<10):
                print("%d: (%.1f, %.1f) %.4f"%(i, keypoints_list[index[0][0]][0], 
                                                  keypoints_list[index[0][0]][1], 
                                                  peak_orb[i]))
                it = it+1
            top_two_hundred_orb.append((keypoints_list[index[0][0]][0], 
                                        keypoints_list[index[0][0]][1]))    
        if(count==200):
            break
        
        
    '''---------------------------------DISTANCES---------------------------------------'''   
       
    top_two_hundred_harris = np.array(top_two_hundred_harris)
    top_two_hundred_orb = np.array(top_two_hundred_orb)

    image_distance_matrix = np.zeros((top_two_hundred_harris.shape[0],
                                      top_two_hundred_orb.shape[0]))
    
    '''calculate pairwise image distance matrix'''
    for i in range(image_distance_matrix.shape[0]):
        image_distance_matrix[i] = calculate_distance(top_two_hundred_harris[i],
                                                      top_two_hundred_orb)
    
    '''Harris 100 -> ORB 200
    sort ASC all elements in each row'''
    rowwise_sort = np.sort(image_distance_matrix, axis=1)
    rowwise_index_sort = np.argsort(image_distance_matrix, axis=1)
    h20_median_dist = np.median(rowwise_sort[:100,0])
    h20_avg_dist = np.mean(rowwise_sort[:100,0])
    h20_median_rank_dist = np.median(np.abs(rowwise_index_sort[:100,0]-np.arange(100)))
    h20_avg_rank_dist = np.mean(np.abs(rowwise_index_sort[:100,0]-np.arange(100))) 
    
    '''ORB 100 -> Harris 200
    sort ASC all elements in each column'''   
    columnwise_sort = np.sort(image_distance_matrix, axis=0)
    columnwise_index_sort = np.argsort(image_distance_matrix, axis=0)
    o2h_median_dist = np.median(columnwise_sort[0,:100])
    o2h_avg_dist = np.mean(columnwise_sort[0,:100])
    o2h_median_rank_dist = np.median(np.abs(columnwise_index_sort[0,:100]-np.arange(100)))
    o2h_avg_rank_dist = np.mean(np.abs(columnwise_index_sort[0,:100]-np.arange(100))) 
    
    '''print the formatted outputs'''
    print("\nHarris keypoint to ORB distances:")
    print("Median distance: %.1f"%(h20_median_dist))
    print("Averange distance: %.1f"%(h20_avg_dist))
    print("Median index difference: %.1f"%(h20_median_rank_dist))
    print("Average index difference: %.1f"%(h20_avg_rank_dist))
    print("\nORB keypoint to Harris distances:")
    print("Median distance: %.1f"%(o2h_median_dist))
    print("Averange distance: %.1f"%(o2h_avg_dist))
    print("Median index difference: %.1f"%(o2h_median_rank_dist))
    print("Average index difference: %.1f"%(o2h_avg_rank_dist))
    
    
    
    '''--------------------------------DRAW KEYPOINT----------------------------------------'''
    
    '''filename processing'''
    file_name_ext = img
    file_extension = file_name_ext[-4:]
    file_name = file_name_ext[:-4]
    
    
    
    '''HARRIS MEASURE
       Extract all indices '''
    HARRIS_indices = top_two_hundred_harris
    cols,rows = HARRIS_indices[:,0], HARRIS_indices[:,1]
    
    '''Generate Harries keypoints'''
    kp_size = 4*sigma
    HARRIS_keypoints = [
        cv2.KeyPoint(cols[i], rows[i], kp_size)
        for i in range(len(rows))
    ]
    
    '''Draw harris keypoints on output image and write the output file'''
    out_im_harris = cv2.drawKeypoints(im, HARRIS_keypoints, None)
    new_file_name_harris = file_name + "_harris" + file_extension
    cv2.imwrite(new_file_name_harris, out_im_harris)
    
    '''ORB MEASURE
       Generate ORB keypoints '''
    ORB_indices = top_two_hundred_orb
    ys,xs = ORB_indices[:,1], ORB_indices[:,0]   
    ORB_keypoints = [
        cv2.KeyPoint(xs[i], ys[i], kp_size)
        for i in range(len(xs))
    ]
    
    '''Draw ORB keypoints on output image and write the output file'''
    out_im_ORB = cv2.drawKeypoints(im, ORB_keypoints, None)
    new_file_name_orb = file_name + "_orb" + file_extension
    cv2.imwrite(new_file_name_orb, out_im_ORB)
    
    
        
        
        
        
    
    
    
    
    