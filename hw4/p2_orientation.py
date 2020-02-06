# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 02:20:58 2019

@author: nafis
"""

import cv2
import numpy as np
import math

import sys


if __name__ == "__main__":
    
    args = sys.argv
    sigma = float(args[1])
    img = args[2]
    points = args[3]
    
    '''---------------------------------IMPORTS AND PRE-CALCULATION---------------------------------------'''
    
    ''' import image as grayscale'''
    im_name = img
    im = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    im_name2 = points
    im_keypoints = np.loadtxt(im_name2).astype(np.uint32)
    
    ''' gaussian smoothing'''
    sigma_v = 2*sigma
    ksize = (4*int(sigma)+1,4*int(sigma)+1)
    im_s = cv2.GaussianBlur(im.astype(np.float32), ksize, sigma)
    
    '''  Derivative kernels '''
    kx,ky = cv2.getDerivKernels(1,1,3)
    kx = np.transpose(kx/2)
    ky = ky/2
    
    '''  Derivatives '''
    im_dx = cv2.filter2D(im_s,-1,kx)
    im_dy = cv2.filter2D(im_s,-1,ky)
    
    '''Magnitude and Orientation'''
    im_magnitude = np.sqrt(im_dx**2+im_dy**2)
    rad2deg = 180/math.pi
    im_direction = np.arctan2(im_dy, im_dx) * rad2deg
    im_direction[im_direction<0] += 360
    
    '''---------------------------------HISTOGRAM CALCULATION---------------------------------------''' 
    
    '''necessary values for weight calculation'''
    window = int(np.round(2*sigma_v))
    ksize = 2*window+1
    gauss_one_d = cv2.getGaussianKernel(ksize, sigma_v)
    gauss_two_d = gauss_one_d * gauss_one_d.T
    
    '''Bin Matrix will contain all the histograms for each keypoint. There are two histograms for each keypoint,
    one has the original values, other one has smoothed values. The first row of the Bin Matrix contains the Angle
    ranges from -180 to 180. Each column corresponds to the bin with boundaries [j, j+10).'''
    bin_matrix = np.zeros((9, 36))
    bin_matrix[0] = np.linspace(-180, 170, 36)
    
    '''keep track of the keypoint'''
    num_keypoint = 1
    
    '''This will loop through every keypoint and calculate all the necessary values for the Bin Matrix.
        Loop will iterate 4 times for wisconsin example and for 4 keypoints'''
    for keypoint in im_keypoints: 
        
        '''declaring the bin vector - equivalent to one row of bin matrix'''
        bin_vector = np.zeros(36)
        
        '''calculate neighbor range, also calculate magnitude and orientation of that neighborhood slice'''
        nbr_range = np.s_[keypoint[0]-window:keypoint[0]+window+1, 
                          keypoint[1]-window:keypoint[1]+window+1]
        nbr_mag = im_magnitude[nbr_range]
        nbr_orientation = im_direction[nbr_range]
        
        '''form weight matrix and flatten this array'''
        W = np.multiply(nbr_mag, gauss_two_d)
        W = np.ravel(W)
        
        '''flatten the orientation vector with angles of all pixels in a keypoints neighbourhood'''
        nbr_orientation = np.ravel(nbr_orientation)
        
        '''the following loop will basically calculate the splitted votes and add them to corresponding bins
           Loop will iterate - 4 x 17 x 17 times for wisconsin example'''
        for i in range(nbr_orientation.shape[0]):
            
            '''current and neighbour bin calculation'''
            bin_ = (nbr_orientation[i]/10) - 0.5
            current_bin = np.floor(bin_) 
            nbr_bin = (current_bin + 1)
            
            '''fraction calculate for weight splitting'''
            fn = bin_ - current_bin                     
            fc = 1-fn                                   
            
            '''mod bin number to wrap up, 36-->0 and -1-->35'''
            current_bin = int(current_bin%36)
            nbr_bin = int(nbr_bin%36)
            
            '''weight split'''
            wc = W[i] * fc
            wn = W[i] * fn
            
            '''add votes to bin'''
            bin_vector[current_bin] += wc
            bin_vector[nbr_bin] +=wn
        
        '''the following segment will add the actual votes and smoothed votes to the Bin Matrix'''
        
        '''add the actual vote for a keypoint (Row index no 1,3,5,7)'''
        bin_matrix[num_keypoint] = np.concatenate((bin_vector[18:],bin_vector[:18]))
        
        '''pad a vector with upto two level wrap ups and calculate the smoothed vote value
        if a vector = [1 2 3 4], then padded vector will be = [3 4 1 2 3 4 1 2]'''
        padded_vector = np.concatenate((bin_matrix[num_keypoint,-2:],bin_matrix[num_keypoint],bin_matrix[num_keypoint,:2]))
        bin_matrix[num_keypoint+1] = (((padded_vector[1:-3] + padded_vector[3:-1])/2)+
                                                padded_vector[2:-2])/2
        '''For each keypoint, there are two rows in the Bin Matrix, that is why we are incrementing by two'''
        num_keypoint +=2
    

    '''---------------------------------CALCULATE PEAK VALUES AND PEAK LOCATIONS---------------------------------------'''

    
    '''this is a list of list, each element list is variable length, 
    contains peaks for each keypoint'''
    list_of_peak_lists = []
    
    '''this is a list of list, each element list is variable length, 
    contains corresponding thetas for each keypoint'''
    list_of_theta_lists = []
    
    '''contains the number of strong peak orientations, 
    for each keypoint'''
    list_of_strong_orientation = []
    
    
    ''' the following loop is used to find all the peaks and thetas. 
        We are working with smoothed vote values only,
        that's why, this loop will only traverse 2,4,6,8th row index of Bin Matrix.
        Loop will iterate 4 times, for wisconsin example with 4 keypoints'''
    for i in range(2, bin_matrix.shape[0], 2):
        
        ''' make 1 level padding for circular wrap up
            array  = smoothed vote value vector for each keypoint
            if vector is = [1 2 3 4], then this wrap up will be = [4 1 2 3 4 1]'''
        array = np.concatenate((bin_matrix[i,-1:],bin_matrix[i],bin_matrix[i,:1]))
        
        '''the following segment will find the peaks, a.k.a local maximas for each keypoint, and only keep the peaks
        that are within 80% of the top peak.'''
        local_maxima = (array[1:-1] > array[:-2]) & (array[1:-1] > array[2:])
        peaks = array[1:-1][local_maxima]
        peaks = np.sort(peaks)
        peaks = peaks[::-1]
        count = peaks[peaks>0.80*peaks.max()].size
        
        '''the following segment will fit the parabola, calculate the offset value X, and then use that to calculate
        theta. I have a negative theta shift of 180 because I have worked with 0-360 range initially.'''     
        a1 = bin_matrix[i]
        a2 = peaks 
        peak_indices = (a1[:, None] == a2).argmax(axis=0)
        peak_right = peak_indices+1
        peak_right[peak_right>35] = 0
        peak_left = peak_indices - 1
        peak_left[peak_left<0] = 35
        a = (a1[peak_right] + a1[peak_left])/2 - a1[peak_indices]
        b = (a1[peak_right] - a1[peak_left])/2
        c = a1[peak_indices]
        X = -b / (2*a)
        fx = a* (X**2) + b*X + c      
        theta = (peak_indices + X + 0.5) * 10
        theta -= 180
         
        '''save all the corresponding peak and theta values and number of strong orientations for each keypoints.'''
        list_of_peak_lists.append(list(peaks))
        list_of_theta_lists.append(list(theta))
        list_of_strong_orientation.append(count)
        
    '''---------------------------------PRINT VALUES---------------------------------------'''
    
    '''this loop is used for printing only, will iterate 4 times for wisconsin example with 4 kpts'''
    for i in range(im_keypoints.shape[0]):
        print ("\n Point %d: (%d,%d)"%(i, im_keypoints[i,0], im_keypoints[i,1]))
        print("Histograms:")
        
        '''the following loop will run for 36 times - goes through each bin in histograms'''
        for j in range(bin_matrix.shape[1]):
            print("[%d,%d]: %.2f %.2f"%(bin_matrix[0,j], bin_matrix[0,j]+10, 
                                        bin_matrix[2*i+1,j], bin_matrix[2*i+2,j]))
        
        '''the following loop will run for at max 6 times for the list with 6 peaks'''    
        for j in range(len(list_of_peak_lists[i])):
            print("Peak %d: theta %.1f value %.2f"%(j, list_of_theta_lists[i][j],
                                                    list_of_peak_lists[i][j]))
            
        print("Number of strong orientation peaks: %d"%(list_of_strong_orientation[i]))
       
        
    '''NOTE to Audrey: although I have used few for loops, I haven't iterated through all the pixels of the image
    and have tried to use sufficient amount of vector programming. The for loops I have used, I believe will not add
    that much complexity.'''
    
        
            
        
        
        
        
        
        
        
      