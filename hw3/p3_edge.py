# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 15:39:47 2019

@author: nafis
"""


'''
Steps-
1. Import the image as grayscale
2. Apply Gaussian Smoothing (ksize = 2.5*sigma+1, sigma=2), use cv2.GaussianBlur, --->Image should be np.float32
3. Calculate SobelX, SobelY (ksize = 4*sigma+1)
4. Calculate Gradient Magnitude using sqrt(SobelX**2 + SobelY**2)
5. Calculate Gradient Direction using arctan2(SobelY, SobelX), use Rad2Deg conversion
6. Calculate Directions of each pixel in Gradient Magnitude matrix and get ahead and behind neighbors
7. Compare each pixel with its neighbors and if not maximum, suppress with 0 (use seperate array for the record)
8. Now apply thresholding (given in HW Statement)
9. Count num_pixels that survived, and other stuff for printing
'''

import numpy as np
import cv2
import sys

#this function returns ahead_x, ahead_y, behind_x, behind_y, dir_string -->> Calculate in terms of radians
def calculate_neighbours(angle_in_degree, px, py):
    if angle_in_degree>-22.5 and angle_in_degree<=22.5:
        return px+1, py, px-1, py, "EW"
    elif angle_in_degree>22.5 and angle_in_degree<=67.5:
        return px+1, py+1, px-1, py-1, "NWSE"
    elif angle_in_degree>67.5 and angle_in_degree<=112.5:
        return px, py+1, px, py-1, "NS"
    elif angle_in_degree>112.5 and angle_in_degree<=157.5:
        return px-1, py+1, px+1, py-1, "NESW"
    elif angle_in_degree>157.5 or angle_in_degree<=-157.5:
        return px-1, py, px+1, py, "EW"
    elif angle_in_degree>-157.5 and angle_in_degree<=-112.5:
        return px-1, py-1, px+1, py+1, "NWSE"
    elif angle_in_degree>-112.5 and angle_in_degree<=-67.5:
        return px, py-1, px, py+1, "NS"
    elif angle_in_degree>-67.5 and angle_in_degree<=-22.5:
        return px+1, py-1, px-1, py+1, "NESW"

if __name__ == "__main__":
    args = sys.argv
    sigma = np.float64(args[1])
    filename = args[2]    
    img = cv2.imread(filename).astype(np.float32)
    #filename="tree_sky.jpg"

    
    #Gaussian Smoothing 
    ksize = 2*np.floor(2.5*sigma)+1
    rad2deg = 180 / np.pi
    img_s = cv2.GaussianBlur(img, (int(ksize),int(ksize)), sigma)    
    
    #x,y derivative and gradient magnitude & direction 
    im_dx = cv2.Sobel(cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY),cv2.CV_64F,1,0)
    im_dy = cv2.Sobel(cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY),cv2.CV_64F,0,1)
    im_gm = np.sqrt(im_dx**2 + im_dy**2)        #gradient magnitude
    im_dir = np.arctan2(im_dy, im_dx)*rad2deg   #gradient direction (in degrees)
    im_clr = img
    
    #generate the gradient color image and write
    for i in range(im_gm.shape[0]):
        for j in range(im_gm.shape[1]):
            if i==0 or j==0 or i==im_gm.shape[0]-1 or j==im_gm.shape[1]-1 or im_gm[i][j]<1.0:
                im_clr[i,j,:] = np.array([0,0,0])
            else:
                ax, ay, bx, by, dir_str = calculate_neighbours(im_dir[i,j], i, j)
                if dir_str=="EW":
                    im_clr[i,j,:] = np.array([0,0,255])
                elif dir_str=="NESW":
                    im_clr[i,j,:] = np.array([255,255,255])
                elif dir_str=="NS":
                    im_clr[i,j,:] = np.array([255,0,0])
                elif dir_str=="NWSE":
                    im_clr[i,j,:] = np.array([0,255,0])
                    
    #file name processing and writing
    file_name_ext = filename
    file_extension = file_name_ext[-4:]
    file_name = file_name_ext[:-4]
    new_file_name_dir = file_name + "_dir" + file_extension
    cv2.imwrite(new_file_name_dir, im_clr)
    
    #generate gradient magnitude image
    im_gm_f = im_gm * 255.0/im_gm.max()
    
    #file name processing and writing
    new_file_name_grd = file_name + "_grd" + file_extension
    cv2.imwrite(new_file_name_grd, im_gm_f)
    
    #seperate array to save the nms survivors
    im_gm_nms = np.zeros_like(im_gm)
    
    #non-maximum suppression attempt
    for i in range(im_gm.shape[0]):
        if (im_gm[i]>0).sum()==0:        #if the row does not have any non-zero gradient, skip
            continue;
        for j in range(im_gm.shape[1]):
            nbr_ahd_x, nbr_ahd_y, nbr_bhnd_x, nbr_bhnd_y, dir_str = calculate_neighbours(im_dir[i,j], i, j) #get nbrs and dir_str
            if ((0<=nbr_ahd_x<=im_gm.shape[0]-1) and (0<=nbr_bhnd_x<=im_gm.shape[0]-1) and (0<=nbr_ahd_y<=im_gm.shape[1]-1) and (0<=nbr_bhnd_y<=im_gm.shape[1]-1)):
                if im_gm[i,j]>=im_gm[nbr_ahd_x, nbr_ahd_y] and im_gm[i,j]>=im_gm[nbr_bhnd_x, nbr_bhnd_y]: #local maxima
                    im_gm_nms[i,j] = im_gm[i,j]
     

    #print
    print("Number after non-maximum:",(im_gm_nms>0).sum())
    
    im_gm_nms[im_gm_nms<1]=0
    print("Number after 1.0 threshold:",(im_gm_nms>0).sum())
    
    mu = np.sum(im_gm_nms[im_gm_nms>0])/((im_gm_nms>0).sum())
    s = np.std(im_gm_nms[im_gm_nms>0])
    print("mu: {:.2f}".format(mu))
    print("s: {:.2f}".format(s))
    
    threshold = np.minimum((mu + 0.5*s),(30/sigma))
    print("threshold: {:.2f}".format(threshold))
    
    print("Number after threshold:",(im_gm_nms>threshold).sum())
    
    im_gm_nms[im_gm_nms<threshold] = 0
    
    #file name processing and writing
    new_file_name_thr = file_name + "_thr" + file_extension
    cv2.imwrite(new_file_name_thr, im_gm_nms)
    
                

    
    
    
    