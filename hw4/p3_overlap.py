# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:00:46 2019

@author: nafis
"""

import numpy as np
import math
import sys

def form_rotation_matrix(deg2rad, rx, ry, rz):
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(deg2rad*rx), (-1)*math.sin(deg2rad*rx)],
                   [0, math.sin(deg2rad*rx),math.cos(deg2rad*rx)]])
    Ry = np.array([[math.cos(deg2rad*ry), 0, math.sin(deg2rad*ry)],
                   [0, 1, 0],
                   [(-1)*math.sin(deg2rad*ry), 0, math.cos(deg2rad*ry)]])
    Rz = np.array([[math.cos(deg2rad*rz), (-1)*math.sin(deg2rad*rz), 0],
                   [math.sin(deg2rad*rz), math.cos(deg2rad*rz), 0],
                   [0, 0, 1]])
    
    return Rx@Ry@Rz

def form_K_Matrix(S, Uc, Vc):
    return np.array([[S, 0, Uc],
                     [0, S, Vc],
                     [0, 0, 1]])
    

if __name__=="__main__":
    
    '''------------------------------Part(a)-----------------------------------'''
    '''command prompt arguments'''
    args = sys.argv
    param_file = args[1]
    
    
    '''import the data from input text file (splitted word by word)'''
    with open(param_file) as f:
        data = [np.float64(word) for line in f for word in line.split()]
        
    '''data assignment to corresponding variables'''
    rx1, ry1, rz1 = data[0], data[1], data[2]
    s1, ic1, jc1 = data[3], data[4], data[5]
    rx2, ry2, rz2 = data[6], data[7], data[8]
    s2, ic2, jc2 = data[9], data[10], data[11]
    N = data[12]
    
    '''rotation matrix formation 3x3 (rx, ry, rz)'''
    deg2rad = (math.pi/180)
    R1 = form_rotation_matrix(deg2rad, rx1, ry1, rz1)
    R2 = form_rotation_matrix(deg2rad, rx2, ry2, rz2)
    
    '''K matrix formation'''
    K1 = form_K_Matrix(s1, jc1, ic1)
    K2 = form_K_Matrix(s2, jc2, ic2)
    
    '''Form (both) H Matrices'''
    H21 = K2@R2@R1.T@np.linalg.inv(K1) 
    H12 = K1@R1@R2.T@np.linalg.inv(K2)
    
    '''normalize - froebnius'''
    H21 /= np.linalg.norm(H21)
    H12 /= np.linalg.norm(H12)
    
    '''scale by 1000'''
    H21 *= 1000
    H12 *= 1000
    
    '''deal with ambiguity (making sure last element (2,2) stays positive)'''
    (rows, cols) = H21.shape
    if (H21[rows-1, cols-1] < 0):
        H21 = -H21
    if (H12[rows-1, cols-1] < 0):
        H12 = -H12
    
    '''print'''
    print("Matrix: H_21")
    for i in range(H21.shape[0]): 
        print("%.3f, %.3f, %.3f"%(H21[i,0], H21[i,1], H21[i,2]))

    '''------------------------------Part(b)-----------------------------------'''
    
    '''image size'''
    im_rows = 4000
    im_cols = 6000
    
    '''build U1 matrix -> remember to flip the coordinated while cv2->numpy'''
    U1 = np.array([[0,0,1],
                   [im_cols,0,1],
                   [0,im_rows,1],
                   [im_cols,im_rows,1]])
    U1 = U1.T
    
    '''mapping from U1 to U2 using H21'''
    U2 = H21@U1
    
    '''unaugment by dividing with the last row'''
    U2 /= U2[2,:]
    
    '''calculate boundary'''
    lower_right_coord = np.max(U2[0:2,:], axis=1)
    upper_left_coord = np.min(U2[0:2,:], axis=1)
    
    '''print'''
    print("Upper left: %.1f %.1f"%(upper_left_coord[1], upper_left_coord[0]))
    print("Lower right: %.1f %.1f"%(lower_right_coord[1], lower_right_coord[0]))
    
    '''------------------------------Part(c)-----------------------------------'''
    
    '''calculate row samples'''
    del_r = im_rows/N
    first_r_sample = del_r/2
    row_samples = np.linspace(first_r_sample, im_rows-first_r_sample, 
                              num=int(np.ceil((im_rows-first_r_sample)/del_r)), 
                              endpoint=True)
    
    
    '''calculate column samples'''
    del_c = im_cols/N
    first_c_sample = del_c/2
    col_samples = np.linspace(first_c_sample, im_cols-first_c_sample,
                              num=int(np.ceil((im_cols-first_c_sample)/del_c)),
                              endpoint=True)
    
    '''form sampled coordinate matrix'''
    rows, cols = np.meshgrid(col_samples, row_samples)
    rows = np.ravel(rows)
    cols = np.ravel(cols)
    coords = np.stack((rows, cols), axis=0)
    
    '''form homogeneous coordinates'''
    coords_aug = np.vstack((coords, np.ones_like(rows)))
    
    '''mapping from U1 to U2'''
    coords21 = H21@coords_aug
    coords21 /= coords21[2,:]
    coords12 = H12@coords_aug
    coords12 /= coords12[2,:]
    
    '''create boundary'''
    bound_rows = np.arange(0,4000)
    bound_cols = np.arange(0,6000)
    
    '''calculate count'''
    count21=0
    count12=0
    for i in range(coords21.shape[1]):
        if int(coords21[1,i]) in bound_rows and int(coords21[0,i]) in bound_cols:
            count21 +=1
        if int(coords12[1,i]) in bound_rows and int(coords12[0,i]) in bound_cols:
            count12 +=1
            
    '''print'''
    print("H_21 overlap count", count21)
    print("H_21 overlap fraction %.3f"%(count21/coords21.shape[1]))
    print("H_12 overlap count", count12)
    print("H_12 overlap fraction %.3f"%(count12/coords12.shape[1]))
    
    
    '''------------------------------Part(d)-----------------------------------'''
    
    '''calculate d'''
    U2_line_cord = np.array([3000,2000,1]).T
    d = R2.T@np.linalg.inv(K2)@U2_line_cord
    
    '''resolving ambiguity'''
    if d[d.shape[0]-1]<0:
        d = -d
    
    '''print'''
    print("Image 2 center direction: (%.3f, %.3f, %.3f)"%(d[0], d[1], d[2]))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    