# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 03:58:27 2019

@author: nafis
"""

import cv2
import numpy as np
import sys


if __name__ == "__main__":
    
    args = sys.argv
    image = args[1]
    rect = args[2]
    
    #import image
    im = cv2.imread(image)
    
    #import rectangles
    with open(rect) as f:
        data = [int(word) for line in f for word in line.split()]
        
    #build mask matrix
    data = np.array(data)
    data = data.reshape(-1, 5)
    
    #resize image 
    img = cv2.resize(im,(500, 500))
    copy_img = img
    
    #initial mask
    mask = np.zeros(img.shape[:2],np.uint8)
    
    #internal model matrices
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    #defining outer boundary (coords format-> x1, y1, x2, y2)
    rect = tuple((data[0,1],data[0,2],data[0,3],data[0,4]))
    
    #first grabcut with rectangular boundary
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    
    #mask manipulate (coords format-> y1, x1, y2, x2)
    for i in range(1,data.shape[0]):
        
        #must foreground if 1, else background if 0
        if data[i,0] == 1:
            mask[data[i,1]:data[i,3],data[i,2]:data[i,4]] = 1    
        elif data[i,0] == 0:
            mask[data[i,1]:data[i,3],data[i,2]:data[i,4]] = 0
    
    #second time masking with mask only for inner rectangles
    mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,
                                              5,cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')   
    img = img*mask[:,:,np.newaxis]
    
    #write final_image
    cv2.imwrite("final_image.jpg", img)
    
    #draw outer rectangles on original image
    copy_img = cv2.rectangle(copy_img, (data[0,1], data[0,2]), (data[0,3], data[0,4]), 
                             (255,0,0), 2)
    
    #draw inner rectangles on original image
    for i in range(1,data.shape[0]):
        if data[i,0] == 1:
            color = (0,255,0)
        elif data[i,0] == 0:
            color = (0,0,255)
        copy_img = cv2.rectangle(copy_img, (data[i,2], data[i,1]), (data[i,4], data[i,3]), 
                             color, 1)
        
    #write image with rectangles
    cv2.imwrite("rect_on_image.jpg", copy_img)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
