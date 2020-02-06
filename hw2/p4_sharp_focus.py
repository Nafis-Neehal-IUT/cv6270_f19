# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:17:31 2019

@author: nafis
"""

import numpy as np
import cv2
import os
import sys


#import a folder of images and create an image array
def load_images_from_folder(folder):
    images = []
    file_list = []

    #first get only the filenames
    for filename in os.listdir(folder):
        file_list.append(filename)
    
    #sort them lexicographically, appraently this will create problem in Submitty
    file_list.sort()
    
    #for sorted file list, now import as image files
    for file in file_list:
        img = cv2.imread(os.path.join(folder,file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        if img is not None:
            images.append(img)
    
    return images, file_list

#this function will print everything according to format
def print_stuff(x, y, finalImage, E):
    counter=0
    print("")
    print("Energies at ({:.0f}".format(x)+", {:.0f})".format(y))
    for E_of_image in E:
        print("{:.0f}: ".format(counter)+"{:.1f}".format(E_of_image[x][y]))
        counter = counter + 1
    print("RGB: ({:.0f}".format(finalImage[x,y,0])+", {:.0f}".format(finalImage[x, y, 1])+", {:.0f})".format(finalImage[x, y, 2]))
    

if __name__=='__main__':
    args = sys.argv
    image_dir = args[1]
    output_file = args[2]
    sigma = float(args[3])
    p = int(args[4])    
    

    images, files = load_images_from_folder(image_dir)
    h = np.floor(2.5*sigma)
    ksize = int(2*h+1)
    gray_images = []
    E = []
    
    #calculate Energy for each image
    for img in (images):        
        #create sobelx and sobely for image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_images.append(gray)
        sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0)
        sobely = cv2.Sobel(gray,cv2.CV_64F,0,1)
        gradient_magnitude = (sobelx**2) + (sobely**2)
        weighted_function = cv2.GaussianBlur(gradient_magnitude, (ksize,ksize), sigma)
        E.append(weighted_function)
    
    #apply energy matrix on each of the channel of a RGB image     
    prod_sum_R = 0
    prod_sum_G = 0
    prod_sum_B = 0
    e_sum = 0
    M,N = images[0].shape[0], images[0].shape[1]
    finalImage = np.zeros((M,N,3))
    for i in range(len(images)):
        prod_sum_R = prod_sum_R + ((E[i]**p)*images[i][:,:,0])
        prod_sum_G = prod_sum_G + ((E[i]**p)*images[i][:,:,1])
        prod_sum_B = prod_sum_B + ((E[i]**p)*images[i][:,:,2])
        e_sum = e_sum + (E[i]**p)
    finalImage[:,:,0] = prod_sum_R/e_sum
    finalImage[:,:,1] = prod_sum_G/e_sum
    finalImage[:,:,2] = prod_sum_B/e_sum
    
    finalImage = np.round(finalImage).astype('uint8')
    
    #printing stuff
    print("Results:")
    print_stuff(M//4, N//4, finalImage, E)
    print_stuff(M//4, (3*N)//4, finalImage, E)
    print_stuff((3*M)//4, N//4, finalImage, E)
    print_stuff((3*M)//4, (3*N)//4, finalImage, E)
    
    print("Wrote "+output_file+"")
    
    cv2.imwrite(output_file, finalImage)
    

    
        
    
