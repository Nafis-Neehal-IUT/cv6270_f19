# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 03:44:20 2019

@author: nafis
"""

import numpy as np
import os 
import cv2
import pickle

'''
This function will load all the images from a folder 
images - list of all images
files - list of all image file names
'''

def load_images_from_folder(folder):
    images  = []
    files   = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            files.append(filename)
    return images, files

'''
This function will generate the descriptor for a single image. 
'''
def generate_single_image_descriptor(img, numBins, bH, bW, windowH, windowW, 
                                     topLeft, bottomRight, label):
    
    '''calculate all the image blocks using the top-right and bottom-left'''
    imageBlocks     = [img[topLeft[i,0]:bottomRight[i,0], 
                       topLeft[i,1]:bottomRight[i,1],:] 
                        for i in range(bH*bW)]
    
    '''calculate all the pixel blocks - merge all three channels into one continuous 
    block with 3 columns side-by-side, each row for one pixel, each column - 1 channel'''
    pixelBlocks     = [im.reshape(windowH*windowW,3) for im in imageBlocks]
    
    '''generate histograms for all the block'''
    histograms      = [np.histogramdd(pixelBlock,(numBins,numBins,numBins))[0] 
                        for pixelBlock in pixelBlocks]
    
    '''flatten each of the histogram for each block'''
    histogramsR     = [np.ravel(hist) for hist in histograms]
    
    '''merge all the histograms of the full image into a single vector'''
    finalDescriptor = np.ravel(histogramsR)
    
    '''generate the final descriptor with class label as the last column'''
    finalDescriptor = np.concatenate((finalDescriptor, np.array([label])))
    
    return finalDescriptor

'''
This function will generate class descriptor for each class by merging descriptor
of single image which belongs to that class. 
'''
def generate_save_class_descriptor(images, outputFILE, label):
    
    finalDescriptor = []
    
    for img in images:
    
        #input properties
        H, W = img.shape[0], img.shape[1]
        
        #decision parameters
        numBins = 4 
        bH      = 4     
        bW      = 4      
        
        #other parameters for window
        delH    = H / (bH+1)
        delW    = W / (bW+1)
        windowH = int(2*delH)
        windowW = int(2*delW)
        
        #calculate top-left and bottom-right borders for each window
        topLeft     = np.array([(n*delH, m*delW) for n in range(bH) 
                        for m in range(bW)]).astype(np.uint8) #vectorize
        bottomRight = topLeft + np.array([windowH, windowW])
        bottomRight = bottomRight.astype(np.uint32)
        
        #generate final descriptor for each image
        finalDescriptor.append(generate_single_image_descriptor(img, numBins, 
                                                                bH, bW, windowH, 
                                                                windowW, topLeft, 
                                                                bottomRight, label))
    
    np.savetxt(outputFILE, finalDescriptor, fmt='%1.2f', delimiter='\t')
        
    return finalDescriptor

'''
This function will generate the output descriptor file's URL for each class.
'''
def generate_input_output_file(outputClass, *args):
    
    inputDIR        = 'hw6_data/train/' + outputClass
    images, files   = load_images_from_folder(inputDIR)
    outputFILE      = inputDIR + '/descriptor-'+ outputClass +'.txt'
    
    return images, outputFILE

if __name__ == "__main__":
    
    #----------------------------------------------------------------------------------------------------
    '''
    Generate all the descriptors for 5 different training data classes seperately
    '''
    grassImages, grassOutputFILE            = generate_input_output_file('grass')
    grassDescriptor                         = generate_save_class_descriptor(grassImages, 
                                                                             grassOutputFILE, 1)
    
    oceanImages, oceanOutputFILE            = generate_input_output_file('ocean')
    oceanDescriptor                         = generate_save_class_descriptor(oceanImages, 
                                                                             oceanOutputFILE, 2)
    
    redcarpetImages, redcarpetOutputFILE    = generate_input_output_file('redcarpet')
    redcarpetDescriptor                     = generate_save_class_descriptor(redcarpetImages, 
                                                                             redcarpetOutputFILE, 3)
    
    roadImages, roadOutputFILE              = generate_input_output_file('road')
    roadDescriptor                          = generate_save_class_descriptor(roadImages, 
                                                                             roadOutputFILE, 4)
    
    wheatfieldImages,  wheatfieldOutputFILE = generate_input_output_file('wheatfield')
    wheatfieldDescriptor                    = generate_save_class_descriptor(wheatfieldImages,  
                                                                             wheatfieldOutputFILE, 5)
    

    '''
    Now save all the descriptors together in a single file.
    Last column is the class label.
    1 = grass, 2 = ocean, 3 = redcarpet, 4 = road, 5 = wheatfield
    '''
    
    datasetDescriptor = np.vstack((grassDescriptor, oceanDescriptor, redcarpetDescriptor, 
                                   roadDescriptor, wheatfieldDescriptor))
    
    with open("trainDesc.pkl", "wb") as f:
        pickle.dump(datasetDescriptor,f)
    
    #----------------------------------------------------------------------------------------------------
    '''
    Generate classwise descriptors for all test data
    '''    
    
    grassImages, grassOutputFILE            = generate_input_output_file('grass', 'test')
    grassDescriptor                         = generate_save_class_descriptor(grassImages, 
                                                                             grassOutputFILE, 1)
    
    oceanImages, oceanOutputFILE            = generate_input_output_file('ocean', 'test')
    oceanDescriptor                         = generate_save_class_descriptor(oceanImages, 
                                                                             oceanOutputFILE, 2)
    
    redcarpetImages, redcarpetOutputFILE    = generate_input_output_file('redcarpet', 'test')
    redcarpetDescriptor                     = generate_save_class_descriptor(redcarpetImages, 
                                                                             redcarpetOutputFILE, 3)
    
    roadImages, roadOutputFILE              = generate_input_output_file('road', 'test')
    roadDescriptor                          = generate_save_class_descriptor(roadImages, 
                                                                             roadOutputFILE, 4)
    
    wheatfieldImages,  wheatfieldOutputFILE = generate_input_output_file('wheatfield', 'test')
    wheatfieldDescriptor                    = generate_save_class_descriptor(wheatfieldImages,  
                                                                             wheatfieldOutputFILE, 5)
    

    '''
    Now save all the descriptors together in a single file.
    Last column is the class label.
    1 = grass, 2 = ocean, 3 = redcarpet, 4 = road, 5 = wheatfield
    '''
    
    datasetDescriptor = np.vstack((grassDescriptor, oceanDescriptor, 
                                   redcarpetDescriptor, roadDescriptor, 
                                   wheatfieldDescriptor))


    with open("testDesc.pkl", "wb") as f:
        pickle.dump(datasetDescriptor,f)
    
    
    
    