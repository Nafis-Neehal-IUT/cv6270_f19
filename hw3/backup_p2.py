# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:13:25 2019

@author: nafis
"""


'''Steps-
1. Import image as np.float32
2. Compute SobelX, SobelY (Convert to Gray, ksize=5 default)
3. Compute energy matrix E = sum(abs(sobelx)+abs(sobely))
4. Compute W matrix - seam cost function - to choose seam pixels --> STUCK
5. Trace back W to find actual seams
6. Once a seam formed, remove from image (use concatenation to stitch disconnected segments)
7. Choose Vertical/Horizontal seam depending on image size -> to make it a square 
'''


import numpy as np
import cv2
import sys

if __name__ == '__main__':
    
    #read file as np.float32
    args = sys.argv
    file_name = args[1]
    img = cv2.imread(file_name).astype(np.float32)
    
    #decide v/h seam
    rows, cols = img.shape[0], img.shape[1]
    
    C_first = None
    C_second = None
    C_last = None
    E_first = None
    E_Second = None
    E_Last = None
    Org_img = img
    seam_type = None
    
    if cols>rows:
        seam_type = "vertical"
        num_iterate = cols-rows
        for n in range(num_iterate):
            #calculate gradient
            sobelx = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),cv2.CV_64F,1,0)
            sobely = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),cv2.CV_64F,0,1)
            
            #energy matrix calculate (E)
            E = np.absolute(sobelx) + np.absolute(sobely)
        
            #calculate W
            W = np.zeros_like(E)
            rw, cw = W.shape[0], W.shape[1]
            W[:,0] = W[:,cw-1] = np.inf 
            for i in range(W.shape[0]):         #traversing rows of W
                if i==0:
                    W[i, 1:(cw-1)] = E[i, 1:(cw-1)]
                else:
                    W[i, 1:(cw-1)] = E[i, 1:(cw-1)] + np.min((W[(i-1), 0:(cw-2)], W[(i-1), 1:(cw-1)], W[(i-1), 2:(cw)]), axis=0)
            
            #calculate seam vector
            C = np.zeros((rw), dtype=int)       #column vector with size of E's column
            for i in range(rw-1,-1,-1):         #traversing rows of C in reverse order
                if i == rw-1:
                    C[i] = np.argmin(W[i])
                else:
                    val_min = np.min(np.array([W[i,C[i+1]-1], W[i,C[i+1]], W[i,C[i+1]+1]]))
                    X = np.argwhere(W[i]==val_min)
                    if(X.shape[0]>1):
                        C[i] = X[0]
                    else:
                        C[i] = X
            
            #calculate average energy of the seam
            sum_eng = 0
            for i in range(E.shape[0]):
                sum_eng = sum_eng + E[i,C[i]]
            sum_eng = sum_eng/E.shape[0]
                
            #output first seam on image
            if n==0:
                C_first = C
                E_first = sum_eng
                copy_img = Org_img
                for i in range(img.shape[0]):
                    copy_img[i,C_first[i],0] = 0
                    copy_img[i,C_first[i],1] = 0
                    copy_img[i,C_first[i],2] = 255
                first_seam = np.uint8(copy_img)        
                cv2.imwrite("seam_first.jpg", first_seam)
            
            #save for n=1 (2nd seam)
            if n==1:
                C_second = C
                E_Second = sum_eng
            
            if n==(num_iterate-1):
                C_last = C
                E_Last = sum_eng
                
            #remove seam column using boolean masking
            boolean_mask = np.ones_like(img, dtype=bool)
            for i in range(boolean_mask.shape[0]):
                boolean_mask[i,C[i]] = False
            img = img[boolean_mask].reshape((img.shape[0], img.shape[1]-1, img.shape[2]))
            
            
            
    else:
        seam_type = "horizontal"
        num_iterate = rows-cols
        for n in range(num_iterate):
            #calculate gradient
            sobelx = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),cv2.CV_64F,1,0)
            sobely = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),cv2.CV_64F,0,1)
            
            #energy matrix calculate (E)
            E = np.absolute(sobelx) + np.absolute(sobely)
        
            #calculate W
            W = np.zeros_like(E)
            rw, cw = W.shape[0], W.shape[1]
            W[0,:] = W[rw-1,:] = np.inf 
            for i in range(W.shape[1]):         #traversing cols of W
                if i==0:
                    W[1:(rw-1), i] = E[1:(rw-1), i]
                else:
                    W[1:(rw-1), i] = E[1:(rw-1), i] + np.min((W[0:(rw-2), (i-1)], W[1:(rw-1), (i-1)], W[2:(rw), (i-1)]), axis=0)
            
            #calculate seam vector
            C = np.zeros((cw), dtype=int)       #column vector with size of E's column
            for i in range(cw-1,-1,-1):         #traversing cols of C in reverse order
                if i == cw-1:
                    C[i] = np.argmin(W[:,i])
                else:
                    val_min = np.min(np.array([W[C[i+1]-1, i], W[C[i+1], i], W[C[i+1]+1, i]]))
                    X = np.argwhere(W[:,i]==val_min)
                    if(X.shape[0]>1):
                        C[i] = X[0]
                    else:
                        C[i] = X
            
            #calculate average energy of the seam
            sum_eng = 0
            for i in range(E.shape[1]):
                sum_eng = sum_eng + E[C[i],i]
            sum_eng = sum_eng/E.shape[1]
                
            #output first seam on image
            if n==0:
                C_first = C
                E_first = sum_eng
                copy_img = Org_img
                for i in range(img.shape[1]):
                    copy_img[C_first[i],i,0] = 0
                    copy_img[C_first[i],i,1] = 0
                    copy_img[C_first[i],i,2] = 255
                    
                first_seam = np.uint8(copy_img)        
                cv2.imwrite("foo_seam.jpg", first_seam)
            
            #save for n=1 (2nd seam)
            if n==1:
                C_second = C
                E_Second = sum_eng
            
            if n==(num_iterate-1):
                C_last = C
                E_Last = sum_eng
                
            #remove seam column using boolean masking
            boolean_mask = np.ones_like(img, dtype=bool)
            for i in range(boolean_mask.shape[1]):
                boolean_mask[C[i], i] = False
            
            img = img[boolean_mask].reshape((img.shape[0]-1, img.shape[1], img.shape[2]))
            
            
            #test
            if n>10 and n<20:
                cv2.imshow('rands',np.uint8(img))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
    #print stuff
    
    print("")
    print("Points on seam 0:")
    print(seam_type)
    print("0, {}".format(C_first[0]))
    middle_index = int(np.round(C_first.shape[0]/2))
    print("{}, {}".format(middle_index, C_first[middle_index]))
    last_index = C_first.shape[0]-1
    print("{}, {}".format(last_index, C_first[last_index]))
    print("Energy of seam 0: {:.2f}".format(E_first))
    
    print("")
    print("Points on seam 1:")
    print(seam_type)
    print("0, {}".format(C_second[0]))
    middle_index = int(np.round(C_second.shape[0]/2))
    print("{}, {}".format(middle_index, C_second[middle_index]))
    last_index = C_second.shape[0]-1
    print("{}, {}".format(last_index, C_second[last_index]))
    print("Energy of seam 0: {:.2f}".format(E_Second))
    
    print("")
    print("Points on seam {}:".format(num_iterate-1))
    print(seam_type)
    print("0, {}".format(C_last[0]))
    middle_index = int(np.round(C_last.shape[0]/2))
    print("{}, {}".format(middle_index, C_last[middle_index]))
    last_index = C_last.shape[0]-1
    print("{}, {}".format(last_index, C_last[last_index]))
    print("Energy of seam 0: {:.2f}".format(E_Last))
    
    
    final_image = np.uint8(img)        
    cv2.imwrite("foo_final.jpg", final_image)
    
    
    
    
    
    
    
    
    
    
    