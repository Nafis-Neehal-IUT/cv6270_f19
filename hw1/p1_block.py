#Author: Nafis Neehal (661990881)

#step 0: import all image and libraries and declare variables

import cv2 #for image r/w/show
import numpy as np
import math
import sys

def main():
    #extract data from command line arguments
    arguments = sys.argv
    img = cv2.imread(arguments[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    M, N = img.shape[:2]
    m, n, b = int(arguments[2]), int(arguments[3]), int(arguments[4])
    s_m = M/m
    s_n = N/n

    #step 1: calculate downsized image float
    downsized_image = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            submatrix_original = img[math.floor(i*s_m):math.floor((i+1)*s_m), math.floor(j*s_n):math.floor((j+1)*s_n)]
            average_original = submatrix_original.mean()
            downsized_image[i,j] = average_original
    downsized_image_original = downsized_image

    #step 2: calculate binary downsized image float
    binary_downsized_image = downsized_image
    threshold = np.median(binary_downsized_image)
    binary_downsized_image = np.where(binary_downsized_image>=threshold, 255, 0) #no for loops used

    #step 3: calculate blocked image 
    upsample_row = m*b
    upsample_col = n*b
    blocked_downsized_image = np.zeros((upsample_row, upsample_col))
    blocked_binary_downsized_image = np.zeros((upsample_row, upsample_col))

    downsized_image = downsized_image.astype(np.uint8)  #convert downsized image from float to uint8 before blocked image
    binary_downsized_image = binary_downsized_image.astype(np.uint8)

    row_start = 0
    row_end = 0+b
    col_start = 0
    col_end = 0+b

    for i in range(m):
        for j in range(n):
            blocked_downsized_image[row_start:row_end, col_start:col_end] = downsized_image[i,j]
            blocked_binary_downsized_image[row_start:row_end, col_start:col_end] = binary_downsized_image[i,j]
            col_start = col_start + b
            col_end = col_end + b
        col_start = 0
        col_end = 0+b
        row_start = row_start + b
        row_end = row_end + b

    #step 4:converting all output images to uint8 from float32 or float 64
    blocked_downsized_image = blocked_downsized_image.astype(np.uint8)
    blocked_binary_downsized_image = blocked_binary_downsized_image.astype(np.uint8)

    #step 4.5: file name processing
    file_name_ext = arguments[1]
    file_extension = file_name_ext[-4:]
    file_name = file_name_ext[:-4]
    new_file_name_b = file_name + "_b" + file_extension
    new_file_name_g = file_name + "_g" + file_extension


    #step 5: write all images
    cv2.imwrite(new_file_name_g, blocked_downsized_image)
    cv2.imwrite(new_file_name_b, blocked_binary_downsized_image)

    #step 6: text outputs
    print("Downsized images are", downsized_image.shape[:2])
    print("Block images are ", blocked_downsized_image.shape[:2])

    avg_value = downsized_image_original[m//3, n//3]
    print("Average intensity at", (m//3, n//3), "is", '{0:.2f}'.format(avg_value))

    avg_value = downsized_image_original[m//3, (2*n)//3]
    print("Average intensity at", (m//3, (2*n)//3), "is", '{0:.2f}'.format(avg_value))

    avg_value = downsized_image_original[(2*m)//3, n//3]
    print("Average intensity at", ((2*m)//3, n//3), "is", '{0:.2f}'.format(avg_value))

    avg_value = downsized_image_original[(2*m)//3,(2*n)//3]
    print("Average intensity at", ((2*m)//3, (2*n)//3), "is", '{0:.2f}'.format(avg_value))
    
    print("Binary threshold:", '{0:.2f}'.format(threshold))

    print("Wrote image", new_file_name_g)
    print("Wrote image", new_file_name_b)

if __name__ == '__main__':
   main()