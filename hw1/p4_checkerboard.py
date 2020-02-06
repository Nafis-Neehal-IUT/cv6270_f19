#Author: Nafis Neehal (661990881)

import numpy as np
import sys
import cv2

def image_crop(img, filename):
	row, col = img.shape[:2]
	if row>col:
		difference = row-col
		difference = int(difference/2)
		sub_image = img[difference:difference+col,:]
		string_TL = "("+ str(difference) + "," + str(0) + ")"
		string_BR = "("+str(difference+col-1) + "," + str(col-1) + ")"
		print("Image",filename,"cropped at",string_TL,"and",string_BR)
		return sub_image
	elif row<col:
		difference = col-row
		difference = int(difference/2)
		sub_image = img[:,difference: difference+row]
		string_TL = "("+ str(0) + "," + str(difference) + ")" 
		string_BR = "("+str(row-1) + "," + str(difference+row-1) + ")"
		print("Image",filename,"cropped at",string_TL,"and",string_BR)
		return sub_image
	else:
		return sub_image

def image_concatenate(img1, img2, M, N):
	#form an even row (0, 2, ...)
	for i in range(N-1):
		if i==0:
			even_row = np.concatenate((img1, img2), axis=1)
			odd_row = np.concatenate((img2, img1), axis=1)
		elif i!=0 and i%2==0:
			even_row = np.concatenate((even_row, img2), axis=1)
			odd_row = np.concatenate((odd_row, img1), axis=1)
		else:
			even_row = np.concatenate((even_row, img1), axis=1)
			odd_row = np.concatenate((odd_row, img2), axis=1)

	#concatenate even and odd row
	#print(even_row.shape, odd_row.shape)
	unit_block = np.concatenate((even_row, odd_row), axis=0)

	#tile and repeat M-1 times
	main_image = np.tile(unit_block, (int(M/2),1, 1))
	return main_image

if __name__ == '__main__':

	#import images and load arguments
	arguments = sys.argv
	input_image_1 = cv2.imread(arguments[1])
	input_image_2 = cv2.imread(arguments[2])
	output_image_name = arguments[3]
	M = int(arguments[4]) #output image rows
	N = int(arguments[5]) #output image columns
	s = int(arguments[6]) #small box image size

	#image cropping of image 1
	input_image_1_cropped = image_crop(input_image_1, arguments[1])
	input_image_1_resized = cv2.resize(input_image_1_cropped, (s, s))
	print("Resized from", input_image_1_cropped.shape, "to", input_image_1_resized.shape)
	
	
	#image resizing of image 2
	input_image_2_cropped = image_crop(input_image_2, arguments[2])
	input_image_2_resized = cv2.resize(input_image_2_cropped, (s, s))
	print("Resized from", input_image_2_cropped.shape, "to", input_image_2_resized.shape)

	#image concatenate
	main_image = image_concatenate(input_image_1_resized, input_image_2_resized, M, N)

	#image write 
	cv2.imwrite(output_image_name, main_image)
	print("The checkerboard with dimensions", (M*s), "X", (N*s), "was output to", output_image_name)