import numpy as np 
import cv2
import math
import sys

def calculate_normalized_distance_matrix():
	row_dimension_matrix = np.arange(M)
	rd = np.tile(row_dimension_matrix, (N,1))
	row_dimension_matrix = rd.transpose()
	column_dimension_matrix = np.arange(N)
	column_dimension_matrix = np.tile(column_dimension_matrix, (M, 1))


# def shade_left():
def shade_right(input_image, M, N, direction):
	# addition_value = 1 / N
	# right_shaded_image = np.zeros((M, N, 3))
	# alpha_mask = np.linspace(0, 1, endpoint=True, num = N) #evenly spaced values between 0 and 1, total N
	# alpha_mask = np.tile(alpha_mask, (M, 1)) #stack values for M rows
	# for i in range(3):
	# 	right_shaded_image[:,:,i] = np.multiply(input_image[:,:,i], alpha_mask)
	# print(input_image)
	# print(right_shaded_image)
	# cv2.imshow("input",input_image)
	# cv2.waitKey(0)
	# cv2.imshow("shade", right_shaded_image)
	# cv2.waitKey(0)




# def shade_top():
# def shade_bottom():
# def shade_center():

def main():
	#imports from command lines
	arguments = sys.argv
	input_image = cv2.imread(arguments[1])
	M, N = input_image.shape[:2]
	output_image = arguments[2]
	direction = arguments[3]
	print("Image Shape: ", M, N)

	if direction=="right":
		shade_right(input_image, M, N, direction)

if __name__ == '__main__':
   main()