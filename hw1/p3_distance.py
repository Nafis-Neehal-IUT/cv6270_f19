# NEED FORMATTING
# NEED average color component based comparison

import numpy as np 
import os
import sys
import cv2

def load_images_from_folder(folder):
    images = []
    file = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            file.append(filename)
    return images, file

def generate_channels(img):
	red_channel = np.ravel(img[:,:,0])
	green_channel = np.ravel(img[:,:,1])
	blue_channel = np.ravel(img[:,:,2])
	return red_channel, green_channel, blue_channel

def generate_distance(h1, h2):
	distance_squared = np.sum((h1-h2)**2)
	rs_distance = np.sqrt(distance_squared)
	return rs_distance

def find_max_min_value(dist_dict):
	minimum = min(dist_dict.items(), key=lambda x: x[1])
	maximum = max(dist_dict.items(), key=lambda x: x[1])
	return minimum, maximum


def avg_color_vector_comparison(images, file):
	counter = 0
	#average color vector storage dictionary
	avg_dict = {}
	for img in (images):
		r, g, b = generate_channels(img)
		r_mean = np.mean(r)
		g_mean = np.mean(g)
		b_mean = np.mean(b)
		avg_dict[file[counter]] = np.array([r_mean, g_mean, b_mean])
		counter = counter + 1

	#average color vector similarity
	avg_dist_dict = {}
	for i in range(len(file)):
		for j in range(len(file)):
			if j>i:
				distance = generate_distance(avg_dict[file[i]], avg_dict[file[j]])
				avg_dist_dict[(file[i], file[j])] = distance

	minimum, maximum = find_max_min_value(avg_dist_dict)

	#printing portion
	print("Using distance between color averages.")
	print("Closest pair is {}".format(tuple(sorted(minimum[0]))))
	print("Minimum distance is", '{0:.3f}'.format(minimum[1]))
	print("Furthest pair is", tuple(sorted(maximum[0])))
	print("Maximmum distance is", '{0:.3f}'.format(maximum[1]))

def histogram_comparison(images, file):

	hist_dict = {}
	counter = 0

	#calculate histogram for each image and save into dictionary
	for img in (images):
		edges = np.linspace(0, 255, 9)
		r, g, b = generate_channels(img)
		hist, e = np.histogramdd([r, g, b], bins=[edges, edges, edges])
		normalized_hist = hist/np.sum(hist)
		hist_dict[file[counter]]=normalized_hist
		counter = counter + 1
	
	#calculate paired distances
	dist_dict = {}
	for i in range(len(file)):
		for j in range(len(file)):
			if j>i:
				distance = generate_distance(hist_dict[file[i]], hist_dict[file[j]])
				dist_dict[(file[i], file[j])] = distance 
	minimum, maximum = find_max_min_value(dist_dict)

	#printing portion
	print("Using distance between histograms.")
	print("Closest pair is", tuple(sorted(minimum[0])))
	print("Minimum distance is", '{0:.3f}'.format(minimum[1]))
	print("Furthest pair is", tuple(sorted(maximum[0])))
	print("Maximmum distance is", '{0:.3f}'.format(maximum[1]))

if __name__ == '__main__':
	#load all iamges into images list
	arguments = sys.argv
	images, file = load_images_from_folder(arguments[1])
	avg_color_vector_comparison(images, file)
	print("")
	histogram_comparison(images, file)







