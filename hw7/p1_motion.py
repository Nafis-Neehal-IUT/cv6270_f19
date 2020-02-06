# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 02:07:34 2019

@author: nafis
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift

'''
This function is to show figures using Matplotlib.
'''
def show_image(im, filename):
    plt.figure()
    plt.imshow(im[...,::-1].astype(np.uint32)) 
    plt.show()
    plt.savefig(filename)
    
'''
This function will create amd apply ORB descriptor on both images and return
keypoints and descriptors.
'''
def apply_ORB(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    return kp1, des1, kp2, des2

'''
This function will create a BF matcher object which will match descriptors from
a pair of images, sort those matches based on distance (ASC) and return the 
match list. 
'''
def BF_matcher_apply(des1, des2, threshold):
    
    '''
    find all the keypoint matches between two images
    used Hamming Distance here
    '''
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x:x.distance)
    
    '''
    Applied another level of thresholding for selecting only those matched
    keypoints which have distance less than a specified threshold
    as BF matcher does not allow this specified thresholding
    '''
    matcher_thresh = lambda threshold:[m.distance<=threshold for m in matches]
    match_mask = np.array(matcher_thresh(threshold))
    match_array = np.array(matches)
    thresholded_matches = match_array[match_mask==1]
    
    return matches, thresholded_matches

'''
This function will draw matched keypoints on both images and connect them with
line. It will return the image with keypoints and lines drawn over it.
'''
def draw_match(im1, kp1, im2, kp2, matches, *num_matches):
    
    draw_params = dict(matchColor = (0,255,0), 
                       singlePointColor = None,
                       flags = 2)
    
    if(num_matches):
        drawnMatch = cv2.drawMatches(im1,kp1,im2,kp2,matches[:num_matches[0]],None,
                                     **draw_params)
    else:
        drawnMatch = cv2.drawMatches(im1,kp1,im2,kp2,matches,None,**draw_params)
        
    return drawnMatch

'''
This function will generate keypoint locations of matches generated
by BF Matcher.
'''
def keypoint_location_generate(matches, kp, sd):
    if sd=='s':
        kp_locs = np.float32([ kp[m.queryIdx].pt for m in matches])
    elif sd=='d':
        kp_locs = np.float32([ kp[m.trainIdx].pt for m in matches])
    return kp_locs
    #return kp_locs.reshape(-1,1,2)

'''
This function will generate the optical flow
'''

def calculate_optical_flow(im1, im2, p0):
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (17,17),
                      maxLevel = 1,
                      criteria = (cv2.TERM_CRITERIA_COUNT | 
                                  cv2.TERM_CRITERIA_EPS, 10, 0.03))
    
    p1, stat, err = cv2.calcOpticalFlowPyrLK(im1, im2, p0, None, **lk_params)
    
    idxs = stat==1
    idxs = idxs[...,-1]
    p_new = p1[idxs]
    p_old = p0[idxs]
    
    return p_new, p_old


'''
This function will draw the Optical Flow tracks on the source 
image. Random color scheme has been used here.
'''
def draw_tracks(old_frame, frame, p_new, p_old):
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    
    # draw the tracks
    for i,(new,old) in enumerate(zip(p_new,p_old)):
        num_color = i%100
        a,b = old.ravel()
        c,d = new.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[num_color].tolist(),3)
        old_frame = cv2.circle(old_frame,(a,b),5,color[num_color].tolist(), 2)
    img = cv2.add(old_frame,mask)
    
    return img
    
'''
This function will threshold the initially generated motion vector based on
its magnitude. 
'''
def thresholding_motion_vector(u, threshold):
    
    #number of motion vectors
    num_motion_vectors = u.shape[0]

    #calculate the magnitude of the motion vector
    u_magnitude = np.linalg.norm(u, axis=1)
    
    #sort in descending order    
    idxs = np.argsort(-u_magnitude)
    
    percentage = int(np.floor(num_motion_vectors*threshold))
    idxs = idxs[:percentage]
    
    return idxs

'''
This function will calculate a normal to each and every motion vector which will
in terms be used for calculating the intersection points. 
'''
def calculate_normal_motion_vector(u):
    #normal matrix to motion vectors
    n = np.zeros_like(u)
    n[:, 0] = u[:, 1]
    n[:, 1] = (-1)*u[:, 0]
    
    n_mag = np.linalg.norm(n, axis=1)[...,np.newaxis]
    
    n = n/n_mag
    
    return n

'''
This function calculates the initial most probable intersection point using
RANSAC algorithm.
'''
def calc_intersection(p, n):
    np.random.seed(1)
    prod_sum = lambda a,b: np.sum(np.multiply(a,b), axis=1)
    counter = 0
    max_inlier = 0
    final_intersect = None
    pts = None
    
    while(counter<100):
        
        #calculate intersection point for a sample
        idx = np.random.randint(0, p.shape[0], size=2)
        ns = n[idx]
        ps = p[idx]
        nps = prod_sum(ps, ns)
        intersect = np.linalg.pinv(ns)@nps
        
        #calculate number of inliers
        dist = np.abs(prod_sum((p-intersect),n))

        num_inlier = len(dist[dist<=30.0])
        
        if(num_inlier>max_inlier):
            max_inlier = num_inlier
            final_intersect = intersect
            pts = ps
        
        counter +=1
        
    return final_intersect, num_inlier, pts, dist

'''
This function will by applying least squares method, find a fit to the inliers
of the primary intersection point.
'''
def calc_least_squares(p, n, idxs):
    prod_sum = lambda a,b: np.sum(np.multiply(a,b), axis=1)
    ns = n[idxs][:,-1,:]
    ps = p[idxs][:,-1,:]
    nps = prod_sum(ps, ns)
    intersect = np.linalg.pinv(ns)@nps
    
    return intersect
    
'''
This function is used for label counting in mean-shift
'''
def bincount_where(y, threshold):
    counts = np.bincount(y)
    return y[np.in1d(y,np.where(counts>=threshold)[0])]

'''
This function will load input files
'''
def input_file():
    
    #camera moves
    f1 = "data/000005_10.png"
    f2 = "data/000005_11.png"
    
    return f1, f2

if __name__ == "__main__":
    
    #import input file
    f1, f2 = input_file()
    
    #import these images
    im1 = cv2.imread(f1)
    im2 = cv2.imread(f2)
    
    #generate keypoints using ORB
    kp1, des1, kp2, des2 = apply_ORB(im1, im2)
    
    #matches
    matches, thresholded_matches = BF_matcher_apply(des1, des2, 1.0)
    
    #draw keypoint matches
    im_match = draw_match(im1, kp1, im2, kp2, matches)
    
    #generate keypoint locations
    p0 = keypoint_location_generate(matches, kp1, 's')
    
    #generate optical flows
    p_new, p_old = calculate_optical_flow(im1, im2, p0)
    
    #generate motion vector
    u = (p_new-p_old)
    
    #threshold the motion vector with top 50% value based on magnitude
    top_idxs = thresholding_motion_vector(u, .90)
    
    #optical flow plotted image
    im_flow = draw_tracks(np.copy(im1), np.copy(im2), p_new[top_idxs], 
                          p_old[top_idxs])
    
    show_image(im_flow, "output1.jpg")
    
    #normal vector to u
    n = calculate_normal_motion_vector(u)
    
    #calculate the best point for intersection
    foe, foe_inlier, pts, dist = calc_intersection(p_old[top_idxs], n[top_idxs])
    
    #Red color
    color = [0,0,255]

    #inlier and outlier calculation
    inlier_indices = np.argwhere(dist<=30.0)
    outlier_indices = np.argwhere(dist>30.0)
    
    #camera moving detection
    num_inliers = len(inlier_indices)
    camera_moving_threshold = 0.20* len(dist)
    
    camera_flag=0
    
    if  num_inliers>=camera_moving_threshold :
        
        camera_flag=1
        
        #final intersection calculation
        final_foe = calc_least_squares(p_old, n, top_idxs[inlier_indices])
        
        #plot the intersection point in the flow image
        im_flow = cv2.circle(im_flow, (final_foe[0], final_foe[1]), 10, color, 5)
        
        show_image(im_flow, "Output1.jpg")
    
    #--------------------------------------------------------------------------------

    #apply mean shift clustering on the outliers
    outlier_pts = p_old[top_idxs[outlier_indices]][:,-1,:]
    
    clustering = MeanShift(bandwidth=2).fit(outlier_pts)
    labels = clustering.labels_
    labels2 = bincount_where(labels, 3)
    unique_labels = np.unique(labels2)

    clusterd_indices = np.argwhere(np.isin(labels, np.unique(labels2)))
    outlier_clusters = outlier_pts[clusterd_indices][:,-1,:]
    
    for i, label in enumerate(labels2):
        np.random.seed(label)
        color = np.random.randint(0,255,(1,3))
        im = cv2.circle(np.copy(im1), (int(outlier_clusters[i,0]), int(outlier_clusters[i,1])), 1, color[0].tolist(), 1)
        
    for i, label in enumerate(unique_labels):
        color = [0,0,255]
        min_x, min_y = int(np.min(outlier_clusters[labels2==label][:,0]))-10,int(np.min(outlier_clusters[labels2==label][:,1]))-10
        max_x, max_y = int(np.max(outlier_clusters[labels2==label][:,0]))+10,int(np.max(outlier_clusters[labels2==label][:,1]))+10
        img = cv2.rectangle(im, (min_x, min_y), (max_x, max_y), color, 2)
        
    show_image(img, "Output2.jpg")
    
    #-------------------------------------------------------------------------------
    print("Number of keypoints from ORB after BF Matcher:", len(p0))
    print("Initial Intersection Vector:", foe)
    print("Initial Number of inliers:", foe_inlier)
    print("Initial Number of outliers:", len(outlier_indices))
    print("Camera Moving Threshold: %0.2f"%camera_moving_threshold)
    statement = "is moving" if camera_flag==1 else "is not moving"
    print("Camera "+ statement)
    print("Final Intersection Point:", final_foe)
    print("Number of clusters:", len(unique_labels))
    print("Clustering Threshold: ",5)
    
    
    
    
    
    
    
    
    
    
    
    