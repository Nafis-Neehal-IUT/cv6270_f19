# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:05:21 2019

@author: nafis
"""

import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

'''
This function will load all images and their filenames from a folder and save
them in seperate lists.
'''
def load_images_from_folder(folder):
    images = []
    file = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            file.append(filename)
    return images, file

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
def BF_matcher_apply(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x:x.distance)
    return matches

'''
This function is to show figures using Matplotlib.
'''
def show_image(im):
    plt.figure()
    plt.imshow(im[...,::-1].astype(np.uint8)) 
    plt.show() 

'''
This function will generate pyramidal weight for a rectangular shaped image.
Center value/values will have weight 1 and all border values will have weight
0.
'''
def pyramid(n1, n2):
    r1 = np.arange(n1)
    d1 = np.minimum(r1,r1[::-1])
    r2 = np.arange(n2)
    d2 = np.minimum(r2,r2[::-1])
    mat = np.minimum.outer(d1,d2)
    mat = mat / np.max(mat)
    return mat[:,:,np.newaxis]

'''
This function will generate keypoint locations of matches generated
by BF Matcher.
'''
def keypoint_location_generate(matches, kp, sd):
    if sd=='s':
        kp_locs = np.float32([ kp[m.queryIdx].pt for m in matches])
    elif sd=='d':
        kp_locs = np.float32([ kp[m.trainIdx].pt for m in matches])
    return kp_locs.reshape(-1,1,2)

'''
This function will apply Homography Matrix on the borders of the Source image 
and return the new mapped coordinates and original source border values.
'''
def map_homography(src_h, src_w, H):
    src_border= np.array([[0, 0],[0, src_h-1], [src_w-1,0], [src_w-1, src_h-1]]).T
    src_border_hom = np.vstack((src_border, np.ones((1,4))))
    src_mapped_hom = H@src_border_hom
    src_mapped = src_mapped_hom / src_mapped_hom[2]
    src_mapped = src_mapped[:2,:]
    return src_border, src_mapped

'''
This function will build a translation matrix that will basically take the top-left
of the mapped source image to the (0,0) coordinate of the composite image.
'''
def build_translation_matrix(point):
    tx = -point[0]
    ty = -point[1]
    T = np.array([[1,0,tx], [0,1,ty], [0,0,1]])
    return T        

'''
This function will calculate the dimension of a rectangle if given the top-left
and bottom-right corner coordinates of that rectangle.
'''
def calculate_boundary(p1, p2):
    return (np.abs(p1[0]-p2[0])+1),(np.abs(p1[1]-p2[1])+1)

'''
This function will draw matched keypoints on both images and connect them with
line. It will return the image with keypoints and lines drawn over it.
'''
def draw_match(m, im1, kp1, im2, kp2, matches):
    m_r = m.ravel().tolist()
    
    draw_params = dict(matchColor = (0,255,0), 
                       singlePointColor = None,
                       matchesMask = m_r, 
                       flags = 2)

    drawnMatch = cv2.drawMatches(im1,kp1,im2,kp2,matches,None,**draw_params)
    return drawnMatch

'''
This is basically a optional helper function to sub-plot 5 result images - 
1. Keypoint Map after F
2. Keypoint Map after H
3. Warped Source
4. Warped Destination
5. Final Image
'''
def plot_image_grid(im1, im2, im3, im4, im5):
  
    fig = plt.figure()
   
    gridspec.GridSpec(3,3)
    
    plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=1)
    plt.imshow(im2[...,::-1])
    plt.title('Keypoint Map after H')
  
    plt.subplot2grid((3,3), (1,0), colspan=2, rowspan=2)
    plt.imshow(im5[...,::-1])
    plt.title('Final Result')
    
    plt.subplot2grid((3,3), (0,2))
    plt.imshow(im1[...,::-1])
    plt.title('Keypoint Map after F')
   
    plt.subplot2grid((3,3), (1,2))
    plt.imshow(im3[...,::-1])
    plt.title('Warped Source')
    
    plt.subplot2grid((3,3), (2,2))
    plt.imshow(im4[...,::-1])
    plt.title('Warped Destination')
    
    fig.tight_layout()
    fig.set_size_inches(w=11,h=7)
    
    
    #fig.savefig(fig_name)
    
    plt.show()

def check_inliner_match(m1, m2, epsilon):
    num_m1 = m1[m1==1].sum()
    num_m2 = m2[m2==1].sum()
    if (num_m2-epsilon) <= num_m1 <= (num_m2+epsilon):
        return True
    return False

def file_output(f1, f2, img, out_dir):
    #file name processing and writing
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    file_extension = f1[-4:]
    file_name_1 = f1[:-4]
    file_name_2 = f2[:-4]
    file_name = (file_name_1 + "_" + file_name_2) if file_name_1<file_name_2 else (file_name_2+"_"+file_name_1)
    file_name = out_dir + "\\" + file_name + file_extension
    print(" Image wrote:",file_name)
    cv2.imwrite(file_name, img)
    
if __name__ == "__main__":
    
    #load images from folder
    #current_folder = 'images/tree-mrc'
    args = sys.argv
    
    in_dir = args[1]
    out_dir = args[2]
    
    images, file = load_images_from_folder(in_dir)
    
    #all image pairs
    image_pairs = [(images[i],images[j]) for i in range(len(images)) 
                    for j in range(i+1, len(images))]
    file = [(file[i],file[j]) for i in range(len(file)) 
                    for j in range(i+1, len(file))]
    
    #Homography list
    H_list = []
    
    #for all image pairs   
    for i in range (len(image_pairs)):
        
        #Image source print
        print("Comparing",file[i][0],"and",file[i][1])
        
        #image alias
        src_img = image_pairs[i][0].astype(np.float32)
        dest_img = image_pairs[i][1].astype(np.float32)
        
        #Find keypoint matches
        kp1, des1, kp2, des2 = apply_ORB(src_img.astype(np.uint8), dest_img.astype(np.uint8))
        matches = BF_matcher_apply(des1, des2)
        
        print("Number of matches from BF Matcher:", len(matches))
        print("Number of keypoints:",len(kp1))
        
        #Fundamental matrix generate and image show
        '''
        Decision: At least 25% matches of total number of keypoints (500)
        Which means, At least 125 matches as default #KP = 500
        '''
        if(len(matches)>0.25*len(kp1)):
            print("Decision 1: At least 25% of total keypoints have been matched!")
            kp1_locs = keypoint_location_generate(matches, kp1, 's')
            kp2_locs = keypoint_location_generate(matches, kp2, 'd')
            
            #Fundamental Matrix generate and image show
            F, mF = cv2.findFundamentalMat(kp1_locs, kp2_locs, cv2.FM_RANSAC, 2, 0.99)
            FMatch = draw_match(mF, src_img.astype(np.uint8), kp1, 
                                dest_img.astype(np.uint8), kp2, matches)
          
            print(" Number of Inliers after F:",mF[mF==1].sum())
            '''
            Decision: 
                Take - Weighted Average of number of matches, match percentage 
                       and number of errors in match
                Add - Add the previous value with (weight*inlier percentage). 
                      This will be the confidence value.
                Threshold - If confidence > threshold, then they have same scene.
            '''
            num_matches = len(matches)
            num_kp = len(kp1)
            match_percentage = (num_matches / num_kp) * 100
            num_inliers = len(mF[mF==1])
            inlier_percentage = (num_inliers / num_matches)*100
            num_match_errors = num_matches - num_inliers
            decision_array = np.array([num_matches, match_percentage, num_match_errors, 
                                       inlier_percentage], dtype=np.float32)
            decision_array = decision_array / np.linalg.norm(decision_array)
            weights_same_scene = np.array([0.05, 0.3, 0.05, 0.6])
            confidence = ((decision_array[:3]@weights_same_scene[:3])/
                          (weights_same_scene[:3].sum()))+(weights_same_scene[3]*decision_array[3])
            
            #print decision parameters
            print(" Match Percentage (of total keypoints) :{:.2f}%".format(match_percentage))
            print(" Inlier Percentage :{:.2f}%:".format(inlier_percentage))
            print(" Number of errors in matches:", num_match_errors)
            print(" Confidence Value: %.2f"%confidence)
            
            confidence_threshold = 0.35
            if(confidence>=confidence_threshold):
                
                print("Decision 2: Confidence is higher than Threshold", confidence_threshold)
                
                #Homography matrix generate and image show
                H, mH = cv2.findHomography(kp1_locs, kp2_locs, cv2.RANSAC, 5.0)
                HMatch = draw_match(mH, src_img.astype(np.uint8), kp1, 
                                    dest_img.astype(np.uint8), kp2, matches)
                
                print(" Number of inliers after H:",mH[mH==1].sum())
                
                #ONLY used for Multi_image_mosaic(Img 1->2, Img 2->3)
                if(i==0 or i==3):
                    H_list.append(H)
                
                epsilon = 40
                if check_inliner_match(mF, mH, epsilon):
                    print(" Matching Threshold Epsilon:",epsilon)
                    print("Decision 3: H and F produce almost same number of matches, so building MOSAIC!")
                    
                    #Map source image to destination
                    src_border, src_mapped = map_homography(src_img.shape[0], src_img.shape[1], H)
                    
                    #find border of the composite
                    candidates = np.concatenate((src_border, src_mapped), axis = 1)         
                    top_left = (int(np.min(candidates[0])), int(np.min(candidates[1])))
                    bottom_right = (int(np.max(candidates[0])), int(np.max(candidates[1])))
                    
                    #translation matrix build
                    T = build_translation_matrix(top_left)
                    
                    #Matrix with translation for both images to map to composite
                    M_src = T@H
                    M_dest = T.astype(np.float64)
                    
                    #apply Warp perspective
                    boundSize = calculate_boundary(top_left, bottom_right)
                    Wrp_src = cv2.warpPerspective(src_img, M_src, boundSize, flags = cv2.INTER_NEAREST)
                    Wrp_dest = cv2.warpPerspective(dest_img, M_dest, boundSize, flags = cv2.INTER_NEAREST)
                    
                    #blend -- Weighted Average
                    wght_src = pyramid(src_img.shape[0], src_img.shape[1])
                    wght_dest = pyramid(dest_img.shape[0], dest_img.shape[1])
                    wght_src = cv2.warpPerspective(wght_src, M_src, boundSize, flags = cv2.INTER_NEAREST)[:,:,np.newaxis]
                    wght_dest = cv2.warpPerspective(wght_dest, M_dest, boundSize, flags = cv2.INTER_NEAREST)[:,:,np.newaxis]
                   
                    W =  wght_src*Wrp_src +  wght_dest*Wrp_dest
                    denom =  wght_src +  wght_dest
                    denom[denom==0] = 1
                    W = W/denom
                    
                    
                    plot_image_grid(FMatch.astype(np.uint8), HMatch.astype(np.uint8), 
                                    Wrp_src.astype(np.uint8), Wrp_dest.astype(np.uint8), 
                                    W.astype(np.uint8))
                    
                    
                    file_output(file[i][0], file[i][1], W, out_dir)
                else:
                    print(" Mosaic cannot be formed between:",file[i][0],"and",file[i][1])
            else:
                print(" Image pair does not pass the confidence threshold.")
        else:
            print(" Image pair does not have suffient amount of matches")
        
        print("____________________________________________")
    
    
    '''
    ------------------------------------------------------------------------------------
    Multi Image Mosaic for first 3 images in 'tree-mrc' folder
    This portion is commented out INTENTIONALLY as this is very specific to the input.
    Output attached in the report. The code segment works fully. It can be uncommented to
    test on the "tree-mrc" folder.
    ------------------------------------------------------------------------------------
    left = images[0]
    right = images[2]
    anchor = images[1]
    
    H12 = H_list[0]
    H32 = np.linalg.inv(H_list[1])
    
    s1_b, s1_m = map_homography(left.shape[0], left.shape[1], H12)
    s2_b, s2_m = map_homography(right.shape[0], right.shape[1], H32)
        
    candidates = np.concatenate((s1_b, s1_m, s2_m), axis = 1)         
    top_leftm = (int(np.min(candidates[0])), int(np.min(candidates[1])))
    bottom_rightm = (int(np.max(candidates[0])), int(np.max(candidates[1])))
    
    Tm = build_translation_matrix(top_left)
    
    M_s1 = Tm@H12
    M_s2 = Tm@H32
    M_d = Tm.astype(np.float64)

    
    boundSizem = calculate_boundary(top_leftm, bottom_rightm)
    Wrp_s1 = cv2.warpPerspective(left, M_s1, boundSizem, flags = cv2.INTER_NEAREST)
    Wrp_s2 = cv2.warpPerspective(right, M_s2, boundSizem, flags = cv2.INTER_NEAREST)
    Wrp_dm = cv2.warpPerspective(anchor, M_d, boundSizem, flags = cv2.INTER_NEAREST)
    
    #blend -- Weighted Average
    wght_s1 = pyramid(left.shape[0], left.shape[1])
    wght_s2 = pyramid(right.shape[0], right.shape[1])
    wght_dm = pyramid(anchor.shape[0], anchor.shape[1])
    wght_s1 = cv2.warpPerspective(wght_s1, M_s1, boundSizem, flags = cv2.INTER_NEAREST)[:,:,np.newaxis]
    wght_s2 = cv2.warpPerspective(wght_s2, M_s2, boundSizem, flags = cv2.INTER_NEAREST)[:,:,np.newaxis]
    wght_dm = cv2.warpPerspective(wght_dm, M_d, boundSizem, flags = cv2.INTER_NEAREST)[:,:,np.newaxis]
   
    Wm =  wght_s1*Wrp_s1 +  wght_s2*Wrp_s2 + wght_dm*Wrp_dm
    denomm =  wght_s1 +  wght_s2 + wght_dm
    denomm[denomm==0] = 1
    Wm = Wm/denomm
    
    show_image(Wm)
    '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    