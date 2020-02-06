# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:48:38 2019

@author: nafis
"""

import numpy as np
import sys
import math

def calculate_line_parameters(sample_matrix, dataset):
    #line parameter calculation
    mu = np.mean(sample_matrix, axis=0) #mean of the sample 2x2 matrix
    centered_data =  sample_matrix - mu #make zero mean
    u_i = centered_data[:,0] #first column of zero mean matrix
    v_i = centered_data[:,1] #second column of zero mean matrix
    numerator = (2*np.sum((u_i)*(v_i))) #numerator
    denominator = np.sum(u_i**2-v_i**2) #denominator
    theta = (1/2) * math.atan(numerator/denominator)  
    theta_2 = theta + math.pi/2
    rho = mu[0]*math.cos(theta) + mu[1]*math.sin(theta)
    rho_2 = mu[0]*math.cos(theta_2) + mu[1]*math.sin(theta_2)
    
    #choose theta from objective function (lower value)
    distance_matrix = ((dataset@np.array([math.cos(theta), math.sin(theta)]).T) - rho)**2
    distance_matrix_2 = ((dataset@np.array([math.cos(theta_2), math.sin(theta_2)]).T) - rho_2)**2
    
    obj_1 = np.sum(distance_matrix)
    obj_2 = np.sum(distance_matrix_2)
    
    if obj_1 > obj_2:       #because we are minimizing objective function
        theta = theta_2
        rho = rho_2
        distance_matrix = distance_matrix_2
    
    return theta, rho, distance_matrix

if __name__=='__main__':
    
    #RANSAC algorithm
    args = sys.argv
    filename = args[1]
    samples = int(args[2])
    tau = float(args[3])
    if(args[4]):
        seed = int(args[4])
        np.random.seed(seed)

    
    dataset = np.loadtxt(filename, delimiter=' ')
    N = dataset.shape[0] #number of samples in the dataset
    k_max = 0
    theta_hat = 0
    rho_hat = 0
    
    for i in range(samples):
        index = np.random.randint(0, N, 2)
        if index[0]==index[1]:
            continue
        else:
            #enter RANSAC
            sample_1 = dataset[index[0]] #first sample randomly picked
            sample_2 = dataset[index[1]] #second sample randomly picked
            sample_matrix = np.vstack([sample_1, sample_2]) #sample matrix vertically stacked 2x2
            
            #calculate line parameters
            theta, rho, distance_matrix = calculate_line_parameters(sample_matrix, dataset)
            
            
            #distance compute and K calculate
            k = (distance_matrix<(tau**2)).sum()

            if k>k_max:
                k_max=k
                theta_hat = theta
                rho_hat = rho
                a = math.cos(theta)
                b = math.sin(theta)
                c = -rho
                
                inlier_avg = sum([x for x in distance_matrix if x<tau**2])/k_max
                outlier_avg = sum([x for x in distance_matrix if x>=tau**2])/(distance_matrix.shape[0]-k_max)
                
                np.set_printoptions(precision=3)
                print("Sample {:.0f}".format(i)+":")
                print("indices ({:.0f}".format(index[0])+",{:.0f})".format(index[1]))
                print("line ({:.3f}".format(a)+",{:.3f}".format(b)+",{:.3f})".format(c))
                print("inliers {:.0f}".format(k_max))
                print()
    
    
    print("avg inlier dist {:.3f}".format(inlier_avg))
    print("avg outlier dist {:.3f}".format(outlier_avg))
            
            
            
    