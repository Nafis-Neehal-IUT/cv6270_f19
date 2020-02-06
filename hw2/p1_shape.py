# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:14:21 2019

@author: nafis
"""

import numpy as np
import sys
import math
import matplotlib.pyplot as plt

def calculate_min_max_value(dataset):
    return np.min(dataset, axis=0), np.max(dataset, axis=0)

def calculate_min_max_axis(dataset):
    #center data
    centered_data = dataset - np.mean(dataset, axis=0)
    
    #calculate covariance matrix
    covariance_matrix = np.cov(dataset.T, bias=True)
    
    #eigenvalues and eigenvectors of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    #minimum and maximum axis 
    if eigenvalues[0]>eigenvalues[1]:
        minimum_axis = eigenvectors[:,1]
        maximum_axis = eigenvectors[:,0]
        s_min = np.sqrt(eigenvalues[1])
        s_max = np.sqrt(eigenvalues[0])
    else:
        minimum_axis = eigenvectors[:,0]
        maximum_axis = eigenvectors[:,1]
        s_min = np.sqrt(eigenvalues[0])
        s_max = np.sqrt(eigenvalues[1])
        
    return minimum_axis, maximum_axis, s_min, s_max

def calculate_point_normal_form(dataset):
    #closest point form best fitting line
    centered_data = dataset - np.mean(dataset, axis=0)
    u_i = centered_data[:,0]
    v_i = centered_data[:,1]
    mu = np.mean(dataset, axis=0)
    
    numerator = (2*np.sum((u_i)*(v_i)))
    denominator = np.sum(u_i**2-v_i**2)
    
    #calculate options for theta and rho simultaneously
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
        
    return rho, theta, distance_matrix

def plot_stuff(dataset):
    #plot graph and output image
    x = dataset[:,0]
    y = dataset[:,1]
    y_new = (-a/b)*x - (c/b)
    plt.axis('equal')
    plt.scatter(dataset[:,0], dataset[:,1])
    plt.plot(scalex=True, scaley=True)
    plt.plot(x, y_new, color='red')
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.title("Problem 1 Output")
    plt.savefig(output_image)
    plt.close()


if __name__== '__main__':
    
    #file and data import
    args = sys.argv
    filename = sys.argv[1]
    tau = float(sys.argv[2])
    output_image = sys.argv[3]
    dataset = np.loadtxt(filename, delimiter=' ')
    
    '''PART (a)'''
    #minimum and maximum value find
    minimum_value, maximum_value = calculate_min_max_value(dataset)
    print("min: ({0:.3f}".format(minimum_value[0])+",{0:.3f})".format(minimum_value[1]))
    print("max: ({0:.3f}".format(maximum_value[0])+",{0:.3f})".format(maximum_value[1]))
    
    
    '''PART (b)'''
    com = np.mean(dataset, axis=0)
    print("com: ({0:.3f}".format(com[0])+",{0:.3f})".format(com[1]))
    
    
    '''PART (c) & (d)'''
    #Find out PCA axis 1,2 -> 1 being most variance, 2 -> being least variance and PRINT
    minimum_axis, maximum_axis, s_min, s_max = calculate_min_max_axis(dataset)
    print("min axis: ({0:.3f}".format(minimum_axis[0])+",{0:.3f})".format(minimum_axis[1])+", sd {:.3f}".format(s_min))
    print("max axis: ({0:.3f}".format(maximum_axis[0])+",{0:.3f})".format(maximum_axis[1])+", sd {:.3f}".format(s_max))

    
    '''PART (e)'''
    rho, theta, distance_matrix = calculate_point_normal_form(dataset)
    print("closest point: rho {:.3f},".format(rho)+" theta {:.3f}".format(theta))
    
    '''PART (f)'''
    #implicit form best fitting line
    #calculate a, b, c
    a = math.cos(theta)
    b = math.sin(theta)
    c = -rho
    print("implicit: a {:.3f},".format(a) + " b {:.3f},".format(b) + " c {:.3f}".format(c))
    
    '''PART (g)'''
    #shape decision
    #compare s_min & tau* s_max
    if s_min< tau * s_max:
        print("best as line")
    else:
        print("best as ellipse")
        
    '''PART (h)'''
    plot_stuff(dataset)
    
    
    
    
    

    




    
    
    
    
    

    
