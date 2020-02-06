# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 01:18:15 2019

@author: nafis
"""

import numpy as np
import math
import sys

def decision(x, y):
    #image size is 4000(y axis)x6000(x axis)
    statement = "inside" if (0<x<4000 and 0<y<6000) else "outside"
    return statement


if __name__ == '__main__':
    
    #read params file word by word as float
    args = sys.argv
    params_file = args[1]
    points_file = args[2]
    
    with open(params_file) as f:
        data = [np.float64(word) for line in f for word in line.split()]
        
    #data assignment
    rx, ry, rz = data[0], data[1], data[2]
    tx, ty, tz = data[3], data[4], data[5]
    f, d, ic, jc = data[6], data[7], data[8], data[9]
    
    #rotation matrix formation 3x3 (rx, ry, rz)
    #180deg = pi rad 
    deg2rad = (math.pi/180)
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(deg2rad*rx), (-1)*math.sin(deg2rad*rx)],
                   [0, math.sin(deg2rad*rx),math.cos(deg2rad*rx)]])
    Ry = np.array([[math.cos(deg2rad*ry), 0, math.sin(deg2rad*ry)],
                   [0, 1, 0],
                   [(-1)*math.sin(deg2rad*ry), 0, math.cos(deg2rad*ry)]])
    Rz = np.array([[math.cos(deg2rad*rz), (-1)*math.sin(deg2rad*rz), 0],
                   [math.sin(deg2rad*rz), math.cos(deg2rad*rz), 0],
                   [0, 0, 1]])
    R = Rx@Ry@Rz
    
    #translation vector formation (tx, ty, tz)
    t = np.array([tx, ty, tz]).reshape(3,1) 
    
    #k matrix formation 3x4 (f, d, vc = ic, uc = jc)
    Sx = (f/d) * 1000 #micron to mm conversion
    Sy = (f/d) * 1000 #micron to mm conversion
    Uc = jc #column value
    Vc = ic #row value
    
    K = np.array([[Sx, 0, Uc],
                 [0, Sy, Vc],
                 [0, 0, 1]])
    
    #Camera Matrix build (K, R, T)
    M = K@np.hstack((R.T, -R.T@t))
        
    #-------------------------------------------------------------------------#
    
    #input image files as vectors of 3 elements
    points = np.loadtxt(points_file)
    num_cols = points.shape[0]
    points  = np.hstack((points, np.ones(num_cols).reshape(num_cols,1)))
    projection = M@points.T
    v = projection[1]/projection[2] #[V'/W'], X Axis
    u = projection[0]/projection[2] #[U'/W'], Y axis
    
    
    #print
    print("Matrix M:")
    for i in range(M.shape[0]): #traverse rows for formatted printing
        print("{:.2f},".format(M[i,0]), "{:.2f},".format(M[i,1]), "{:.2f},".format(M[i,2]), "{:.2f}".format(M[i,3]))
    
    print("Projections:")
    for i in range(points.shape[0]):
        print("{:.0f}:".format(i), "{:.1f}".format(points[i,0]), 
              "{:.1f}".format(points[i,1]), "{:.1f}".format(points[i,2]),
              "=>", "{:.1f}".format(v[i]), "{:.1f}".format(u[i]), 
              decision(v[i], u[i]))
    
    print("visible:", *[x[0] for x in np.argwhere(projection[2]>0)])
    print("hidden:", *[x[0] for x in np.argwhere(projection[2]<0)])
    
    #-------------------------------------------------------------------------#
    
    
        
    