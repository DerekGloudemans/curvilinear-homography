#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 09:59:49 2022

@author: derek
"""

import cv2 
import scipy
import numpy as np
import scipy
from scipy import interpolate
import time

global x_pos,y_pos
global clicked
clicked = False

global moved
moved = True
x_pos = 0
y_pos = 0


# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    global x_pos,y_pos
    global clicked
    global moved
    
    # checking for left mouse clicks
    if event == cv2.EVENT_MOUSEMOVE:
 
        # displaying the coordinates
        # on the Shell
        x_pos = x
        y_pos = y
        moved = True
 
    elif event == cv2.EVENT_LBUTTONDOWN:
        clicked = True
        moved = True
        
# driver function
if __name__=="__main__":
 


    jitter = 3
    x = np.arange(0,1000,10) + (np.random.rand(100)*2*jitter - jitter)
    y = np.arange(0,1000,10)*0+500 + (np.random.rand(100)*2* jitter - jitter)
    z = np.arange(0,1000,10)
    
    x_jit = 0
    y_jit = 0
    for i in range(len(x)):
        x_jit += (np.random.rand()*2*jitter)
        y_jit += (np.random.rand()*2*jitter - jitter)
        x[i] += x_jit
        y[i] += y_jit
        
    x = x.tolist()
    y = y.tolist()
    
    x = [0,100,200,300,400]
    y = [0,100,200,300,400]
    # x_prime_spline = scipy.interpolate.UnivariateSpline(z,x)
    # y_prime_spline = scipy.interpolate.UnivariateSpline(z,y)
    
    # x_prime = x_prime_spline(z)
    # y_prime = y_prime_spline(z)
    
    # x_prime_spline = scipy.interpolate.SmoothBivariateSpline(x,y,z)
    
    tck, u = interpolate.splprep([x, y], s=0, per=False)
    
    im = (np.ones([1080,1920,3])*255).astype(np.uint8)
    im_orig = im
    
    cv2.namedWindow("Frame")
    cv2.setMouseCallback('Frame', click_event)

    f_idx = 0
    while True:
        f_idx += 1
        
        if clicked:
            x.append(x_pos)
            y.append(y_pos)
            clicked = False
            tck, u = interpolate.splprep([x, y], s=0, per=False)
        
        im = im_orig.copy()
                
        start = time.time()
        
        # evaluate the spline fits for 1000 evenly spaced distance values
        x_prime, y_prime = interpolate.splev(np.linspace(0, 1, 1000), tck)

        # find closest point on spline
        xpos_rep = np.ones(len(x_prime)) * x_pos
        ypos_rep = np.ones(len(x_prime)) * y_pos
        
        dist = ((x_prime - xpos_rep)**2 + (y_prime - ypos_rep)**2)**0.5
        min_dist,min_idx= np.min(dist),np.argmin(dist)
        min_point = (int(x_prime[min_idx]),int(y_prime[min_idx]))
        
        int_dist = ((x_prime[1:] - x_prime[:-1])**2 + (y_prime[1:] - y_prime[:-1])**2)**0.5 # by convention dist[0] will be 0, so dist[i] = sum(int_dist[0:i])

    
        x_along_spline = int(np.sum(int_dist[0:min_idx]))
        y_along_spline = int(min_dist)
        
        end = time.time()
        
        # plot the included spline
        for n in range(min_idx):
            point = (int(x_prime[n]),int(y_prime[n]))
            point2 = (int(x_prime[n+1]),int(y_prime[n+1]))
        
            cv2.line(im,point,point2,(250,150,200),2)
            
        # plot the y_coord tie
        cv2.line(im,min_point,(x_pos,y_pos),(250,150,200),2)


        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, 'Cart. Coords: ({},{})'.format(x_pos,y_pos), (10,950), font, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(im, 'Curvil. Coords: ({},{})'.format(x_along_spline,y_along_spline), (10,900), font, 1, (250,150,200), 2, cv2.LINE_AA)
        cv2.putText(im, 'Spline Comp Time: {:.2f}ms'.format(end*1000-start*1000),(10,850), font, 1, (0,0,0), 2, cv2.LINE_AA)

        # plot the spline
        for n in range(len(x_prime)-1):
            point = (int(x_prime[n]),int(y_prime[n]))
            point2 = (int(x_prime[n+1]),int(y_prime[n+1]))
        
            cv2.line(im,point,point2,(0,0,0),1)
        
        # draw crosshairs
        cv2.line(im,(x_pos,0),(x_pos,im.shape[0]),(0,0,255),1)
        cv2.line(im,(0,y_pos),(im.shape[1],y_pos),(0,0,255),1)

        cv2.imshow("Frame",im)

        if moved:
            moved = False
            
            cv2.imwrite("/home/derek/Desktop/temp_frames/{}.png".format(str(f_idx).zfill(5)),im)
            # setting mouse handler for the image
            # and calling the click_event() function
         
        # wait for a key to be pressed to exit
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
     
        # close the window
    cv2.destroyAllWindows()