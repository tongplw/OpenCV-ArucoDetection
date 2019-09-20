#!/usr/bin/env python
"""This script shows an example of using the PyWavefront module."""
import ctypes
import sys
sys.path.append('..')
import pyglet
from pyglet.gl import *
from pywavefront import visualization
import pywavefront
import numpy as np
import cv2
import cv2.aruco as aruco
import glob

# resource path
import os
path = os.path.abspath('..')
res = path + '/res'

def calibrate():
    cap = cv2.VideoCapture(0)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # checkerboard of size (9 x 7) is used
    objp = np.zeros((7*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

    # arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # resizing for faster detection
        frame = cv2.resize(frame, (640, 480))
        # using a greyscale picture, also for faster detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,7), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(frame, (9,7), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'

        # Display the resulting frame
        cv2.imshow('Calibration',frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(10)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    
    #create a file to store data
    from lxml import etree
    from lxml.builder import E
    global fname
    with open(fname, "w") as f:
        f.write("{'ret':"+str(ret)+", 'mtx':"+str(list(mtx))+', "dist":'+str(list(dist))+'}')
        f.close()

    
#test wheater already calibrated or not
fname = res + "/calibration_parameters.txt"
try:
    f = open(fname, "r")
    f.read()
    f.close()
except:
    calibrate()


cap = cv2.VideoCapture(0)

#importing aruco dictionary
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

#calibration parameters
f = open(fname, "r")
ff = [i for i in f.readlines()]
f.close()
from numpy import array
parameters = eval(''.join(ff))
mtx = array(parameters['mtx'])
dist = array(parameters['dist'])

# Create absolute path from this module
file_abspath = os.path.join(os.path.dirname(__file__), res + '/box.obj')

red = green = blue = 0
meshes = pywavefront.Wavefront(file_abspath)
window = pyglet.window.Window()
lightfv = ctypes.c_float * 4

tvec = [[[0, 0, 0]]]
rvec = [[[0, 0, 0]]]

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250 )
markerLength = 0.25   # Here, our measurement unit is centimetre.
parameters = cv2.aruco.DetectorParameters_create()
parameters.adaptiveThreshConstant = 10

def a():
    global tvec, rvec, aruco_dict, parameters, mtx, dist
    
    while True:

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if np.all(ids != None):
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)

            for i in range(0, ids.size):
                aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)

            aruco.drawDetectedMarkers(frame, corners)
        else:
            tvec = [[[0, 0, 0]]]
            rvec = [[[0, 0, 0]]]
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

import threading    
x = threading.Thread(target=a)
x.start()

@window.event
def on_resize(width, height):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., float(width)/height, 1., 100.)
    glMatrixMode(GL_MODELVIEW)
    return True


@window.event
def on_draw():
    window.clear()
    glLoadIdentity()

    glLightfv(GL_LIGHT0, GL_POSITION, lightfv(-1.0, 1.0, 1.0, 0.0))
    glEnable(GL_LIGHT0)

    glTranslated(tvec[0][0][0]*50, -tvec[0][0][1]*50, -tvec[0][0][2]*30)
    glRotatef(red, 1, 0, 0)
    glRotatef(green, 0, 1, 0)
    glRotatef(blue, 0, 0, 1)
    
    glEnable(GL_LIGHTING)

    visualization.draw(meshes)


def update(dt):
    global red, green, blue
    red = rvec[0][0][0]*24
    green = -rvec[0][0][1]*20
    blue = -rvec[0][0][2]*24

pyglet.clock.schedule(update)
pyglet.app.run()
