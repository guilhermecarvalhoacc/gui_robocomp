#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__      = "Matheus Dib, Fabio de Miranda"
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import sys
from math import degrees, asin

if len(sys.argv) > 1:
    arg = sys.argv[1]
    try:
        input_source=int(arg) # se for um device
    except:
        input_source=str(arg) # se for nome de arquivo
else:   
    input_source = 0

cap = cv2.VideoCapture(input_source)
# Parameters to use when opening the webcam.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

lower = 0
upper = 1
#parametros para identificar e desenhar a linha entre os circulos
m_raio = 0
sm_raio = 0    
posicao_centro1 = (0,0)  
posicao_centro2 = (0,0)
contador = 0
lista_distancias = []
# print("Press q to QUIT")
# Returns an image containing the borders of the image
# sigma is how far from the median we are setting the thresholds
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # A gaussian blur to get rid of the noise in the image
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #blur = gray
    # Detect the edges present in the image
    bordas = auto_canny(blur)
    circles = []
    # Obtains a version of the edges image where we can draw in color
    bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
    # HoughCircles - detects circles using the Hough Method. For an explanation of
    # param1 and param2 please see an explanation here http://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
    circles = None
    circles=cv2.HoughCircles(bordas,cv2.HOUGH_GRADIENT,2,40,param1=50,param2=100,minRadius=5,maxRadius=60)

    if circles is not None:  
        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
            if i[2] >= m_raio and i[2] > 15:
                sm_raio = m_raio
                posicao_centro2 = posicao_centro1
                m_raio = i[2]
                posicao_centro1 = (int(i[0]),int(i[1]))
            
        #draw the outer circle
        #cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
        cv2.circle(bordas_color, posicao_centro1, m_raio,(255,255,0),2)
        cv2.circle(bordas_color, posicao_centro2, sm_raio,(0,255,0),2)
        # draw the center of the circle
        cv2.circle(bordas_color,posicao_centro1,2,(0,0,255),3)
        #draw line between circles 
        cv2.line(bordas_color,posicao_centro1,posicao_centro2 ,(255,50,100),5)
        #euclidian distance
        deltax = (abs(posicao_centro2[0] - posicao_centro1[0]))
        deltay = (abs(posicao_centro2[1] - posicao_centro1[1]))
        distancia_centros = (deltax**2 + deltay**2)**(1/2)

        if distancia_centros >= 20:
            distancia_real = 14.5*606.896/distancia_centros
            sin_alfa = deltay/distancia_centros
            alfa  = degrees(asin(sin_alfa))



        #reseta 
    m_raio = 2
    sm_raio = 2


    # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    font = cv2.FONT_HERSHEY_SIMPLEX
    #More drawing functions @ http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

    # Display the resulting frame
    cv2.imshow('Detector de circulos',bordas_color)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

