import cv2
import matplotlib.pyplot as plt
import numpy as np

#Resized image dimensions
m=600
n=800

image = cv2.imread('wood.jpg')#import the image from the folder
img = cv2.resize(image,(m,n))#resize the image to fit in the output box
copy_1 = img.copy()#create a copy of orignal image
blurred_hsv = cv2.GaussianBlur(img,(5,5),0)#Use gaussian blur to remove the noise
hsv = cv2.cvtColor(blurred_hsv,cv2.COLOR_RGB2HSV)#convert the image into HSV image to detect the white color
#plt.imshow(hsv)#optional to print the image

#120 for Black,purple 73 for blue,new,pink Different values for different background
sensitivity = 120#Depending upon the different lighting conditions the value may change
lower_white = np.array([0,0,255-sensitivity])#lower threshold for white in HSV
upper_white = np.array([255,sensitivity,255])#upper threshold for white in HSV

mask = cv2.inRange(hsv,lower_white,upper_white)#To select the values in the range 

_,contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#To find the continuous boundaries

for cnt in contours:
    area = cv2.contourArea(cnt)#To find contour area
    #if the area of the contour select the largest value of the area
    if area > 10000:
        peri = cv2.arcLength(cnt, True)#select the continous line 
        approx = cv2.approxPolyDP(cnt,0.05*peri,True)#Draw a polygon with the continous lines
        print(len(approx))#optional to print the length of the apprx 
        if len(approx) == 4:#If it == 4 means it is a rectangle 
            target = approx
            break

cv2.drawContours(img,[approx],-1,(0,255,0),2)#superimpose the contour we found and draw it on orignal image
cv2.imshow('Orignal',copy_1)
cv2.imshow('Page',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
