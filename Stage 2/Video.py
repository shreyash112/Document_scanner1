import cv2
import numpy as np

cap = cv2.VideoCapture('http://192.168.0.100:4747/video')#capture the video from droid camera
#to store the video 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('new.mp4',fourcc, 20.0, (1200,880))

while(1):

    ret, frame = cap.read()# read the image from image frame by frame 
    frame = cv2.resize(frame,(1200,880))#resze the frame
    blurred_hsv = cv2.GaussianBlur(frame,(3,3),0)#apply gaussian blur to remove noise
    hsv = cv2.cvtColor(blurred_hsv, cv2.COLOR_BGR2HSV)#convert it into hsv image

    # define range of white color in HSV
    sensitivity = 150
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    #Find the the continous lines(contours) in from the thresholded image
    _,contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)#Calculate area of contours
        #print(area)
        #if only area is greater than the 10000 calculate the perimeter and draw the polygon across the contours
        if area > 10000:    
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.1*peri, True)
            #print(len(approx))
            
    frame = cv2.drawContours(frame,[approx],-1,(0,255,0),3)#draw the contours of thickness 3 on the frame
    out.write(frame)#write the video on the new.mp4
    cv2.imshow('frame',frame)#display the output video

    #to exit from the output screen
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break

cap.release()#release the camera
out.release()#release the video recording
cv2.destroyAllWindows()

