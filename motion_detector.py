import cv2
import numpy as np

cap = cv2.VideoCapture("vtest.avi") #use 0 for webcamera
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

#for .avi format
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')#specify the video codec, e.g:fourcc is a 4-byte code.
out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280,720))

#for .mp4 format
#fourcc = cv2.VideoWriter_fourcc(*"X264")
#out = cv2.VideoWriter("output.mp4", fourcc, 5.0, (1280,720))

ret, frame1 = cap.read() #reading frame1/image1
ret, frame2 = cap.read() #reading frame2

while cap.isOpened():
    frame_diff = cv2.absdiff(frame1, frame2) #for finding absolute difference between frame1 and frame2.
    diff_gray = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)#converting frame color(3 channels) to grayscale(1 channel) color(grayscale because to easily get contours)
    blur = cv2.GaussianBlur(diff_gray, (5,5), 0)#blurring gray frame to reduce noise(outer pixels) of frame so that it becomes smooth
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)#thresholding grayscale frame/image returns only 0 or 255.
    dilated = cv2.dilate(thresh, None, iterations=3)#Dilation means Adding pixels to the boundaries of objects in an image.
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#finding boundary/outline

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)#returns the coordinates and width and height of the bounding rectangle.

        if cv2.contourArea(contour) < 1550: #cv.contourArea(contour) returns the area bound by the contour.
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame1, "Status: {}".format('Motion detected'), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0,0,255), 2)


    image = cv2.resize(frame1, (1280,720))
    out.write(image)
    cv2.imshow("Result", frame1)#for capture each frame
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(60) == ord('z'):  #cv2.waitKey() pauses each frame in given time (ms) in case of image pass 0
        break

cv2.destroyAllWindows()
cap.release()
out.release()
