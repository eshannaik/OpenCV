import cv2
import time

import object_detection as od

wCam,hCam = 1024,720

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(3,wCam)
cap.set(4,hCam)

cTime = 0
pTime = 0

d = od.object_detection()

while 1:
	_,img = cap.read()

	box,confidence,class_id = d.object_d(img,0.8,0.6,True)

	cTime=time.time()
	fps = 1/(cTime-pTime)
	pTime=cTime
	cv2.rectangle(img,(5,40),(100,5),(0,0,0),-1)
	cv2.putText(img,f'FPS:{int(fps)}',(10,35),cv2.FONT_HERSHEY_PLAIN,2,(123,12,153),3)

	cv2.imshow("Image",img)

	if cv2.waitKey(20) & 0xFF==27 or cv2.getWindowProperty("Image",0)==-1:
		break

cap.release()