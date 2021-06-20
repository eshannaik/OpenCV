import cv2
import autopy
import numpy as np
import time

import hd_module as hdm

frameR=100 # Frame reduction
smoothening = 2

wCam,hCam = 640,480
wScreen,hScreen = autopy.screen.size()

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(3,wCam)
cap.set(4,hCam)

d = hdm.HandDetector(maxHands=1)
cTime=0
pTime=0

xp,yp = 0,0
xc,yc = 0,0 

while 1:
	_,img = cap.read()

	img = d.find_hands(img)
	lmlist,bbox = d.position(img)

	if len(lmlist)!= 0:
		x1,y1 = lmlist[8][1:]
		x2,y2 = lmlist[12][1:]

		fingers = d.fingersUp()

		cv2.rectangle(img,(frameR+50,frameR-50),(wCam-frameR+50,hCam-frameR-50),(255,0,255),2)

		#Only index finger is up - moving mode
		if fingers[1]==1 and fingers[2]==0:
			#Converting coordinates
			x3 = np.interp(x1,(frameR+50,wCam-frameR+50),(0,wScreen))
			y3 = np.interp(y1,(frameR-50,hCam-frameR-50),(0,hScreen))

			#Smoothen the values
			xc = xp+(x3-xp)/smoothening
			yc = yp+(y3-yp)/smoothening

			autopy.mouse.move(wScreen - xc,yc)
			cv2.circle(img,(x1,y1),10,(255,0,255),cv2.FILLED)
			xp,yp = xc,yc

		# index and middle finger are up - clicking mode
		if fingers[1]==1 and fingers[2]==1:
			length,img,lineInfo =d.find_distance(img,8,12)
			# print(length)
			if length < 20:
				cv2.circle(img,(lineInfo[4],lineInfo[5]),10,(0,255,0),cv2.FILLED)
				autopy.mouse.click()

	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime
	cv2.rectangle(img,(5,465),(65,440),(0,0,0),-1)
	cv2.putText(img,f'FPS :{int(fps)}',(10,460),cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)

	cv2.imshow("Image",img)

	if cv2.waitKey(20) & 0xFF==27 or cv2.getWindowProperty("Image",0)==-1:
		break

cap.release()
cv2.destroyAllWindows()

