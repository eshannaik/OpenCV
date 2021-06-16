import cv2
import mediapipe
import time

import hd_module as hdm

cTime=0
pTime=0

d = hdm.HandDetector(detectionCon=0.75)

cap = cv2.VideoCapture(0,cv2.D_SHOW)

while 1:
	_,img = cap.read()

	img = d.find_hands(img)
	lmlist = d.position(img,draw=False)

	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime
	cv2.putText(img,f'FPS :{int(fps)}',(10,25),cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)

	cv2.imshow("WebCame",img)

	if cv2.waitKey(20) & 0xFF==27 or cv2.getWindowsProperty("WebCame",0) == -1:
		break

cap.release()
cv2.destroyAllWindows()