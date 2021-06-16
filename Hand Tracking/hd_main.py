import cv2
import mediapipe as mp
import time

import hd_module as htm

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

pTime=0
cTime=0

detector = htm.HandDetector()

while True:
	_,img = cap.read()

	img = detector.find_hands(img)
	lmlist = detector.position(img)

	#print location of landmark at tip of middle finger
	if len(lmlist) != 0:
		print(lmlist[16])

	#fps
	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime
	cv2.putText(img,str(int(fps)),(0,35),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(255,0,255),2)

	cv2.imshow("Image",img)
	if cv2.waitKey(20) & 0xFF == 27 or cv2.getWindowProperty("Image",0) == -1:
		break

cap.release()
cv2.destroyAllWindows()