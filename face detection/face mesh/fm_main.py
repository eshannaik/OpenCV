import cv2
import mediapipe as mp 
import time 
import argparse

import fm as fm

ap = argparse.ArgumentParser()
ap.add_argument("-v","--video",required=True,help="Video Path")
args = vars(ap.parse_args())
video_path = args["video"]

cap = cv2.VideoCapture(video_path)

cTime=0
pTime=0

d = fm.face_mesh()

while 1:
	_,img = cap.read()

	img = d.find_face(img)
	lmlist = d.find_landmarks(img)
	# print(lmlist)
	img = cv2.resize(img,(1024,712))

	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime=cTime

	cv2.putText(img,f'FPS : {int(fps)}',(10,45),2,cv2.FONT_HERSHEY_PLAIN,(255,0,255),2)

	cv2.imshow("Video",img)

	if cv2.waitKey(30) & 0xff == 27 or cv2.getWindowProperty('Video', 0) == -1:
		break

cap.release()
cv2.destroyAllWindows()