import cv2
import argparse
import time

import pe_module as p

ap = argparse.ArgumentParser()
ap.add_argument('-v','--video',required="True",help="Video Path")
args=vars(ap.parse_args())
video_path = args['video']

cap = cv2.VideoCapture(video_path)

cTime = 0
pTime = 0

pe = p.pose_estimation()

while 1:
	_,img = cap.read()
	new_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

	img = pe.estimation(img)
	lmlist = pe.points(img)
	# print(lmlist)
	img = cv2.resize(img,(1024,712))

	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime
	cv2.rectangle(img,(5,40),(220,5),(0,0,0),-1)
	cv2.putText(img,f'FPS : {int(fps)}',(10,35),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(255,255,0),2)

	cv2.imshow('Image',img)	

	if cv2.waitKey(20) & 0xFF == 27 or cv2.getWindowProperty("Image",0) == -1:
		break

cap.release()
cv2.destroyAllWindows()