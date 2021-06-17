import cv2
import argparse
import time
import numpy as np

import pe_module as p

ap = argparse.ArgumentParser()
ap.add_argument('-v','--video',required="True",help="Video Path")
ap.add_argument('-p1',help="Landmark 3")
ap.add_argument('-p2',help="Landmark 3")
ap.add_argument('-p3',help="Landmark 3")
args=vars(ap.parse_args())
video_path = args['video']
p1=args['p1']
p2=args['p2']
p3=args['p3']

# For image comment below line
cap = cv2.VideoCapture(video_path)
# Webcam
# cap = cv2.VideoCapture(0,cv2.CAP_DSHOWS)

cTime = 0
pTime = 0

pe = p.pose_estimation()
count=-1
direction=0

while 1:
	_,img = cap.read()
	# for image uncomment below line and comment above line
	# img = cv2.imread(video_path)

	img = pe.estimation(img,False)
	lmlist = pe.points(img,False)
	# print(lmlist)
	
	if len(lmlist) != 0:
		# Right Hand
		# angel = pe.findAngle(img,12,14,16)
		# Left Hand
		angel = pe.findAngle(img,11,13,15)
		# Left Leg
		# angel = pe.findAngle(img,23,25,27)
		# Right Leg
		# angel = pe.findAngle(img,24,26,28)

		per = np.interp(angel,(210,320),(0,100))
		# print(angel,' = ', per)

		bar = np.interp(angel,(210,320),(650,100)) 

		#checking for curl
		color = (0,255,255)
		if per == 100:
			color = (255,255,0)
			if direction==0:
				count += 0.5
				direction = 1

		if per==0:
			color = (255,255,0)
			if direction==1:
				count += 0.5
				direction = 0
	# print(count)

	# Drawing the rep count
	cv2.rectangle(img,(130,920),(400,850),(0,0,0),-1)
	cv2.putText(img,f'Curls : {int(count)}',(147,900),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(255,0,255),2)

	#Drawing the bar
	cv2.rectangle(img,(1800,100),(1850,650),color,3)
	cv2.rectangle(img,(1800,int(bar)),(1850,650),color,cv2.FILLED)
	cv2.putText(img,f'{int(per)}%',(1800,75),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,color,2)

	img = cv2.resize(img,(1024,712))

	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime
	cv2.rectangle(img,(5,40),(155,5),(0,0,0),-1)
	cv2.putText(img,f'FPS : {int(fps)}',(10,35),cv2.FONT_HERSHEY_PLAIN,2,(255,255,0),2)

	cv2.imshow('Image',img)	

	if cv2.waitKey(20) & 0xFF == 27 or cv2.getWindowProperty("Image",0) == -1:
		break

# cap.release()
cv2.destroyAllWindows()