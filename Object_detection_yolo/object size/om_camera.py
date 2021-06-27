import cv2
import time
import numpy as np

import odm as o

wCam,hCam=1024,720

#Load Acura
parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

#Load Object Detector
d = o.HomogeneousDetector() 

#Read Image
# img = cv2.imread("./phone_aruco_marker.jpg")

#Camera
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,wCam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,hCam)

cTime=0
pTime=0

while True:
	_,img = cap.read()

	#Get aruco marker
	corner,_,_ = cv2.aruco.detectMarkers(img,aruco_dict,parameters=parameters)
	# print(corner)

	if corner:
	int_corner=np.int0(corner)
	cv2.polylines(img,int_corner,True,(255,255,0),2)

		#Aruco Parameter
	aru_perimeter = cv2.arcLength(corner[0],True)
	# print(int(aru_perimeter))

		#Pixel to cm
	pixel_cm_ratio = aru_perimeter/20

		#detect objects
	c = d.detect_object(img)
		# print(c)

		#Draw Contours
	for cnt in c:
			#get rect
			#x and y are centre points
		rect = cv2.minAreaRect(cnt)
			# print(rect)
		(x,y),(w,h),angle = rect

			#Get width and height  of the objects by applying  the ratio pixel to cm
		object_w = w/pixel_cm_ratio
		object_h = h/pixel_cm_ratio
				
			#get box
		box = cv2.boxPoints(rect)
		box = np.int0(box)
			# print(box)

			#Draw polygon
		cv2.circle(img,(int(x),int(y)),3,(0,0,255),3)
		cv2.polylines(img,[box],True,(255,255,0),2)
		cv2.putText(img,"Width {} cm".format(round(object_w,1)),(int(x),int(y-15)),cv2.FONT_HERSHEY_COMPLEX,1,(100,23,123),2)
		cv2.putText(img,"Height {} cm".format(round(object_h,1)),(int(x),int(y+15)),cv2.FONT_HERSHEY_COMPLEX,1,(100,23,123),2)

	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime
	cv2.rectangle(img,(5,35),(70,15),(0,0,0),-1)
	cv2.putText(img,f'FPS :{int(fps)}',(10,30),cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)

	cv2.imshow("Image", img)
	cv2.waitKey(1)

	if cv2.waitKey(20) & 0xFF == 27 or cv2.getWindowProperty("Image",0) == -1:
		break
		s
cap.release()