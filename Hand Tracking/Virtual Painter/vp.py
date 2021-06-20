import cv2
import time
import os
import numpy as np

import hd_module as hdm

wCam,hCam = 640,480
brushThickness = 8
eraserThickness = 60

pTime=0
cTime=0

d=hdm.HandDetector(detectionCon=0.75)

folderPath = "./Design"
myList = os.listdir(folderPath)

overlayList=[]
for i in myList:
	image = cv2.imread(f'{folderPath}/{i}')
	overlayList.append(image)

# print(len(overlayList))

header = overlayList[3]
drawColor = (0,140,255)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(3,wCam)
cap.set(4,hCam)

xp=0
yp=0

imgCanvas = np.zeros((480,640,3),np.uint8)

while True:
	_,img = cap.read()
	img = cv2.flip(img,1)

	img = d.find_hands(img)
	lmlist = d.position(img,False)

	if len(lmlist) != 0:
		x1,y1 = lmlist[8][1:]
		x2,y2 = lmlist[12][1:]
		# print(lmlist)

		fingers=d.fingersUp()
		# print(fingers)

		if fingers[1] and fingers[2]:
			xp,yp = 0,0

			if y1 < 125:
				if 0 < x1 < 110:
					header = overlayList[3]
					drawColor = (0,140,255) #orange
				elif 110 < x2 < 220:
					header = overlayList[4]
					drawColor = (0,0,255) #red
				elif 220 < x2 < 330:
					header = overlayList[0]
					drawColor = (255,0,0) # blue
				elif 330 < x2 < 420:
					header = overlayList[2]
					drawColor = (0,128,0) #green
				elif 420 < x2 < 520:
					header = overlayList[5]
					drawColor = (88,85,80) #grey
				elif 520 < x2 < 640:
					header = overlayList[1]
					drawColor = (0,0,0) #eraser

			cv2.rectangle(img,(x1,y1-10),(x2,y2+10),(255,0,255),cv2.FILLED)

		if fingers[1] and fingers[2]==False:
			cv2.circle(img,(x1,y1),8,drawColor,cv2.FILLED)

			if xp==0 and yp==0:
				xp,yp=x1,y1

			if drawColor==(0,0,0):
				cv2.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
				cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)
			else:
				cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
				cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)

			xp,yp=x1,y1

		# if all (x>=1 for x in fingers):
		# 	imgCanvas = np.zeros((480,640,3),np.uint8)

	imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
	_, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV) #inverts the image
	imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
	img = cv2.bitwise_and(img,imgInv) # brings it back to color
	img = cv2.bitwise_or(img,imgCanvas) # add the two images 

	img[0:125,0:640]=header

	# cTime = time.time()
	# fps = 1/(cTime-pTime)
	# pTime = cTime
	# cv2.rectangle(img,(5,465),(70,440),(0,0,0),-1)
	# cv2.putText(img,f'FPS :{int(fps)}',(10,460),cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)

	# img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)

	cv2.imshow("Image",img)

	if cv2.waitKey(20) & 0xFF==27 or cv2.getWindowProperty("Image",0)==-1:
		break

cap.release()
cv2.destroyAllWindows()