import cv2
import mediapipe
import time
import os

import hd_module as hdm

cTime=0
pTime=0

d = hdm.HandDetector(detectionCon=0.75)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

folder_path = "images"
myList = os.listdir(folder_path)

#Getting the images
overlayList =[]
for imPath in myList:
	image = cv2.imread(f'{folder_path}/{imPath}')
	image = cv2.resize(image,(150,150))
	overlayList.append(image)

print(myList)

fingerTip = [4,8,12,16,20]

while 1:
	_,img = cap.read()

	img = d.find_hands(img)
	lmlist = d.position(img,draw=False)

	if len(lmlist) != 0:
		finger = []

		#Detection how many fingers are open
		#Thumb
		if lmlist[fingerTip[0]][1] < lmlist[fingerTip[0]-1][1]: # for left hand make it <
			finger.append(1)
		else:
			finger.append(0)

		#4 Fingers
		for id_point in range(1,5):
			if lmlist[fingerTip[id_point]][2] < lmlist[fingerTip[id_point]-2][2]:
				finger.append(1)
			else:
				finger.append(0)

		totalFingers= finger.count(1)
		# print(totalFingers)

		#Using images to put the overlay
		img[0:150,0:150] = overlayList[totalFingers-1] 

		#Using putText to make overlay
		cv2.rectangle(img,(5,325),(85,250),(0,255,0),-1)
		cv2.putText(img,f'{int(totalFingers)}',(15,320),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),2)

	#fps
	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime
	cv2.rectangle(img,(5,465),(65,440),(0,0,0),-1)
	cv2.putText(img,f'FPS :{int(fps)}',(10,460),cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)

	cv2.imshow("WebCame",img)

	if cv2.waitKey(20) & 0xFF==27 or cv2.getWindowProperty("WebCame",0) == -1:
		break

cap.release()
cv2.destroyAllWindows()