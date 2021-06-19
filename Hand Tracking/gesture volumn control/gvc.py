import cv2
import time
import numpy as np
import math 
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import hd_module as htm

wCam,hCam = 630,480

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3,wCam)
cap.set(4,hCam)

pTime=0
cTime=0

detector = htm.HandDetector(maxHands=1,detectionCon=0.8)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]

volBar = 400
volPer = 0

while True:
	_,img = cap.read()

	img = detector.find_hands(img)
	lmlist = detector.position(img,False)

	if len(lmlist) != 0 :
		# print(lmlist[4],lmlist[8])

		x1,y1 = lmlist[4][1],lmlist[4][2]
		x2,y2 = lmlist[8][1],lmlist[8][2]
		cx,cy = (x1+x2)//2,(y1+y2)//2

		cv2.circle(img,(x1,y1),7,(255,0,255),cv2.FILLED)
		cv2.circle(img,(x2,y2),7,(255,0,255),cv2.FILLED)
		cv2.line(img,(x1,y1),(x2,y2),(255,0,255),2)
		cv2.circle(img,(cx,cy),7,(255,0,255),cv2.FILLED)

		length = math.hypot(x2-x1,y2-y1)
		#print length

		# Converting the range from 50-265 to -65-0
		vol = np.interp(length,[50,150],[minVol,maxVol])
		volBar = np.interp(length,[50,150],[400,150])
		volPer = np.interp(length,[50,150],[0,100])
		# print(vol)
		volume.SetMasterVolumeLevel(vol, None) 

		cv2.rectangle(img,(50,150),(85,400),(255,0,0),3)
		cv2.rectangle(img,(50,int(volBar)),(85,400),(255,0,0),cv2.FILLED)
		cv2.putText(img,f'{int(volPer)} %',(40, 450), cv2.FONT_HERSHEY_COMPLEX,1, (255, 0, 0), 3)

		if length<50:
			cv2.circle(img,(cx,cy),7,(0,255,0),cv2.FILLED)

	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime
	cv2.rectangle(img,(5,465),(65,440),(0,0,0),-1)
	cv2.putText(img,f'FPS :{int(fps)}',(10,460),cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)

	cv2.imshow("Image",img)
	if cv2.waitKey(20) & 0xFF == 27 or cv2.getWindowProperty("Image",0) == -1:
		break

cap.release()
cv2.destroyAllWindows()