import cv2
import time
import argparse
import bs as b
import math

wCam,hCam = 1024,720

p = b.body_size()

#Image
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Image location")
args = vars(ap.parse_args())
img_path = args['image']
img = cv2.imread(img_path)

# Video camera
# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# cap.set(3,wCam)
# cap.set(4,hCam)

cTime=0
pTime=0

#Image
img = p.detection_draw(img,False)

lmlist = p.landmarks(img)

img = cv2.resize(img,(720,720))

if len(lmlist)!= 0:
	# print(lmlist[0])

	#Shoulder
	l_shoulder_x,l_shoulder_y = lmlist[11][1],lmlist[11][2]
	r_shoulder_x,r_shoulder_y = lmlist[12][1],lmlist[12][2]

	shoulder_y = (l_shoulder_y - r_shoulder_y)
	shoulder_x = (l_shoulder_x - r_shoulder_x)

	#Waist
	l_waist_x,l_waist_y = lmlist[23][1],lmlist[23][2]
	r_waist_x,r_waist_y = lmlist[24][1],lmlist[24][2]

	waist_x = (l_waist_y - r_waist_y)
	waist_y = (l_waist_x - r_waist_x)

	cv2.line(img,(r_shoulder_x,r_shoulder_y),(l_shoulder_x,l_shoulder_y),(255,0,255),2)

	print("ratio :",f'{format((shoulder_x/waist_x),".1f")}')

	#Index Finger
	r_index_x,r_index_y = lmlist[20][1],lmlist[20][2]
	l_index_x,l_index_y = lmlist[19][1],lmlist[19][2]

	#arm length
	r_arm_x = r_shoulder_x - r_index_x
	r_arm_y = r_shoulder_y - r_index_y
	r_arm = math.sqrt((r_arm_x*r_arm_x) + (r_arm_y*r_arm_y))

	l_arm_x = r_shoulder_x - r_index_x
	l_arm_y = r_shoulder_y - r_index_y
	l_arm = math.sqrt((l_arm_x*l_arm_x) + (l_arm_y*l_arm_y))

	print("ratio (left arm/right arm) (1 if arms are'nt straight or seen) :",f'{format((l_arm/r_arm),".1f")}')

	#Foot 
	r_foot_x,r_foot_y = lmlist[30][1],lmlist[30][2]
	l_foot_x,l_foot_y = lmlist[29][1],lmlist[29][2]

	#Leg Length
	r_leg_x = r_waist_x - r_foot_x
	r_leg_y = r_foot_y - r_waist_y
	r_leg = math.sqrt((r_leg_x*r_leg_x) + (r_leg_y*r_leg_y))

	l_leg_x = r_waist_x - r_foot_x
	l_leg_y = r_foot_y - r_waist_y
	l_leg = math.sqrt((l_leg_x*l_leg_x) + (l_leg_y*l_leg_y))

	print("ratio (left leg/right leg) (1 if legs are'nt straight or seen) :",f'{format((l_leg/r_leg),".1f")}')

	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime
	cv2.rectangle(img,(5,40),(150,5),(0,0,0),-1)
	cv2.putText(img,f'FPS : {int(fps)}',(10,35),cv2.FONT_HERSHEY_PLAIN,2,(255,255,0),2)

	cv2.imshow("Image",img)
	cv2.waitKey(0)

	if cv2.waitKey(20) & 0xFF == 27 or cv2.getWindowProperty("Image",0)==-1:
		cv2.destroyAllWindows()

# Camera
# while 1:
# 	_,img = cap.read()

# 	img = p.detection_draw(img,False)

# 	lmlist = p.landmarks(img)

# 	if len(lmlist)!= 0:
# 		# print(lmlist[0])

# 		# if lmlist[11]==11:
# 		l_shoulder_x,l_shoulder_y = lmlist[11][1],lmlist[11][2]

# 		# if lmlist[12]==12:
# 		r_shoulder_x,r_shoulder_y = lmlist[12][1],lmlist[12][2]

# 		l_waist_x,l_waist_y = lmlist[23][1],lmlist[23][2]

# 		r_waist_x,r_waist_y = lmlist[24][1],lmlist[24][2]

# 		shoulder_y = (l_shoulder_y - r_shoulder_y)
# 		shoulder_x = (l_shoulder_x - r_shoulder_x)

# 		waist_x = (l_waist_y - r_waist_y)
# 		waist_y = (l_waist_x - r_waist_x)

# 		# print(shoulder_x,"inches")
# 		# print(waist_x,"inches")
# 		print("ratio :",f'{format(shoulder_x/waist_x,".1f")}')

# 	cTime = time.time()
# 	fps = 1/(cTime-pTime)
# 	pTime = cTime
# 	cv2.rectangle(img,(5,40),(150,5),(0,0,0),-1)
# 	cv2.putText(img,f'FPS : {int(fps)}',(10,35),cv2.FONT_HERSHEY_PLAIN,2,(255,255,0),2)

# 	cv2.imshow("Image",img)

# 	if cv2.waitKey(20) & 0xFF == 27 or cv2.getWindowProperty("Image",0)==-1:
# 		break

# cap.release()

