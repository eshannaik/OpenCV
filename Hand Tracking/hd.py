import cv2
import mediapipe as mp 
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mpHands = mp.solutions.hands
hands = mpHands.Hands() # recognize hands
mpDraw = mp.solutions.drawing_utils # drawing the connections

pTime=0
cTime=0

while True:
	_,img = cap.read()
	new_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # RGB as the module works only when RGB is given

	results = hands.process(new_img) #detect the hand

	if results.multi_hand_landmarks: # if there is a hand
		for h in results.multi_hand_landmarks: #taking each hand
			for id_point,lm in enumerate(h.landmark): # id and landmark location on the finger
				height,width = img.shape[:2]
				cx,cy = int(lm.x*width),int(lm.y*height) # getting the pixel location of the landmarks

				if id_point%4==0 and id_point != 0:
					cv2.circle(img,(cx,cy),10,(255,255,0),cv2.FILLED) # drawing a circle around each finger tip

			mpDraw.draw_landmarks(img,h,mpHands.HAND_CONNECTIONS) # drawing the connections between the landmarks

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