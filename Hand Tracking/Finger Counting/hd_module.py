import cv2
import mediapipe as mp 
import time


class HandDetector():
	def __init__ (self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
		self.mode= mode
		self.maxHands = maxHands
		self.detectionCon = detectionCon
		self.trackCon = trackCon

		self.mpHands = mp.solutions.hands
		self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon) # recognize hands
		self.mpDraw = mp.solutions.drawing_utils # drawing the connections


	def find_hands(self,img,draw=True): # detects the hands and draws a connection between all landmarks
		new_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # RGB as the module works only when RGB is given

		self.results = self.hands.process(new_img) #detect the hand

		if self.results.multi_hand_landmarks: # if there is a hand
			for h in self.results.multi_hand_landmarks: #taking each hand
				if draw:
					self.mpDraw.draw_landmarks(img,h,self.mpHands.HAND_CONNECTIONS) # drawing the connections between the landmarks

		return img


	def position(self,img,draw=True): # finger tips

		lmlist = []

		if self.results.multi_hand_landmarks:

			for h in self.results.multi_hand_landmarks:
				for id_point,lm in enumerate(h.landmark): # id and landmark location on the finger
						height,width = img.shape[:2]
						cx,cy = int(lm.x*width),int(lm.y*height) # getting the pixel location of the landmarks

						lmlist.append([id_point,cx,cy])

						if draw:
							if id_point%4==0 and id_point != 0:
								cv2.circle(img,(cx,cy),10,(255,255,0),cv2.FILLED) # drawing a circle around each finger tip

		return lmlist

if __name__ == "__main__":
	main()