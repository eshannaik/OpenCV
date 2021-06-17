import cv2
import argparse
import mediapipe as mp 
import time
import math

class pose_estimation():
	def __init__(self,mode=False,upper_body=False,landmark_smooth=True,detectionCon=0.5,trackCon=0.5):
		self.mode = mode
		self.upper_body = upper_body
		self.landmark_smooth = landmark_smooth
		self.detectionCon = detectionCon
		self.trackCon = trackCon

		self.mpDraw = mp.solutions.drawing_utils
		self.mpPose = mp.solutions.pose
		self.pose = self.mpPose.Pose(self.mode,self.upper_body,self.landmark_smooth,self.detectionCon,self.trackCon)

	def estimation(self,img,draw=True):
		new_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

		self.results = self.pose.process(new_img)

		if self.results.pose_landmarks:
			if draw:
				self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

		return img

	def points(self,img,draw=True):
		self.lmlist=[]
		# points=[17,18,19,20]
		if self.results.pose_landmarks:
			for id_points,p in enumerate(self.results.pose_landmarks.landmark):
				height,width = img.shape[:2]
				cx,cy = int(p.x * width),int(p.y *height)

				#storing the x and y coordinates of the landmarks landmarks
				self.lmlist.append([id_points,cx,cy])

				# if draw:
				# 	if id_points in points:
				# 		cv2.circle(img,(cx,cy),3,(255,0,0),cv2.FILLED)

		return self.lmlist

	def findAngle(self,img,p1,p2,p3,draw=True):
		#getting the x and y coordinates of the input points
		x1,y1 = self.lmlist[p1][1:]
		x2,y2 = self.lmlist[p2][1:]
		x3,y3 = self.lmlist[p3][1:]

		#getting the angle
		angle = math.degrees(math.atan2(y3-y2,x3-x2) - math.atan2(y1-y2,x1-x2))

		if angle<0:
			angle += 360 

		if draw:
			#draw circles on landmarks
			cv2.circle(img,(x1,y1),10,(0,0,255),cv2.FILLED)
			cv2.circle(img,(x1,y1),15,(0,0,255))

			cv2.circle(img,(x2,y2),10,(0,0,255),cv2.FILLED)
			cv2.circle(img,(x2,y2),15,(0,0,255))

			cv2.circle(img,(x3,y3),10,(0,0,255),cv2.FILLED)
			cv2.circle(img,(x3,y3),15,(0,0,255))

			#draw lines between the landmarks
			cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
			cv2.line(img,(x2,y2),(x3,y3),(255,255,255),3)

			#angle of curl
			#cv2.putText(img,f'{int(angle)}',(x2-20,y2+50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)

		return angle

if __name__ == "__main__":
	main()