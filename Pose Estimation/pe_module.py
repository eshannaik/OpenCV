import cv2
import argparse
import mediapipe as mp 
import time

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
		self.results = self.pose.process(img)

		if self.results.pose_landmarks:
			# for p in self.results.pose_landmarks:
			if draw:
				self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

		return img

	def points(self,img,draw=True):
		lmlist=[]
		points=[17,18,19,20]
		if self.results.pose_landmarks:
			for id_points,p in enumerate(self.results.pose_landmarks.landmark):
				height,width = img.shape[:2]
				cx,cy = int(p.x * width),int(p.y *height)

				lmlist.append([id_points,cx,cy])

				if draw:
					if id_points in points:
						cv2.circle(img,(cx,cy),3,(255,0,0),cv2.FILLED)

		return lmlist

if __name__ == "__main__":
	main()