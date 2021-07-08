import cv2
import mediapipe as mp

class body_size():
	def __init__(self,mode=False,upper_body=False,landmark_smooth=True,detectionCon=0.5,trackCon=0.5):
		self.mode=mode
		self.upper_body=upper_body
		self.landmark_smooth=landmark_smooth
		self.detectionCon=detectionCon
		self.trackCon=trackCon

		self.mpDraw = mp.solutions.drawing_utils
		self.mpPose = mp.solutions.pose
		self.pose = self.mpPose.Pose(self.mode,self.upper_body,self.landmark_smooth,self.detectionCon,self.trackCon)

	def detection_draw(self,img,draw=True):
		new_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		self.results = self.pose.process(new_img)

		if self.results.pose_landmarks:
			if draw:
				self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

		return img

	def landmarks(self,img):
		lmlist=[]

		if self.results.pose_landmarks:
			for id_point,lm in enumerate(self.results.pose_landmarks.landmark):
				height,width=img.shape[:2]
				cx,cy = int(lm.x * width),int(lm.y * height)

				lmlist.append([id_point,cx,cy])

		return lmlist

if __name__ == "__main__":
	main()
