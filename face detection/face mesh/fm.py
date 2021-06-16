import cv2
import mediapipe as mp 
import time
import argparse

class face_mesh():

	def __init__(self,mode=False,max_num_faces=2,detectionCon=0.5,trackCon=0.5):
		self.mode = mode
		self.max_num_faces=max_num_faces
		self.detectionCon=detectionCon
		self.trackCon=trackCon

		self.mpdraw = mp.solutions.drawing_utils
		self.mpFaceMesh = mp.solutions.face_mesh
		self.faceMesh = self.mpFaceMesh.FaceMesh(self.mode,self.max_num_faces,self.detectionCon,self.trackCon)
		self.drawSpec = self.mpdraw.DrawingSpec(thickness=2,circle_radius=3)

	def find_face(self,img,draw=True):
		new_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

		self.results = self.faceMesh.process(new_img)

		if self.results.multi_face_landmarks:
			for f in self.results.multi_face_landmarks:
				if draw:
					self.mpdraw.draw_landmarks(img,f,self.mpFaceMesh.FACE_CONNECTIONS,self.drawSpec,self.drawSpec)

		return img

	def find_landmarks(self,img,draw=True):

		lmlist=[]

		if self.results.multi_face_landmarks:
			for face_num,f in enumerate(self.results.multi_face_landmarks):
				for id_point,lm in enumerate(f.landmark):
					height,width = img.shape[:2]
					cx,cy = int(lm.x * width),int(lm.y * height)

					lmlist.append([face_num,id_point,cx,cy])

		return lmlist

if __name__ == "__main__":
	main()