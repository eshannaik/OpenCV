import cv2
import time
import mediapipe as mp

class face_detection():
	def __init__(self,mode=False,max_num_faces=2,detectionCon=0.5,trackCon=0.5):
		self.detectionCon=detectionCon

		self.mp_faceDetection = mp.solutions.face_detection
		self.fD = self.mp_faceDetection.FaceDetection(self.detectionCon)
		self.mp_draw = mp.solutions.drawing_utils

	def find_face(self,img,draw=True):
		new_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

		self.results = self.fD.process(new_image)

		if self.results.detections:
			for f in self.results.detections:
				if draw:
					self.mp_draw.draw_detection(img,f)

		return img

	def find_landmarks(self,img):
		lmlist = []

		if self.results.detections:
			# for face_num,f in enumerate(self.results.detections):
			for id_point,lm in enumerate(self.results.detections):
				bboxc = lm.location_data.relative_bounding_box
				h,w = img.shape[:2]
				bbox = int(bboxc.xmin * w),int(bboxc.ymin * h),int(bboxc.width * w),int(bboxc.height * h)

				cv2.rectangle(img,bbox,(255,0,255),2)
				cv2.putText(img,f'{int(lm.score[0] * 100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)

				lmlist.append([id_point,bbox,lm.score])

		return img,lmlist

def main():
	wCam = 640
	hCam = 480

	import fd as fdm

	cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
	cap.set(3,wCam)
	cap.set(4,hCam)

	f = fdm.face_detection(detectionCon=0.7)

	cTime=0
	pTime=0

	while True:
		_,img = cap.read()

		img = cv2.flip(img,1)

		img = f.find_face(img,False)
		img,lmlist = f.find_landmarks(img)

		cTime = time.time()
		fps = 1/(cTime-pTime)
		pTime = cTime
		cv2.rectangle(img,(5,465),(70,440),(0,0,0),-1)
		cv2.putText(img,f'FPS :{int(fps)}',(10,460),cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)

		cv2.imshow("Image",img)
		
		if cv2.waitKey(20) & 0xFF == 27 or cv2.getWindowProperty("Image",0) == -1:
			break

	cap.release()


if __name__ == "__main__":
	main()