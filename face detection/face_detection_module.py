import cv2
import time
import mediapipe as mp
import argparse

class detection():

	def face_image(self,image):
		face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
		eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
		smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
		
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray,1.1,10)
		eyes = eye_cascade.detectMultiScale(gray,1.1,10)
		smile = smile_cascade.detectMultiScale(gray,1.1,100)

		for x,y,w,h in faces:
			cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.putText(image,'Face',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

		for x,y,w,h in eyes:
			cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
			cv2.putText(image,'Eyes',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

		for x,y,w,h in smile:
			cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
			cv2.putText(image,'Smile',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

		cv2.imshow('Image',image)
		cv2.waitKey()

	def camera_video_face(self):
		cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
		face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
		eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
		smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

		cTime=0
		pTime=0

		while 1 :	
			_,img = cap.read()
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			face = face_cascade.detectMultiScale(gray,1.1,10)
			eyes = eye_cascade.detectMultiScale(gray,1.1,10)
			smile = smile_cascade.detectMultiScale(gray,1.1,100)

			cTime = time.time()
			fps=1/(cTime-pTime)
			pTime=cTime

			for x,y,w,h in face:
				cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
				cv2.putText(img,'Face',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

			for x,y,w,h in eyes:
				cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
				cv2.putText(img,'Eyes',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

			for x,y,w,h in smile:
				cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
				cv2.putText(img,'Smile',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

			cv2.putText(img,f'FPS : {int(fps)}',(10,35),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(255,0,255),2)

			cv2.imshow('Camera',img)

			if cv2.waitKey(20) & 0xFF == 27 or cv2.getWindowProperty('Camera', 0) == -1:
				break

		cap.release()

	def video_face(self,cap):
		face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
		eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
		smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

		cTime=0
		pTime=0

		while True:	
			_,img = cap.read()
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			face = face_cascade.detectMultiScale(gray,1.1,4)
			eyes = eye_cascade.detectMultiScale(gray,1.1,90)
			smile = smile_cascade.detectMultiScale(gray,1.1,130)

			cTime = time.time()
			fps=1/(cTime-pTime)
			pTime=cTime

			for x,y,w,h in face:
				cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
				cv2.line(img,(x,y))

			for x,y,w,h in eyes:
				cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
				cv2.putText(img,'Eyes',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

			for x,y,w,h in smile:
				cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
				cv2.putText(img,'Smile',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

			cv2.putText(img,f'FPS : {int(fps)}',(10,35),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(255,0,255),2)

			cv2.imshow('Video',img)	

			if cv2.waitKey(30) & 0xff == 27 or cv2.getWindowProperty('Video', 0) == -1:
				break

		cap.release()

def main():
	n=0

	print("To recognize the faces in an image input image")
	print("To recognize the faces in an video input video")
	print("To recognize the faces from your webcam input camera")

	# ap=argparse.ArgumentParser()
	# ap.add_argument("-i","--image",help="Image Path")
	# ap.add_argument("-c",help="Webcam")
	# ap.add_argument("-v","--video",help="Video Path")
	# args=vars(ap.parse_args())

	d = detection()
	
	while n==0 :
		a = input("Find faces in an image or video or webcam (image/video/camera) : \n")

		if a=="image":
			print("Write the name as ./____ if the image is in the same folder")
			image = input("Input Image file path :")

			i = cv2.imread(image,1)
			# i = args["image"]

			d.face_image(i)
			n=1

		elif a=="camera":
			d.camera_video_face()
			n=1

		elif a=="video":
			print("Write the name as ./____ if the video is in the same folder")
			video = input("Input Video file path (mp4):")

			cap = cv2.VideoCapture(video)
			# cap = cv2.VideoCapture(args["video"])

			d.video_face(cap)
			n=1

		else:
			print("Please enter either image or video or camera\n")

if __name__ == "__main__":
	main()







