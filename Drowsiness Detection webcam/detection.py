import cv2
import numpy as np
from keras.models import load_model
from pygame import mixer

lbl = ['Closed','Open']
score=0
l_pred=[99]
r_pred =[99]
thick=2

#sound
mixer.init()
sound = mixer.Sound('./alarm.wav')

#model
model = load_model('./models/cnncat2.h5')

#webcam
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

#haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

while(1):
	_,img = cap.read()
	height,width = img.shape[:2]

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	face = face_cascade.detectMultiScale(gray,1.1,5)
	right = left_eye_cascade.detectMultiScale(gray)
	left = right_eye_cascade.detectMultiScale(gray)

	cv2.rectangle(img,(0,height-50),(200,height),(0,0,0),thickness=cv2.FILLED)

	for x,y,w,h in face:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

	for x,y,w,h in left:
		l_eye = img[y:y+h,x:x+w] # getting boundary box of eyes

		#preprocessing the image
		l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
		l_eye = cv2.resize(l_eye,(24,24))
		l_eye = l_eye/255 
		l_eye = l_eye.reshape(24,24,-1)
		l_eye = np.expand_dims(l_eye,axis=0)

		l_pred= model.predict_classes(l_eye)

		if(l_pred[0]==1):
			lbl='Open'
		elif(l_pred[0]==0):
			lbl='Closed' 

		break

	for x,y,w,h in right:
		r_eye = img[y:y+h,x:x+w] #getting boundary box of eyes

		#preprocessing the image
		r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
		r_eye = cv2.resize(r_eye,(24,24))
		r_eye = r_eye/255
		r_eye = r_eye.reshape(24,24,-1)
		r_eye = np.expand_dims(r_eye,axis=0)

		r_pred = model.predict_classes(r_eye)

		if(r_pred[0]==1):
			lbl='Open'
		elif(r_pred[0]==0):
			lbl='Closed'

		break


	if(r_pred[0]==0 and l_pred[0]==0):
		score=score+1
		cv2.putText(img,"Closed",(10,height-20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2,cv2.LINE_AA)
	elif(r_pred[0]==1 or l_pred[0]==1):
		score=score-1 
		cv2.putText(img,"Open",(10,height-20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2,cv2.LINE_AA)

	cv2.putText(img,'Score:'+str(score),(100,height-20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),1,cv2.LINE_AA)

	if(score < 0):
		score = 0

	if(score > 10):
		cv2.imwrite("./img.png",img)
		try:
			sound.play()
		except:
			pass 

		#thickness of the red frame around webcam when person is asleep
		if(thick<16):
			thick=thick+2
		else :
			thick=thick-2
			if(thick<2):
				thick=2

		cv2.rectangle(img,(0,0),(width,height),(0,0,255),thick)

	cv2.imshow("Camera",img);

	if cv2.waitKey(20) & 0xFF == 27 or cv2.getWindowProperty('Camera',0) == -1:
		break;

cap.release()
cv2.destroyAllWindows()